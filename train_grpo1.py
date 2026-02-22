import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
    SkipBatchSampler, init_model

warnings.filterwarnings('ignore')
import math

class DynamicAdvantageScaler:

    def __init__(self,
                 clip_range=(-10, 10),
                 momentum=0.9,
                 warmup_steps=100,
                 min_std_threshold=0.01,
                 max_std_threshold=5.0):
        self.clip_min, self.clip_max = clip_range
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold

        # 运行统计量
        self.step_count = 0
        self.running_mean = 0.0
        self.running_std = 1.0
        self.running_reward_mean = 0.0
        self.running_reward_std = 1.0

        # 训练进度相关
        self.training_progress = 0.0
        self.last_reward_stability = 1.0

    def update_running_statistics(self, rewards, advantages):

        if self.step_count < self.warmup_steps:

            self.running_mean = advantages.mean().item()
            self.running_std = max(advantages.std().item(), self.min_std_threshold)
            self.running_reward_mean = rewards.mean().item()
            self.running_reward_std = max(rewards.std().item(), self.min_std_threshold)
        else:

            alpha = 1.0 - self.momentum
            self.running_mean = self.momentum * self.running_mean + alpha * advantages.mean().item()
            self.running_std = self.momentum * self.running_std + alpha * max(advantages.std().item(),
                                                                              self.min_std_threshold)
            self.running_reward_mean = self.momentum * self.running_reward_mean + alpha * rewards.mean().item()
            self.running_reward_std = self.momentum * self.running_reward_std + alpha * max(rewards.std().item(),
                                                                                            self.min_std_threshold)

        self.running_std = max(self.min_std_threshold, min(self.running_std, self.max_std_threshold))
        self.running_reward_std = max(self.min_std_threshold, min(self.running_reward_std, self.max_std_threshold))

        self.step_count += 1
        self.last_reward_stability = 1.0 / (self.running_reward_std + 1e-8)

        return self.running_mean, self.running_std

    def compute_dynamic_advantages(self, rewards, grouped_rewards, current_step, total_steps):

        batch_size, num_gen = grouped_rewards.shape


        self.training_progress = min(1.0, current_step / max(total_steps, 1))


        mean_r = grouped_rewards.mean(dim=1, keepdim=True).repeat(1, num_gen).flatten()
        std_r = grouped_rewards.std(dim=1, keepdim=True).repeat(1, num_gen).flatten()


        std_r = torch.clamp(std_r, min=self.min_std_threshold, max=self.max_std_threshold)


        if self.step_count > self.warmup_steps:

            stability_factor = torch.sigmoid(torch.tensor(self.last_reward_stability * 0.1))


            dynamic_epsilon = 0.1 * (1.0 - self.training_progress) + 0.001 * self.training_progress

            if stability_factor > 0.7:
                epsilon = 1e-4
            else:
                epsilon = dynamic_epsilon


            advantages = (rewards - mean_r) / (std_r + epsilon)


            early_train_factor = 1.0 - min(1.0, self.step_count / 1000.0)
            dynamic_clip_max = self.clip_max * (0.3 + 0.7 * early_train_factor)
            dynamic_clip_min = self.clip_min * (0.3 + 0.7 * early_train_factor)

            advantages = torch.clamp(advantages, dynamic_clip_min, dynamic_clip_max)
        else:

            advantages = (rewards - mean_r) / (std_r + 1e-4)
            advantages = torch.clamp(advantages, self.clip_min, self.clip_max)


        if self.step_count > self.warmup_steps and self.running_std > 1e-8:

            blend_factor = 0.7
            global_mean = blend_factor * self.running_mean + (1 - blend_factor) * advantages.mean().item()
            global_std = blend_factor * self.running_std + (1 - blend_factor) * max(advantages.std().item(),
                                                                                    self.min_std_threshold)

            advantages = (advantages - global_mean) / (global_std + 1e-8)
        else:

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        advantages = torch.clamp(advantages, -5.0, 5.0)


        self.update_running_statistics(rewards, advantages)

        return advantages

    def get_metrics(self):

        return {
            'adv_mean': self.running_mean,
            'adv_std': self.running_std,
            'reward_mean': self.running_reward_mean,
            'reward_std': self.running_reward_std,
            'training_progress': self.training_progress,
            'reward_stability': self.last_reward_stability
        }
class AdaptiveGradientClipper:
    def __init__(self,
                 max_norm=1.0,
                 norm_type=2.0,
                 growth_factor=1.1,
                 shrink_factor=0.9,
                 window_size=100,
                 tolerance=0.1,
                 min_norm=0.01,
                 max_norm_limit=10.0):

        self.initial_max_norm = max_norm
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.window_size = window_size
        self.tolerance = tolerance
        self.min_norm = min_norm
        self.max_norm_limit = max_norm_limit

        # 历史记录
        self.gradient_norms = []
        self.clip_counts = []
        self.step_count = 0
        self.total_clips = 0

        # 统计信息
        self.avg_grad_norm = 0.0
        self.grad_norm_std = 0.0
        self.clip_frequency = 0.0

    def compute_gradient_norm(self, parameters):
        """计算梯度范数"""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]

        if len(parameters) == 0:
            return torch.tensor(0.0)

        if self.norm_type == float('inf'):
            total_norm = max(p.grad.detach().abs().max() for p in parameters)
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), self.norm_type) for p in parameters]),
                self.norm_type
            )

        return total_norm

    def adaptive_clip_grad_norm_(self, parameters, current_loss=None):

        if self.max_norm <= 0:
            return torch.tensor(0.0)

        # 1. 计算当前梯度范数
        total_norm = self.compute_gradient_norm(parameters)
        original_norm = total_norm.item()

        # 2. 记录历史
        self.gradient_norms.append(original_norm)
        if len(self.gradient_norms) > self.window_size:
            self.gradient_norms.pop(0)

        # 3. 计算统计信息
        if len(self.gradient_norms) >= 10:
            recent_norms = self.gradient_norms[-10:]
            self.avg_grad_norm = sum(recent_norms) / len(recent_norms)
            self.grad_norm_std = math.sqrt(sum((n - self.avg_grad_norm) ** 2 for n in recent_norms) / len(recent_norms))

        # 4. 动态调整裁剪阈值
        if len(self.gradient_norms) >= 10:
            # 检测梯度爆炸
            if original_norm > self.max_norm * 3.0:
                new_norm = self.max_norm * self.shrink_factor
                self.max_norm = max(self.min_norm, min(new_norm, self.max_norm_limit))
                Logger(f'[梯度爆炸] 检测到梯度范数 {original_norm:.4f} > {self.max_norm * 3:.4f}, '
                       f'降低阈值至 {self.max_norm:.4f}')

            # 检测梯度消失
            elif original_norm < self.max_norm * 0.1 and self.grad_norm_std < 0.01:
                new_norm = self.max_norm * self.growth_factor
                self.max_norm = max(self.min_norm, min(new_norm, self.max_norm_limit))
                Logger(f'[梯度消失] 检测到梯度范数 {original_norm:.4f} < {self.max_norm * 0.1:.4f}, '
                       f'提高阈值至 {self.max_norm:.4f}')

            # 基于裁剪频率调整
            elif self.step_count % 50 == 0 and len(self.clip_counts) >= 50:
                recent_clips = self.clip_counts[-50:]
                self.clip_frequency = sum(recent_clips) / len(recent_clips)

                if self.clip_frequency > 0.3:  # 裁剪过多
                    self.max_norm *= 1.05
                    Logger(f'[裁剪过多] 频率 {self.clip_frequency:.3f} > 0.3, '
                           f'提高阈值至 {self.max_norm:.4f}')
                elif self.clip_frequency < 0.05:  # 裁剪过少
                    self.max_norm *= 0.95
                    Logger(f'[裁剪过少] 频率 {self.clip_frequency:.3f} < 0.05, '
                           f'降低阈值至 {self.max_norm:.4f}')

                # 确保在合理范围内
                self.max_norm = max(self.min_norm, min(self.max_norm, self.max_norm_limit))

        # 5. 执行裁剪
        clip_occurred = 0
        if original_norm > self.max_norm:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, norm_type=self.norm_type)
            clip_occurred = 1
            self.total_clips += 1

        self.clip_counts.append(clip_occurred)
        if len(self.clip_counts) > self.window_size:
            self.clip_counts.pop(0)

        self.step_count += 1
        return total_norm

    def get_metrics(self):
        """获取裁剪器指标"""
        return {
            'current_max_norm': self.max_norm,
            'avg_grad_norm': self.avg_grad_norm,
            'grad_norm_std': self.grad_norm_std,
            'clip_frequency': self.clip_frequency,
            'total_clips': self.total_clips,
            'step_count': self.step_count
        }

    def reset(self):
        """重置裁剪器状态"""
        self.max_norm = self.initial_max_norm
        self.gradient_norms.clear()
        self.clip_counts.clear()
        self.step_count = 0
        self.total_clips = 0
        self.avg_grad_norm = 0.0
        self.grad_norm_std = 0.0
        self.clip_frequency = 0.0

class EnhancedRewardCalculator:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.weights = {
            'format': {'initial': 1.0, 'current': 1.0, 'decay': 0.999},
            'mark': {'initial': 1.0, 'current': 1.0, 'decay': 0.998},
            'model': {'initial': 1.0, 'current': 1.0, 'decay': 0.995},
            'answer': {'initial': 0.6 if args.reasoning == 1 else 0.0,
                       'current': 0.6 if args.reasoning == 1 else 0.0,
                       'decay': 0.99}
        }

        self.logic_indicators = {
            'sequential': ['首先', '其次', '然后', '接着', '最后', '第一步', '第二步', '第三步'],
            'causal': ['因为', '所以', '因此', '因而', '于是', '从而', '导致'],
            'contrast': ['但是', '然而', '尽管', '虽然', '反之', '相反'],
            'conclusion': ['总之', '综上所述', '总而言之', '总的来说', '因此可得']
        }

        self.reward_stats = {}
        for key in self.weights.keys():
            self.reward_stats[key] = {
                'mean': 0.0,
                'std': 1.0,
                'count': 0,
                'min': float('inf'),
                'max': float('-inf')
            }

        self.total_samples = 0

    def _calculate_enhanced_format_reward(self, response):
        patterns = [
            r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$",
            r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        ]

        format_score = 0.0

        # 基础格式匹配
        for pattern in patterns:
            if re.match(pattern, response, re.S):
                format_score += 0.3
                break

        # 结构完整性检查
        if "<think>" in response and "</think>" in response and "<answer>" in response and "</answer>" in response:
            format_score += 0.2

            # 检查标记顺序
            try:
                think_start = response.index("<think>")
                think_end = response.index("</think>")
                answer_start = response.index("<answer>")
                answer_end = response.index("</answer>")

                if think_start < think_end < answer_start < answer_end:
                    format_score += 0.2
            except ValueError:
                pass

        return min(1.0, format_score)

    def _calculate_logic_coherence(self, response):
        """计算逻辑连贯性分数"""
        logic_score = 0.0

        think_match = re.search(r"<think>\n(.*?)\n</think>", response, re.S)
        if not think_match:
            return 0.0

        thinking_text = think_match.group(1)

        for category, indicators in self.logic_indicators.items():
            count = sum(1 for indicator in indicators if indicator in thinking_text)
            if count > 0:
                logic_score += min(0.1, count * 0.02)


        sentences = re.split(r'[。！？；]', thinking_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 3:
            logic_score += 0.1
            if len(sentences) >= 5:
                logic_score += 0.1


        words = thinking_text.split()
        if len(words) > 0:
            word_freq = {}
            for word in words:
                if len(word) > 1:
                    word_freq[word] = word_freq.get(word, 0) + 1
            total_words = len(words)
            repeated_words = sum(1 for count in word_freq.values() if count > 3)
            repetition_ratio = repeated_words / total_words if total_words > 0 else 0

            if repetition_ratio > 0.3:
                logic_score -= 0.1 * repetition_ratio

        return max(0.0, min(0.5, logic_score))

    def _calculate_mark_reward(self, response):
        reward = 0.0
        markers = [
            ("<think>", 0.25, 0.1),
            ("</think>", 0.25, 0.1),
            ("<answer>", 0.25, 0.1),
            ("</answer>", 0.25, 0.1)
        ]

        for marker, correct_reward, repeat_penalty in markers:
            count = response.count(marker)
            if count == 1:
                reward += correct_reward
            elif count > 1:
                reward -= repeat_penalty * (count - 1)

        return max(0.0, reward)

    def _calculate_answer_quality(self, response):
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if not answer_match:
            return 0.0

        answer = answer_match.group(1).strip()

        if not answer or len(answer) < 2:
            return 0.0

        quality_score = 0.0


        quality_score += 0.1

        if any(char.isdigit() for char in answer):
            quality_score += 0.1


        if len(answer) > 10:
            quality_score += 0.1

        if '.' in answer or '。' in answer or '!' in answer or '？' in answer:
            quality_score += 0.1

        return min(0.4, quality_score)

    def _calculate_novelty_reward(self, response, response_history):
        if len(response_history) < 5:
            return 0.0

        recent_responses = response_history[-5:]

        current_words = set(response.lower().split())
        similarities = []

        for hist_response in recent_responses:
            hist_words = set(hist_response.lower().split())
            if current_words and hist_words:
                similarity = len(current_words.intersection(hist_words)) / len(current_words.union(hist_words))
                similarities.append(similarity)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)

            novelty = 1.0 - avg_similarity
            return max(0.0, min(0.2, novelty * 0.2))

        return 0.0

    def _update_reward_stats(self, reward_type, value):
        stats = self.reward_stats[reward_type]
        old_count = stats['count']
        stats['count'] += 1


        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)


        old_mean = stats['mean']
        stats['mean'] = old_mean + (value - old_mean) / stats['count']

        if stats['count'] > 1:

            old_M2 = stats['std'] * max(1, old_count - 1)
            delta = value - old_mean
            delta2 = value - stats['mean']
            new_M2 = old_M2 + delta * delta2
            stats['std'] = math.sqrt(new_M2 / max(1, stats['count'] - 1))

    def _normalize_reward(self, reward_type, value, use_robust=True):

        stats = self.reward_stats[reward_type]

        if stats['count'] < 10 or stats['std'] < 1e-8:
            return value

        if use_robust:

            median = stats['mean']
            iqr = stats['std'] * 1.349
            if iqr > 1e-8:
                return (value - median) / iqr
            else:
                return (value - median) / (stats['std'] + 1e-8)
        else:

            return (value - stats['mean']) / (stats['std'] + 1e-8)

    def _update_dynamic_weights(self, training_progress):
         for key, weight_info in self.weights.items():

            if key in ['format', 'mark']:

                decay_factor = weight_info['decay'] ** self.total_samples
                weight_info['current'] = weight_info['initial'] * (0.3 + 0.7 * decay_factor)
            else:
                growth_factor = min(2.0, 1.0 + training_progress)
                weight_info['current'] = weight_info['initial'] * growth_factor

    def calculate_enhanced_rewards(self, prompts, responses, reward_model, reward_tokenizer, response_history=None):

        if response_history is None:
            response_history = []

        batch_size = len(prompts)
        num_responses = len(responses)

        # 初始化奖励张量
        total_rewards = torch.zeros(num_responses, device=self.device)
        reward_components = {
            'format': torch.zeros(num_responses, device=self.device),
            'logic': torch.zeros(num_responses, device=self.device),
            'mark': torch.zeros(num_responses, device=self.device),
            'answer': torch.zeros(num_responses, device=self.device),
            'novelty': torch.zeros(num_responses, device=self.device),
            'model': torch.zeros(num_responses, device=self.device)
        }

        # 1. 计算组件奖励
        for i, response in enumerate(responses):
            # 格式奖励
            format_reward = self._calculate_enhanced_format_reward(response)
            reward_components['format'][i] = format_reward

            # 逻辑连贯性奖励
            logic_reward = self._calculate_logic_coherence(response)
            reward_components['logic'][i] = logic_reward

            # 标记奖励
            mark_reward = self._calculate_mark_reward(response)
            reward_components['mark'][i] = mark_reward

            # 答案质量奖励
            answer_reward = self._calculate_answer_quality(response)
            reward_components['answer'][i] = answer_reward

            # 新颖性奖励
            novelty_reward = self._calculate_novelty_reward(response, response_history)
            reward_components['novelty'][i] = novelty_reward

            # 更新历史
            response_history.append(response)
            if len(response_history) > 100:  # 保持历史窗口大小
                response_history.pop(0)

            # 更新统计
            self._update_reward_stats('format', format_reward)
            self._update_reward_stats('mark', mark_reward)
            self._update_reward_stats('answer', answer_reward)

        self.total_samples += num_responses

        # 2. 奖励模型评分
        with torch.no_grad():
            scale = 3.0

            for i in range(batch_size):
                for j in range(self.args.num_generations):
                    response_idx = i * self.args.num_generations + j
                    response = responses[response_idx]
                    prompt = prompts[i]

                    # 提取对话历史
                    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                    matches = re.findall(pattern, prompt, re.DOTALL)
                    messages = [{"role": role, "content": content.strip()} for role, content in matches]

                    # 完整回答评分
                    tmp_chat = messages + [{"role": "assistant", "content": response}]
                    try:
                        score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        score = max(min(score, scale), -scale)
                    except Exception as e:
                        Logger(f"奖励模型评分失败: {e}")
                        score = 0.0

                    reward_components['model'][response_idx] = score
                    self._update_reward_stats('model', score)

                    # 如果是推理模型，额外评估答案部分
                    if self.args.reasoning == 1:
                        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                        if answer_match:
                            answer_content = answer_match.group(1).strip()
                            if answer_content:  # 确保答案非空
                                tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                                try:
                                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                                    answer_score = max(min(answer_score, scale), -scale)
                                except Exception as e:
                                    Logger(f"答案部分评分失败: {e}")
                                    answer_score = 0.0

                                # 动态加权：训练后期更关注答案质量
                                training_progress = min(1.0, self.total_samples / 10000.0)
                                self._update_dynamic_weights(training_progress)

                                answer_weight = self.weights['answer']['current']
                                blended_score = score * (1.0 - answer_weight) + answer_score * answer_weight

                                reward_components['model'][response_idx] = blended_score


        training_progress = min(1.0, self.total_samples / 10000.0)
        self._update_dynamic_weights(training_progress)

        for comp_name in reward_components:
            if comp_name in self.weights:
                weight = self.weights[comp_name]['current']
            elif comp_name == 'logic':

                weight = 0.5 + 0.5 * training_progress
            elif comp_name == 'novelty':

                weight = 0.1 + 0.2 * training_progress
            else:
                weight = 1.0


            comp_rewards = reward_components[comp_name]
            if self.reward_stats.get(comp_name, {}).get('count', 0) > 10:
                normalized = self._normalize_reward(comp_name, comp_rewards)
            else:
                normalized = comp_rewards


            total_rewards += normalized * weight

        return total_rewards, reward_components

    def get_reward_statistics(self):

        stats = {}
        for comp_name, comp_stats in self.reward_stats.items():
            if comp_stats['count'] > 0:
                stats[comp_name] = {
                    'mean': comp_stats['mean'],
                    'std': comp_stats['std'],
                    'count': comp_stats['count'],
                    'min': comp_stats['min'],
                    'max': comp_stats['max']
                }
        return stats

    def get_current_weights(self):

        weights = {}
        for comp_name, weight_info in self.weights.items():
            weights[comp_name] = weight_info['current']
        return weights



def adaptive_beta_scheduler(current_step, total_steps, initial_beta=0.02,
                            min_beta=0.005, max_beta=0.1, warmup_ratio=0.1):

    progress = current_step / total_steps

    if progress < warmup_ratio:
        # 预热阶段：线性增加到初始值
        warmup_progress = progress / warmup_ratio
        return initial_beta * warmup_progress
    else:
        # 训练阶段：根据进度调整
        train_progress = (progress - warmup_ratio) / (1 - warmup_ratio)

        if train_progress < 0.3:
            # 前期：保持较高beta以稳定训练
            return initial_beta
        elif train_progress < 0.7:
            # 中期：逐渐降低beta以增强探索
            decay_ratio = (train_progress - 0.3) / 0.4
            return initial_beta * (1.0 - 0.5 * decay_ratio)
        else:
            # 后期：保持较低beta以微调
            return max(min_beta, initial_beta * 0.5)
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    if not hasattr(calculate_rewards, 'enhanced_calculator'):
        calculate_rewards.enhanced_calculator = EnhancedRewardCalculator(args, args.device)
        calculate_rewards.response_history = []

    total_rewards, components = calculate_rewards.enhanced_calculator.calculate_enhanced_rewards(
        prompts, responses, reward_model, reward_tokenizer, calculate_rewards.response_history
    )

    return total_rewards


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer,
                     start_step=0, wandb=None, dynamic_scaler=None, gradient_clipper=None):
    if dynamic_scaler is None:
        dynamic_scaler = DynamicAdvantageScaler(
            warmup_steps=args.adv_warmup_steps,
            min_std_threshold=args.min_std_threshold,
            max_std_threshold=args.max_std_threshold
        )

    if gradient_clipper is None:
        gradient_clipper = AdaptiveGradientClipper(
            max_norm=args.grad_clip,
            growth_factor=args.clip_growth_factor,
            shrink_factor=args.clip_shrink_factor
        )
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(
            args.device)  # input_ids: [B, P], attention_mask: [B, P]
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)  # [B*num_gen, P+R]

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]

        def get_per_token_logps(mdl, input_ids, n_keep):
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)

        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]

        # 使用动态优势标准化
        current_global_step = epoch * iters + step
        total_steps = args.epochs * iters
        advantages = dynamic_scaler.compute_dynamic_advantages(
            rewards, grouped_rewards, current_global_step, total_steps
        )

        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
                    torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(
                1)).int()  # [B*num_gen, R]

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        current_beta = adaptive_beta_scheduler(
            current_global_step,
            total_steps,
            initial_beta=args.beta,
            min_beta=args.min_beta,
            max_beta=args.max_beta
        )

        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(
            1) - current_beta * per_token_kl)  # [B*num_gen, R
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                grad_norm = gradient_clipper.adaptive_clip_grad_norm_(model.parameters(), loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            # 获取更多指标
            grad_norm = gradient_clipper.gradient_norms[-1] if gradient_clipper.gradient_norms else 0.0
            clip_ratio = sum(gradient_clipper.clip_counts[-100:]) / len(
                gradient_clipper.clip_counts) if gradient_clipper.clip_counts else 0.0

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, '
                   f'Reward: {avg_reward_val:.4f}, Avg Len: {avg_len_val:.2f}, '
                   f'LR: {current_lr:.8f}, Grad Norm: {grad_norm:.4f}, '
                   f'Clip Ratio: {clip_ratio:.3f}, Beta: {current_beta:.4f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr,
                    "grad_norm": grad_norm,
                    "clip_ratio": clip_ratio,
                    "advantages_mean": advantages.mean().item(),
                    "advantages_std": advantages.std().item(),
                    "beta": current_beta,
                    "dynamic_clip_threshold": gradient_clipper.max_norm
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    # 原参数8e-8
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    # 原参数1536
    parser.add_argument("--max_gen_len", type=int, default=800, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    # 原参数为8
    parser.add_argument("--num_generations", type=int, default=2, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    # ========== 新增：动态优势标准化参数 ==========
    parser.add_argument("--adv_warmup_steps", type=int, default=100,
                        help="动态标准化预热步数")
    parser.add_argument("--min_std_threshold", type=float, default=0.01,
                        help="最小标准差阈值")
    parser.add_argument("--max_std_threshold", type=float, default=5.0,
                        help="最大标准差阈值")

    # ========== 新增：自适应梯度裁剪参数 ==========
    parser.add_argument("--clip_growth_factor", type=float, default=1.1,
                        help="梯度裁剪阈值增长因子")
    parser.add_argument("--clip_shrink_factor", type=float, default=0.9,
                        help="梯度裁剪阈值收缩因子")
    parser.add_argument("--min_grad_norm", type=float, default=0.01,
                        help="最小梯度范数限制")
    parser.add_argument("--max_grad_norm_limit", type=float, default=10.0,
                        help="最大梯度范数限制")

    # ========== 新增：动态beta调度参数 ==========
    parser.add_argument("--min_beta", type=float, default=0.005,
                        help="最小KL惩罚系数")
    parser.add_argument("--max_beta", type=float, default=0.1,
                        help="最大KL惩罚系数")

    # ========== 新增：生成参数 ==========
    parser.add_argument("--sampling_temperature", type=float, default=0.8,
                        help="生成时的温度参数")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="生成时的top-p参数")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch);
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 初始化增强组件
        dynamic_scaler = DynamicAdvantageScaler(
            warmup_steps=args.adv_warmup_steps,
            min_std_threshold=args.min_std_threshold,
            max_std_threshold=args.max_std_threshold
        ) if args.enable_dynamic_scaling else None

        gradient_clipper = AdaptiveGradientClipper(
            max_norm=args.grad_clip,
            growth_factor=args.clip_growth_factor,
            shrink_factor=args.clip_shrink_factor,
            min_norm=args.min_grad_norm,
            max_norm_limit=args.max_grad_norm_limit
        ) if args.enable_adaptive_clipping else None

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model, reward_model, reward_tokenizer,
                             start_step, wandb, dynamic_scaler, gradient_clipper)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer,
                             0, wandb, dynamic_scaler, gradient_clipper)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()