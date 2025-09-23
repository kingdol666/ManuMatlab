import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

from .run_matlab_simulation import run_matlab_script, clear_matlab_engine_workspace
from .Models import ScriptType, RollDirection

# ---------------- FilmCastingEnv (Final Version) ----------------
class FilmCastingEnv:
    def __init__(self, n_rolls=5, target_temp=150.0, config_path='config/config.json', action_bounds=None, use_custom_directions=False, custom_directions=None):
        self.n_rolls = n_rolls
        # 修复目标温度：target_temp应该是摄氏度，转换为开尔文
        # 工业薄膜铸造的合理目标温度应该在100-200°C范围内
        self.target_temp_kelvin = target_temp + 273.15  # 150°C -> 423.15K
        self.base_config = self._load_initial_config(config_path)
        self.temp_points = 20  # Sample 20 points from T1 for state representation
        self.use_custom_directions = use_custom_directions
        self.custom_directions = custom_directions if custom_directions is not None else []

        # --- Input Validation for Custom Directions ---
        if self.use_custom_directions and len(self.custom_directions) < self.n_rolls - 1:
            raise ValueError(
                f"When using custom directions, `custom_directions` must have a length of at least `n_rolls - 1`. "
                f"Provided length: {len(self.custom_directions)}, Required length: {self.n_rolls - 1}"
            )

        # 增强的State: 22维特征，包含基础、历史、空间、动作反馈和过程特征
        # 基础特征(5) + 历史特征(8) + 空间特征(4) + 动作反馈特征(3) + 过程特征(2) = 22维
        self.action_dim = 3 if self.use_custom_directions else 4  # [temp, contact_time, cooling_time, (optional) direction]
        self.state_dim = 22  # 增强的22维状态空间
        print(f"Enhanced state dimension: {self.state_dim} (includes history, spatial, action feedback, and process features, action_dim: {self.action_dim})")
        self._setup_action_bounds(action_bounds)
    
        
        self.reset()

    def _setup_action_bounds(self, action_bounds):
        if action_bounds:
            temp_min_k = action_bounds['temp_min'] + 273.15
            temp_max_k = action_bounds['temp_max'] + 273.15
            contact_min, contact_max = action_bounds['contact_min'], action_bounds['contact_max']
            cooling_min, cooling_max = action_bounds['cooling_min'], action_bounds['cooling_max']
        else:
            temp_min_k, temp_max_k = 100 + 273.15, 200 + 273.15
            contact_min, contact_max = 0.1, 5.0
            cooling_min, cooling_max = 0.1, 5.0
        
        action_low_list = [temp_min_k, contact_min, cooling_min]
        action_high_list = [temp_max_k, contact_max, cooling_max]
        
        if not self.use_custom_directions:
            action_low_list.append(-1.0)
            action_high_list.append(1.0)
            
        self.action_low = np.array(action_low_list, dtype=np.float32)
        self.action_high = np.array(action_high_list, dtype=np.float32)

    def _load_initial_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Print a visible warning so users know defaults are used
            print("Warning: config.json not found or invalid, using defaults.")
            return {"main_params": {"T0": 400.0}}

    def reset(self):
        self.current_step = 0
        self.last_T1 = None
        self.last_cooling_T_matrix = None
        
        # 增强状态空间的重置逻辑
        t0_temp = self.base_config.get('main_params', {}).get('T0', 400.0)
        self.initial_temp_kelvin = t0_temp  # Store initial temperature

        self.current_temp_distribution = np.full(self.temp_points, t0_temp, dtype=np.float32)
        self.action_history = []
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        
        # 历史记录变量用于增强状态表示
        self.prev_temp_error = None
        self.prev_std = None
        
        # 增强状态的历史记录变量
        self.temp_history = []  # 温度统计历史
        self.action_effect_history = []  # 动作效果历史
        self.reward_history = []  # 奖励历史
        self.temp_trend_history = []  # 温度趋势历史
        
        return self._get_state()

    def _get_state(self):
        """
        构建22维增强状态表示
        基础特征(5) + 历史特征(8) + 空间特征(4) + 动作反馈特征(3) + 过程特征(2) = 22维
        """
        t0_temp = self.base_config.get('main_params', {}).get('T0', 400.0)
        
        # === 1. 基础特征 (5维) ===
        target_temp_norm = self.target_temp_kelvin / t0_temp
        mean_current_temp = np.mean(self.current_temp_distribution)
        uniformity_error = np.std(self.current_temp_distribution)
        temp_diff = mean_current_temp - self.target_temp_kelvin
        
        uniformity_norm = uniformity_error / 50.0
        temp_diff_norm = temp_diff / t0_temp
        mean_temp_norm = mean_current_temp / t0_temp
        step_norm = self.current_step / self.n_rolls if self.n_rolls > 0 else 0.0
        progress_norm = self.current_step / max(1, self.n_rolls - 1)
        
        # === 2. 历史特征 (8维) ===
        # 温度统计历史 (2维)
        temp_mean_trend = np.mean([h['mean_temp'] for h in self.temp_history[-3:]]) / t0_temp if self.temp_history else mean_temp_norm
        temp_std_trend = np.mean([h['std_temp'] for h in self.temp_history[-3:]]) / 50.0 if self.temp_history else uniformity_norm
        
        # 动作效果历史 (3维)
        recent_temp_change = np.mean([h['temp_change'] for h in self.action_effect_history[-2:]]) / t0_temp if self.action_effect_history else 0.0
        recent_std_change = np.mean([h['std_change'] for h in self.action_effect_history[-2:]]) / 50.0 if self.action_effect_history else 0.0
        action_effectiveness = np.mean([h['effectiveness'] for h in self.action_effect_history[-3:]]) if self.action_effect_history else 0.0
        
        # 奖励历史 (2维)
        recent_reward_trend = np.mean(self.reward_history[-3:]) if self.reward_history else 0.0
        reward_improvement = (self.reward_history[-1] - self.reward_history[-2]) if len(self.reward_history) >= 2 else 0.0
        
        # 温度趋势历史 (1维)
        temp_trend_slope = np.mean(self.temp_trend_history[-3:]) if self.temp_trend_history else 0.0
        
        # === 3. 空间特征 (4维) ===
        temp_min_norm = np.min(self.current_temp_distribution) / t0_temp
        temp_max_norm = np.max(self.current_temp_distribution) / t0_temp
        temp_range_norm = (np.max(self.current_temp_distribution) - np.min(self.current_temp_distribution)) / t0_temp
        temp_median_norm = np.median(self.current_temp_distribution) / t0_temp
        
        # === 4. 动作反馈特征 (3维) ===
        # 确保动作反馈特征始终为3维，兼容3维和4维动作空间
        if np.any(self.last_action):
            # 归一化动作到[-1, 1]范围
            last_action_norm = self.last_action / np.maximum(np.abs(self.last_action).max(), 1e-6)
        else:
            # 初始化为与action_dim相同维度的零向量
            last_action_norm = np.zeros(self.action_dim)
        
        # 确保输出为3维：取前3维或补零到3维
        if len(last_action_norm) >= 3:
            last_action_norm = last_action_norm[:3]
        else:
            last_action_norm = np.pad(last_action_norm, (0, 3 - len(last_action_norm)), 'constant')
        
        # === 5. 过程特征 (2维) ===
        convergence_indicator = max(0, 1 - uniformity_norm)  # 收敛指标
        exploration_factor = max(0, 1 - progress_norm)  # 探索因子
        
        # === 组合22维状态 ===
        state = np.array([
            # 基础特征 (5维)
            target_temp_norm, mean_temp_norm, uniformity_norm, temp_diff_norm, step_norm,
            # 历史特征 (8维)
            temp_mean_trend, temp_std_trend, recent_temp_change, recent_std_change, 
            action_effectiveness, recent_reward_trend, reward_improvement, temp_trend_slope,
            # 空间特征 (4维)
            temp_min_norm, temp_max_norm, temp_range_norm, temp_median_norm,
            # 动作反馈特征 (3维)
            last_action_norm[0], last_action_norm[1], last_action_norm[2],
            # 过程特征 (2维)
            convergence_indicator, exploration_factor
        ]).astype(np.float32)
        
        # 确保状态维度正确
        assert len(state) == 22, f"State dimension mismatch: expected 22, got {len(state)}"
        return state

    def step(self, action):
        # First, clip the action provided by the agent against its action space bounds.
        clipped_action = np.clip(action, self.action_low, self.action_high)
        # Store the agent's action (which has self.action_dim dimensions) for the next state representation.
        self.last_action = clipped_action.copy()

        # If using custom directions, construct the full 4D action for the simulation.
        if self.use_custom_directions:
            if self.current_step > 0:
                # Get the user-defined direction for the current step.
                direction = self.custom_directions[self.current_step - 1]
            else:
                # The first step has no preceding direction choice.
                direction = 1.0
            # Append the direction to the agent's action to create the full action.
            full_action = np.append(clipped_action, direction).astype(np.float32)
        else:
            # If not using custom directions, the agent's action is already the full action.
            full_action = clipped_action

        # Store the full action in the history for saving results.
        self.action_history.append(full_action.copy())
        
        # 安全地构建current_params，避免索引越界
        current_params = {
            "temp": float(full_action[0]), 
            "contact_time": float(full_action[1]), 
            "cooling_time": float(full_action[2]), 
            "direction": float(full_action[3]) if len(full_action) > 3 else 1.0  # 默认方向为1.0
        }
        try:
            # Heating
            heating_folder = "升温1" if self.current_step == 0 else ("升温3" if current_params['direction'] < 0 else "升温5")
            mesh_path_heating = os.path.join('matlabScripts', heating_folder, 'shijian_rechuandao_mesh.m')
            main_path_heating = os.path.join('matlabScripts', heating_folder, 'shijian_rechuandao_main.m')
            
            input_vars_heating = {'T_GunWen_Input': current_params['temp'], 't_up_input': current_params['contact_time']}
            if self.last_T1 is not None:
                input_vars_heating['T1'] = self.last_T1

            # Run heating mesh and main scripts
            run_matlab_script(mesh_path_heating, [], self.base_config.get('mesh_params', {}), input_vars_heating)
            heating_output = run_matlab_script(main_path_heating, ['T'], self.base_config.get('main_params', {}), input_vars_heating)
            clear_matlab_engine_workspace()

            if 'T' not in heating_output or heating_output['T'] is None:
                raise RuntimeError("Heating step failed to return 'T'.")
            
            T_array = np.array(heating_output['T'])
            temp_after_heating = T_array[:, -1].reshape(-1, 1) if T_array.ndim == 2 and T_array.shape[1] > 0 else T_array.tolist()

            # Cooling
            cooling_folder = '冷却2'
            mesh_path_cooling = os.path.join('matlabScripts', cooling_folder, 'shijian_rechuandao_mesh.m')
            main_path_cooling = os.path.join('matlabScripts', cooling_folder, 'shijian_rechuandao_main.m')

            input_vars_cooling = {'t_up_input': current_params['cooling_time'], 'T1': temp_after_heating}

            # Run cooling mesh and main scripts
            run_matlab_script(mesh_path_cooling, [], self.base_config.get('mesh_params', {}), input_vars_cooling)
            cooling_output = run_matlab_script(main_path_cooling, ['T'], self.base_config.get('main_params', {}), input_vars_cooling)
            clear_matlab_engine_workspace()
            if 'T' not in cooling_output or cooling_output['T'] is None:
                raise RuntimeError("Cooling step failed to return 'T'.")

            self.last_cooling_T_matrix = np.array(cooling_output['T'])
            if self.last_cooling_T_matrix.ndim == 2 and self.last_cooling_T_matrix.shape[1] > 0:
                self.last_T1 = self.last_cooling_T_matrix[:, -1].reshape(-1, 1)
            else:
                self.last_T1 = self.last_cooling_T_matrix.tolist()
            
            t1_flat = np.array(self.last_T1).flatten()
            indices = np.linspace(0, len(t1_flat) - 1, self.temp_points, dtype=int)
            self.current_temp_distribution = t1_flat[indices].astype(np.float32)
            
            self.current_step += 1
            done = self.current_step >= self.n_rolls
            reward = self._calculate_reward(done)
            
            # 更新历史记录用于增强状态表示
            self._update_history_records(clipped_action, reward)
            
            new_state = self._get_state()
            
            return new_state, float(reward), bool(done), {}
        except Exception as e:
            print(f"MATLAB simulation step error: {e}")
            return self._get_state(), -1000.0, True, {}
        finally:
            # Ensure the MATLAB workspace is cleared after each step, regardless of success or failure
            print("matlab finished")

    def _update_history_records(self, action, reward):
        """更新历史记录用于增强状态表示"""
        t0_temp = self.base_config.get('main_params', {}).get('T0', 400.0)
        
        # 计算当前温度统计
        current_mean_temp = np.mean(self.current_temp_distribution)
        current_std_temp = np.std(self.current_temp_distribution)
        
        # 1. 更新温度历史 (保持最近5步)
        temp_record = {
            'mean_temp': current_mean_temp,
            'std_temp': current_std_temp,
            'step': self.current_step
        }
        self.temp_history.append(temp_record)
        if len(self.temp_history) > 5:
            self.temp_history.pop(0)
        
        # 2. 计算并更新动作效果历史 (保持最近5步)
        if len(self.temp_history) >= 2:
            prev_temp_record = self.temp_history[-2]
            temp_change = current_mean_temp - prev_temp_record['mean_temp']
            std_change = current_std_temp - prev_temp_record['std_temp']
            
            # 计算动作有效性 (温度向目标靠近且均匀性改善)
            target_approach = -(abs(current_mean_temp - self.target_temp_kelvin) - 
                               abs(prev_temp_record['mean_temp'] - self.target_temp_kelvin))
            uniformity_improvement = -(current_std_temp - prev_temp_record['std_temp'])
            effectiveness = target_approach + uniformity_improvement
            
            action_effect_record = {
                'action': action.copy(),
                'temp_change': temp_change,
                'std_change': std_change,
                'effectiveness': effectiveness,
                'step': self.current_step
            }
            self.action_effect_history.append(action_effect_record)
            if len(self.action_effect_history) > 5:
                self.action_effect_history.pop(0)
        
        # 3. 更新奖励历史 (保持最近5步)
        self.reward_history.append(reward)
        if len(self.reward_history) > 5:
            self.reward_history.pop(0)
        
        # 4. 更新温度趋势历史
        if len(self.temp_history) >= 3:
            recent_temps = [h['mean_temp'] for h in self.temp_history[-3:]]
            temp_trend = (recent_temps[-1] - recent_temps[0]) / 2  # 简单的趋势斜率
            self.temp_trend_history.append(temp_trend)
            if len(self.temp_trend_history) > 5:
                self.temp_trend_history.pop(0)

    def _calculate_reward(self, done):
        """简化的奖励函数 - 基于target温度和std温度的逐步加权设计"""
        # 使用当前温度分布计算奖励
        if self.current_temp_distribution is None or len(self.current_temp_distribution) == 0:
            return -1.0  # 归一化后的最低奖励

        mean_current_temp = float(np.mean(self.current_temp_distribution))
        std_current = float(np.std(self.current_temp_distribution))

        # 计算与目标温度的距离差异
        target_diff = abs(mean_current_temp - self.target_temp_kelvin)
        
        # 步骤权重：随步骤递增，强调后期精度
        step_weight = (self.current_step + 1) / self.n_rolls if self.n_rolls > 0 else 1.0
        
        # 1. Target温度奖励：使用更敏感的奖励函数
        # 使用分段函数：小差异时给予更高奖励梯度
        if target_diff <= 5.0:  # 5K以内给予高奖励
            target_reward = 1.0 - (target_diff / 5.0) * 0.3  # 0.7-1.0范围
        elif target_diff <= 20.0:  # 5-20K中等奖励
            target_reward = 0.7 - ((target_diff - 5.0) / 15.0) * 0.5  # 0.2-0.7范围
        else:  # 20K以上低奖励
            target_reward = max(0.0, 0.2 - (target_diff - 20.0) / 50.0)  # 0-0.2范围
        
        # 2. Std温度奖励：标准差越小越好（均匀性）
        if std_current <= 2.0:  # 2K以内给予高奖励
            std_reward = 1.0 - (std_current / 2.0) * 0.3  # 0.7-1.0范围
        elif std_current <= 10.0:  # 2-10K中等奖励
            std_reward = 0.7 - ((std_current - 2.0) / 8.0) * 0.5  # 0.2-0.7范围
        else:  # 10K以上低奖励
            std_reward = max(0.0, 0.2 - (std_current - 10.0) / 20.0)  # 0-0.2范围
        
        # 3. 改进奖励：与上一步比较
        improvement_reward = 0.0
        if hasattr(self, 'prev_temp_error') and self.prev_temp_error is not None:
            temp_improvement = self.prev_temp_error - target_diff
            std_improvement = (self.prev_std - std_current) if hasattr(self, 'prev_std') and self.prev_std is not None else 0.0
            improvement_reward = (temp_improvement + std_improvement) / 10.0  # 归一化改进奖励
            improvement_reward = np.clip(improvement_reward, -0.5, 0.5)
        
        # 更新历史记录
        self.prev_temp_error = target_diff
        self.prev_std = std_current
        
        # 4. 逐步加权：权重随步骤增加
        target_weight = 0.5 + 0.3 * step_weight  # 0.5 -> 0.8
        std_weight = 0.3 + 0.2 * step_weight     # 0.3 -> 0.5
        improvement_weight = 0.2  # 固定改进权重
        
        # 5. 组合奖励
        total_reward = (target_weight * target_reward + 
                       std_weight * std_reward + 
                       improvement_weight * improvement_reward)
        
        # 6. 归一化到[-1, 1]范围
        # 理论最大值约为 0.8 * 1.0 + 0.5 * 1.0 + 0.2 * 0.5 = 1.4
        # 理论最小值约为 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * (-0.5) = -0.1
        normalized_reward = (total_reward - 0.65) / 0.75  # 将[-0.1, 1.4]映射到[-1, 1]
        normalized_reward = np.clip(normalized_reward, -1.0, 1.0)
        
        # 调试日志
        print(f"Step {self.current_step}/{self.n_rolls} (Weight: {step_weight:.2f}) - "
              f"Temp: {mean_current_temp:.2f}K (Target: {self.target_temp_kelvin:.2f}K) | "
              f"Target_diff: {target_diff:.2f}K, Std: {std_current:.2f}K | "
              f"Rewards: Target={target_reward:.3f} (w={target_weight:.2f}), "
              f"Std={std_reward:.3f} (w={std_weight:.2f}), "
              f"Improvement={improvement_reward:.3f} (w={improvement_weight:.2f}) | "
              f"Total: {total_reward:.3f} -> Normalized: {normalized_reward:.3f}")

        return float(normalized_reward)

# ---------------- SumTree for Prioritized Replay Buffer ----------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

# ---------------- Prioritized Replay Buffer ----------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 0.01
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if isinstance(data, np.ndarray) and data.size == 0:
                s = random.uniform(0, self.tree.total())
                (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            if priority > self.max_priority:
                self.max_priority = priority
    
    def clear(self):
        self.tree = SumTree(self.capacity)

    def __len__(self):
        return self.tree.n_entries
    
    def __iter__(self):
        return iter(self.tree.data[:self.tree.n_entries])

# ---------------- Actor / Critic / DDPG (Final Version) ----------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 两层隐藏层结构 + BatchNorm1d
        self.layer1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, action_dim)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 第一层
        x = self.layer1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.layer2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # 输出层
        return torch.tanh(self.output_layer(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 两层隐藏层结构 + BatchNorm1d
        input_dim = state_dim + action_dim
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, 1)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        
        # 第一层
        x = self.layer1(xu)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.layer2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # 输出层
        return self.output_layer(x)

class TD3:
    def __init__(self, state_dim, action_dim, action_low, action_high, device, log_updated=None, replay_buffer_size=1000000, batch_size=128):
        self.device = device
        self.log_updated = log_updated
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low_t = torch.from_numpy(action_low).to(device)
        self.action_high_t = torch.from_numpy(action_high).to(device)
        self.action_low_np = action_low
        self.action_high_np = action_high

        # --- Actor Networks ---
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # 优化学习率：适中的学习率平衡学习速度和稳定性
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-4, weight_decay=1e-5)

        # --- Critic Networks (Twin) ---
        self.critic1 = Critic(state_dim, action_dim).to(device)
        print("--- Actor Architecture ---")
        print(self.actor)
        print("--- Critic Architecture ---")
        print(self.critic1)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        # Critic学习率稍高于Actor，加快价值函数学习
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=8e-4, weight_decay=1e-5) 

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=8e-4, weight_decay=1e-5)

        # --- Prioritized Experience Replay ---
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)
        self.beta_initial = 0.4
        self.beta_frames = 1000000
        self.beta = self.beta_initial

        self.batch_size = batch_size
        self.discount = 0.99
        self.tau = 0.005
        
        # --- 改进的探索策略 ---
        # 更慢的噪声衰减，给模型更多时间探索
        self.initial_exploration_noise_std = 0.15  # 适度增加初始噪声，确保充分探索
        self.final_exploration_noise_std = 0.01   # 提高最终噪声，保持持续探索
        self.exploration_decay_steps = 1000000    # 延长衰减步数，避免过早收敛
        self.exploration_noise_std = self.initial_exploration_noise_std
        
        # TD3 specific parameters
        # 调整policy_noise以适应归一化的action空间[-1,1]
        self.policy_noise = 0.1  # 保持适中的policy noise
        self.noise_clip = 0.05   # 保持合理的噪声范围
        self.policy_freq = 2     # 保持延迟更新频率
        self.total_it = 0
        
        # 打印维度范围信息，验证相对变换效果
        action_ranges = self.action_high_np - self.action_low_np
        action_centers = (self.action_high_np + self.action_low_np) * 0.5
        print(f"Action dimensions optimization:")
        print(f"  Original ranges: {action_ranges}")
        print(f"  Action centers: {action_centers}")
        print(f"  Relative transform: All dimensions normalized to [-1, 1] for equal training magnitude")
        print(f"  This ensures balanced gradient updates across all action dimensions")

    def _scale_action_torch(self, action_norm_t):
        """
        将归一化的action [-1, 1] 转换为实际的action值
        使用相对变换，确保各维度处于相同量级的训练空间
        """
        # 计算各维度的范围
        action_ranges = self.action_high_t - self.action_low_t
        
        # 使用相对变换：action_norm_t 在 [-1, 1] 范围内代表相对于中心点的偏移比例
        action_centers = (self.action_high_t + self.action_low_t) * 0.5
        
        # 将归一化的动作转换为实际动作
        # action_norm_t = 0 对应中心值，±1 对应边界值
        scaled_action = action_centers + action_norm_t * action_ranges * 0.5
        
        return scaled_action

    def _scale_action_numpy(self, action_norm_np):
        """
        将归一化的action [-1, 1] 转换为实际的action值 (numpy版本)
        使用相对变换，确保各维度处于相同量级的训练空间
        """
        # 计算各维度的范围
        action_ranges = self.action_high_np - self.action_low_np
        
        # 使用相对变换：action_norm_np 在 [-1, 1] 范围内代表相对于中心点的偏移比例
        action_centers = (self.action_high_np + self.action_low_np) * 0.5
        
        # 将归一化的动作转换为实际动作
        # action_norm_np = 0 对应中心值，±1 对应边界值
        scaled_action = action_centers + action_norm_np * action_ranges * 0.5
        
        return scaled_action
    
    def _normalize_action_torch(self, action_t):
        """
        将实际动作值反向归一化到[-1, 1]范围
        使用相对变换，确保各维度处于相同量级的训练空间
        """
        # 计算各维度的范围和中心点
        action_ranges = self.action_high_t - self.action_low_t
        action_centers = (self.action_high_t + self.action_low_t) * 0.5
        
        # 将实际动作转换为归一化动作
        # 相对于中心点的偏移，除以半范围，得到 [-1, 1] 的归一化值
        normalized_action = (action_t - action_centers) / (action_ranges * 0.5)
        
        return torch.clamp(normalized_action, -1.0, 1.0)

    def select_action(self, state, noise=True):
        self.actor.eval()
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action_norm = self.actor(state_t).cpu().data.numpy().flatten()
        self.actor.train()
        if noise:
            # 维度自适应的探索噪声
            # 由于使用了相对变换，所有维度在[-1,1]范围内具有相同的量级
            # 可以使用统一的噪声标准差，确保各维度的探索强度一致
            noise_val = np.random.normal(0, self.exploration_noise_std, size=self.action_dim)
            action_norm = np.clip(action_norm + noise_val, -1.0, 1.0)
        action = self._scale_action_numpy(action_norm)
        return action.astype(np.float32)

    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state.copy(), action.copy(), float(reward), next_state.copy(), float(done))

    def train(self, best_trajectories=None):
        if len(self.replay_buffer) < self.batch_size: return
        self.total_it += 1
        
        # Update beta for PER
        self.beta = self.beta_initial + (1.0 - self.beta_initial) * (self.total_it / self.beta_frames)
        self.beta = min(1.0, self.beta)

        # Decay exploration noise
        self.exploration_noise_std = self.final_exploration_noise_std + \
            (self.initial_exploration_noise_std - self.final_exploration_noise_std) * \
            max(0, 1 - self.total_it / self.exploration_decay_steps)

        # Set networks to training mode
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        # --- Sample from Prioritized Replay Buffer ---
        batch, batch_indices, is_weights = self.replay_buffer.sample(self.batch_size, self.beta)
        
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action_raw = torch.FloatTensor(np.array(action)).to(self.device)
        # 将实际动作值归一化到[-1, 1]范围用于训练
        action = self._normalize_action_torch(action_raw)
        reward = torch.FloatTensor(np.array(reward)).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).view(-1, 1).to(self.device)
        is_weights = torch.FloatTensor(np.array(is_weights)).view(-1, 1).to(self.device)

        with torch.no_grad():
            # --- Target Policy Smoothing ---
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action_norm = self.actor_target(next_state)
            next_action_norm_clipped = (next_action_norm + noise).clamp(-1.0, 1.0)
            # 使用归一化的next_action与Critic网络保持一致
            next_action = next_action_norm_clipped

            # --- Clipped Double-Q Learning: Compute the target Q value ---
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q)

        # --- Critic 1 Update ---
        current_Q1 = self.critic1(state, action)
        td_error1 = (current_Q1 - target_Q).abs()
        critic1_loss = (is_weights * nn.MSELoss(reduction='none')(current_Q1, target_Q)).mean()
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # --- Critic 2 Update ---
        current_Q2 = self.critic2(state, action)
        td_error2 = (current_Q2 - target_Q).abs()
        critic2_loss = (is_weights * nn.MSELoss(reduction='none')(current_Q2, target_Q)).mean()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # --- Update Priorities in Replay Buffer ---
        # Use the average of the two TD errors for the priority
        priorities = ((td_error1 + td_error2) / 2).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(batch_indices, priorities)

        # --- Delayed Policy Updates ---
        if self.total_it % self.policy_freq == 0:
            # --- Actor Update ---
            action_norm_pred = self.actor(state)
            # 使用归一化的action与Critic网络保持一致
            q_value_pred = self.critic1(state, action_norm_pred)
            actor_loss = -q_value_pred.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Log losses
            if self.log_updated:
                self.log_updated.emit(f"Losses - Actor: {actor_loss.item():.4f}, Critic1: {critic1_loss.item():.4f}, Critic2: {critic2_loss.item():.4f}", "info")

            # --- Soft Update Target Networks ---
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self, filepath):
        """仅保存模型和优化器状态到.pth文件"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Model and optimizer states saved to {filepath}")

    def load_checkpoint(self, filepath):
        """仅从.pth文件加载模型和优化器状态"""
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        print(f"Model and optimizer states loaded from {filepath}")

    def save_best_model(self, filepath):
        """Saves the actor model's weights."""
        torch.save(self.actor.state_dict(), filepath)
        print(f"Best model weights saved to {filepath}")

    def load_model(self, filepath):
        """Loads the actor model's weights for inference."""
        if not os.path.exists(filepath):
            print(f"Model weights file not found: {filepath}")
            return
        # Load weights into the main actor and the target actor
        self.actor.load_state_dict(torch.load(filepath, map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        # Set models to evaluation mode
        self.actor.eval()
        self.actor_target.eval()
        print(f"Model weights loaded from {filepath}")


# ---------------- Utilities ----------------
def save_training_state(agent, training_params, rewards, episode, output_path):
    """Saves the complete training state, including the PER buffer's priorities."""
    # Serialize the PER buffer state, including the SumTree
    replay_buffer_data_list = [
        (s.tolist(), a.tolist(), float(r), ns.tolist(), float(d))
        for s, a, r, ns, d in agent.replay_buffer
    ]
    replay_buffer_state = {
        'tree': agent.replay_buffer.tree.tree.tolist(),
        'data': replay_buffer_data_list,
        'n_entries': int(agent.replay_buffer.tree.n_entries),
        'write': int(agent.replay_buffer.tree.write),
        'max_priority': float(agent.replay_buffer.max_priority)
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    state = {
        'training_params': convert_to_json_serializable(training_params),
        'rewards': [float(r) for r in rewards],  # 确保rewards中的所有值都是Python float
        'episode': int(episode),
        'replay_buffer_state': replay_buffer_state
    }
    
    # Save model weights to a .pth file
    agent_path = os.path.splitext(output_path)[0] + ".pth"
    agent.save_checkpoint(agent_path)
    
    # Save everything else to the JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    print(f"Training state (including PER buffer) saved to {output_path}")

def load_training_state(filepath, agent):
    """Loads the complete training state, including the PER buffer's priorities."""
    if not os.path.exists(filepath):
        print(f"Training state file not found: {filepath}")
        return None, [], 0

    with open(filepath, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    # Convert loaded data to ensure proper types
    def ensure_proper_types(obj):
        """确保加载的数据具有正确的Python类型"""
        if isinstance(obj, dict):
            return {key: ensure_proper_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ensure_proper_types(item) for item in obj]
        else:
            return obj
    
    training_params = ensure_proper_types(state['training_params'])
    rewards = [float(r) for r in state['rewards']]  # 确保rewards是float列表
    start_episode = int(state['episode'])
    
    # Load the PER buffer state
    replay_buffer_state = state.get('replay_buffer_state')
    if replay_buffer_state:
        # 检查保存的经验池大小与当前经验池大小是否匹配
        saved_capacity = len(replay_buffer_state['data'])
        current_capacity = agent.replay_buffer.capacity
        saved_n_entries = int(replay_buffer_state['n_entries'])
        
        print(f"Loading replay buffer: saved_capacity={saved_capacity}, current_capacity={current_capacity}, saved_entries={saved_n_entries}")
        
        # 如果容量不匹配，需要适配处理
        if saved_capacity != current_capacity or saved_n_entries > current_capacity:
            print(f"Warning: Replay buffer size mismatch. Saved: {saved_capacity}, Current: {current_capacity}")
            print("Adapting replay buffer to new size...")
            
            # 重新创建适合当前容量的缓冲区
            agent.replay_buffer.clear()
            
            # 加载数据，但限制在当前容量内
            loaded_data = replay_buffer_state['data']
            max_entries_to_load = min(len(loaded_data), current_capacity, saved_n_entries)
            
            # 如果保存的数据太多，只取最新的数据
            if len(loaded_data) > current_capacity:
                # 取最后的 current_capacity 个经验
                start_idx = len(loaded_data) - current_capacity
                loaded_data = loaded_data[start_idx:]
                print(f"Taking the last {current_capacity} experiences from {len(replay_buffer_state['data'])} saved experiences")
            
            # 逐个添加经验到新的缓冲区
            for i, (s_list, a_list, r, ns_list, d) in enumerate(loaded_data[:max_entries_to_load]):
                state_np = np.array(s_list, dtype=np.float32)
                action_np = np.array(a_list, dtype=np.float32)
                reward_val = float(r)
                next_state_np = np.array(ns_list, dtype=np.float32)
                done_val = float(d)
                agent.replay_buffer.push(state_np, action_np, reward_val, next_state_np, done_val)
            
            print(f"Successfully loaded {len(agent.replay_buffer)} experiences into new buffer")
            
        else:
            # 容量匹配，可以直接加载
            # 重建 SumTree
            saved_tree = replay_buffer_state['tree']
            if len(saved_tree) == len(agent.replay_buffer.tree.tree):
                agent.replay_buffer.tree.tree = np.array(saved_tree, dtype=np.float64)
            else:
                print("Warning: SumTree size mismatch, rebuilding priorities...")
                agent.replay_buffer.clear()
                # 重新添加所有经验
                for s_list, a_list, r, ns_list, d in replay_buffer_state['data'][:saved_n_entries]:
                    state_np = np.array(s_list, dtype=np.float32)
                    action_np = np.array(a_list, dtype=np.float32)
                    reward_val = float(r)
                    next_state_np = np.array(ns_list, dtype=np.float32)
                    done_val = float(d)
                    agent.replay_buffer.push(state_np, action_np, reward_val, next_state_np, done_val)
                print(f"Rebuilt replay buffer with {len(agent.replay_buffer)} experiences")
                return training_params, rewards, start_episode
            
            # 重建数据数组
            loaded_data = replay_buffer_state['data']
            agent.replay_buffer.tree.data = np.zeros(agent.replay_buffer.capacity, dtype=object)
            
            for i, (s_list, a_list, r, ns_list, d) in enumerate(loaded_data[:saved_n_entries]):
                if i >= agent.replay_buffer.capacity:
                    break
                state_np = np.array(s_list, dtype=np.float32)
                action_np = np.array(a_list, dtype=np.float32)
                reward_val = float(r)
                next_state_np = np.array(ns_list, dtype=np.float32)
                done_val = float(d)
                agent.replay_buffer.tree.data[i] = (state_np, action_np, reward_val, next_state_np, done_val)

            # 设置缓冲区状态，确保不超过当前容量
            agent.replay_buffer.tree.n_entries = min(saved_n_entries, current_capacity)
            agent.replay_buffer.tree.write = min(int(replay_buffer_state['write']), current_capacity - 1)
            agent.replay_buffer.max_priority = float(replay_buffer_state.get('max_priority', 1.0))
            
            print(f"PER buffer loaded with {agent.replay_buffer.tree.n_entries} experiences and their priorities.")
    else:
        # Fallback for old save format
        replay_buffer_list = state.get('replay_buffer', [])
        agent.replay_buffer.clear()
        for s_list, a_list, r, ns_list, d in replay_buffer_list:
            state_np = np.array(s_list, dtype=np.float32)
            action_np = np.array(a_list, dtype=np.float32)
            reward_val = float(r)
            next_state_np = np.array(ns_list, dtype=np.float32)
            done_val = float(d)
            agent.push(state_np, action_np, reward_val, next_state_np, done_val)
        print(f"Replay buffer loaded with {len(agent.replay_buffer)} experiences (old format, priorities reset).")

    # Load model weights from the .pth file
    agent_path = os.path.splitext(filepath)[0] + ".pth"
    if os.path.exists(agent_path):
        agent.load_checkpoint(agent_path)
    else:
        print(f"Warning: Agent weights file not found at {agent_path}. Starting with fresh weights.")

    print(f"Training state loaded from {filepath}. Resuming from episode {start_episode + 1}.")
    return training_params, rewards, start_episode


def save_best_params_as_json(action_history, file_path):
    models_list = []
    for i, action in enumerate(action_history):
        params = {"temp": float(action[0]), "contact_time": float(action[1]), "cooling_time": float(action[2]), "direction": float(action[3])}
        roll_direction = RollDirection.INITIAL if i == 0 else (RollDirection.REVERSE if params['direction'] < 0 else RollDirection.FORWARD)
        models_list.append({"script_type": ScriptType.HEATING, "roll_direction": roll_direction, "T_GunWen": params['temp'], "t_up": params['contact_time']})
        models_list.append({"script_type": ScriptType.COOLING, "roll_direction": RollDirection.INITIAL, "T_GunWen": params['temp'], "t_up": params['cooling_time']})
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(models_list, f, ensure_ascii=False, indent=2)
        print(f"Best parameters saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return None

def save_exploration_reward_plot(exploration_rewards, plot_path='exploration_reward_curve.png', title='Exploration Reward Curve'):
    """
    保存探索得分图表
    
    Args:
        exploration_rewards: 探索得分列表，每个元素代表一轮的得分
        plot_path: 图表保存路径
        title: 图表标题
    """
    if not exploration_rewards:
        print("No exploration rewards to plot")
        return None
        
    plt.figure(figsize=(12, 6))
    rounds = list(range(1, len(exploration_rewards) + 1))
    plt.plot(rounds, exploration_rewards, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('轮数 (Round)')
    plt.ylabel('探索得分 (Exploration Reward)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    if exploration_rewards:
        avg_reward = sum(exploration_rewards) / len(exploration_rewards)
        max_reward = max(exploration_rewards)
        min_reward = min(exploration_rewards)
        plt.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.7, label=f'平均值: {avg_reward:.4f}')
        plt.legend()
        
        # 添加文本注释
        plt.text(0.02, 0.98, f'最大值: {max_reward:.4f}\n最小值: {min_reward:.4f}\n平均值: {avg_reward:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Exploration reward curve saved to {plot_path}")
    return plot_path
