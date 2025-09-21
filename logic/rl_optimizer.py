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
    def __init__(self, n_rolls=5, target_temp=25.0, config_path='config/config.json', action_bounds=None, use_custom_directions=False, custom_directions=None):
        self.n_rolls = n_rolls
        self.target_temp_kelvin = target_temp + 273.15
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

        # State: [target_temp_norm, current_temp_norm, uniformity_norm, temp_diff_norm, mean_temp_norm, min_temp_norm, max_temp_norm, step_norm]
        self.action_dim = 3 if self.use_custom_directions else 4  # [temp, contact_time, cooling_time, (optional) direction]
        self.state_dim = 1 + self.temp_points + 1 + 1 + 3 + 1 # target_norm, temp_dist, uniformity, temp_diff, mean, min, max, step_norm
        print(f"State dimension calculated: {self.state_dim} (temp_points: {self.temp_points}, action_dim: {self.action_dim})")
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
        
        # Initialize error tracking for the new reward function
        t0_temp = self.base_config.get('main_params', {}).get('T0', 400.0)
        self.last_final_target_error = np.abs(t0_temp - self.target_temp_kelvin)
        self.last_uniformity_error = 0.0  # Initial uniformity is perfect (zero std dev)

        self.current_temp_distribution = np.full(self.temp_points, t0_temp, dtype=np.float32)
        self.action_history = []
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        t0_temp = self.base_config.get('main_params', {}).get('T0', 400.0)
        target_temp_norm = self.target_temp_kelvin / t0_temp
        current_temp_norm = self.current_temp_distribution / t0_temp
        
        # --- Add descriptive statistics of the current temperature distribution ---
        mean_current_temp = np.mean(self.current_temp_distribution)
        min_current_temp = np.min(self.current_temp_distribution)
        max_current_temp = np.max(self.current_temp_distribution)
        uniformity_error = np.std(self.current_temp_distribution)
        temp_diff = mean_current_temp - self.target_temp_kelvin

        # --- Normalize all features ---
        uniformity_norm = uniformity_error / 50.0  # Assuming a max std dev of 50K
        temp_diff_norm = temp_diff / t0_temp
        mean_temp_norm = mean_current_temp / t0_temp
        min_temp_norm = min_current_temp / t0_temp
        max_temp_norm = max_current_temp / t0_temp

        step_norm = self.current_step / self.n_rolls if self.n_rolls > 0 else 0.0

        state = np.concatenate([
            [target_temp_norm],
            current_temp_norm,
            [uniformity_norm],
            [temp_diff_norm],
            [mean_temp_norm],
            [min_temp_norm],
            [max_temp_norm],
            [step_norm]
        ]).astype(np.float32)
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

        # Store the full 4D action in the history for saving results.
        self.action_history.append(full_action.copy())
        current_params = {"temp": float(full_action[0]), "contact_time": float(full_action[1]), "cooling_time": float(full_action[2]), "direction": float(full_action[3])}
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
            new_state = self._get_state()
            
            return new_state, float(reward), bool(done), {}
        except Exception as e:
            print(f"MATLAB simulation step error: {e}")
            return self._get_state(), -1000.0, True, {}
        finally:
            # Ensure the MATLAB workspace is cleared after each step, regardless of success or failure
            print("matlab finished")

    def _calculate_reward(self, done):
        if self.last_cooling_T_matrix is None or self.last_cooling_T_matrix.shape[1] < 1:
            return -200.0

        current_temp_distribution = self.last_cooling_T_matrix[:, -1]
        mean_current_temp = float(np.mean(current_temp_distribution))
        std_current = float(np.std(current_temp_distribution))
        mean_err = float(abs(mean_current_temp - self.target_temp_kelvin))

        # --- 1. Define Normalization and Reward Parameters ---
        max_mean_err = 50.0  # Max expected error in Kelvin for normalization
        max_std = 5.0      # Max expected std dev in Kelvin for normalization
        
        # --- 2. Calculate Step-wise Progressive Reward Components ---
        # The penalty for errors becomes harsher as we approach the final step.
        step_progress = self.current_step / self.n_rolls
        k_mean = 1.0 + step_progress * 4.0  # Sharpness for mean error penalty (1 -> 5)
        k_std = 1.0 + step_progress * 4.0   # Sharpness for std error penalty (1 -> 5)

        # Normalize errors to [0, 1]
        mean_err_norm = np.clip(mean_err / max_mean_err, 0.0, 1.0)
        std_norm = np.clip(std_current / max_std, 0.0, 1.0)

        # Calculate exponential scores for mean and std, each contributing up to 0.5
        score_mean = 0.5 * np.exp(-k_mean * mean_err_norm)
        score_std = 0.5 * np.exp(-k_std * std_norm)
        
        base_reward = score_mean + score_std  # Max base reward is 1.0

        # --- 3. Calculate Improvement-based Reward ---
        # Reward the agent for reducing the error compared to the previous step.
        mean_err_improvement = self.last_final_target_error - mean_err
        std_improvement = self.last_uniformity_error - std_current
        
        # Scale the improvement to a small bonus/penalty
        improvement_reward = (mean_err_improvement / max_mean_err) * 0.1 + \
                             (std_improvement / max_std) * 0.1
        
        # --- 4. Calculate Final Step Bonus ---
        final_bonus = 0.0
        if done:
            # Give a significant smooth bonus for achieving a good final state
            # The bonus is higher if the final error and std are very low
            if mean_err < 4.0 and std_current < 0.5:
                final_mean_score = np.exp(-0.1 * mean_err) # Decays slowly
                final_std_score = np.exp(-0.5 * std_current) # Decays faster
                final_bonus = 1.0 * final_mean_score * final_std_score

        # --- 5. Combine all reward components ---
        total_reward = base_reward + improvement_reward + final_bonus
        
        # --- 6. Update last errors for the next step's calculation ---
        self.last_final_target_error = mean_err
        self.last_uniformity_error = std_current

        # Log for debugging
        print(f"Step {self.current_step}/{self.n_rolls} - Reward: {total_reward:.4f} (Base: {base_reward:.2f}, Improv: {improvement_reward:.2f}, Bonus: {final_bonus:.2f}) | Mean Err: {mean_err:.2f}K, Std: {std_current:.2f}K")

        return float(np.clip(total_reward, -1.0, 2.0)) # Clip to a reasonable range

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
        self.layer1 = nn.Linear(state_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        return torch.tanh(self.layer3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x = torch.relu(self.bn1(self.layer1(xu)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        return self.layer3(x)

class TD3:
    def __init__(self, state_dim, action_dim, action_low, action_high, device, bc_weight=0.2, log_updated=None, replay_buffer_size=1000000, batch_size=16):
        self.device = device
        self.log_updated = log_updated
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bc_weight = bc_weight
        self.action_low_t = torch.from_numpy(action_low).to(device)
        self.action_high_t = torch.from_numpy(action_high).to(device)
        self.action_low_np = action_low
        self.action_high_np = action_high

        # --- Actor Networks ---
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        # --- Critic Networks (Twin) ---
        self.critic1 = Critic(state_dim, action_dim).to(device)
        print("--- Actor Architecture ---")
        print(self.actor)
        print("--- Critic Architecture ---")
        print(self.critic1)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # --- Prioritized Experience Replay ---
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)
        self.beta_initial = 0.4
        self.beta_frames = 1000000
        self.beta = self.beta_initial

        self.batch_size = batch_size
        self.discount = 0.99
        self.tau = 0.005
        
        # --- Exploration Noise Decay ---
        self.initial_exploration_noise_std = 0.1
        self.final_exploration_noise_std = 0.1
        self.exploration_decay_steps = 500000
        self.exploration_noise_std = self.initial_exploration_noise_std
        
        # TD3 specific parameters
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def _scale_action_torch(self, action_norm_t):
        return self.action_low_t + (action_norm_t + 1.0) * 0.5 * (self.action_high_t - self.action_low_t)

    def _scale_action_numpy(self, action_norm_np):
        return self.action_low_np + (action_norm_np + 1.0) * 0.5 * (self.action_high_np - self.action_low_np)

    def select_action(self, state, noise=True):
        self.actor.eval()
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action_norm = self.actor(state_t).cpu().data.numpy().flatten()
        self.actor.train()
        if noise:
            # Apply decayed exploration noise
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
        
        # --- Behavioral Cloning (BC) Augmentation ---
        # If expert trajectories are provided, augment the batch with them
        if best_trajectories:
            flat_best_trajectories = [exp for traj in best_trajectories for exp in traj]
            if flat_best_trajectories:
                # Replace a portion of the batch with expert data
                num_expert_samples = min(len(flat_best_trajectories), self.batch_size // 4) # e.g., 25% expert data
                expert_samples = random.sample(flat_best_trajectories, num_expert_samples)
                
                # Replace the first `num_expert_samples` in the batch
                batch = expert_samples + batch[num_expert_samples:]
                # For expert data, IS weights are 1.0 as they are always included
                is_weights[:num_expert_samples] = [1.0] * num_expert_samples

        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).view(-1, 1).to(self.device)
        is_weights = torch.FloatTensor(np.array(is_weights)).view(-1, 1).to(self.device)

        with torch.no_grad():
            # --- Target Policy Smoothing ---
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action_norm = self.actor_target(next_state)
            next_action_norm_clipped = (next_action_norm + noise).clamp(-1.0, 1.0)
            next_action = self._scale_action_torch(next_action_norm_clipped)

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
            action_pred = self._scale_action_torch(action_norm_pred)
            q_value_pred = self.critic1(state, action_pred)
            pg_loss = -q_value_pred.mean()

            # --- Behavioral Cloning (BC) Loss on High-Reward Samples ---
            bc_loss = 0.0
            if best_trajectories:
                # Find which of the current batch samples are from the expert trajectories
                # This is a simplified check; for robustness, consider a more sophisticated way to tag expert data
                expert_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
                # Heuristic: check if the reward is high, assuming expert trajectories have higher rewards
                # A more robust method would be to pass expert flags during sampling
                # For now, we apply BC loss on the expert samples we added to the batch
                if 'num_expert_samples' in locals() and num_expert_samples > 0:
                    expert_mask[:num_expert_samples] = True

                if expert_mask.any():
                    expert_state = state[expert_mask]
                    expert_action = action[expert_mask]
                    
                    expert_action_pred_norm = self.actor(expert_state)
                    expert_action_pred = self._scale_action_torch(expert_action_pred_norm)
                    
                    bc_loss = nn.MSELoss()(expert_action_pred, expert_action)

            # Combine the losses
            actor_loss = pg_loss + bc_loss * self.bc_weight

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
        (s.tolist(), a.tolist(), r, ns.tolist(), d)
        for s, a, r, ns, d in agent.replay_buffer
    ]
    replay_buffer_state = {
        'tree': agent.replay_buffer.tree.tree.tolist(),
        'data': replay_buffer_data_list,
        'n_entries': agent.replay_buffer.tree.n_entries,
        'write': agent.replay_buffer.tree.write,
        'max_priority': agent.replay_buffer.max_priority
    }
    
    state = {
        'training_params': training_params,
        'rewards': rewards,
        'episode': episode,
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
    
    training_params = state['training_params']
    rewards = state['rewards']
    start_episode = state['episode']
    
    # Load the PER buffer state
    replay_buffer_state = state.get('replay_buffer_state')
    if replay_buffer_state:
        agent.replay_buffer.tree.tree = np.array(replay_buffer_state['tree'])
        
        # Reconstruct data with numpy arrays
        loaded_data = replay_buffer_state['data']
        agent.replay_buffer.tree.data = np.zeros(agent.replay_buffer.capacity, dtype=object)
        for i, (s_list, a_list, r, ns_list, d) in enumerate(loaded_data):
            state_np = np.array(s_list, dtype=np.float32)
            action_np = np.array(a_list, dtype=np.float32)
            next_state_np = np.array(ns_list, dtype=np.float32)
            agent.replay_buffer.tree.data[i] = (state_np, action_np, r, next_state_np, d)

        agent.replay_buffer.tree.n_entries = replay_buffer_state['n_entries']
        agent.replay_buffer.tree.write = replay_buffer_state['write']
        agent.replay_buffer.max_priority = replay_buffer_state.get('max_priority', 1.0)
        print(f"PER buffer loaded with {agent.replay_buffer.tree.n_entries} experiences and their priorities.")
    else:
        # Fallback for old save format
        replay_buffer_list = state.get('replay_buffer', [])
        agent.replay_buffer.clear()
        for s_list, a_list, r, ns_list, d in replay_buffer_list:
            state_np = np.array(s_list, dtype=np.float32)
            action_np = np.array(a_list, dtype=np.float32)
            next_state_np = np.array(ns_list, dtype=np.float32)
            agent.push(state_np, action_np, r, next_state_np, d)
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

def save_reward_plot(rewards, plot_path='rl_reward_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reinforcement Learning Reward Curve')
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Reward curve saved to {plot_path}")
    return plot_path
