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

        # State: [target_temp_norm, current_temp_norm, uniformity_norm, temp_diff_norm, mean_temp_norm, min_temp_norm, max_temp_norm]
        self.action_dim = 3 if self.use_custom_directions else 4  # [temp, contact_time, cooling_time, (optional) direction]
        self.state_dim = 1 + self.temp_points + 1 + 1 + 3 # target_norm, temp_dist, uniformity, temp_diff, mean, min, max
        print(f"State dimension calculated: {self.state_dim} (temp_points: {self.temp_points}, action_dim: {self.action_dim})")
        self._setup_action_bounds(action_bounds)
        
        self.last_target_error = None
        self.last_uniformity_error = None
        
        # --- Reward Shaping Parameters ---
        self.w_trajectory = 20.0   # Base weight for the trajectory following reward
        self.w_uniformity_penalty = 2.0      # Penalty for non-uniform temperature, scaled by progress
        self.log_base = 1.5                  # Controls the curve of the logarithmic temperature gradient
        
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
            
            ("Warning: config.json not found or invalid, using defaults.")
            return {"main_params": {"T0": 400.0}}

    def reset(self):
        self.current_step = 0
        self.last_T1 = None
        self.last_cooling_T_matrix = None
        
        # Initialize error tracking for the new reward function
        t0_temp = self.base_config.get('main_params', {}).get('T0', 400.0)
        self.last_target_error = np.abs(t0_temp - self.target_temp_kelvin)
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

        state = np.concatenate([
            [target_temp_norm],
            current_temp_norm,
            [uniformity_norm],
            [temp_diff_norm],
            [mean_temp_norm],
            [min_temp_norm],
            [max_temp_norm]
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
            return -200.0  # Heavy penalty for simulation failure

        # --- Calculate current state errors ---
        current_temp_distribution = self.last_cooling_T_matrix[:, -1]
        mean_current_temp = np.mean(current_temp_distribution)
        uniformity_error = np.std(current_temp_distribution)

        # --- Logarithmic Temperature Gradient ---
        t0 = self.base_config.get('main_params', {}).get('T0', 400.0)
        progress = (self.current_step + 1) / self.n_rolls
        log_progress = np.log(1 + (self.log_base - 1) * progress) / np.log(self.log_base)
        step_target_temp = t0 - (t0 - self.target_temp_kelvin) * log_progress

        # --- Reward Calculation ---
        # 1. Trajectory Reward: Reward for being close to the step-wise target.
        temp_diff = np.abs(mean_current_temp - step_target_temp)
        trajectory_reward = self.w_trajectory * np.exp(-0.1 * temp_diff)

        # 2. Uniformity Penalty: Penalize non-uniform temperature. The penalty gets stronger over time.
        uniformity_penalty = -self.w_uniformity_penalty * uniformity_error * progress

        reward = trajectory_reward + uniformity_penalty

        # --- Update trackers for next step ---
        self.last_uniformity_error = uniformity_error
        
        # --- Final Step Bonus ---
        if done:
            target_error = np.abs(mean_current_temp - self.target_temp_kelvin)
            final_bonus = 1000 * np.exp(-0.5 * target_error - 0.8 * uniformity_error)
            reward += final_bonus
            print(f"Final Step - Mean Temp: {mean_current_temp:.2f}K, Final Bonus: {final_bonus:.2f}")
        
        print(f"Step {self.current_step}/{self.n_rolls} - Reward: {reward:.2f}, Temp Diff: {temp_diff:.2f}, STD: {uniformity_error:.2f}")

        return float(reward)

# ---------------- Actor / Critic / DDPG (Final Version) ----------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, action_dim)
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        return torch.tanh(self.layer3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 1)
    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x = torch.relu(self.bn1(self.layer1(xu)))
        x = torch.relu(self.bn2(self.layer2(x)))
        return self.layer3(x)

class TD3:
    def __init__(self, state_dim, action_dim, action_low, action_high, device, bc_weight=0.2):
        self.device = device
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

        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 256
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
        self.replay_buffer.append((state.copy(), action.copy(), float(reward), next_state.copy(), float(done)))

    def train(self, best_trajectories=None):
        if len(self.replay_buffer) < self.batch_size: return
        self.total_it += 1
        
        # Decay exploration noise
        self.exploration_noise_std = self.final_exploration_noise_std + \
            (self.initial_exploration_noise_std - self.final_exploration_noise_std) * \
            max(0, 1 - self.total_it / self.exploration_decay_steps)

        # 明确设置所有网络为训练模式，确保梯度计算和参数更新正常进行
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        # --- Stratified Sampling by Temperature ---
        # Group experiences into bins based on mean temperature
        experiences_by_temp = {}
        # Unpack state to get mean temperature for binning
        # State: [target_temp_norm, current_temp_norm, uniformity_norm, temp_diff_norm, mean_temp_norm, min_temp_norm, max_temp_norm]
        # The mean_temp_norm is the -4th element from the end of the state vector
        for exp in self.replay_buffer:
            state = exp[0]
            mean_temp_norm = state[-4]
            # Bin experiences into 10 bins based on normalized mean temperature
            bin_index = int(mean_temp_norm * 10)
            if bin_index not in experiences_by_temp:
                experiences_by_temp[bin_index] = []
            experiences_by_temp[bin_index].append(exp)

        batch = []
        
        # Distribute batch size among available temperature bins
        available_bins = sorted(experiences_by_temp.keys())
        if not available_bins:
            return # Not enough data to train

        samples_per_bin = self.batch_size // len(available_bins)
        remainder = self.batch_size % len(available_bins)

        for i, bin_index in enumerate(available_bins):
            num_samples = samples_per_bin + (1 if i < remainder else 0)
            bin_experiences = experiences_by_temp[bin_index]
            
            if len(bin_experiences) <= num_samples:
                # If not enough experiences for this bin, take all of them
                batch.extend(bin_experiences)
            else:
                # Sample randomly from this bin's experiences
                batch.extend(random.sample(bin_experiences, num_samples))
        
        flat_best_trajectories = []
        if best_trajectories:
            for traj in best_trajectories:
                flat_best_trajectories.extend(traj)
            batch.extend(flat_best_trajectories)
        
        # --- Log sampled data ---
        log_entry = {
            "training_iteration": self.total_it,
            "sampled_batch": [
                {
                    "state": s.tolist(),
                    "action": a.tolist(),
                    "reward": r,
                    "next_state": ns.tolist(),
                    "done": d
                }
                for s, a, r, ns, d in batch
            ]
        }
        try:
            with open("sampled_batch_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging sampled batch: {e}")

        random.shuffle(batch)

        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).view(-1, 1).to(self.device)

        with torch.no_grad():
            # --- Target Policy Smoothing ---
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # Compute the next action from the target actor and add noise
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
        critic1_loss = nn.MSELoss()(current_Q1, target_Q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # --- Critic 2 Update ---
        current_Q2 = self.critic2(state, action)
        critic2_loss = nn.MSELoss()(current_Q2, target_Q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Delayed Policy Updates ---
        if self.total_it % self.policy_freq == 0:
            # --- Actor Update with Behavioral Cloning from High-Reward Samples ---
            
            # 1. Standard Policy Gradient (PG) Loss
            # This encourages the actor to output actions that the critic estimates to have high Q-values.
            action_norm_pred = self.actor(state)
            action_pred = self._scale_action_torch(action_norm_pred)
            q_value_pred = torch.min(self.critic1(state, action_pred), self.critic2(state, action_pred))
            pg_loss = -q_value_pred.mean()

            # 2. Behavioral Cloning (BC) Loss on High-Reward Samples
            bc_loss = 0.0
            if best_trajectories:
                flat_best_trajectories = [exp for traj in best_trajectories for exp in traj]
                expert_samples = [s for s in batch if s in flat_best_trajectories]
                if expert_samples:
                    expert_state, expert_action, _, _, _ = zip(*expert_samples)
                    expert_state = torch.FloatTensor(np.array(expert_state)).to(self.device)
                    expert_action = torch.FloatTensor(np.array(expert_action)).to(self.device)
                    
                    expert_action_pred_norm = self.actor(expert_state)
                    expert_action_pred = self._scale_action_torch(expert_action_pred_norm)
                    
                    bc_loss = nn.MSELoss()(expert_action_pred, expert_action)

            # 3. Combine the losses
            actor_loss = pg_loss + bc_loss * self.bc_weight

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

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
    """保存完整的训练状态到JSON文件，包括经验池"""
    # 将numpy数组转换为列表以便JSON序列化
    replay_buffer_list = [
        (s.tolist(), a.tolist(), r, ns.tolist(), d)
        for s, a, r, ns, d in agent.replay_buffer
    ]
    
    state = {
        'training_params': training_params,
        'rewards': rewards,
        'episode': episode,
        'replay_buffer': replay_buffer_list
    }
    
    # 保存模型权重到.pth文件
    agent_path = os.path.splitext(output_path)[0] + ".pth"
    agent.save_checkpoint(agent_path)
    
    # 保存包括经验池在内的所有其他信息到JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    print(f"Training state (including replay buffer) saved to {output_path}")

def load_training_state(filepath, agent):
    """从JSON文件加载完整的训练状态，包括经验池"""
    if not os.path.exists(filepath):
        print(f"Training state file not found: {filepath}")
        return None, [], 0

    with open(filepath, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    training_params = state['training_params']
    rewards = state['rewards']
    start_episode = state['episode']
    
    # 从JSON文件加载经验池
    replay_buffer_list = state.get('replay_buffer', [])
    agent.replay_buffer.clear()
    for s_list, a_list, r, ns_list, d in replay_buffer_list:
        state_np = np.array(s_list, dtype=np.float32)
        action_np = np.array(a_list, dtype=np.float32)
        next_state_np = np.array(ns_list, dtype=np.float32)
        agent.push(state_np, action_np, r, next_state_np, d)
    print(f"Replay buffer loaded with {len(agent.replay_buffer)} experiences from JSON.")

    # 从.pth文件加载模型权重
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
