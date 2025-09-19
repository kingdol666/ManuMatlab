import json
import os
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from logic.rl_optimizer import (
    FilmCastingEnv, TD3,
    save_best_params_as_json, save_reward_plot,
    save_training_state, load_training_state
)
from logic.run_matlab_simulation import start_shared_engine, stop_shared_engine, restart_shared_engine

# 定义一个阈值，当MATLAB调用次数超过该值时，重启引擎以释放内存和资源
MATLAB_RESTART_INTERVAL = 1000  # 每50调用后重启
SAVE_INTERVAL = 5  # 每5个episode保存一次

class RlOptimizationThread(QThread):
    """在单独的线程中运行强化学习优化，并实时报告进度"""
    log_updated = pyqtSignal(str, str)
    optimization_finished = pyqtSignal(str, str)
    error_occurred = pyqtSignal(dict)
    finished = pyqtSignal() # 线程完成信号

    # 在 __init__ 中增加一个高分阈值（你也可以从外部参数传入）
    def __init__(self, num_episodes, n_rolls, target_temp, bounds, checkpoint_path=None, 
                 use_custom_directions=False, custom_directions=None):
        super().__init__()
        self.num_episodes = num_episodes
        self.n_rolls = n_rolls
        self.target_temp = target_temp
        self.bounds = bounds
        self.checkpoint_path = checkpoint_path
        self.use_custom_directions = use_custom_directions
        self.custom_directions = custom_directions if custom_directions is not None else []
        self.result_dir = "RLresult"
        self.output_path = os.path.join(self.result_dir, f"best_params_{n_rolls}rolls_{num_episodes}eps.json")
        self._is_running = True
        self.high_score_threshold = 450  # 仅当奖励高于此阈值时，更新最佳动作

    def stop(self):
        """请求停止线程"""
        self.log_updated.emit("正在停止优化...", "warning")
        self._is_running = False

    def run(self):
        """执行优化任务，并在每个 episode 后发射信号"""
        os.makedirs(self.result_dir, exist_ok=True)
        matlab_step_counter = 0  # 初始化MATLAB调用计数器
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_updated.emit(f"Using device: {device}", "info")
            
            start_shared_engine()
            self.log_updated.emit("共享MATLAB引擎已启动。", "info")

            env = FilmCastingEnv(
                n_rolls=self.n_rolls,
                target_temp=self.target_temp,
                action_bounds=self.bounds,
                use_custom_directions=self.use_custom_directions,
                custom_directions=self.custom_directions
            )
            agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim,
                        action_low=env.action_low, action_high=env.action_high, device=device)
            
            rewards, best_reward, best_params = [], -np.inf, None
            best_action_history = None # To store the sequence of actions from the best episode
            best_trajectories = [] # Store the top 4 best trajectories
            start_episode = 0

            # --- 加载检查点 ---
            if self.checkpoint_path:
                self.log_updated.emit(f"正在从 {self.checkpoint_path} 加载检查点...", "info")
                # First, load parameters from the JSON to correctly initialize the agent
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                training_params = saved_state.get('training_params', {})
                
                if training_params:
                    self.n_rolls = training_params.get('n_rolls', self.n_rolls)
                    self.target_temp = training_params.get('target_temp', self.target_temp)
                    self.bounds = training_params.get('bounds', self.bounds)
                    
                    # Re-initialize env and agent with loaded parameters BEFORE loading weights
                    env = FilmCastingEnv(
                        n_rolls=self.n_rolls,
                        target_temp=self.target_temp,
                        action_bounds=self.bounds,
                        use_custom_directions=self.use_custom_directions,
                        custom_directions=self.custom_directions
                    )
                    agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim,
                                action_low=env.action_low, action_high=env.action_high, device=device)

                    # Now, load the full state (weights, buffer, etc.)
                    _, rewards, start_episode = load_training_state(self.checkpoint_path, agent)
                    self.log_updated.emit("检查点加载成功。", "success")
                else:
                    self.log_updated.emit("加载检查点失败，开始新的训练。", "warning")

            # --- 数据预采集阶段 ---
            pre_collect_size = 128 # 增加预
            if not self.checkpoint_path or len(agent.replay_buffer) < pre_collect_size:
                self.log_updated.emit(f"经验池大小不足 {pre_collect_size}，正在进行数据预采集...", "info")
            while len(agent.replay_buffer) < pre_collect_size:
                if not self._is_running: break
                state = env.reset()
                for step in range(self.n_rolls):
                    if not self._is_running: break
                    action = np.random.uniform(low=env.action_low, high=env.action_high, size=env.action_dim).astype(np.float32)
                    next_state, reward, done, _ = env.step(action)

                    if reward < -999: # Check for simulation error
                        self.log_updated.emit(f"数据采集中检测到异常奖励值 ({reward:.2f})。", "error")
                        self.log_updated.emit("正在重启MATLAB引擎...", "warning")
                        stop_shared_engine()
                        matlab_step_counter = 0
                        self.log_updated.emit("MATLAB引擎重启完成。", "info")
                        break
                    
                    agent.push(state, action, reward, next_state, done)
                    state = next_state
                    if done: break
                
                if 'reward' in locals() and reward < -999:
                    continue
                matlab_step_counter += 1
                
                progress_msg = f"数据预采集中... ({len(agent.replay_buffer)}/{pre_collect_size})"
                self.log_updated.emit(progress_msg, "info")

                if matlab_step_counter >= MATLAB_RESTART_INTERVAL:
                    self.log_updated.emit(f"达到 {MATLAB_RESTART_INTERVAL} 次模拟，正在重启MATLAB引擎...", "warning")
                    restart_shared_engine()
                    matlab_step_counter = 0
                    self.log_updated.emit("MATLAB引擎重启完成。", "info")
            
            self.log_updated.emit(f"数据预采集完成，经验池大小: {len(agent.replay_buffer)}。开始正式训练...", "info")

            # --- 正式训练循环 ---
            # track consecutive episodes with no improvement
            no_improve_counter = 0
            NO_IMPROVE_THRESHOLD = 10  # 连续多少个 episode 无提升则触发探索策略
            EXTRA_RANDOM_EXPLORATIONS = 3
            EXTRA_GUIDED_EXPLORATIONS = 2

            for episode in range(start_episode, self.num_episodes):
                if not self._is_running:
                    self.log_updated.emit("优化过程已被用户中断。", "warning")
                    # --- 保存中断时的状态 ---
                    self.log_updated.emit("正在保存中断前的训练状态...", "info")
                    training_params = {
                        'num_episodes': self.num_episodes,
                        'n_rolls': self.n_rolls,
                        'target_temp': self.target_temp,
                        'bounds': self.bounds,
                        'use_custom_directions': self.use_custom_directions,
                        'custom_directions': self.custom_directions
                    }
                    save_dir = "modelSave"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    checkpoint_save_path = os.path.join(save_dir, f"training_state_{self.n_rolls}rolls_{self.num_episodes}eps.json")
                    save_training_state(agent, training_params, rewards, episode, checkpoint_save_path)
                    self.log_updated.emit(f"训练状态已保存至 {checkpoint_save_path}", "success")
                    break

                state = env.reset()
                episode_reward = 0
                current_trajectory = [] # Add this line to track the current episode's trajectory

                for step in range(self.n_rolls):
                    if not self._is_running: break
                    
                    # Standard Exploration: Use the policy with noise
                    action = agent.select_action(state, noise=True)

                    next_state, reward, done, _ = env.step(action)
                    
                    # 检查奖励值是否异常 (包括超时)
                    if reward < -999:
                        self.log_updated.emit(f"检测到异常奖励值 ({reward:.2f})，可能表示MATLAB模拟出错或超时。", "error")
                        self.log_updated.emit("正在跳过当前回合，此经验将不会被学习。", "warning")
                        stop_shared_engine()
                        break  # 中断当前 episode 的内部循环

                    # Store the experience in the current trajectory as well as the main buffer
                    current_trajectory.append((state, action, reward, next_state, done))
                    agent.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    if done:
                        break
                
                # --- 在每个 Episode 结束后进行训练 ---
                if self._is_running and episode_reward > -999: # Do not train on failed episodes
                    # Perform a single training step, leveraging the best_trajectories for prioritized replay
                    agent.train(best_trajectories)

                matlab_step_counter += 1 # 每个episode结束后计数器+1
                
                rewards.append(episode_reward)
                
                if episode_reward > best_reward and episode_reward >= self.high_score_threshold:
                    best_reward = episode_reward
                    best_params = env.action_history
                    best_action_history = env.action_history.copy()  # 只保存高分episode的动作作为引导策略
                    
                    # Add the new best trajectory and keep the list sorted
                    best_trajectories.append(current_trajectory)
                    best_trajectories = sorted(best_trajectories, key=lambda traj: sum(exp[2] for exp in traj), reverse=True)
                    
                    # Keep only the top 4
                    if len(best_trajectories) > 4:
                        best_trajectories = best_trajectories[:4]
                    
                    # --- Instantly save best parameters and plot upon finding a better solution ---
                    self.log_updated.emit(f"发现新的最优奖励: {best_reward:.4f}！正在保存参数和模型权重...", "success")
                    save_best_params_as_json(best_params, self.output_path)
                    
                    # Save the best model weights
                    model_save_dir = "modelSave"
                    os.makedirs(model_save_dir, exist_ok=True)
                    model_path = os.path.join(model_save_dir, f"best_model_{self.n_rolls}rolls_{self.num_episodes}eps.pth")
                    agent.save_best_model(model_path)

                    plot_path = os.path.join(self.result_dir, f"rl_reward_curve_{self.n_rolls}rolls_{self.num_episodes}eps.png")
                    save_reward_plot(rewards, plot_path)
                    # reset no-improve counter on improvement
                    no_improve_counter = 0
                else:
                    # increment when no improvement
                    no_improve_counter += 1

                # 如果连续多轮没有提升，触发额外探索策略
                if no_improve_counter >= NO_IMPROVE_THRESHOLD and self._is_running:
                    self.log_updated.emit(f"检测到 {no_improve_counter} 轮无提升，触发额外探索策略...", "warning")

                    def run_exploration_episode():
                        nonlocal matlab_step_counter
                        if not self._is_running:
                            return None
                        s = env.reset()
                        ep_r = 0.0
                        for _ in range(self.n_rolls):
                            if not self._is_running: break
                            a = np.random.uniform(low=env.action_low, high=env.action_high, size=env.action_dim).astype(np.float32)
                            ns, r, d, _ = env.step(a)
                            if r < -999:
                                self.log_updated.emit(f"探索过程中检测到异常奖励值 ({r:.2f})，正在重启MATLAB引擎...", "error")
                                stop_shared_engine()
                                matlab_step_counter = 0
                                self.log_updated.emit("MATLAB引擎重启完成。", "info")
                                return None
                            agent.push(s, a, r, ns, d)
                            s = ns
                            ep_r += r
                            if d: break
                        matlab_step_counter += 1
                        return ep_r

                    # 进行N次完全随机探索
                    total_explorations = EXTRA_RANDOM_EXPLORATIONS + EXTRA_GUIDED_EXPLORATIONS
                    for i in range(total_explorations):
                        if not self._is_running: break
                        rr = run_exploration_episode()
                        self.log_updated.emit(f"额外随机探索 {i+1}/{total_explorations} 完成，reward={rr}", "info")

                    # 重置计数器并继续训练
                    no_improve_counter = 0

                # --- Save good policies based on score gradient ---
                if episode_reward > 650:
                    base_dir = os.path.join(self.result_dir, "good_policies")
                    if episode_reward > 850:
                        policy_dir = os.path.join(base_dir, "850+")
                    elif episode_reward > 750:
                        policy_dir = os.path.join(base_dir, "750-850")
                    else: # 650-750
                        policy_dir = os.path.join(base_dir, "650-750")
                    
                    os.makedirs(policy_dir, exist_ok=True)
                    filename = f"episode_{episode + 1}_reward_{episode_reward:.2f}.json"
                    filepath = os.path.join(policy_dir, filename)
                    save_best_params_as_json(env.action_history, filepath)
                    self.log_updated.emit(f"高分策略已保存至: {filepath}", "success")

                # Log progress and replay buffer size
                progress_msg = (
                    f"Episode {episode + 1}/{self.num_episodes} | "
                    f"Reward: {episode_reward:.4f} | "
                    f"Best Reward: {best_reward:.4f} | "
                    f"Replay Buffer: {len(agent.replay_buffer)}"
                )
                self.log_updated.emit(progress_msg, "info")

                # --- 定期保存检查点 ---
                if (episode + 1) % SAVE_INTERVAL == 0:
                    self.log_updated.emit(f"达到 {SAVE_INTERVAL} 个 episodes，正在保存训练状态...", "info")
                    training_params = {
                        'num_episodes': self.num_episodes,
                        'n_rolls': self.n_rolls,
                        'target_temp': self.target_temp,
                        'bounds': self.bounds,
                        'use_custom_directions': self.use_custom_directions,
                        'custom_directions': self.custom_directions
                    }
                    save_dir = "modelSave"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    checkpoint_save_path = os.path.join(save_dir, f"training_state_{self.n_rolls}rolls_{self.num_episodes}eps.json")
                    save_training_state(agent, training_params, rewards, episode + 1, checkpoint_save_path)
                    self.log_updated.emit(f"训练状态已保存至 {checkpoint_save_path}", "success")

                # --- 周期性重启MATLAB引擎 ---
                if matlab_step_counter >= MATLAB_RESTART_INTERVAL:
                    self.log_updated.emit(f"达到 {MATLAB_RESTART_INTERVAL} 次模拟，正在执行MATLAB引擎热插拔以优化性能...", "warning")
                    restart_shared_engine()
                    matlab_step_counter = 0  # 重置计数器
                    self.log_updated.emit("MATLAB引擎热插拔完成。", "info")

            if best_params is not None and self._is_running:
                self.log_updated.emit("\n全部优化流程完成。", "success")
                # Final save is already done in real-time, just emit the signal with the final paths
                json_path = self.output_path
                plot_path = os.path.join(self.result_dir, f"rl_reward_curve_{self.n_rolls}rolls_{self.num_episodes}eps.png")
                self.optimization_finished.emit(json_path, plot_path)
            elif not self._is_running:
                pass
            else:
                self.log_updated.emit("未能找到有效的最优参数。", "warning")

        except Exception as e:
            error_info = {
                "type": "强化学习优化错误",
                "message": str(e),
                "solution": "请检查控制台输出以获取详细的错误信息。确保MATLAB环境已正确配置。"
            }
            self.error_occurred.emit(error_info)
        finally:
            stop_shared_engine()
            self.log_updated.emit("共享MATLAB引擎已关闭。", "info")
            self.finished.emit()
