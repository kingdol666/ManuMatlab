# Standard library imports
from __future__ import annotations
import os
import json
import pickle
from typing import TYPE_CHECKING

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# 尝试设置支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 'SimHei' 是黑体, 'Microsoft YaHei' 是微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
except Exception as e:
    print(f"设置中文字体失败: {e}")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from PyQt6.QtCore import Qt, QThread, QMetaObject, QTimer, Q_ARG
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QMessageBox, QVBoxLayout, QScrollArea, QWidget,
    QComboBox, QLabel, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QGridLayout, QPushButton
)
# Project module imports
from .Models import ScriptType, RollDirection
if TYPE_CHECKING:
    from simulation_gui import SimulationGUI
from .plot_utils import get_combined_contour_data
from .run_matlab_simulation import load_config


class VisualizationModel:
    """Holds the state and data for visualization."""

    def __init__(self):
        """Initializes the visualization model."""
        self.visualization_active = False
        self.current_step = 0
        self.film_position = 0
        self.roll_angle = 0
        self.live_visualization_active = False
        self.animation_frame = 0
        self.avg_temperatures = []
        self.visualization_source_data = []
        self.visualization_models = []
        self.temperature_curve_data = None
        self.current_simulation_time = 0
        self.total_simulation_time = 0

    def clear(self):
        """Clears all visualization data."""
        self.visualization_active = False
        self.current_step = 0
        self.film_position = 0
        self.roll_angle = 0
        self.live_visualization_active = False
        self.animation_frame = 0
        self.avg_temperatures = []
        self.visualization_source_data = []
        self.visualization_models = []
        self.temperature_curve_data = None
        self.current_simulation_time = 0
        self.total_simulation_time = 0


class VisualizationManager:
    """Manages all visualization-related functionalities for the Simulation GUI."""

    def __init__(self, gui: 'SimulationGUI'):
        """
        Initializes the VisualizationManager.

        Args:
            gui: The main SimulationGUI instance.
        """
        self.gui: 'SimulationGUI' = gui
        self.gradient_dialog = None
        self.temperature_curve_dialog = None

        # Encapsulated visualization state and data model
        self.model = VisualizationModel()

        # --------- 动画优化相关属性 ---------
        self.dynamic_initialized = False
        self.anim_ax = None
        self.film_segment_artist = None
        self.roll_indicator_infos = []
        self.avg_temp_texts = []
        # --------- 异步线程相关 ---------
        self.worker_thread = None
        self.vis_worker = None
        # --------- 节流相关 ---------
        self.last_update_time = 0
        self.throttle_timer = QTimer()
        self.throttle_timer.setInterval(33)  # 每秒最多更新约 30 次
        self.throttle_timer.setSingleShot(True)
        self.throttle_timer.timeout.connect(self._throttled_update)
        self.pending_frame_data = None

    def clear_data(self):
        """Clears all visualization data."""
        self.model.clear()

    # ---------------- Live Visualization / Schematic Methods ----------------
    def toggle_live_contour_visibility(self, state):
        """Toggles the visibility of the live contour plot."""
        self.gui.live_contour_group.setVisible(state == Qt.CheckState.Checked.value)
        self.draw_live_visualization()

    def draw_live_visualization(self):
        """Draws the live contour plot on its dedicated canvas."""
        if not self.gui.show_contour_checkbox.isChecked():
            self.gui.live_contour_figure.clear()
            self.gui.live_contour_canvas.draw()
            return

        self.gui.live_contour_figure.clear()
        ax = self.gui.live_contour_figure.add_subplot(111)
        ax.set_title('实时等高线图')
        ax.set_xlabel('工艺流程 (时间 / s)')
        ax.set_ylabel('薄膜厚度 (m)')

        if not self.gui.simulation_step_results:
            ax.text(0.5, 0.5, "等待仿真数据...", ha='center', va='center', transform=ax.transAxes)
            self.gui.live_contour_canvas.draw()
            return

        all_t_min, all_t_max = float('inf'), float('-inf')
        valid_steps_data = []

        for step_result in self.gui.simulation_step_results:
            output = step_result.get('output', {})
            if all(k in output for k in ['t', 'JXYV', 'BN2', 'T']):
                try:
                    T_data = np.array(output['T'])
                    if T_data.size > 0:
                        all_t_min = min(all_t_min, np.nanmin(T_data))
                        all_t_max = max(all_t_max, np.nanmax(T_data))
                        valid_steps_data.append(step_result)
                except Exception as e:
                    print(f"Live viz error (pass 1) step {step_result.get('model_id', 'N/A')}: {e}")

        if not valid_steps_data:
            ax.text(0.5, 0.5, "无有效数据", ha='center', va='center', transform=ax.transAxes)
            self.gui.live_contour_canvas.draw()
            return

        cumulative_time = 0
        mesh = None

        for step_result in valid_steps_data:
            output = step_result['output']
            try:
                t_duration = float(np.array(output['t']).item())
                JXYV, BN2, T_data = np.array(output['JXYV']), np.array(output['BN2']), np.array(output['T'])
                
                BN2_int = BN2.astype(int).flatten() - 1
                valid_indices = BN2_int[(BN2_int >= 0) & (BN2_int < JXYV.shape[0]) & (BN2_int < T_data.shape[0])]
                if valid_indices.size == 0:
                    cumulative_time += t_duration
                    continue

                num_time_steps = T_data.shape[1]
                time_array = np.linspace(0, t_duration, num_time_steps) + cumulative_time
                y_coords = JXYV[valid_indices, 1]
                
                xxxx, yyyy = np.meshgrid(time_array, y_coords)
                tttt = T_data[valid_indices, :]

                mesh = ax.contourf(xxxx, yyyy, tttt, cmap='jet', levels=np.linspace(all_t_min, all_t_max, 100))
                cumulative_time += t_duration
            except Exception as e:
                print(f"Live viz error (pass 2) step {step_result.get('model_id', 'N/A')}: {e}")

        # Set axis limits to be tight
        if valid_steps_data:
            # Find overall y limits
            y_min, y_max = float('inf'), float('-inf')
            for step_result in valid_steps_data:
                output = step_result['output']
                JXYV, BN2 = np.array(output['JXYV']), np.array(output['BN2'])
                BN2_int = BN2.astype(int).flatten() - 1
                valid_indices = BN2_int[(BN2_int >= 0) & (BN2_int < JXYV.shape[0])]
                if valid_indices.size > 0:
                    y_coords = JXYV[valid_indices, 1]
                    y_min = min(y_min, y_coords.min())
                    y_max = max(y_max, y_coords.max())
            
            if y_min != float('inf'):
                ax.set_xlim(0, cumulative_time)
                ax.set_ylim(y_min, y_max) 

        # If animation is active, draw the progress pointer
        if self.model.visualization_active:
            ax.axvline(x=self.model.current_simulation_time, color='r', linestyle='--', linewidth=2)

        # Draw vertical lines to separate models
        boundary_time = 0
        for step_result in valid_steps_data:
            output = step_result['output']
            try:
                t_duration = float(np.array(output['t']).item())
                boundary_time += t_duration
                ax.axvline(x=boundary_time, color='k', linestyle=':', linewidth=1)
            except Exception as e:
                print(f"Live viz error (drawing boundary) step {step_result.get('model_id', 'N/A')}: {e}")

        if mesh:
            self.gui.live_contour_figure.colorbar(mesh, ax=ax, label='温度 (K)')
        self.gui.live_contour_figure.tight_layout()
        self.gui.live_contour_canvas.draw()

    def draw_model_schematic(self):
        """Draws a static schematic of the current model sequence based on time."""
        self.gui.visualization_figure.clear()
        ax = self.gui.visualization_figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('模型序列示意图')
        ax.set_xlabel('工艺流程 (时间 / s)')
        ax.get_yaxis().set_visible(False)

        if not self.gui.model_manager.simulation_models:
            ax.text(0.5, 0.5, "请在“模型管理”选项卡中添加模型",
                    ha='center', va='center', transform=ax.transAxes)
            self.gui.visualization_canvas.draw()
            return

        total_time = sum(model.t_up for model in self.gui.model_manager.simulation_models)
        if total_time <= 0:
            ax.text(0.5, 0.5, "模型总时长为0，无法绘制示意图",
                    ha='center', va='center', transform=ax.transAxes)
            self.gui.visualization_canvas.draw()
            return

        display_width = total_time * 1.8
        
        x_center = total_time / 2
        ax.set_xlim(x_center - display_width / 2, x_center + display_width / 2)
        ax.set_ylim(-5, 5)
        ax.plot([0, total_time], [0, 0], 'b-', linewidth=2, label='Film Path')

        cumulative_time = 0
        for i, model in enumerate(self.gui.model_manager.simulation_models):
            model_duration = model.t_up
            x_start = cumulative_time
            x_center = cumulative_time + model_duration / 2

            if model.script_type == ScriptType.HEATING:
                roll_radius = model_duration * 0.45
                y_center = 1.2
                if model.roll_direction == RollDirection.REVERSE:
                    y_center = -1.2
                
                roll = plt.Circle((x_center, y_center), roll_radius, color='gray', alpha=0.7)
                ax.add_patch(roll)
                ax.text(x_center, y_center, f"R{i+1}", ha='center', va='center', color='white', weight='bold')

            elif model.script_type == ScriptType.COOLING:
                gap = model_duration * 0.1
                cooling_box = Rectangle((x_start + gap / 2, -0.5), model_duration - gap, 1, color='lightblue', alpha=0.5)
                ax.add_patch(cooling_box)

            label = f"#{i+1}: {model.script_type}\n{model.roll_direction if model.script_type == ScriptType.HEATING else ''}"
            ax.text(x_center, -3.5, label, ha='center', va='center', fontsize=9)
            
            cumulative_time += model_duration
            if cumulative_time < total_time:
                ax.axvline(x=cumulative_time, color='k', linestyle=':', linewidth=0.5)

        self.gui.visualization_canvas.draw()
        
    def _setup_animation_canvas(self):
        """Prepares the static background and dynamic artists for optimized animation."""
        # Clear and set up figure/axes
        self.gui.visualization_figure.clear()
        self.anim_ax = self.gui.visualization_figure.add_subplot(111)
        ax = self.anim_ax
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('动态仿真过程')
        ax.set_xlabel('工艺流程 (时间 / s)')
        ax.get_yaxis().set_visible(False)

        total_time = self.model.total_simulation_time
        if total_time <= 0:
            return

        display_width = total_time * 1.8
        x_center = total_time / 2
        ax.set_xlim(x_center - display_width / 2, x_center + display_width / 2)
        ax.set_ylim(-5, 5)

        # Static film path
        ax.plot([0, total_time], [0, 0], 'b-', linewidth=2)

        # Draw models and create dynamic artists
        cumulative_time = 0
        self.roll_indicator_infos.clear()
        self.avg_temp_texts.clear()

        for i, model in enumerate(self.model.visualization_models):
            model_duration = model.t_up
            x_start = cumulative_time
            x_center_model = x_start + model_duration / 2

            if model.script_type == ScriptType.HEATING:
                roll_radius = model_duration * 0.45
                y_center = 1.2 if model.roll_direction in [RollDirection.FORWARD, RollDirection.INITIAL] else -1.2
                roll = plt.Circle((x_center_model, y_center), roll_radius, color='gray', alpha=0.7)
                ax.add_patch(roll)
                ax.text(x_center_model, y_center, f"R{i+1}", ha='center', va='center', color='white', weight='bold')

                # Indicator line (dynamic)
                indicator_line, = ax.plot([x_center_model, x_center_model + roll_radius * 0.8],
                                          [y_center, y_center], 'k-', linewidth=2)
                direction_factor = 1 if model.roll_direction in [RollDirection.FORWARD, RollDirection.INITIAL] else -1
                self.roll_indicator_infos.append({'line': indicator_line,
                                                  'x_center': x_center_model,
                                                  'y_center': y_center,
                                                  'roll_radius': roll_radius,
                                                  'direction': direction_factor})
            elif model.script_type == ScriptType.COOLING:
                gap = model_duration * 0.1
                cooling_box = Rectangle((x_start + gap / 2, -0.5), model_duration - gap, 1, color='lightblue', alpha=0.5)
                ax.add_patch(cooling_box)

            # Placeholder for average temperature
            avg_text = ax.text(x_center_model, -3.5, '', ha='center', va='center', fontsize=9, color='red', visible=False)
            self.avg_temp_texts.append({'text': avg_text,
                                        'start': cumulative_time,
                                        'end': cumulative_time + model_duration})

            cumulative_time += model_duration

        # Film segment rectangle (dynamic)
        film_segment_width = 0.02 * total_time
        self.film_segment_artist = Rectangle((0, -0.1), film_segment_width, 0.2, color='cyan', alpha=0.8)
        ax.add_patch(self.film_segment_artist)

        self.gui.visualization_canvas.draw()
        self.dynamic_initialized = True


    def start_visualization(self):
        """Starts the dynamic visualization of simulation results."""
        if not self.gui.simulation_step_results:
            QMessageBox.warning(self.gui, "警告", "没有可用的仿真结果数据")
            return

        self.model.visualization_source_data = list(self.gui.simulation_step_results)
        self.model.visualization_models = list(self.gui.model_manager.simulation_models)
            
        self.model.visualization_active = True
        self.model.current_step = 0
        self.model.animation_frame = 0
        self.model.total_simulation_time = sum(model.t_up for model in self.model.visualization_models)
        self.model.current_simulation_time = 0
        # 初始化优化画布
        self.dynamic_initialized = False
        self._setup_animation_canvas()
        
        self.model.avg_temperatures = []
        for step_result in self.model.visualization_source_data:
            avg_temp = None
            if 'output' in step_result and 'T' in step_result['output']:
                T = np.array(step_result['output']['T'])
                if T.size > 0:
                    avg_temp = np.mean(T)
            self.model.avg_temperatures.append(avg_temp)

        self.gui.start_visualization_button.setEnabled(False)
        self.gui.stop_visualization_button.setEnabled(True)
        self.gui.next_frame_button.setEnabled(True)
        self.gui.previous_frame_button.setEnabled(True)
        self.gui.pause_visualization_button.setEnabled(True)
        self.gui.pause_visualization_button.setText("暂停")
        self.gui.current_status_label.setText("可视化运行中")
        
        # 启动后台可视化线程
        from threads.visualization_worker import VisualizationWorker
        if self.worker_thread and self.worker_thread.isRunning():
            self.vis_worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.worker_thread = QThread()
        self.vis_worker = VisualizationWorker(self.model.total_simulation_time)
        self.vis_worker.moveToThread(self.worker_thread)

        # 信号连接
        self.vis_worker.update_frame.connect(self._on_worker_update_frame)
        self.vis_worker.finished.connect(self.worker_thread.quit)

        # 线程启动后在工作线程上下文初始化定时器
        self.worker_thread.started.connect(self.vis_worker.start)
        # 启动线程
        self.worker_thread.start()
        
    def stop_visualization(self):
        """Stops the dynamic visualization."""
        self.model.visualization_active = False
        if self.vis_worker:
            QMetaObject.invokeMethod(self.vis_worker, "stop", Qt.ConnectionType.QueuedConnection)
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        
        self.gui.start_visualization_button.setEnabled(True)
        self.gui.stop_visualization_button.setEnabled(False)
        self.gui.next_frame_button.setEnabled(False)
        self.gui.previous_frame_button.setEnabled(False)
        self.gui.pause_visualization_button.setEnabled(False)
        self.gui.current_status_label.setText("可视化已停止")
        self.draw_model_schematic()

    def restart_visualization(self):
        """Restarts the visualization, typically after a simulation run completes."""
        if self.model.visualization_active:
            self.stop_visualization()
        if self.gui.simulation_step_results:
            self.start_visualization()

    def next_frame(self):
        """Advances to the next frame in the animation via worker."""
        if self.vis_worker:
            QMetaObject.invokeMethod(self.vis_worker, "next_frame", Qt.ConnectionType.QueuedConnection)

    def previous_frame(self):
        """Goes back to the previous frame via worker."""
        if self.vis_worker:
            QMetaObject.invokeMethod(self.vis_worker, "previous_frame", Qt.ConnectionType.QueuedConnection)

    def _tick_animation(self):
        """Timer-triggered function to update the animation."""
        if self.model.visualization_active:
            self.model.animation_frame += 1
            self._update_animation_frame()
            self.draw_live_visualization()

    def _on_worker_update_frame(self, frame_idx, current_time):
        """响应后台线程的帧更新信号，并进行节流处理"""
        self.pending_frame_data = (frame_idx, current_time)
        if not self.throttle_timer.isActive():
            self.throttle_timer.start()

    def _throttled_update(self):
        """节流后的实际更新函数"""
        if self.pending_frame_data is None:
            return
        
        frame_idx, current_time = self.pending_frame_data
        self.model.animation_frame = frame_idx
        self.model.current_simulation_time = current_time
        
        self._update_animation_frame()
        self.draw_live_visualization()
        
        self.pending_frame_data = None

    def plot_temperature_profile(self):
        """Plots the temperature profile of the last column of T_data with enhanced visualization."""
        if not self.model.visualization_source_data:
            QMessageBox.warning(self.gui, "警告", "没有可用的可视化数据。")
            return

        last_step_result = self.model.visualization_source_data[-1]
        output = last_step_result.get('output', {})
        if not all(k in output for k in ['T', 'JXYV', 'BN2']):
            QMessageBox.warning(self.gui, "警告", "仿真结果缺少'T', 'JXYV', 或 'BN2'数据，无法绘制精确的温度分布。")
            return

        try:
            T_data = np.array(output['T'])
            JXYV = np.array(output['JXYV'])
            BN2 = np.array(output['BN2'])

            if T_data.ndim != 2 or T_data.shape[1] == 0:
                QMessageBox.warning(self.gui, "警告", "温度数据'T'格式不正确。")
                return

            # 从BN2和JXYV获取沿厚度方向的精确坐标
            BN2_int = BN2.astype(int).flatten() - 1  # MATLAB to Python indexing
            valid_indices = BN2_int[(BN2_int >= 0) & (BN2_int < JXYV.shape[0])]
            
            if valid_indices.size == 0:
                QMessageBox.warning(self.gui, "警告", "未能从'BN2'和'JXYV'中提取有效的坐标。")
                return

            # 距离是网格中的实际y坐标
            distance = JXYV[valid_indices, 1]
            
            # 提取最后一个时间步的温度数据
            # 假设T_data的行与JXYV的行对应
            if T_data.shape[0] == JXYV.shape[0]:
                temperature_profile_kelvin = T_data[valid_indices, -1]
            else:
                # 如果T_data的行数与BN2的长度匹配，则假定它已经被预先筛选
                if T_data.shape[0] == len(valid_indices):
                    temperature_profile_kelvin = T_data[:, -1]
                else:
                    QMessageBox.warning(self.gui, "警告", "温度数据'T'的行数与网格坐标不匹配。")
                    return

            temperature_profile_celsius = temperature_profile_kelvin - 273.15

            plt.style.use('seaborn-v0_8-whitegrid')
            dialog = QDialog(self.gui)
            dialog.setWindowTitle("最终温度分布")
            dialog.setMinimumSize(800, 600)
            layout = QVBoxLayout(dialog)
            
            fig = Figure(figsize=(8, 6), dpi=100)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

            ax = fig.add_subplot(111)

            # Set axis limits with padding
            temp_min, temp_max = np.min(temperature_profile_celsius), np.max(temperature_profile_celsius)
            padding = (temp_max - temp_min) * 0.1
            ax.set_xlim(temp_min - padding, temp_max + padding)
            ax.set_ylim(np.min(distance), np.max(distance))

            # Add a gradient background
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('coolwarm'), 
                      extent=[xmin, xmax, ymin, ymax], alpha=0.4)

            # Plot the main temperature curve on top
            ax.plot(temperature_profile_celsius, distance, marker='o', markersize=5, linestyle='-', 
                    linewidth=2.5, color='#2c3e50', label='温度分布')

            # Enhance labels and title
            ax.set_xlabel("温度 (°C)", fontsize=12, fontweight='bold')
            ax.set_ylabel("距离膜底部距离 (m)", fontsize=12, fontweight='bold')
            ax.set_title("沿膜厚度方向的最终温度分布", fontsize=16, fontweight='bold', pad=20)

            # Customize grid and ticks
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=10)

            fig.tight_layout()
            canvas.draw()
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"绘制温度分布图时出错: {e}")

    def show_realtime_temperature_gradient(self):
        """Shows a new window with the realtime temperature gradient."""
        if not self.model.visualization_active:
            QMessageBox.warning(self.gui, "警告", "请先开始可视化。")
            return

        self.gradient_dialog = QDialog(self.gui)
        self.gradient_dialog.setWindowTitle("实时温度梯度")
        self.gradient_dialog.setMinimumSize(900, 600)
        
        layout = QVBoxLayout(self.gradient_dialog)
        self.gradient_figure = Figure(figsize=(8, 6), dpi=100)
        self.gradient_canvas = FigureCanvas(self.gradient_figure)
        self.gradient_canvas.setObjectName("gradient_canvas")
        layout.addWidget(self.gradient_canvas)
        
        self.gradient_dialog.show()

    def _update_animation_frame(self):
        """Updates dynamic artists for the current animation frame without redrawing static background."""
        if not (self.model.visualization_active and self.dynamic_initialized):
            return

        # Update current simulation time based on frame index
        animation_duration_frames = 200
        time_ratio = (self.model.animation_frame % animation_duration_frames) / animation_duration_frames
        self.model.current_simulation_time = time_ratio * self.model.total_simulation_time

        # Update film segment position
        self.film_segment_artist.set_x(self.model.current_simulation_time - self.film_segment_artist.get_width() / 2)

        # Update roll indicator lines
        for info in self.roll_indicator_infos:
            angle_rad = np.radians(self.model.animation_frame * 5 * info['direction'])
            indicator_x = info['x_center'] + info['roll_radius'] * 0.8 * np.cos(angle_rad)
            indicator_y = info['y_center'] + info['roll_radius'] * 0.8 * np.sin(angle_rad)
            info['line'].set_data([info['x_center'], indicator_x], [info['y_center'], indicator_y])

        # Update average temperature text visibility/content
        for idx, txt_info in enumerate(self.avg_temp_texts):
            if txt_info['start'] <= self.model.current_simulation_time < txt_info['end']:
                if idx < len(self.model.avg_temperatures) and self.model.avg_temperatures[idx] is not None:
                    txt_info['text'].set_text(f"Avg Temp: {self.model.avg_temperatures[idx]:.2f} K")
                    txt_info['text'].set_visible(True)
            else:
                txt_info['text'].set_visible(False)

        # Efficient redraw
        self.gui.visualization_canvas.draw_idle()

        # Update gradient dialog if visible
        if hasattr(self, 'gradient_dialog') and self.gradient_dialog and self.gradient_dialog.isVisible():
            self._update_gradient_figure()
        return

    def on_playback_speed_changed(self, text):
        """Handles changes in the playback speed dropdown."""
        if self.vis_worker:
            try:
                speed_multiplier = float(text.replace('x', ''))
                QMetaObject.invokeMethod(self.vis_worker, "set_speed_multiplier",
                                         Qt.ConnectionType.QueuedConnection,
                                         Q_ARG(float, speed_multiplier))
            except (ValueError, IndexError):
                pass  # Ignore invalid values

    def toggle_visualization_pause(self):
        """Toggles the pause/resume state of the visualization."""
        if not self.model.visualization_active:
            return

        if self.vis_worker:
            QMetaObject.invokeMethod(self.vis_worker, "pause", Qt.ConnectionType.QueuedConnection)
            if self.gui.pause_visualization_button.text() == "暂停":
                self.gui.pause_visualization_button.setText("继续")
                self.gui.current_status_label.setText("可视化已暂停")
            else:
                self.gui.pause_visualization_button.setText("暂停")
                self.gui.current_status_label.setText("可视化运行中")

    def _update_gradient_figure(self):
        """Updates the temperature gradient figure."""
        if not hasattr(self, 'gradient_figure'):
            return
            
        self.gradient_figure.clear()
        ax = self.gradient_figure.add_subplot(111)
        
        cumulative_time = 0
        current_model_index = -1
        for i, model in enumerate(self.model.visualization_models):
            model_duration = model.t_up
            if self.model.current_simulation_time >= cumulative_time and self.model.current_simulation_time < cumulative_time + model_duration:
                current_model_index = i
                break
            cumulative_time += model_duration
        
        if current_model_index == -1:
            current_model_index = len(self.model.visualization_models) - 1

        if current_model_index >= 0 and current_model_index < len(self.model.visualization_source_data):
            step_result = self.model.visualization_source_data[current_model_index]
            if 'output' in step_result and all(k in step_result['output'] for k in ['t', 'JXYV', 'BN2', 'T']):
                output = step_result['output']
                t = output['t'].item() if isinstance(output['t'], np.ndarray) and output['t'].size == 1 else float(output['t'])
                JXYV, BN2, T_data = np.array(output['JXYV']), np.array(output['BN2']), np.array(output['T'])
                BN2_int = np.array(BN2).astype(int).flatten() - 1
                valid_indices = BN2_int[(BN2_int >= 0) & (BN2_int < JXYV.shape[0]) & (BN2_int < T_data.shape[0])]
                
                if valid_indices.size > 0:
                    num_time_steps = T_data.shape[1]
                    time_array = np.linspace(0, t, num_time_steps)
                    y_coords = JXYV[valid_indices, 1]
                    xxxx, yyyy = np.meshgrid(time_array, y_coords)
                    tttt = T_data[valid_indices, :]

                    if xxxx.size > 0:
                        contour = ax.contourf(xxxx, yyyy, tttt, cmap='jet', levels=20)
                        ax.set_xlim(np.min(time_array), np.max(time_array))
                        ax.set_ylim(np.min(y_coords), np.max(y_coords))
                        self.gradient_figure.colorbar(contour, ax=ax)
                        ax.set_title(f"模型 #{current_model_index + 1} 温度梯度")
                        ax.set_xlabel("时间 (s)")
                        ax.set_ylabel("薄膜厚度 (m)")

        self.gradient_canvas.draw()

    def save_visualization_state(self, key=None):
        """Saves the current simulation results to a pickle file."""
        if not self.gui.simulation_step_results:
            QMessageBox.warning(self.gui, "警告", "没有可保存的结果。")
            return

        file_path, _ = QFileDialog.getSaveFileName(self.gui, "保存仿真结果", "simulation_results.pkl", "Pickle Files (*.pkl)")
        if not file_path:
            return

        temperature_data = self._calculate_temperature_curve_data()

        state = {
            "simulation_step_results": self.gui.simulation_step_results,
            "simulation_models": [self.gui.model_manager._model_to_dict(m) for m in self.gui.model_manager.simulation_models],
            "average_temperatures": self.model.avg_temperatures,
        }
        
        if temperature_data:
            state["temperature_curve_data"] = temperature_data

        try:
            with open(file_path, "wb") as f:
                pickle.dump(state, f)
            QMessageBox.information(self.gui, "成功", f"仿真结果已保存到 {file_path}")
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"保存失败: {e}")

    def load_visualization_state(self):
        """Loads simulation results from a pickle file."""
        file_path, _ = QFileDialog.getOpenFileName(self.gui, "加载仿真结果", "", "Pickle Files (*.pkl)")
        if not file_path:
            return

        try:
            with open(file_path, "rb") as f:
                state = pickle.load(f)
            
            self.clear_data()

            self.gui.simulation_step_results = state.get("simulation_step_results", [])
            self.model.temperature_curve_data = state.get("temperature_curve_data")
            self.model.avg_temperatures = state.get("average_temperatures", [])
            
            self.gui.model_manager.simulation_models.clear()
            model_data = state.get("simulation_models", [])
            for item in model_data:
                model = self.gui.model_manager._dict_to_model(item)
                self.gui.model_manager.simulation_models.append(model)
            
            self.gui.model_manager._update_model_table()

            self.visualize_step_results()
            
            QMessageBox.information(self.gui, "成功", f"成功加载仿真结果")
            self.restart_visualization()
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"加载失败: {e}")

    # ---------------- Result Visualization Methods ----------------
    def _calculate_temperature_curve_data(self):
        """Calculates the data needed for the temperature curves plot."""
        if not self.gui.simulation_step_results:
            return None

        try:
            all_T_arrays = [res['output']['T'] for res in self.gui.simulation_step_results if 'output' in res and 'T' in res['output']]
            if not all_T_arrays:
                return None

            num_rows = all_T_arrays[0].shape[0]
            if not all(arr.shape[0] == num_rows for arr in all_T_arrays):
                self.gui.update_log("警告: 各步骤的温度数据行数不一致，无法拼接。")
                return None

            combined_T = np.hstack(all_T_arrays)
            time_steps = np.arange(combined_T.shape[1])
            
            row_indices = [30, num_rows // 2, num_rows - 200]
            row_labels = ['膜底部', '中间层', '膜顶部']
            
            model_boundaries = []
            current_time_step = 0
            for i, res in enumerate(self.gui.simulation_step_results):
                if 'output' in res and 'T' in res['output']:
                    num_steps = res['output']['T'].shape[1]
                    model_name = f"ID:{res['model_id']} {res['model_type']}"
                    model_boundaries.append({'time_step': current_time_step, 'name': model_name})
                    current_time_step += num_steps
            
            return {
                "combined_T": combined_T,
                "time_steps": time_steps,
                "row_indices": row_indices,
                "row_labels": row_labels,
                "model_boundaries": model_boundaries,
                "num_rows": num_rows
            }
        except Exception as e:
            self.gui.update_log(f"错误: 计算温度曲线数据时出错: {e}")
            return None

    def _get_valid_step_data(self):
        """Helper function to process and validate simulation step results."""
        valid_steps = []
        all_y_min, all_y_max = float('inf'), float('-inf')
        all_t_min, all_t_max = float('inf'), float('-inf')

        for step_result in self.gui.simulation_step_results:
            output = step_result.get('output', {})
            if all(k in output for k in ['t', 'Ns', 'JXYV', 'BN2', 'T']):
                try:
                    t = output['t'].item() if isinstance(output['t'], np.ndarray) and output['t'].size == 1 else float(output['t'])
                    JXYV, BN2, T_data = np.array(output['JXYV']), np.array(output['BN2']), np.array(output['T'])
                    BN2_int = BN2.astype(int).flatten() - 1
                    valid_indices = BN2_int[(BN2_int >= 0) & (BN2_int < JXYV.shape[0]) & (BN2_int < T_data.shape[0])]
                    if valid_indices.size == 0:
                        continue
                    
                    num_time_steps = T_data.shape[1]
                    time_array = np.linspace(0, t, num_time_steps)
                    y_coords = JXYV[valid_indices, 1]
                    xxxx, yyyy = np.meshgrid(time_array, y_coords)
                    tttt = T_data[valid_indices, :]

                    if xxxx.size == 0:
                        continue

                    all_y_min = min(all_y_min, np.min(yyyy))
                    all_y_max = max(all_y_max, np.max(yyyy))
                    all_t_min = min(all_t_min, np.nanmin(tttt))
                    all_t_max = max(all_t_max, np.nanmax(tttt))
                    valid_steps.append({'step_result': step_result, 'xxxx': xxxx, 'yyyy': yyyy, 'tttt': tttt})
                except Exception as e:
                    self.gui.update_log(f"处理步骤 {step_result.get('model_id', 'N/A')} 数据时出错: {e}", "error")

        return valid_steps, all_y_min, all_y_max, all_t_min, all_t_max

    def visualize_step_results(self):
        """Visualizes the results of each simulation step."""
        if not self.gui.simulation_step_results:
            QMessageBox.warning(self.gui, "警告", "没有可可视化的步骤结果")
            return

        self.gui.figure.clear()
        valid_steps, all_y_min, all_y_max, all_t_min, all_t_max = self._get_valid_step_data()

        if not valid_steps:
            QMessageBox.warning(self.gui, "警告", "没有有效的步骤结果可以可视化")
            return

        num_steps = len(valid_steps)
        cols = min(num_steps, 2)
        rows = (num_steps + cols - 1) // cols
        
        # Use tight y-range for all subplots for consistent scale without padding
        y_range = [all_y_min, all_y_max]
        
        t_levels = np.linspace(all_t_min, all_t_max, 20)
        
        for i, step_data in enumerate(valid_steps):
            ax = self.gui.figure.add_subplot(rows, cols, i + 1)
            contour = ax.contourf(step_data['xxxx'], step_data['yyyy'], step_data['tttt'], levels=t_levels, cmap='jet', extend='both')
            ax.set_xlabel('Time/s')
            ax.set_ylabel('PBAT Thickness (m)')
            ax.set_ylim(y_range)
            # Manually set x-limits to remove padding
            ax.set_xlim(step_data['xxxx'].min(), step_data['xxxx'].max())
            step_result = step_data['step_result']
            ax.set_title(f"步骤 {step_result['model_id']}: {step_result['model_type']}")
            ax.grid(True, linestyle='--', alpha=0.6)

        if valid_steps:
            self.gui.figure.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.5)
            cbar_ax = self.gui.figure.add_axes([0.9, 0.15, 0.03, 0.7])
            mappable = self.gui.figure.axes[0].collections[0]
            self.gui.figure.colorbar(mappable, cax=cbar_ax)

        self.gui.canvas.draw()
        self.gui.update_log(f"已显示所有 {len(valid_steps)} 个步骤的可视化结果")

    def visualize_combined_contour(self):
        """Visualizes the combined temperature contour plot of all steps."""
        if not self.gui.simulation_step_results:
            QMessageBox.warning(self.gui, "警告", "没有可可视化的步骤结果")
            return

        plot_data, all_t_min, all_t_max, boundary_times = get_combined_contour_data(self.gui.simulation_step_results)

        if not plot_data:
            QMessageBox.warning(self.gui, "警告", "没有有效的步骤结果可以可视化")
            return

        self.gui.figure.clear()
        ax = self.gui.figure.add_subplot(111)
        ax.set_title('整体温度云图')
        ax.set_xlabel('工艺流程 (时间 / s)')
        ax.set_ylabel('薄膜厚度 (m)')

        mesh = None
        levels = np.linspace(all_t_min, all_t_max, 100)
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for data in plot_data:
            mesh = ax.contourf(data['xxxx'], data['yyyy'], data['tttt'], cmap='jet', levels=levels)
            x_min = min(x_min, data['xxxx'].min())
            x_max = max(x_max, data['xxxx'].max())
            y_min = min(y_min, data['yyyy'].min())
            y_max = max(y_max, data['yyyy'].max())

        if mesh:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        for boundary_time in boundary_times:
            ax.axvline(x=boundary_time, color='k', linestyle=':', linewidth=1)

        if mesh:
            self.gui.figure.colorbar(mesh, ax=ax, label='温度 (K)')
        
        self.gui.figure.tight_layout()
        self.gui.canvas.draw()
        self.gui.update_log("已显示整体温度云图")

    def save_combined_contour_plot(self, folder_path):
        """Generates and saves the combined contour plot."""
        if not self.gui.simulation_step_results:
            return

        plot_data, all_t_min, all_t_max, boundary_times = get_combined_contour_data(self.gui.simulation_step_results)

        if not plot_data:
            return

        fig = Figure(figsize=(12, 8), dpi=150)
        ax = fig.add_subplot(111)
        ax.set_title('整体温度云图')
        ax.set_xlabel('工艺流程 (时间 / s)')
        ax.set_ylabel('薄膜厚度 (m)')

        mesh = None
        levels = np.linspace(all_t_min, all_t_max, 100)
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for data in plot_data:
            mesh = ax.contourf(data['xxxx'], data['yyyy'], data['tttt'], cmap='jet', levels=levels)
            x_min = min(x_min, data['xxxx'].min())
            x_max = max(x_max, data['xxxx'].max())
            y_min = min(y_min, data['yyyy'].min())
            y_max = max(y_max, data['yyyy'].max())

        if mesh:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        for boundary_time in boundary_times:
            ax.axvline(x=boundary_time, color='k', linestyle=':', linewidth=1)

        if mesh:
            fig.colorbar(mesh, ax=ax, label='温度 (K)')
        
        fig.tight_layout()
        file_path = os.path.join(folder_path, "combined_contour.png")
        fig.savefig(file_path)
        plt.close(fig)

    def show_temperature_curves(self, key=None):
        """Displays temperature curves for key locations in a new window."""
        if self.model.temperature_curve_data:
            curve_data = self.model.temperature_curve_data
            self.gui.update_log("使用加载的温度曲线数据绘图。")
        else:
            if not self.gui.simulation_step_results:
                QMessageBox.warning(self.gui, "警告", "没有可用于绘图的结果。")
                return
            curve_data = self._calculate_temperature_curve_data()
            if not curve_data:
                QMessageBox.warning(self.gui, "警告", "无法计算温度曲线数据。")
                return
            self.gui.update_log("从当前仿真结果计算温度曲线数据并绘图。")

        try:
            combined_T = np.array(curve_data["combined_T"])
            time_steps = np.array(curve_data["time_steps"])
            row_indices = curve_data["row_indices"]
            row_labels = curve_data["row_labels"]
            num_rows = curve_data["num_rows"]
            model_boundaries = curve_data["model_boundaries"]

            if self.temperature_curve_dialog is None:
                self.temperature_curve_dialog = QDialog(self.gui)
                self.temperature_curve_dialog.setWindowTitle("温度变化曲线")
                self.temperature_curve_dialog.setMinimumSize(900, 600)
                
                layout = QVBoxLayout(self.temperature_curve_dialog)
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)
                layout.addWidget(scroll_area)

                container = QWidget()
                scroll_area.setWidget(container)
                container_layout = QVBoxLayout(container)
                
                figure = Figure(figsize=(5, 12), dpi=100)
                figure.patch.set_alpha(0)
                canvas = FigureCanvas(figure)
                canvas.setObjectName("temperature_curve_canvas")
                container_layout.addWidget(canvas)
                
                self.temperature_curve_dialog.figure = figure
                self.temperature_curve_dialog.canvas = canvas
            
            figure = self.temperature_curve_dialog.figure
            canvas = self.temperature_curve_dialog.canvas
            figure.clear()

            ax = figure.add_subplot(111)
            ax.patch.set_alpha(0)
            
            for i, row_idx in enumerate(row_indices):
                if row_idx < num_rows:
                    ax.plot(time_steps, combined_T[row_idx, :], label=row_labels[i])
            
            ax.set_xlabel('总时间步')
            ax.set_ylabel('温度 (K)')
            ax.set_title('关键位置温度随时间变化曲线')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

            if self.gui.show_model_ids_checkbox.isChecked():
                for boundary in model_boundaries:
                    ax.axvline(x=boundary['time_step'], color='r', linestyle='--')
                    ax.text(boundary['time_step'] + 5, ax.get_ylim()[1] * 0.9, boundary['name'], color='r', rotation=90)
            
            figure.tight_layout(pad=3.0)
            canvas.draw()
            
            self.temperature_curve_dialog.show()
            self.temperature_curve_dialog.raise_()
            self.temperature_curve_dialog.activateWindow()

        except Exception as e:
            self.gui.update_log(f"错误: 绘制温度变化曲线时出错: {e}")
            QMessageBox.critical(self.gui, "错误", f"绘制温度变化曲线时出错: {e}")

    def save_all_plots(self):
        """Saves all visualization plots to a specified folder."""
        if not self.gui.simulation_step_results:
            QMessageBox.warning(self.gui, "警告", "没有可保存的图表。")
            return

        folder_path = QFileDialog.getExistingDirectory(self.gui, "选择保存文件夹")
        if not folder_path:
            return

        try:
            self.save_step_results_plots(folder_path)
            self.save_temperature_curves_plot(folder_path)
            self.save_combined_contour_plot(folder_path)
            QMessageBox.information(self.gui, "成功", f"所有图表已保存到: {folder_path}")
            self.gui.update_log(f"所有图表已保存到: {folder_path}")
        except Exception as e:
            self.gui.update_log(f"错误: 保存图表时出错: {e}")
            QMessageBox.critical(self.gui, "错误", f"保存图表时出错: {e}")

    def save_step_results_plots(self, folder_path):
        """Generates and saves plots for all step results."""
        if not self.gui.simulation_step_results:
            return

        valid_steps, all_y_min, all_y_max, all_t_min, all_t_max = self._get_valid_step_data()

        if not valid_steps:
            return

        # Use tight y-range for all subplots for consistent scale without padding
        y_range = [all_y_min, all_y_max]
        t_levels = np.linspace(all_t_min, all_t_max, 20)
        
        for i, step_data in enumerate(valid_steps):
            fig = Figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111)
            contour = ax.contourf(step_data['xxxx'], step_data['yyyy'], step_data['tttt'], levels=t_levels, cmap='jet', extend='both')
            ax.set_ylim(y_range)
            ax.set_xlabel('Time/s')
            ax.set_ylabel('PBAT Thickness (m)')
            step_result = step_data['step_result']
            ax.set_title(f"步骤 {step_result['model_id']}: {step_result['model_type']}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(step_data['xxxx'].min(), step_data['xxxx'].max())
            fig.colorbar(contour, ax=ax)
            fig.tight_layout()
            
            file_path = os.path.join(folder_path, f"step_{step_result['model_id']}_contour.png")
            fig.savefig(file_path)
            plt.close(fig)

    def save_temperature_curves_plot(self, folder_path):
        """Generates and saves the temperature curves plot."""
        if not self.gui.simulation_step_results:
            return

        try:
            all_T_arrays = [res['output']['T'] for res in self.gui.simulation_step_results if 'output' in res and 'T' in res['output']]
            if not all_T_arrays:
                return

            num_rows = all_T_arrays[0].shape[0]
            if not all(arr.shape[0] == num_rows for arr in all_T_arrays):
                return

            combined_T = np.hstack(all_T_arrays)
            
            fig = Figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111)
            time_steps = np.arange(combined_T.shape[1])
            
            row_indices = [30, num_rows // 2, num_rows -200]
            row_labels = ['膜底部', '中间层', '膜顶部']
            
            for i, row_idx in enumerate(row_indices):
                if row_idx < num_rows:
                    ax.plot(time_steps, combined_T[row_idx, :], label=row_labels[i])
            
            ax.set_xlabel('总时间步')
            ax.set_ylabel('温度 (K)')
            ax.set_title('关键位置温度随时间变化曲线')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()
            
            file_path = os.path.join(folder_path, "temperature_curves.png")
            fig.savefig(file_path)
            plt.close(fig)

        except Exception as e:
            print(f"Error saving temperature curves plot: {e}")

    def display_all_step_parameters(self):
        """Displays output parameters for all steps in a dialog."""
        if not hasattr(self.gui, 'simulation_step_results') or not self.gui.simulation_step_results:
            QMessageBox.warning(self.gui, "警告", "没有可显示的步骤结果")
            return

        dialog = QDialog(self.gui)
        dialog.setWindowTitle("步骤参数")
        dialog.setMinimumSize(1000, 800)
        layout = QVBoxLayout(dialog)

        step_selector = QComboBox()
        step_selector.addItem("所有步骤")
        for step_result in self.gui.simulation_step_results:
            model_type = step_result['model_type']
            roll_direction = step_result.get('roll_direction', '')
            display_text = f"步骤 {step_result['model_id']}: {model_type}-{roll_direction}"
            step_selector.addItem(display_text)
        layout.addWidget(QLabel("选择步骤:"))
        layout.addWidget(step_selector)

        param_selector = QComboBox()
        param_selector.addItem("所有参数")
        layout.addWidget(QLabel("选择参数:"))
        layout.addWidget(param_selector)

        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_layout.addWidget(text_edit)
        tab_widget.addTab(text_tab, "文本视图")

        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_widget = QTableWidget()
        table_layout.addWidget(table_widget)
        tab_widget.addTab(table_tab, "表格视图")

        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_figure = Figure(figsize=(5, 4), dpi=100)
        chart_canvas = FigureCanvas(chart_figure)
        chart_canvas.setObjectName("chart_canvas")
        chart_layout.addWidget(chart_canvas)
        tab_widget.addTab(chart_tab, "图表视图")
        
        icon_tab = QWidget()
        icon_layout = QVBoxLayout(icon_tab)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        icon_container = QWidget()
        icon_grid = QGridLayout()
        icon_container.setLayout(icon_grid)
        scroll_area.setWidget(icon_container)
        icon_layout.addWidget(scroll_area)
        tab_widget.addTab(icon_tab, "图标视图")

        self.update_step_parameter_display(text_edit, table_widget, chart_figure, icon_grid, "所有步骤", "所有参数", step_selector, param_selector)

        step_selector.currentTextChanged.connect(lambda step: self.update_step_parameter_display(text_edit, table_widget, chart_figure, icon_grid, step, param_selector.currentText(), step_selector, param_selector))
        param_selector.currentTextChanged.connect(lambda param: self.update_step_parameter_display(text_edit, table_widget, chart_figure, icon_grid, step_selector.currentText(), param, step_selector, param_selector))

        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec()

    def update_step_parameter_display(self, text_edit, table_widget, chart_figure, icon_grid, selected_step, selected_param, step_selector, param_selector):
        """Updates the display of step parameters based on user selection."""
        chart_figure.clear()
        self.clear_icon_view(icon_grid)
        
        text_content = ""
        if selected_step == "所有步骤":
            text_content = "所有步骤的参数:\n\n"
            all_params = set()
            for step_result in self.gui.simulation_step_results:
                if 'output' in step_result and step_result['output']:
                    all_params.update(step_result['output'].keys())
            
            param_selector.blockSignals(True)
            param_selector.clear()
            param_selector.addItem("所有参数")
            for param_name in sorted(all_params):
                param_selector.addItem(param_name)
            param_selector.blockSignals(False)
            
            if selected_param != "所有参数":
                text_content += f"参数: {selected_param}\n\n"
                for step_result in self.gui.simulation_step_results:
                    if 'output' in step_result and step_result['output'] and selected_param in step_result['output']:
                        param_value = step_result['output'][selected_param]
                        model_type = step_result['model_type']
                        roll_direction = step_result.get('roll_direction', '')
                        text_content += f"步骤 {step_result['model_id']}: {model_type}-{roll_direction}\n"
                        text_content += self.format_parameter_value(param_value, indent=2)
                        text_content += "\n\n"
                self.display_step_array_in_table(table_widget, selected_param, "所有步骤")
                self.visualize_step_array(chart_figure, selected_param, "所有步骤")
            else:
                for step_result in self.gui.simulation_step_results:
                    model_type = step_result['model_type']
                    roll_direction = step_result.get('roll_direction', '')
                    text_content += f"步骤 {step_result['model_id']}: {model_type}-{roll_direction}\n"
                    if 'output' in step_result and step_result['output']:
                        for param_name, param_value in step_result['output'].items():
                            text_content += f"  {param_name}: {self.format_parameter_value(param_value, indent=4)}\n"
                    else:
                        text_content += "  无结果数据\n"
                    text_content += "\n"
                table_widget.clear()
                table_widget.setRowCount(0)
                table_widget.setColumnCount(0)
                table_widget.setHorizontalHeaderLabels([])
        else:
            step_id = int(selected_step.split(":")[0].split(" ")[1])
            step_result = next((s for s in self.gui.simulation_step_results if s['model_id'] == step_id), None)
            
            if not step_result:
                text_content = f"未找到步骤 {step_id}"
                table_widget.clear()
                table_widget.setRowCount(0)
                table_widget.setColumnCount(0)
                table_widget.setHorizontalHeaderLabels([])
            else:
                model_type = step_result['model_type']
                roll_direction = step_result.get('roll_direction', '')
                text_content = f"步骤 {step_result['model_id']}: {model_type}-{roll_direction} 的参数:\n\n"
                
                if 'output' in step_result and step_result['output']:
                    param_names = list(step_result['output'].keys())
                    param_selector.blockSignals(True)
                    param_selector.clear()
                    param_selector.addItem("所有参数")
                    for param_name in sorted(param_names):
                        param_selector.addItem(param_name)
                    param_selector.blockSignals(False)
                    
                    if selected_param != "所有参数" and selected_param in step_result['output']:
                        param_value = step_result['output'][selected_param]
                        text_content += f"{selected_param}: {self.format_parameter_value(param_value, indent=2)}\n"
                        self.display_step_array_in_table(table_widget, selected_param, step_id)
                        self.visualize_step_array(chart_figure, selected_param, step_id)
                    else:
                        for param_name, param_value in step_result['output'].items():
                            text_content += f"{param_name}: {self.format_parameter_value(param_value, indent=2)}\n"
                        table_widget.clear()
                        table_widget.setRowCount(0)
                        table_widget.setColumnCount(0)
                        table_widget.setHorizontalHeaderLabels([])
                else:
                    text_content += "无结果数据\n"
                    table_widget.clear()
                    table_widget.setRowCount(0)
                    table_widget.setColumnCount(0)
                    table_widget.setHorizontalHeaderLabels([])
        
        text_edit.setText(text_content)
        chart_figure.canvas.draw()
        self.update_icon_view(icon_grid, selected_step, selected_param)
        
    def clear_icon_view(self, icon_grid):
        """Clears the icon view."""
        while icon_grid.count():
            item = icon_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def update_icon_view(self, icon_grid, selected_step, selected_param):
        """Updates the icon view based on selections."""
        if selected_param == "所有参数":
            if selected_step == "所有步骤":
                row, col, max_cols = 0, 0, 4
                for step_result in self.gui.simulation_step_results:
                    if 'output' in step_result and step_result['output']:
                        model_type = step_result['model_type']
                        roll_direction = step_result.get('roll_direction', '')
                        step_title = f"步骤 {step_result['model_id']}: {model_type}-{roll_direction}"
                        step_icon = self.create_step_icon(step_result, step_title)
                        icon_grid.addWidget(step_icon, row, col)
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
            else:
                step_id = int(selected_step.split(":")[0].split(" ")[1])
                step_result = next((s for s in self.gui.simulation_step_results if s['model_id'] == step_id), None)
                if step_result and 'output' in step_result and step_result['output']:
                    model_type = step_result['model_type']
                    roll_direction = step_result.get('roll_direction', '')
                    step_title = f"步骤 {step_result['model_id']}: {model_type}-{roll_direction}"
                    step_icon = self.create_step_icon(step_result, step_title)
                    icon_grid.addWidget(step_icon, 0, 0)
        else:
            if selected_step == "所有步骤":
                row, col, max_cols = 0, 0, 4
                for step_result in self.gui.simulation_step_results:
                    if 'output' in step_result and step_result['output'] and selected_param in step_result['output']:
                        model_type = step_result['model_type']
                        roll_direction = step_result.get('roll_direction', '')
                        step_title = f"步骤 {step_result['model_id']}: {model_type}-{roll_direction}"
                        param_value = step_result['output'][selected_param]
                        param_icon = self.create_parameter_icon(selected_param, param_value, step_title)
                        icon_grid.addWidget(param_icon, row, col)
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
            else:
                step_id = int(selected_step.split(":")[0].split(" ")[1])
                step_result = next((s for s in self.gui.simulation_step_results if s['model_id'] == step_id), None)
                if step_result and 'output' in step_result and step_result['output'] and selected_param in step_result['output']:
                    model_type = step_result['model_type']
                    roll_direction = step_result.get('roll_direction', '')
                    step_title = f"步骤 {step_result['model_id']}: {model_type}-{roll_direction}"
                    param_value = step_result['output'][selected_param]
                    param_icon = self.create_parameter_icon(selected_param, param_value, step_title)
                    icon_grid.addWidget(param_icon, 0, 0)
    
    def create_step_icon(self, step_result, step_title):
        """Creates a widget representing a step icon."""
        icon_widget = QWidget()
        icon_layout = QVBoxLayout(icon_widget)
        icon_widget.setStyleSheet("QWidget { background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; padding: 10px; margin: 5px; min-width: 200px; min-height: 150px; }")
        
        title_label = QLabel(step_title)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_layout.addWidget(title_label)
        
        param_count = len(step_result.get('output', {}))
        param_label = QLabel(f"参数数量: {param_count}")
        param_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_layout.addWidget(param_label)
        
        model_type = step_result.get('model_type', '未知类型')
        type_icon = QLabel("❄️" if model_type == '降温' else "🔥" if model_type == '升温' else "⚙️")
        type_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        type_icon.setStyleSheet("font-size: 24px;")
        icon_layout.addWidget(type_icon)
        
        roll_direction = step_result.get('roll_direction', '未知方向')
        direction_label = QLabel(f"方向: {roll_direction}")
        direction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_layout.addWidget(direction_label)
        
        icon_layout.addStretch()
        return icon_widget
    
    def create_parameter_icon(self, param_name, param_value, step_title):
        """Creates a widget representing a parameter icon."""
        icon_widget = QWidget()
        icon_layout = QVBoxLayout(icon_widget)
        icon_widget.setStyleSheet("QWidget { background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; padding: 10px; margin: 5px; min-width: 200px; min-height: 150px; }")
        
        step_label = QLabel(step_title)
        step_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_label.setWordWrap(True)
        icon_layout.addWidget(step_label)
        
        param_label = QLabel(param_name)
        param_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        param_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        param_label.setWordWrap(True)
        icon_layout.addWidget(param_label)
        
        value_preview = self.format_parameter_value_for_icon(param_value)
        value_label = QLabel(value_preview)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setWordWrap(True)
        value_label.setStyleSheet("font-size: 10px;")
        icon_layout.addWidget(value_label)
        
        type_icon = QLabel()
        if isinstance(param_value, np.ndarray):
            type_icon.setText("📊" if param_value.ndim == 1 else "🔲" if param_value.ndim == 2 else "🔷")
        elif isinstance(param_value, (int, float)): type_icon.setText("🔢")
        elif isinstance(param_value, str): type_icon.setText("📝")
        elif isinstance(param_value, (list, tuple)): type_icon.setText("📋")
        elif isinstance(param_value, dict): type_icon.setText("📚")
        else: type_icon.setText("❓")
        
        type_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        type_icon.setStyleSheet("font-size: 24px;")
        icon_layout.addWidget(type_icon)
        
        icon_layout.addStretch()
        return icon_widget
    
    def format_parameter_value_for_icon(self, param_value):
        """Formats a parameter value for display in an icon."""
        if isinstance(param_value, np.ndarray):
            if param_value.size == 0: return "空数组"
            elif param_value.size <= 4: return f"数组: {param_value.tolist()}"
            else: return f"{param_value.ndim}维数组, 形状: {param_value.shape}"
        elif isinstance(param_value, (int, float)): return f"数值: {param_value}"
        elif isinstance(param_value, str): return f"文本: {param_value[:20]}{'...' if len(param_value) > 20 else ''}"
        elif isinstance(param_value, (list, tuple)):
            if len(param_value) <= 4: return f"{type(param_value).__name__}: {param_value}"
            else: return f"{type(param_value).__name__}, 长度: {len(param_value)}"
        elif isinstance(param_value, dict): return f"字典, 键数: {len(param_value)}"
        else: return f"{type(param_value).__name__}: {str(param_value)[:20]}..."

    def display_step_array_in_table(self, table_widget, param_name, step_id):
        """Displays array data for a step in a table."""
        table_widget.clear()
        
        if step_id == "所有步骤":
            all_data, headers = [], ["步骤ID: 类型-方向"]
            for step_result in self.gui.simulation_step_results:
                if 'output' in step_result and step_result['output'] and param_name in step_result['output']:
                    param_value = step_result['output'][param_name]
                    step_display = f"{step_result['model_id']}: {step_result['model_type']}-{step_result.get('roll_direction', '')}"
                    if isinstance(param_value, np.ndarray):
                        if param_value.ndim == 1:
                            all_data.append([step_display] + param_value.tolist())
                            if len(headers) == 1: headers.extend([f"元素{i}" for i in range(len(param_value))])
                        elif param_value.ndim == 2:
                            all_data.append([step_display] + param_value.flatten().tolist())
                            if len(headers) == 1: headers.extend([f"({i},{j})" for i in range(param_value.shape[0]) for j in range(param_value.shape[1])])
                        else:
                            all_data.append([step_display, f"{param_value.ndim}维数组，形状{param_value.shape}"])
                            if len(headers) == 1: headers.append("信息")
                    else:
                        all_data.append([step_display, str(param_value)])
                        if len(headers) == 1: headers.append("值")
            if all_data:
                table_widget.setRowCount(len(all_data))
                table_widget.setColumnCount(len(headers))
                table_widget.setHorizontalHeaderLabels(headers)
                for i, row_data in enumerate(all_data):
                    for j, value in enumerate(row_data):
                        table_widget.setItem(i, j, QTableWidgetItem(str(value)))
        else:
            step_result = next((s for s in self.gui.simulation_step_results if s['model_id'] == step_id), None)
            if not step_result or 'output' not in step_result or not step_result['output'] or param_name not in step_result['output']:
                table_widget.setRowCount(0); table_widget.setColumnCount(0); return
            param_value = step_result['output'][param_name]
            if isinstance(param_value, np.ndarray):
                if param_value.ndim == 1:
                    table_widget.setRowCount(param_value.size); table_widget.setColumnCount(2)
                    table_widget.setHorizontalHeaderLabels(["索引", "值"])
                    for i in range(param_value.size):
                        table_widget.setItem(i, 0, QTableWidgetItem(str(i))); table_widget.setItem(i, 1, QTableWidgetItem(str(param_value[i])))
                elif param_value.ndim == 2:
                    rows, cols = param_value.shape
                    table_widget.setRowCount(rows); table_widget.setColumnCount(cols)
                    table_widget.setHorizontalHeaderLabels([f"列{j}" for j in range(cols)])
                    for i in range(rows):
                        for j in range(cols):
                            table_widget.setItem(i, j, QTableWidgetItem(str(param_value[i, j])))
                else:
                    table_widget.setRowCount(1); table_widget.setColumnCount(1)
                    table_widget.setHorizontalHeaderLabels(["信息"]); table_widget.setItem(0, 0, QTableWidgetItem(f"{param_value.ndim}维数组，形状{param_value.shape}"))
            else:
                table_widget.setRowCount(1); table_widget.setColumnCount(1)
                table_widget.setHorizontalHeaderLabels(["值"]); table_widget.setItem(0, 0, QTableWidgetItem(str(param_value)))

    def visualize_step_array(self, figure, param_name, step_id):
        """Visualizes array data for a step."""
        figure.clear()
        ax = figure.add_subplot(111)
        
        if step_id == "所有步骤":
            step_ids, step_labels, values = [], [], []
            for step_result in self.gui.simulation_step_results:
                if 'output' in step_result and step_result['output'] and param_name in step_result['output']:
                    param_value = step_result['output'][param_name]
                    step_label = f"{step_result['model_id']}: {step_result['model_type']}-{step_result.get('roll_direction', '')}"
                    val = None
                    if isinstance(param_value, np.ndarray):
                        if param_value.size > 0: val = float(np.mean(param_value))
                    elif isinstance(param_value, (int, float)): val = float(param_value)
                    if val is not None:
                        step_ids.append(step_result['model_id']); step_labels.append(step_label); values.append(val)
            if step_ids and values:
                ax.plot(step_ids, values, 'o-')
                ax.set_xlabel('步骤ID'); ax.set_ylabel(f'{param_name} (平均值)'); ax.set_title(f'所有步骤的 {param_name} 变化'); ax.grid(True)
                if len(step_ids) <= 10:
                    for x, y, label in zip(step_ids, values, step_labels):
                        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            else: ax.text(0.5, 0.5, "无可视化数据", ha='center', va='center')
        else:
            step_result = next((s for s in self.gui.simulation_step_results if s['model_id'] == step_id), None)
            if not step_result or 'output' not in step_result or not step_result['output'] or param_name not in step_result['output']:
                ax.text(0.5, 0.5, "无数据", ha='center', va='center'); return
            param_value = step_result['output'][param_name]
            if isinstance(param_value, np.ndarray):
                if param_value.ndim == 1:
                    ax.plot(param_value, 'o-'); ax.set_xlabel('索引'); ax.set_ylabel(param_name); ax.set_title(f'步骤 {step_id} 的 {param_name}'); ax.grid(True)
                elif param_value.ndim == 2:
                    im = ax.imshow(param_value, cmap='viridis', aspect='auto'); ax.set_xlabel('列索引'); ax.set_ylabel('行索引'); ax.set_title(f'步骤 {step_id} 的 {param_name}'); figure.colorbar(im, ax=ax)
                else: ax.text(0.5, 0.5, f"{param_value.ndim}维数组，无法可视化", ha='center', va='center')
            elif isinstance(param_value, (int, float)): ax.text(0.5, 0.5, f"{param_name}: {param_value}", ha='center', va='center')
            else: ax.text(0.5, 0.5, "无法可视化", ha='center', va='center')

    def format_parameter_value(self, value, indent=0):
        """Formats a parameter value for text display."""
        indent_str = " " * indent
        if isinstance(value, np.ndarray):
            if value.ndim == 0: return f"{indent_str}{float(value)}"
            elif value.ndim == 1:
                if value.size < 10: return f"{indent_str}{value.tolist()}"
                else: return f"{indent_str}[{value[0]}, {value[1]}, ..., {value[-1]}] (共{value.size}个元素)"
            elif value.ndim == 2:
                if value.size < 20: return f"{indent_str}\n{indent_str}" + "\n".join([f"{indent_str}{row}" for row in value.tolist()])
                else: return f"{indent_str}二维数组，形状{value.shape}"
            else: return f"{indent_str}{value.ndim}维数组，形状{value.shape}"
        elif isinstance(value, (list, tuple)):
            if len(value) < 10: return f"{indent_str}{value}"
            else: return f"{indent_str}[{value[0]}, {value[1]}, ..., {value[-1]}] (共{len(value)}个元素)"
        elif isinstance(value, dict):
            result = f"{indent_str}{{\n"
            for k, v in value.items():
                result += f"{indent_str}  {k}: {self.format_parameter_value(v, indent+4)}\n"
            result += f"{indent_str}}}"  
            return result
        else: return f"{indent_str}{value}"
