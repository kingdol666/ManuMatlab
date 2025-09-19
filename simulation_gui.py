# 标准库导入
import os
import sys
import datetime
import json
import markdown

# 第三方库导入
import numpy as np
import scipy.io
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QMainWindow, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget, QMessageBox, QGridLayout, QLabel, QScrollArea, QGraphicsBlurEffect,
    QHBoxLayout, QSizePolicy, QFrame, QStackedWidget, QDialogButtonBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
import torch

# 项目模块导入
from logic.rl_optimizer import TD3, FilmCastingEnv
from config.paths import (
    get_config_file,
    get_styles_file,
    get_icon_file,
    get_logo_file,
    get_temp_matlab_dir,
    get_background_image_file,
    get_check_image_file,
    get_select_image_file,
    get_up_arrow_image_file,
    get_down_arrow_image_file,
    get_uncheck_image_file
)
from logic.Models import ScriptType, RollDirection, SimulationModel
from threads.simulation_thread import SimulationThread
from threads.rl_thread import RlOptimizationThread
from logic.run_matlab_simulation import stop_all_matlab_engines, load_config
from ui.config_tab import create_config_tab
from ui.model_tab import create_model_tab
from ui.run_tab import create_run_tab
from ui.result_tab import create_result_tab
from ui.visualization_tab import create_visualization_tab
from ui.help_tab import create_help_tab
from logic.visualization_manager import VisualizationManager
from logic.model_manager import ModelManager

try:
    # 尝试使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告：无法设置中文字体，可能无法正确显示中文标签")
    
DEFAULT_CONFIG = {
    "mesh_params": {
        "H": 0.0001, "L": 0.0001, "Nx": 20, "Ny": 20
    },
    "main_params": {
        "k": 0.2, "midu": 1238.0, "Cv": 1450.0, "q": 0.0, "alpha": 25.0,
        "alpha1": 25.0, "T_kongqi": 400.0, "T0": 400.0, "t": 0.0, "dt": 0.1
    }
}

class SimulationGUI(QMainWindow):
    """纵拉预热仿真系统的主GUI类"""
    
    def __init__(self):
        super().__init__()
        self.config = self.load_initial_config()
        self.last_T1 = None
        self.simulation_step_results = []
        self.visualization_source_data = []
        self.visualization_models = []
        self.running_simulation_models = []
        self.simulation_thread = None
        self.rl_thread = None
        self.model_manager = ModelManager(self)
        self.visualization_manager = VisualizationManager(self)
        self.agent = None # To hold the loaded RL agent
        self.init_ui()

    def _validate_config(self, config_data):
        """Validates the structure and types of the configuration dictionary."""
        if not isinstance(config_data, dict):
            raise ValueError("配置必须是字典格式。")

        required_keys = {"mesh_params": dict, "main_params": dict}
        for key, key_type in required_keys.items():
            if key not in config_data:
                raise ValueError(f"配置文件缺少 '{key}'。")
            if not isinstance(config_data[key], key_type):
                raise ValueError(f"配置中的 '{key}' 必须是 {key_type.__name__}。")

        required_mesh_params = {"H": float, "L": float, "Nx": int, "Ny": int}
        for param, param_type in required_mesh_params.items():
            if param not in config_data["mesh_params"]:
                raise ValueError(f"配置 'mesh_params' 缺少 '{param}'。")
            
            value = config_data["mesh_params"][param]
            if param_type is float and not isinstance(value, (int, float)):
                 raise ValueError(f"配置参数 '{param}' 必须是数值。")
            elif param_type is int and not isinstance(value, int):
                 raise ValueError(f"配置参数 '{param}' 必须是整数。")

        required_main_params = {
            "k": float, "midu": float, "Cv": float, "q": float, "alpha": float,
            "alpha1": float, "T_kongqi": float, "T0": float, "t": float, "dt": float
        }
        for param, param_type in required_main_params.items():
            if param not in config_data["main_params"]:
                raise ValueError(f"配置 'main_params' 缺少 '{param}'。")
            
            value = config_data["main_params"][param]
            if not isinstance(value, (int, float)):
                raise ValueError(f"配置参数 '{param}' 必须是数值。")
        
        return True

    def load_initial_config(self):
        """Load the initial configuration from the JSON file."""
        config_path = get_config_file()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            self._validate_config(config_data)
            return config_data
        except FileNotFoundError:
            print("警告: 未找到 config.json，将使用默认配置。")
            return DEFAULT_CONFIG
        except (json.JSONDecodeError, ValueError) as e:
            print(f"警告: 加载 config.json 失败: {e}。将使用默认配置。")
            return DEFAULT_CONFIG

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('纵拉预热仿真系统')
        self.setObjectName("MainWindow")
        self.setGeometry(100, 100, 1600, 900)
        
        # 设置应用程序图标
        icon_path = get_icon_file()
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            # 如果图标文件不存在，尝试使用SVG
            svg_icon_path = get_logo_file()
            if os.path.exists(svg_icon_path):
                self.setWindowIcon(QIcon(svg_icon_path))
        
        # Apply stylesheet
        qss_file_path = get_styles_file()
        with open(qss_file_path, "r", encoding="utf-8") as f:
            stylesheet = f.read()
            stylesheet = stylesheet.replace(
                "url(public/images/background1.png)", f"url({get_background_image_file()})"
            ).replace(
                "url(public/images/check.png)", f"url({get_check_image_file()})"
            ).replace(
                "url(public/images/select.png)", f"url({get_select_image_file()})"
            ).replace(
                "url(public/images/up.png)", f"url({get_up_arrow_image_file()})"
            ).replace(
                "url(public/images/down.png)", f"url({get_down_arrow_image_file()})"
            ).replace(
                "url(public/images/uncheck.png)", f"url({get_uncheck_image_file()})"
            )
            self.setStyleSheet(stylesheet)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(1.1)
        main_widget.setGraphicsEffect(blur_effect)
        
        tabs = QTabWidget()
        tabs.addTab(create_config_tab(self), "配置参数")
        tabs.addTab(create_model_tab(self), "模型管理")
        tabs.addTab(create_run_tab(self), "运行仿真")
        tabs.addTab(create_result_tab(self), "结果可视化")
        tabs.addTab(create_visualization_tab(self), "运行可视化")
        self.help_tab = create_help_tab()
        tabs.addTab(self.help_tab, "帮助")
        main_layout.addWidget(tabs)
        self.tabs = tabs
        
        bottom_frame = QFrame()
        bottom_frame.setFixedHeight(60)
        bottom_frame.setObjectName("bottomFrame")
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(20, 10, 20, 10)
        
        help_button = QPushButton("帮助")
        help_button.clicked.connect(self.show_help_tab)
        help_button.setObjectName("helpButton")
        help_button.setFixedSize(80, 30)
        bottom_layout.addWidget(help_button)
        
        bottom_layout.addStretch()
        
        author_label = QLabel("作者邮箱：zhanghaozheng@mail.ustc.edu.cn")
        author_label.setObjectName("authorLabel")
        author_font = QFont()
        author_font.setPointSize(10)
        author_font.setFamily("Microsoft YaHei")
        author_label.setFont(author_font)
        author_label.setStyleSheet("color: #000; background-color: transparent; padding: 5px 10px; border-radius: 3px; font-size: 13px;")
        bottom_layout.addWidget(author_label)
        
        logo_container = QWidget()
        logo_container.setFixedSize(130, 40)
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        logo_label = QLabel()
        logo_label.setObjectName("logoLabel")
        logo_label.setFixedSize(120, 35)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = get_logo_file()
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(100, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            logo_label.setText("USTC Logo")
        logo_layout.addWidget(logo_label)
        bottom_layout.addWidget(logo_container)
        
        main_layout.addWidget(bottom_frame)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self._apply_config_to_ui(self.config)
        self.model_manager.on_script_type_changed(self.script_type_combo.currentText())
        
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("log_text")

        self.visualization_manager.draw_model_schematic()

        # Connect the RL optimization button signal
        self.rl_optimize_button.clicked.connect(self.run_rl_optimization_dialog)
        # The connection for rl_stop_button is now handled within create_run_tab to avoid duplication.
        # self.rl_stop_button.setEnabled(False)

    def load_rl_model(self):
        """Loads a pre-trained RL model for inference."""
        file_path, _ = QFileDialog.getOpenFileName(self, "加载RL模型权重", "modelSave/", "PyTorch Model Files (*.pth)")
        if file_path:
            try:
                # We need to initialize an environment and agent to load the model into.
                # The parameters for these don't matter as much for just loading the weights for inference,
                # but the state and action dimensions must match the saved model.
                # We can get n_rolls from the filename, e.g., "best_model_5rolls_200eps.pth"
                try:
                    filename = os.path.basename(file_path)
                    parts = filename.split('_')
                    n_rolls = int(parts[2].replace('rolls', ''))
                except (IndexError, ValueError):
                    self.update_log("无法从文件名解析辊数，将使用默认值5。", "warning")
                    n_rolls = 5 # Default value if parsing fails

                env = FilmCastingEnv(n_rolls=n_rolls)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim,
                                 action_low=env.action_low, action_high=env.action_high, device=device)
                
                self.agent.load_model(file_path)
                self.update_log(f"成功加载RL模型: {os.path.basename(file_path)}", "success")
                QMessageBox.information(self, "成功", "RL模型已成功加载并准备好用于仿真。")
                
                # Enable the "Run with Agent" button
                self.run_with_agent_button.setEnabled(True)

            except Exception as e:
                self.update_log(f"加载RL模型失败: {e}", "error")
                QMessageBox.critical(self, "错误", f"加载RL模型失败: {e}")
                self.agent = None
                self.run_with_agent_button.setEnabled(False)

    def run_simulation_with_agent(self):
        """Generates a model sequence using the loaded RL agent and runs the simulation."""
        if not self.agent:
            QMessageBox.warning(self, "警告", "请先加载一个RL模型。")
            return

        self.update_log("正在使用加载的Agent生成最优模型序列...", "info")
        
        try:
            # Create a temporary environment to get states and actions
            env = FilmCastingEnv(n_rolls=self.agent.actor.layer1.in_features - self.agent.action_dim - 1 - 20) # Infer n_rolls from model's input layer size
            state = env.reset()
            
            # Clear existing models
            self.model_manager.delete_all_models()
            
            # Generate the sequence of actions from the agent
            for step in range(env.n_rolls):
                action = self.agent.select_action(state, noise=False) # Use deterministic actions for inference
                next_state, _, done, _ = env.step(action)
                state = next_state
                
                # Convert action to model parameters and add to the model manager
                temp, contact_time, cooling_time, direction = action
                
                # Add Heating Model
                roll_dir = RollDirection.INITIAL if step == 0 else (RollDirection.REVERSE if direction < 0 else RollDirection.FORWARD)
                heating_model = SimulationModel(
                    script_type=ScriptType.HEATING,
                    roll_direction=roll_dir,
                    T_GunWen=temp,
                    t_up=contact_time
                )
                self.model_manager.add_model(heating_model, silent=True)
                
                # Add Cooling Model
                cooling_model = SimulationModel(
                    script_type=ScriptType.COOLING,
                    roll_direction=RollDirection.INITIAL, # Cooling direction is always initial
                    T_GunWen=temp, # Use the same temp for consistency, though it's not used by the script
                    t_up=cooling_time
                )
                self.model_manager.add_model(cooling_model, silent=True)

                if done:
                    break
 
            self.update_log(f"成功生成 {len(self.model_manager.simulation_models)} 个模型的序列。", "success")
            
            # Automatically start the simulation
            self.update_log("正在启动使用Agent生成序列的仿真...", "info")
            self.run_simulation()

        except Exception as e:
            self.update_log(f"使用Agent生成序列时出错: {e}", "error")
            QMessageBox.critical(self, "错误", f"使用Agent生成序列时出错: {e}")

    def stop_rl_optimization(self):
        """Stops the currently running RL optimization thread."""
        if self.rl_thread and self.rl_thread.isRunning():
            self.rl_thread.stop()
            self.rl_stop_button.setText("正在停止...")
        else:
            QMessageBox.information(self, "提示", "当前没有正在运行的智能优化任务。")

    def run_rl_optimization_dialog(self):
        """Opens a dialog to get RL optimization parameters and starts the process."""
        dialog = RlOptimizationDialog(self)
        if dialog.exec():
            params = dialog.get_parameters()
            self.update_log(f"Starting RL optimization with parameters: {params}", "info")

            self.rl_thread = RlOptimizationThread(
                num_episodes=params['num_episodes'],
                n_rolls=params['n_rolls'],
                target_temp=params['target_temp'],
                bounds=params['bounds'],
                checkpoint_path=params['checkpoint_path'],
                use_custom_directions=params['use_custom_directions'],
                custom_directions=params['custom_directions']
            )
            self.rl_thread.log_updated.connect(self.update_log)
            self.rl_thread.optimization_finished.connect(self.on_rl_optimization_finished)
            self.rl_thread.error_occurred.connect(self.on_simulation_error)
            self.rl_thread.finished.connect(lambda: self.set_rl_controls_enabled(True))
            
            self.rl_thread.start()
            self.set_rl_controls_enabled(False)

    def on_rl_optimization_finished(self, json_path, plot_path):
        """Callback for when the RL optimization is finished."""
        self.update_log(f"RL optimization finished! Best parameters saved to {json_path}", "success")
        QMessageBox.information(self, "Optimization Complete", f"Optimal process parameters saved to:\n{json_path}\n\nYou can now load them from the 'Model Management' tab.")
        
        # 在主线程中显示奖励曲线图
        try:
            pixmap = QPixmap(plot_path)
            self.result_image_label.setPixmap(pixmap)
            self.tabs.setCurrentWidget(self.result_tab) # 切换到结果可视化选项卡
            self.update_log(f"Displaying reward curve from {plot_path}", "info")
        except Exception as e:
            self.update_log(f"Failed to display reward curve image: {e}", "error")

    def set_rl_controls_enabled(self, is_finished):
        """Enables or disables controls during RL optimization."""
        self.rl_optimize_button.setEnabled(is_finished)
        # self.rl_stop_button.setEnabled(not is_finished)
        self.rl_stop_button.setText("停止优化")
        # 禁用其他可能冲突的选项卡
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) not in ["运行仿真", "结果可视化"]:
                self.tabs.widget(i).setEnabled(is_finished)

    def set_controls_enabled(self, enabled):
        """Enables or disables main controls during long operations."""
        self.tabs.setEnabled(enabled)

    def show_help_tab(self):
        """切换到帮助选项卡"""
        self.tabs.setCurrentWidget(self.help_tab)

    def run_simulation(self):
        """运行仿真"""
        if not self.model_manager.simulation_models:
            QMessageBox.warning(self, "警告", "请先添加至少一个模型")
            return
        if not self._validate_model_sequence():
            return
        self.visualization_manager.clear_data()
        self.update_log("已清除旧的可视化数据。")
        if not self._save_config_to_file():
            self.update_log("无法运行仿真，因为配置保存失败。")
            return
        self.update_log("当前配置已自动保存。")
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait()
        self.running_simulation_models = list(self.model_manager.simulation_models)
        self.simulation_thread = SimulationThread(self.running_simulation_models)
        self.simulation_thread.log_updated.connect(self.update_log)
        self.simulation_thread.progress_updated.connect(self.update_progress)
        self.simulation_thread.simulation_finished.connect(self.on_simulation_finished)
        self.simulation_thread.error_occurred.connect(self.on_simulation_error)
        self.simulation_thread.model_completed.connect(self.on_model_completed)
        self.set_model_tab_enabled(False)
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(self.model_manager.simulation_models))
        self.visualization_manager.model.live_visualization_active = True
        self.simulation_step_results = []
        self.visualization_manager.draw_live_visualization()
        self.simulation_thread.start()
        
    def update_log(self, message, level="info"):
        """更新日志文本"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        color_map = {"info": "#f0f0f0", "warning": "#ffcc00", "error": "#ff4d4d", "success": "#33cc33"}
        color = color_map.get(level, "#f0f0f0")
        log_entry = f'<span style="color: {color};">{message}</span> <span style="color: #888;">[{current_time}]</span>'
        self.log_text.append(log_entry)
        
    def update_progress(self, current, total):
        """更新进度"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            progress_percentage = int((current / total) * 100) if total > 0 else 0
            self.progress_bar.setFormat(f"{progress_percentage}% ({current}/{total})")
            self.update_log(f"仿真进度: {current}/{total} ({progress_percentage}%)", "info")
        
    def on_simulation_finished(self, results):
        """仿真完成时的处理"""
        self.simulation_step_results = results
        if results:
            last_result = results[-1]
            self.simulation_results = last_result['output']
            if 'T1' in last_result['output'] and last_result['output']['T1'] is not None:
                self.last_T1 = last_result['output']['T1']
        self.update_log("所有模型运行完成")
        QMessageBox.information(self, "完成", "仿真运行完成")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(self.progress_bar.maximum())
        self.visualization_manager.model.live_visualization_active = False
        self.running_simulation_models = []
        self.visualization_manager.restart_visualization()
        self.set_model_tab_enabled(True)
        
    def on_model_completed(self, step_result):
        """处理单个模型完成后的结果"""
        if not hasattr(self, 'simulation_step_results'):
            self.simulation_step_results = []
        self.simulation_step_results.append(step_result)
        self.update_log(f"模型 #{step_result['model_id']} 运行完成，结果已添加")
        if self.visualization_manager.model.live_visualization_active:
            self.visualization_manager.draw_live_visualization()
        
    def on_simulation_error(self, error_info):
        """仿真出错时的处理"""
        if isinstance(error_info, dict):
            error_message = f"类型: {error_info.get('type', '未知')}\n\n信息: {error_info.get('message', '无详细信息')}\n\n建议: {error_info.get('solution', '无')}"
        else:
            error_message = str(error_info)
        QMessageBox.critical(self, "错误", error_message)
        if self.visualization_manager.model.visualization_active:
            self.visualization_manager.stop_visualization()
        
        self.visualization_manager.model.live_visualization_active = False
        self.running_simulation_models = []
        self.visualization_manager.draw_model_schematic()
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
        self.set_model_tab_enabled(True)
    
    def clear_results(self):
        """清除所有结果"""
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait()
        self.last_T1 = None
        self.simulation_results = None
        self.simulation_step_results = []
        self.visualization_manager.clear_data()
        self.visualization_manager.stop_visualization()
        stop_all_matlab_engines()
        self.update_log("已清除所有结果并关闭MATLAB引擎")
        self.figure.clear()
        self.canvas.draw()
        temp_dir = get_temp_matlab_dir()
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait()
        
        if self.rl_thread and self.rl_thread.isRunning():
            self.rl_thread.stop()
            self.rl_thread.wait()
            
        event.accept()
        
    def load_results(self):
        """加载结果文件"""
        self.visualization_manager.load_visualization_state()
                
    def _save_config_to_file(self, file_path=None):
        """Internal method to save config to a JSON file."""
        if file_path is None:
            file_path = get_config_file()
        try:
            config_data = {
                "mesh_params": {
                    "H": self.h_edit.value(), "L": self.l_edit.value(),
                    "Nx": self.nx_edit.value(), "Ny": self.ny_edit.value()
                },
                "main_params": {
                    "k": self.k_edit.value(), "midu": self.midu_edit.value(), "Cv": self.cv_edit.value(),
                    "q": self.q_edit.value(), "alpha": self.alpha_edit.value(), "alpha1": self.alpha1_edit.value(),
                    "T_kongqi": self.t_kongqi_edit.value(), "T0": self.t0_edit.value(),
                    "t": self.t_edit.value(), "dt": self.dt_edit.value()
                }
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            if file_path == get_config_file():
                self.config = config_data
            return True
        except Exception as e:
            self.update_log(f"保存配置失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
            return False

    def save_config(self):
        """Saves the configuration parameters to the default config.json."""
        if self._save_config_to_file():
            self.update_log("配置已保存到 config.json")
            QMessageBox.information(self, "成功", "配置参数已保存到默认的 config.json 文件")

    def save_config_as(self):
        """Saves the current configuration to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "配置另存为", "", "JSON Files (*.json)")
        if file_path and self._save_config_to_file(file_path):
            self.update_log(f"配置已另存为: {os.path.basename(file_path)}")
            QMessageBox.information(self, "成功", f"配置已成功保存到 {os.path.basename(file_path)}")

    def load_config_from_file(self):
        """Loads configuration from a JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "从文件加载配置", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
                self._validate_config(new_config)
                self._apply_config_to_ui(new_config)
                self.update_log(f"已从 {os.path.basename(file_path)} 加载配置")
                QMessageBox.information(self, "成功", "配置已成功加载。")
            except (json.JSONDecodeError, ValueError) as e:
                self.update_log(f"加载配置文件失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"加载配置文件失败: {str(e)}")

    def _apply_config_to_ui(self, config_data):
        """Applies a configuration dictionary to the UI widgets."""
        self.k_edit.setValue(config_data["main_params"]["k"])
        self.midu_edit.setValue(config_data["main_params"]["midu"])
        self.cv_edit.setValue(config_data["main_params"]["Cv"])
        self.q_edit.setValue(config_data["main_params"]["q"])
        self.alpha_edit.setValue(config_data["main_params"]["alpha"])
        self.alpha1_edit.setValue(config_data["main_params"]["alpha1"])
        self.t_kongqi_edit.setValue(config_data["main_params"]["T_kongqi"])
        self.t0_edit.setValue(config_data["main_params"]["T0"])
        self.t_edit.setValue(config_data["main_params"]["t"])
        self.dt_edit.setValue(config_data["main_params"]["dt"])
        self.h_edit.setValue(config_data["mesh_params"]["H"])
        self.l_edit.setValue(config_data["mesh_params"]["L"])
        self.nx_edit.setValue(config_data["mesh_params"]["Nx"])
        self.ny_edit.setValue(config_data["mesh_params"]["Ny"])
        self.config = config_data

    def reset_config(self):
        """Resets the configuration parameters to default values."""
        default_config = self.load_initial_config()
        self._apply_config_to_ui(default_config)
        self.update_log("配置已重置为默认值")

    def export_log(self):
        """导出运行日志到文本文件"""
        log_content = self.log_text.toPlainText()
        if not log_content:
            QMessageBox.warning(self, "警告", "日志内容为空，无需导出。")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出日志", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                self.update_log(f"日志已成功导出到: {os.path.basename(file_path)}")
                QMessageBox.information(self, "成功", "日志已成功导出。")
            except Exception as e:
                self.update_log(f"导出日志失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"导出日志失败: {str(e)}")
    
    def save_results(self):
        """保存结果"""
        self.visualization_manager.save_visualization_state()

    def terminate_simulation(self):
        """终止当前正在运行的仿真"""
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.update_log("正在终止仿真...", "warning")
            self.terminate_button.setEnabled(False)
            self.simulation_thread.stop()

    def set_model_tab_enabled(self, enabled):
        """启用或禁用模型管理选项卡中的所有控件"""
        self.model_config_group.setEnabled(enabled)
        self.add_model_button.setEnabled(enabled)
        self.delete_model_button.setEnabled(enabled)
        self.delete_all_models_button.setEnabled(enabled)
        self.save_models_button.setEnabled(enabled)
        self.load_models_button.setEnabled(enabled)
        self.model_table.setEnabled(enabled)

    def _validate_model_sequence(self):
        """校验模型序列是否符合运行要求"""
        models = self.model_manager.simulation_models
        if not models:
            return True
        first_model = models[0]
        if not (first_model.script_type == ScriptType.HEATING and first_model.roll_direction == RollDirection.INITIAL):
            QMessageBox.warning(self, "模型序列无效", "第一个模型必须是升温初始辊。")
            return False
        for i in range(len(models) - 1):
            if models[i].script_type == ScriptType.HEATING and models[i+1].script_type == ScriptType.HEATING:
                QMessageBox.warning(self, "模型序列无效", f"模型 #{i+1} 和 #{i+2} 不能连续为两个升温模型。")
                return False
        for i, model in enumerate(models[1:], start=2):
            if model.script_type == ScriptType.HEATING and model.roll_direction == RollDirection.INITIAL:
                QMessageBox.warning(self, "模型序列无效", f"升温模型 #{i} 不能是初始辊。")
                return False
        return True

class RlOptimizationDialog(QDialog):
    """Dialog for setting RL optimization parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("智能优化参数设置")
        self.setMinimumWidth(400)
        self.checkpoint_path = None # 初始化检查点路径
        self.custom_directions = [] # To store the user-defined roll directions
        
        layout = QVBoxLayout(self)

        # --- 加载检查点 ---
        checkpoint_group = QGroupBox("加载训练状态")
        checkpoint_layout = QHBoxLayout(checkpoint_group)
        self.checkpoint_label = QLabel("未选择文件")
        self.checkpoint_label.setStyleSheet("font-style: italic; color: #888;")
        load_button = QPushButton("加载检查点")
        load_button.clicked.connect(self.load_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_label)
        checkpoint_layout.addStretch()
        checkpoint_layout.addWidget(load_button)
        layout.addWidget(checkpoint_group)
        
        # --- 主参数 ---
        main_params_group = QGroupBox("主要参数")
        form_layout = QFormLayout(main_params_group)
        
        self.n_rolls_input = QSpinBox()
        self.n_rolls_input.setRange(1, 10)
        self.n_rolls_input.setValue(5)
        self.n_rolls_input.valueChanged.connect(self.on_n_rolls_changed)
        form_layout.addRow("辊的数量:", self.n_rolls_input)

        self.num_episodes_input = QSpinBox()
        self.num_episodes_input.setRange(10, 5000)
        self.num_episodes_input.setValue(200)
        self.num_episodes_input.setSingleStep(10)
        form_layout.addRow("训练轮数 (Episodes):", self.num_episodes_input)

        self.target_temp_input = QDoubleSpinBox()
        self.target_temp_input.setRange(0.0, 100.0)
        self.target_temp_input.setValue(25.0)
        self.target_temp_input.setSingleStep(1.0)
        form_layout.addRow("目标温度 (°C):", self.target_temp_input)
        
        layout.addWidget(main_params_group)

        # --- 动作范围参数 ---
        action_params_group = QGroupBox("动作参数范围 (Action Space)")
        action_form_layout = QFormLayout(action_params_group)

        self.temp_min_input = QDoubleSpinBox()
        self.temp_min_input.setRange(50.0, 250.0)
        self.temp_min_input.setValue(100.0)
        action_form_layout.addRow("辊温下限 (°C):", self.temp_min_input)

        self.temp_max_input = QDoubleSpinBox()
        self.temp_max_input.setRange(100.0, 300.0)
        self.temp_max_input.setValue(200.0)
        action_form_layout.addRow("辊温上限 (°C):", self.temp_max_input)

        self.contact_min_input = QDoubleSpinBox()
        self.contact_min_input.setRange(0.1, 10.0)
        self.contact_min_input.setValue(0.1)
        self.contact_min_input.setDecimals(2)
        action_form_layout.addRow("贴辊时间下限 (s):", self.contact_min_input)

        self.contact_max_input = QDoubleSpinBox()
        self.contact_max_input.setRange(0.5, 20.0)
        self.contact_max_input.setValue(5.0)
        self.contact_max_input.setDecimals(2)
        action_form_layout.addRow("贴辊时间上限 (s):", self.contact_max_input)

        self.cooling_min_input = QDoubleSpinBox()
        self.cooling_min_input.setRange(0.1, 10.0)
        self.cooling_min_input.setValue(0.1)
        self.cooling_min_input.setDecimals(2)
        action_form_layout.addRow("冷却时间下限 (s):", self.cooling_min_input)

        self.cooling_max_input = QDoubleSpinBox()
        self.cooling_max_input.setRange(0.5, 20.0)
        self.cooling_max_input.setValue(5.0)
        self.cooling_max_input.setDecimals(2)
        action_form_layout.addRow("冷却时间上限 (s):", self.cooling_max_input)

        layout.addWidget(action_params_group)

        # --- Custom Roll Directions ---
        custom_direction_group = QGroupBox("自定义辊方向")
        custom_direction_layout = QVBoxLayout(custom_direction_group)
        
        self.use_custom_directions_checkbox = QCheckBox("使用自定义方向序列")
        self.use_custom_directions_checkbox.toggled.connect(self.toggle_custom_direction_widgets)
        custom_direction_layout.addWidget(self.use_custom_directions_checkbox)

        self.direction_widgets_container = QWidget()
        direction_layout = QHBoxLayout(self.direction_widgets_container)
        direction_layout.setContentsMargins(0, 5, 0, 0)
        
        self.set_directions_button = QPushButton("设置方向")
        self.set_directions_button.clicked.connect(self.open_direction_selection_dialog)
        direction_layout.addWidget(self.set_directions_button)

        self.directions_label = QLabel("未设置")
        self.directions_label.setStyleSheet("color: #888;")
        direction_layout.addWidget(self.directions_label)
        direction_layout.addStretch()
        
        custom_direction_layout.addWidget(self.direction_widgets_container)
        layout.addWidget(custom_direction_group)
        
        self.toggle_custom_direction_widgets(False) # Initially disable the custom direction widgets

        # --- 按钮 ---
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_n_rolls_changed(self):
        """Resets custom directions when the number of rolls changes."""
        if self.custom_directions:
            self.custom_directions = []
            self.directions_label.setText("未设置 (辊数已更改)")
            self.directions_label.setStyleSheet("color: #ff4d4d;") # Red color to indicate reset
            QMessageBox.warning(self, "注意", "辊的数量已更改，请重新设置自定义方向序列。")

    def toggle_custom_direction_widgets(self, checked):
        """Enables or disables the custom direction setting widgets."""
        self.direction_widgets_container.setEnabled(checked)

    def open_direction_selection_dialog(self):
        """Opens the dialog to select roll directions."""
        n_rolls = self.n_rolls_input.value()
        dialog = DirectionSelectionDialog(n_rolls, self.custom_directions, self)
        if dialog.exec():
            self.custom_directions = dialog.get_directions()
            # Display the selected directions in a compact format
            display_text = ", ".join(["F" if d > 0 else "R" for d in self.custom_directions])
            self.directions_label.setText(f"序列: {display_text}")
            self.directions_label.setStyleSheet("") # Reset style

    def load_checkpoint(self):
        """打开文件对话框以选择检查点文件，并用其参数更新UI"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载训练状态", "modelSave/", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                
                training_params = saved_state.get('training_params')
                if not training_params:
                    raise ValueError("检查点文件不包含 'training_params'。")

                # --- 更新UI元素 ---
                self.n_rolls_input.setValue(training_params.get('n_rolls', 5))
                self.num_episodes_input.setValue(training_params.get('num_episodes', 200))
                self.target_temp_input.setValue(training_params.get('target_temp', 25.0))
                
                bounds = training_params.get('bounds', {})
                self.temp_min_input.setValue(bounds.get('temp_min', 100.0))
                self.temp_max_input.setValue(bounds.get('temp_max', 200.0))
                self.contact_min_input.setValue(bounds.get('contact_min', 0.1))
                self.contact_max_input.setValue(bounds.get('contact_max', 5.0))
                self.cooling_min_input.setValue(bounds.get('cooling_min', 0.1))
                self.cooling_max_input.setValue(bounds.get('cooling_max', 5.0))

                use_custom = training_params.get('use_custom_directions', False)
                self.use_custom_directions_checkbox.setChecked(use_custom)
                if use_custom:
                    self.custom_directions = training_params.get('custom_directions', [])
                    display_text = ", ".join(["F" if d > 0 else "R" for d in self.custom_directions])
                    self.directions_label.setText(f"序列: {display_text}")
                    self.directions_label.setStyleSheet("")
                else:
                    self.custom_directions = []
                    self.directions_label.setText("未设置")

                self.checkpoint_path = file_path
                self.checkpoint_label.setText(os.path.basename(file_path))
                self.checkpoint_label.setStyleSheet("") # 恢复默认样式
                
                QMessageBox.information(self, "成功", f"检查点参数已加载: {os.path.basename(file_path)}\nUI已更新以匹配加载的设置。")

            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"无法加载或解析检查点文件: {e}")
                self.checkpoint_path = None
                self.checkpoint_label.setText("加载失败")
                self.checkpoint_label.setStyleSheet("color: red;")


    def get_parameters(self):
        """Returns the parameters entered by the user."""
        return {
            "n_rolls": self.n_rolls_input.value(),
            "num_episodes": self.num_episodes_input.value(),
            "target_temp": self.target_temp_input.value(),
            "bounds": {
                "temp_min": self.temp_min_input.value(),
                "temp_max": self.temp_max_input.value(),
                "contact_min": self.contact_min_input.value(),
                "contact_max": self.contact_max_input.value(),
                "cooling_min": self.cooling_min_input.value(),
                "cooling_max": self.cooling_max_input.value(),
            },
            "checkpoint_path": self.checkpoint_path,
            "use_custom_directions": self.use_custom_directions_checkbox.isChecked(),
            "custom_directions": self.custom_directions
        }

class DirectionSelectionDialog(QDialog):
    """A dialog to let the user select the direction for each roll."""
    def __init__(self, n_rolls, current_directions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置辊方向序列")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.direction_combos = []
        # The first roll is always initial, so we provide choices for the remaining n_rolls - 1
        for i in range(n_rolls - 1):
            combo = QComboBox()
            combo.addItems(["正向辊 (Forward)", "逆向辊 (Reverse)"])
            # If editing, set to the current direction
            if i < len(current_directions):
                index = 1 if current_directions[i] < 0 else 0
                combo.setCurrentIndex(index)
            
            self.direction_combos.append(combo)
            form_layout.addRow(f"辊 {i + 2} 方向:", combo)
            
        layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_directions(self):
        """Returns the selected directions as a list of floats (+1.0 for Forward, -1.0 for Reverse)."""
        # +1.0 for Forward, -1.0 for Reverse
        return [1.0 if combo.currentIndex() == 0 else -1.0 for combo in self.direction_combos]
