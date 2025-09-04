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
    QHBoxLayout, QSizePolicy, QFrame, QStackedWidget
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

# 项目模块导入
from config.paths import (
    get_config_file,
    get_styles_file,
    get_icon_file,
    get_logo_file,
    get_temp_matlab_dir,
    get_background_image_file,
    get_check_image_file
)
from Models import ScriptType, RollDirection, SimulationModel
from threads.simulation_thread import SimulationThread
from run_matlab_simulation import stop_all_matlab_engines, load_config
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
        self.model_manager = ModelManager(self)
        self.visualization_manager = VisualizationManager(self)
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
        # 应用样式表
        qss_file_path = get_styles_file()
        with open(qss_file_path, "r", encoding="utf-8") as f:
            stylesheet = f.read()
            # Dynamically replace placeholders with absolute paths
            stylesheet = stylesheet.replace(
                "url(public/images/background1.png)",
                f"url({get_background_image_file()})"
            )
            stylesheet = stylesheet.replace(
                "url(public/images/check.png)",
                f"url({get_check_image_file()})"
            )
            self.setStyleSheet(stylesheet)
        
        main_widget = QWidget()

        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)  # 增加边距
        main_layout.setSpacing(16)  # 设置组件间距

        # 添加模糊效果
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
        
        # 添加底部信息区域
        bottom_frame = QFrame()
        bottom_frame.setFixedHeight(60)
        bottom_frame.setObjectName("bottomFrame")
        
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(20, 10, 20, 10)
        
        # 左侧帮助按钮
        help_button = QPushButton("帮助")
        help_button.clicked.connect(self.show_help_tab)
        help_button.setObjectName("helpButton")
        help_button.setFixedSize(80, 30)
        bottom_layout.addWidget(help_button)
        
        # 左侧占位空间
        left_spacer = QLabel()
        left_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bottom_layout.addWidget(left_spacer)
        
        # 中间作者邮箱信息
        author_label = QLabel("作者邮箱：zhanghaozheng@mail.ustc.edu.cn")
        author_label.setObjectName("authorLabel")
        author_label.setStyleSheet("""
            QLabel#authorLabel {
                color: rgba(0, 0, 0, 0.3);
                background-color: transparent;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 13px;
            }
        """)
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 设置作者标签的字体和样式
        author_font = QFont()
        author_font.setPointSize(10)
        author_font.setFamily("Microsoft YaHei")
        author_label.setFont(author_font)
        author_label.setStyleSheet("""
            QLabel#authorLabel {
                color: #000000;
                background-color: transparent;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 13px;
            }
        """)
        bottom_layout.addWidget(author_label)
        
        # 右侧logo区域
        logo_container = QWidget()
        logo_container.setFixedSize(130, 40)
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        logo_label = QLabel()
        logo_label.setObjectName("logoLabel")
        logo_label.setFixedSize(120, 35)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 尝试加载logo图片，如果没有则显示文字logo
        logo_path = get_logo_file()
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(100, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            # 如果没有logo图片，显示文字logo
            logo_label.setText("USTC Logo")
            logo_label.setStyleSheet("""
                QLabel#logoLabel {
                    background-color: #4a90e2;
                    color: white;
                    border-radius: 5px;
                    font-size: 10px;
                    font-weight: bold;
                    font-family: 'Microsoft YaHei';
                }
            """)
        
        logo_layout.addWidget(logo_label)
        bottom_layout.addWidget(logo_container)
        
        main_layout.addWidget(bottom_frame)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Apply the loaded configuration to the UI
        self._apply_config_to_ui(self.config)
        
        self.model_manager.on_script_type_changed(self.script_type_combo.currentText())
        
        # 设置日志文本框为富文本格式
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("log_text")

        self.visualization_manager.draw_model_schematic()

    def show_help_tab(self):
        """切换到帮助选项卡"""
        self.tabs.setCurrentWidget(self.help_tab)

        
    def run_simulation(self):
        """运行仿真"""
        if not self.model_manager.simulation_models:
            QMessageBox.warning(self, "警告", "请先添加至少一个模型")
            return

        # 清除旧的可视化数据
        self.visualization_manager.clear_data()
        self.update_log("已清除旧的可视化数据。")

        # 自动保存当前配置
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
        
        self.run_button.setEnabled(False)
        self.terminate_button.setEnabled(True)
        self.clear_button.setEnabled(False)
        self.save_config_button.setEnabled(False)
        self.reset_config_button.setEnabled(False)
        self.load_config_from_file_button.setEnabled(False)
        self.load_results_button.setEnabled(False)
        self.load_visualization_button.setEnabled(False)
        self.set_model_tab_enabled(False)
        
        # 重置进度条
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(self.model_manager.simulation_models))

        # Start live visualization
        self.visualization_manager.live_visualization_active = True
        self.simulation_step_results = [] # Clear previous results
        self.visualization_manager.draw_live_visualization() # Draw initial empty state
            
        self.simulation_thread.start()
        
    def update_log(self, message, level="info"):
        """更新日志文本"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            "info": "#f0f0f0",
            "warning": "#ffcc00",
            "error": "#ff4d4d",
            "success": "#33cc33"
        }
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
        
        # 完成时设置进度条为100%
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(self.progress_bar.maximum())
            
        # 仿真完成后自动重启可视化
        self.visualization_manager.live_visualization_active = False
        self.running_simulation_models = []
        self.visualization_manager.restart_visualization()
            
        self.run_button.setEnabled(True)
        self.terminate_button.setEnabled(False)
        self.clear_button.setEnabled(True)
        self.save_config_button.setEnabled(True)
        self.reset_config_button.setEnabled(True)
        self.load_config_from_file_button.setEnabled(True)
        self.load_results_button.setEnabled(True)
        self.load_visualization_button.setEnabled(True)
        self.set_model_tab_enabled(True)
        
    def on_model_completed(self, step_result):
        """处理单个模型完成后的结果"""
        # 将结果添加到simulation_step_results列表中
        if not hasattr(self, 'simulation_step_results'):
            self.simulation_step_results = []
        self.simulation_step_results.append(step_result)
        
        # 更新日志
        self.update_log(f"模型 #{step_result['model_id']} 运行完成，结果已添加")

        if self.visualization_manager.live_visualization_active:
            self.visualization_manager.draw_live_visualization()
        
        
    def on_simulation_error(self, error_info):
        """仿真出错时的处理"""
        if isinstance(error_info, dict):
            error_message = f"类型: {error_info.get('type', '未知')}\n\n" \
                            f"信息: {error_info.get('message', '无详细信息')}\n\n" \
                            f"建议: {error_info.get('solution', '无')}"
        else:
            error_message = str(error_info)
            
        QMessageBox.critical(self, "错误", error_message)
        
        # 出错时停止可视化
        if self.visualization_manager.visualization_active:
            self.visualization_manager.stop_visualization()

        self.visualization_manager.live_visualization_active = False
        self.running_simulation_models = []
        self.visualization_manager.draw_model_schematic()
            
        # 出错时重置进度条
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
            
        self.run_button.setEnabled(True)
        self.terminate_button.setEnabled(False)
        self.clear_button.setEnabled(True)
        self.save_config_button.setEnabled(True)
        self.reset_config_button.setEnabled(True)
        self.load_config_from_file_button.setEnabled(True)
        self.load_results_button.setEnabled(True)
        self.load_visualization_button.setEnabled(True)
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
        
        # 停止所有MATLAB引擎
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
                    "H": self.h_edit.value(),
                    "L": self.l_edit.value(),
                    "Nx": self.nx_edit.value(),
                    "Ny": self.ny_edit.value()
                },
                "main_params": {
                    "k": self.k_edit.value(),
                    "midu": self.midu_edit.value(),
                    "Cv": self.cv_edit.value(),
                    "q": self.q_edit.value(),
                    "alpha": self.alpha_edit.value(),
                    "alpha1": self.alpha1_edit.value(),
                    "T_kongqi": self.t_kongqi_edit.value(),
                    "T0": self.t0_edit.value(),
                    "t": self.t_edit.value(),
                    "dt": self.dt_edit.value()
                }
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            # Update the in-memory config only if saving to the default path
            default_path = get_config_file()
            if file_path == default_path:
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
        if file_path:
            if self._save_config_to_file(file_path):
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
            except Exception as e:
                self.update_log(f"处理配置文件时发生未知错误: {str(e)}")
                QMessageBox.critical(self, "错误", f"处理配置文件时发生未知错误: {str(e)}")

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
            self.terminate_button.setEnabled(False) # 防止重复点击
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
