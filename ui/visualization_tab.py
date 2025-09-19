from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QComboBox, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import numpy as np
from logic.Models import ScriptType, RollDirection

def create_visualization_tab(main_window):
    """创建运行可视化选项卡"""
    visualization_widget = QWidget()
    main_layout = QHBoxLayout(visualization_widget)
    main_layout.setContentsMargins(30, 30, 30, 30)
    main_layout.setSpacing(30)

    # --- 左侧：控制面板 ---
    control_panel = QWidget()
    control_panel.setObjectName("controlPanel")
    control_layout = QVBoxLayout(control_panel)
    control_layout.setSpacing(20)
    control_panel.setFixedWidth(300)

    # 可视化控制组
    control_group = QGroupBox("可视化控制")
    control_group_layout = QVBoxLayout(control_group)
    control_group_layout.setContentsMargins(16, 20, 16, 16)
    control_group_layout.setSpacing(12)
    
    main_window.start_visualization_button = QPushButton("开始可视化")
    main_window.start_visualization_button.clicked.connect(main_window.visualization_manager.start_visualization)
    control_group_layout.addWidget(main_window.start_visualization_button)
    
    main_window.stop_visualization_button = QPushButton("停止可视化")
    main_window.stop_visualization_button.clicked.connect(main_window.visualization_manager.stop_visualization)
    main_window.stop_visualization_button.setEnabled(False)
    control_group_layout.addWidget(main_window.stop_visualization_button)
    
    main_window.pause_visualization_button = QPushButton("暂停")
    main_window.pause_visualization_button.clicked.connect(main_window.visualization_manager.toggle_visualization_pause)
    main_window.pause_visualization_button.setEnabled(False)
    control_group_layout.addWidget(main_window.pause_visualization_button)
    
    main_window.show_gradient_button = QPushButton("显示实时温度梯度")
    main_window.show_gradient_button.clicked.connect(main_window.visualization_manager.show_realtime_temperature_gradient)
    control_group_layout.addWidget(main_window.show_gradient_button)
    
    main_window.save_visualization_button = QPushButton("保存可视化")
    main_window.save_visualization_button.clicked.connect(main_window.visualization_manager.save_visualization_state)
    control_group_layout.addWidget(main_window.save_visualization_button)
    
    main_window.load_visualization_button = QPushButton("加载可视化")
    main_window.load_visualization_button.clicked.connect(main_window.visualization_manager.load_visualization_state)
    control_group_layout.addWidget(main_window.load_visualization_button)

    main_window.plot_temperature_profile_button = QPushButton("绘制温度分布")
    main_window.plot_temperature_profile_button.clicked.connect(main_window.visualization_manager.plot_temperature_profile)
    control_group_layout.addWidget(main_window.plot_temperature_profile_button)
    
    control_layout.addWidget(control_group)

    # 显示设置组
    display_group = QGroupBox("显示设置")
    display_group_layout = QVBoxLayout(display_group)
    display_group_layout.setContentsMargins(16, 20, 16, 16)
    display_group_layout.setSpacing(12)

    # 播放速度控制
    speed_control_layout = QHBoxLayout()
    speed_label = QLabel("播放速度:")
    main_window.speed_combo = QComboBox()
    main_window.speed_combo.setObjectName("speedComboBox")
    main_window.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x", "4.0x"])
    main_window.speed_combo.setCurrentText("1.0x")
    main_window.speed_combo.currentTextChanged.connect(
        lambda text: main_window.visualization_manager.on_playback_speed_changed(text)
    )
    speed_control_layout.addWidget(speed_label)
    speed_control_layout.addWidget(main_window.speed_combo)
    display_group_layout.addLayout(speed_control_layout)

    # 帧控制
    frame_control_layout = QHBoxLayout()
    main_window.previous_frame_button = QPushButton("上一步")
    main_window.previous_frame_button.setObjectName("smallButton")
    main_window.previous_frame_button.clicked.connect(main_window.visualization_manager.previous_frame)
    main_window.previous_frame_button.setEnabled(False)
    frame_control_layout.addWidget(main_window.previous_frame_button)

    main_window.next_frame_button = QPushButton("下一步")
    main_window.next_frame_button.setObjectName("smallButton")
    main_window.next_frame_button.clicked.connect(main_window.visualization_manager.next_frame)
    main_window.next_frame_button.setEnabled(False)
    frame_control_layout.addWidget(main_window.next_frame_button)
    display_group_layout.addLayout(frame_control_layout)

    main_window.show_contour_checkbox = QCheckBox("显示实时等高线图")
    main_window.show_contour_checkbox.setChecked(False)
    main_window.show_contour_checkbox.stateChanged.connect(main_window.visualization_manager.toggle_live_contour_visibility)
    display_group_layout.addWidget(main_window.show_contour_checkbox)



    control_layout.addWidget(display_group)
    control_layout.addStretch()

    # --- 右侧：可视化显示区域 ---
    visualization_panel = QWidget()
    visualization_panel.setObjectName("visualizationPanel")
    visualization_layout = QVBoxLayout(visualization_panel)
    visualization_layout.setContentsMargins(0, 0, 0, 0)
    visualization_layout.setSpacing(15)
    
    # 创建实时等高线图区域
    main_window.live_contour_group = QGroupBox("实时等高线图")
    live_contour_layout = QVBoxLayout(main_window.live_contour_group)
    main_window.live_contour_figure = Figure(figsize=(8, 4), dpi=100)
    main_window.live_contour_figure.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
    main_window.live_contour_canvas = FigureCanvas(main_window.live_contour_figure)
    main_window.live_contour_canvas.setObjectName("live_contour_canvas")
    live_contour_layout.addWidget(main_window.live_contour_canvas)
    main_window.live_contour_group.setVisible(False)
    visualization_layout.addWidget(main_window.live_contour_group)
    
    # 创建matplotlib图形
    main_window.visualization_figure = Figure(figsize=(8, 6), dpi=100)
    main_window.visualization_figure.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
    main_window.visualization_figure.patch.set_alpha(0)
    main_window.visualization_canvas = FigureCanvas(main_window.visualization_figure)
    main_window.visualization_canvas.setObjectName("visualization_canvas")
    visualization_layout.addWidget(main_window.visualization_canvas)
    
    # 状态信息
    status_layout = QHBoxLayout()
    status_label = QLabel("当前状态:")
    main_window.current_status_label = QLabel("等待开始")
    main_window.current_status_label.setObjectName("statusLabel")
    
    status_widget = QWidget()
    status_widget.setObjectName("statusWidget")
    status_widget.setLayout(status_layout)
    
    status_layout.addWidget(status_label)
    status_layout.addWidget(main_window.current_status_label)
    status_layout.addStretch()
    
    visualization_layout.addWidget(status_widget)

    main_layout.addWidget(control_panel)
    main_layout.addWidget(visualization_panel, 1)

    # 初始化可视化定时器
    
    main_window.visualization_timer = QTimer()
    
    return visualization_widget
