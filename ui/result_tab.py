from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QPushButton, QCheckBox, QLabel, QStackedWidget
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def create_result_tab(main_window):
    """创建结果可视化选项卡"""
    main_window.result_tab = QWidget()
    main_layout = QHBoxLayout(main_window.result_tab)
    main_layout.setContentsMargins(24, 24, 24, 24)
    main_layout.setSpacing(24)

    # --- 左侧：控制面板 ---
    control_panel = QWidget()
    control_layout = QVBoxLayout(control_panel)
    control_layout.setSpacing(16)
    control_panel.setFixedWidth(280)

    # 结果控制组
    control_group = QGroupBox("结果控制")
    control_group_layout = QVBoxLayout(control_group)
    control_group_layout.setContentsMargins(16, 20, 16, 16)
    control_group_layout.setSpacing(12)
    
    main_window.load_results_button = QPushButton("加载结果")
    main_window.load_results_button.clicked.connect(main_window.visualization_manager.load_visualization_state)
    control_group_layout.addWidget(main_window.load_results_button)
    
    save_button = QPushButton("保存结果")
    save_button.clicked.connect(main_window.visualization_manager.save_visualization_state)
    control_group_layout.addWidget(save_button)
    
    control_layout.addWidget(control_group)

    # 可视化组
    visualize_group = QGroupBox("可视化")
    visualize_group_layout = QVBoxLayout(visualize_group)
    visualize_group_layout.setContentsMargins(16, 20, 16, 16)
    visualize_group_layout.setSpacing(12)

    display_step_button = QPushButton("显示所有步骤结果")
    display_step_button.clicked.connect(main_window.visualization_manager.visualize_step_results)
    visualize_group_layout.addWidget(display_step_button)

    display_combined_contour_button = QPushButton("显示整体温度云图")
    display_combined_contour_button.clicked.connect(main_window.visualization_manager.visualize_combined_contour)
    visualize_group_layout.addWidget(display_combined_contour_button)
    
    display_step_params_button = QPushButton("显示步骤参数")
    display_step_params_button.clicked.connect(main_window.visualization_manager.display_all_step_parameters)
    visualize_group_layout.addWidget(display_step_params_button)

    main_window.show_curves_button = QPushButton("显示温度变化曲线")
    main_window.show_curves_button.clicked.connect(main_window.visualization_manager.show_temperature_curves)
    
    main_window.show_model_ids_checkbox = QCheckBox("显示模型ID")
    main_window.show_model_ids_checkbox.setChecked(True)
    
    curves_layout = QHBoxLayout()
    curves_layout.addWidget(main_window.show_curves_button)
    curves_layout.addStretch()
    visualize_group_layout.addLayout(curves_layout)
    visualize_group_layout.addWidget(main_window.show_model_ids_checkbox)

    save_plots_button = QPushButton("保存所有图片")
    save_plots_button.clicked.connect(main_window.visualization_manager.save_all_plots)
    visualize_group_layout.addWidget(save_plots_button)
    
    control_layout.addWidget(visualize_group)
    control_layout.addStretch()

    # --- 右侧：图表显示区域 (使用 QStackedWidget) ---
    display_panel = QWidget()
    display_layout = QVBoxLayout(display_panel)
    display_layout.setContentsMargins(0, 0, 0, 0)
    
    main_window.result_display_stack = QStackedWidget()
    
    # 页面1: Matplotlib Canvas for charts
    main_window.figure = Figure(figsize=(8, 6), dpi=100)
    main_window.figure.patch.set_alpha(0)
    main_window.canvas = FigureCanvas(main_window.figure)
    main_window.canvas.setObjectName("canvas")
    main_window.result_display_stack.addWidget(main_window.canvas)
    
    # 页面2: QLabel for static images
    main_window.result_image_label = QLabel("No image to display.")
    main_window.result_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    main_window.result_image_label.setObjectName("resultImageLabel")
    main_window.result_display_stack.addWidget(main_window.result_image_label)
    
    display_layout.addWidget(main_window.result_display_stack)

    main_layout.addWidget(control_panel)
    main_layout.addWidget(display_panel, 1)

    return main_window.result_tab
