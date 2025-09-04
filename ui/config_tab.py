from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QDoubleSpinBox, 
    QPushButton, QSpinBox, QGridLayout, QHBoxLayout
)

def create_config_tab(main_window):
    """创建配置参数选项卡"""
    config_widget = QWidget()
    main_layout = QVBoxLayout(config_widget)
    main_layout.setContentsMargins(24, 24, 24, 24)
    main_layout.setSpacing(20)

    # --- 参数设置区域 ---
    params_layout = QGridLayout()
    params_layout.setSpacing(20)

    # 物理参数组
    physical_params_group = QGroupBox("物理参数")
    physical_params_group.setMinimumWidth(350)
    physical_params_layout = QFormLayout(physical_params_group)
    physical_params_layout.setSpacing(15)
    physical_params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    physical_params_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    
    main_window.k_edit = QDoubleSpinBox()
    main_window.k_edit.setRange(0.0001, 10.0000)
    main_window.k_edit.setSingleStep(0.0001)
    main_window.k_edit.setDecimals(4)
    main_window.k_edit.setMinimumWidth(120)
    main_window.k_edit.setToolTip("材料的导热系数，影响热量传递的速度。")
    physical_params_layout.addRow("导热系数 (W/(m·K)):", main_window.k_edit)
    
    main_window.midu_edit = QDoubleSpinBox()
    main_window.midu_edit.setRange(100.0, 10000.0)
    main_window.midu_edit.setDecimals(4)
    main_window.midu_edit.setMinimumWidth(120)
    main_window.midu_edit.setToolTip("材料的密度，影响其质量和热容。")
    physical_params_layout.addRow("密度 (kg/m³):", main_window.midu_edit)
    
    main_window.cv_edit = QDoubleSpinBox()
    main_window.cv_edit.setRange(100.0, 10000.0)
    main_window.cv_edit.setDecimals(4)
    main_window.cv_edit.setMinimumWidth(120)
    main_window.cv_edit.setToolTip("材料的比热容，表示单位质量材料升高单位温度所需的热量。")
    physical_params_layout.addRow("比热容 (J/(kg·K)):", main_window.cv_edit)
    
    main_window.q_edit = QDoubleSpinBox()
    main_window.q_edit.setRange(0.0, 10000.0)
    main_window.q_edit.setDecimals(4)
    main_window.q_edit.setMinimumWidth(120)
    main_window.q_edit.setToolTip("施加在材料表面的热流密度。")
    physical_params_layout.addRow("热流密度 (W/m²):", main_window.q_edit)
    
    main_window.alpha_edit = QDoubleSpinBox()
    main_window.alpha_edit.setRange(0.0, 1000.0)
    main_window.alpha_edit.setDecimals(4)
    main_window.alpha_edit.setMinimumWidth(120)
    main_window.alpha_edit.setToolTip("材料表面与周围介质之间的对流换热系数。")
    physical_params_layout.addRow("对流换热系数 (W/(m²·K)):", main_window.alpha_edit)
    
    main_window.alpha1_edit = QDoubleSpinBox()
    main_window.alpha1_edit.setRange(0.0, 1000.0)
    main_window.alpha1_edit.setDecimals(4)
    main_window.alpha1_edit.setMinimumWidth(120)
    main_window.alpha1_edit.setToolTip("材料另一表面与周围介质之间的对流换热系数。")
    physical_params_layout.addRow("对流换热系数1 (W/(m²·K)):", main_window.alpha1_edit)
    
    main_window.t_kongqi_edit = QDoubleSpinBox()
    main_window.t_kongqi_edit.setRange(273.15, 400.0)
    main_window.t_kongqi_edit.setDecimals(4)
    main_window.t_kongqi_edit.setMinimumWidth(120)
    main_window.t_kongqi_edit.setToolTip("周围空气的温度。")
    physical_params_layout.addRow("空气温度 (K):", main_window.t_kongqi_edit)
    
    main_window.t0_edit = QDoubleSpinBox()
    main_window.t0_edit.setRange(273.15, 400.0)
    main_window.t0_edit.setDecimals(4)
    main_window.t0_edit.setMinimumWidth(120)
    main_window.t0_edit.setToolTip("薄膜的初始温度。")
    physical_params_layout.addRow("薄膜初始温度 (K):", main_window.t0_edit)

    params_layout.addWidget(physical_params_group, 0, 0)

    # 仿真参数组
    simulation_params_group = QGroupBox("仿真参数")
    simulation_params_group.setMinimumWidth(350)
    simulation_params_layout = QFormLayout(simulation_params_group)
    simulation_params_layout.setSpacing(15)
    simulation_params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    simulation_params_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    
    main_window.t_edit = QDoubleSpinBox()
    main_window.t_edit.setRange(0.0, 1000.0)
    main_window.t_edit.setSingleStep(0.1)
    main_window.t_edit.setDecimals(4)
    main_window.t_edit.setMinimumWidth(120)
    main_window.t_edit.setToolTip("仿真开始的初始时间。")
    simulation_params_layout.addRow("初始时间 (s):", main_window.t_edit)
    
    main_window.dt_edit = QDoubleSpinBox()
    main_window.dt_edit.setRange(0.0001, 10.0)
    main_window.dt_edit.setSingleStep(0.01)
    main_window.dt_edit.setDecimals(4)
    main_window.dt_edit.setMinimumWidth(120)
    main_window.dt_edit.setToolTip("仿真计算的时间步长。")
    simulation_params_layout.addRow("时间步长 (s):", main_window.dt_edit)

    params_layout.addWidget(simulation_params_group, 1, 0)

    # 网格划分参数组
    mesh_params_group = QGroupBox("网格划分参数")
    mesh_params_group.setMinimumWidth(350)
    mesh_params_layout = QFormLayout(mesh_params_group)
    mesh_params_layout.setSpacing(15)
    mesh_params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    mesh_params_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

    main_window.h_edit = QDoubleSpinBox()
    main_window.h_edit.setRange(0.0001, 1.0)
    main_window.h_edit.setSingleStep(0.0001)
    main_window.h_edit.setDecimals(6)
    main_window.h_edit.setMinimumWidth(120)
    main_window.h_edit.setToolTip("铸片的厚度。")
    mesh_params_layout.addRow("铸片厚度 (H):", main_window.h_edit)

    main_window.l_edit = QDoubleSpinBox()
    main_window.l_edit.setRange(0.0001, 1.0)
    main_window.l_edit.setSingleStep(0.0001)
    main_window.l_edit.setDecimals(6)
    main_window.l_edit.setMinimumWidth(120)
    main_window.l_edit.setToolTip("铸片的宽度。")
    mesh_params_layout.addRow("铸片宽度 (L):", main_window.l_edit)

    main_window.nx_edit = QSpinBox()
    main_window.nx_edit.setRange(10, 1000)
    main_window.nx_edit.setMinimumWidth(120)
    main_window.nx_edit.setToolTip("X方向上的网格划分数量。不宜过高，25左右即可。(10-1000)")
    mesh_params_layout.addRow("Nx:", main_window.nx_edit)

    main_window.ny_edit = QSpinBox()
    main_window.ny_edit.setRange(10, 1000)
    main_window.ny_edit.setMinimumWidth(120)
    main_window.ny_edit.setToolTip("Y方向上的网格划分数量。不宜过高，25左右即可。(10-1000)")
    mesh_params_layout.addRow("Ny:", main_window.ny_edit)

    params_layout.addWidget(mesh_params_group, 0, 1, 2, 1)
    params_layout.setColumnStretch(0, 1)
    params_layout.setColumnStretch(1, 1)

    main_layout.addLayout(params_layout)

    # --- 按钮区域 ---
    button_layout = QHBoxLayout()
    
    main_window.save_config_button = QPushButton("保存配置")
    main_window.save_config_button.setToolTip("将当前配置保存到默认文件。")
    main_window.save_config_button.clicked.connect(main_window.save_config)
    button_layout.addWidget(main_window.save_config_button)

    main_window.save_config_as_button = QPushButton("另存为...")
    main_window.save_config_as_button.setToolTip("将当前配置保存到指定文件。")
    main_window.save_config_as_button.clicked.connect(main_window.save_config_as)
    button_layout.addWidget(main_window.save_config_as_button)

    main_window.load_config_from_file_button = QPushButton("加载配置...")
    main_window.load_config_from_file_button.setToolTip("从文件中加载配置。")
    main_window.load_config_from_file_button.clicked.connect(main_window.load_config_from_file)
    button_layout.addWidget(main_window.load_config_from_file_button)
    
    main_window.reset_config_button = QPushButton("重置")
    main_window.reset_config_button.setObjectName("resetButton") # 用于特定样式
    main_window.reset_config_button.setToolTip("将配置重置为默认值。")
    main_window.reset_config_button.setObjectName("resetButton") # 用于特定样式
    main_window.reset_config_button.clicked.connect(main_window.reset_config)
    button_layout.addWidget(main_window.reset_config_button)
    
    main_layout.addLayout(button_layout)
    main_layout.addStretch()

    return config_widget
