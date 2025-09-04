from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox,
    QPushButton, QTableWidget, QHeaderView, QHBoxLayout, QLabel
)
from Models import ScriptType, RollDirection
from .delegates import ModelDelegate

def create_model_tab(main_window):
    """创建模型管理选项卡"""
    model_widget = QWidget()
    main_layout = QHBoxLayout(model_widget)
    main_layout.setContentsMargins(24, 24, 24, 24)
    main_layout.setSpacing(24)

    # --- 左侧：模型配置 ---
    left_panel = QWidget()
    left_panel.setMaximumWidth(350)
    left_layout = QVBoxLayout(left_panel)
    left_layout.setSpacing(16)
    
    main_window.model_config_group = QGroupBox("模型配置")
    form_layout = QFormLayout(main_window.model_config_group)
    form_layout.setSpacing(12)
    form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    
    main_window.script_type_combo = QComboBox()
    main_window.script_type_combo.addItems([ScriptType.HEATING, ScriptType.COOLING])
    main_window.script_type_combo.currentTextChanged.connect(main_window.model_manager.on_script_type_changed)
    form_layout.addRow("脚本类型:", main_window.script_type_combo)
    
    main_window.roll_direction_combo = QComboBox()
    main_window.roll_direction_combo.addItems([RollDirection.INITIAL, RollDirection.FORWARD, RollDirection.REVERSE])
    form_layout.addRow("滚动方向:", main_window.roll_direction_combo)
    
    main_window.t_gunwen_input = QDoubleSpinBox()
    main_window.t_gunwen_input.setRange(273.1500, 500.0000)
    main_window.t_gunwen_input.setValue(315.1500)
    main_window.t_gunwen_input.setSingleStep(10)
    main_window.t_gunwen_input.setDecimals(4)
    form_layout.addRow("辊温 (K):", main_window.t_gunwen_input)
    
    main_window.t_up_input = QDoubleSpinBox()
    main_window.t_up_input.setRange(0.1000, 100.0000)
    main_window.t_up_input.setValue(1.3800)
    main_window.t_up_input.setSingleStep(0.01)
    main_window.t_up_input.setDecimals(4)
    form_layout.addRow("时间上限 (s):", main_window.t_up_input)
    
    main_window.add_model_button = QPushButton("添加模型")
    main_window.add_model_button.clicked.connect(main_window.model_manager.add_model)
    
    instruction_label = QLabel("注意：第一个模型必须是正向初始辊")
    instruction_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
    instruction_label.setStyleSheet("color: #888; font-size: 12px; margin-top: 10px;")
    
    left_layout.addWidget(main_window.model_config_group)
    left_layout.addWidget(main_window.add_model_button)
    left_layout.addStretch()
    left_layout.addWidget(instruction_label)

    # --- 右侧：模型列表 ---
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setSpacing(16)
    
    table_group = QGroupBox("模型序列")
    table_layout = QVBoxLayout(table_group)
    table_layout.setContentsMargins(16, 20, 16, 16)
    table_layout.setSpacing(12)
    
    main_window.model_table = QTableWidget()
    main_window.model_table.setColumnCount(5)
    main_window.model_table.setHorizontalHeaderLabels(["ID", "脚本类型", "滚动方向", "辊温 (K)", "时间上限 (s)"])
    main_window.model_table.setAlternatingRowColors(True)
    main_window.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    main_window.model_table.verticalHeader().setDefaultSectionSize(45)
    
    header = main_window.model_table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
    main_window.model_table.setColumnWidth(0, 50)

    delegate = ModelDelegate(main_window.model_table)
    main_window.model_table.setItemDelegate(delegate)
    main_window.model_table.cellChanged.connect(main_window.model_manager.on_model_table_cell_changed)
    
    table_layout.addWidget(main_window.model_table)

    instruction_label_2 = QLabel("双击可以修改模型参数")
    instruction_label_2.setAlignment(Qt.AlignmentFlag.AlignRight)
    instruction_label_2.setStyleSheet("color: #888;")
    table_layout.addWidget(instruction_label_2)
    
    delete_row_widget = QWidget()
    delete_row_layout = QHBoxLayout(delete_row_widget)
    delete_row_layout.setContentsMargins(0, 0, 0, 0)
    delete_row_layout.setSpacing(12)

    main_window.delete_model_button = QPushButton("删除选中模型")
    main_window.delete_model_button.setObjectName("deleteButton")
    main_window.delete_model_button.clicked.connect(main_window.model_manager.delete_model)
    delete_row_layout.addWidget(main_window.delete_model_button)

    main_window.delete_all_models_button = QPushButton("删除所有模型")
    main_window.delete_all_models_button.setObjectName("deleteButton")
    main_window.delete_all_models_button.clicked.connect(main_window.model_manager.delete_all_models)
    delete_row_layout.addWidget(main_window.delete_all_models_button)
    delete_row_layout.addStretch()

    table_layout.addWidget(delete_row_widget)

    save_load_row_widget = QWidget()
    save_load_layout = QHBoxLayout(save_load_row_widget)
    save_load_layout.setContentsMargins(0, 0, 0, 0)
    save_load_layout.setSpacing(12)

    main_window.save_models_button = QPushButton("保存模型序列")
    main_window.save_models_button.clicked.connect(main_window.model_manager.save_model_sequence)
    save_load_layout.addWidget(main_window.save_models_button)

    main_window.load_models_button = QPushButton("加载模型序列")
    main_window.load_models_button.clicked.connect(main_window.model_manager.load_model_sequence)
    save_load_layout.addWidget(main_window.load_models_button)
    save_load_layout.addStretch()

    table_layout.addWidget(save_load_row_widget)
    
    right_layout.addWidget(table_group)

    main_layout.addWidget(left_panel, 1)
    main_layout.addWidget(right_panel, 2)

    return model_widget
