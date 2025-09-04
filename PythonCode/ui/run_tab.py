from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QTextEdit, QHBoxLayout, QLabel, QProgressBar
)
from PyQt6.QtCore import Qt

def create_run_tab(main_window):
    """创建运行仿真选项卡"""
    run_widget = QWidget()
    layout = QVBoxLayout(run_widget)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(20)
    
    # --- 运行控制 ---
    control_group = QGroupBox("运行控制")
    control_layout = QHBoxLayout(control_group)
    control_layout.setContentsMargins(16, 20, 16, 16)
    control_layout.setSpacing(16)
    
    main_window.run_button = QPushButton("运行仿真")
    main_window.run_button.clicked.connect(main_window.run_simulation)
    main_window.run_button.setObjectName("primaryButton")
    control_layout.addWidget(main_window.run_button)

    main_window.terminate_button = QPushButton("终止仿真")
    main_window.terminate_button.clicked.connect(main_window.terminate_simulation)
    main_window.terminate_button.setObjectName("dangerButton")
    main_window.terminate_button.setEnabled(False)  # Initially disabled
    control_layout.addWidget(main_window.terminate_button)
    
    main_window.clear_button = QPushButton("清除所有结果")
    main_window.clear_button.setObjectName("deleteButton")
    main_window.clear_button.clicked.connect(main_window.clear_results)
    control_layout.addWidget(main_window.clear_button)
    
    control_layout.addStretch()
    
    # 添加进度条
    progress_layout = QVBoxLayout()
    progress_label = QLabel("仿真进度")
    progress_label.setStyleSheet("""
        font-weight: 700; 
        color: #1e293b;
        font-size: 14px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(248, 250, 252, 0.6), stop:1 rgba(241, 245, 249, 0.6));
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 8px;
        padding: 6px 12px;
    """)
    progress_layout.addWidget(progress_label)
    
    main_window.progress_bar = QProgressBar()
    main_window.progress_bar.setMinimum(0)
    main_window.progress_bar.setMaximum(100)
    main_window.progress_bar.setValue(0)
    main_window.progress_bar.setTextVisible(True)
    main_window.progress_bar.setFormat("%p% (%v/%m)")
    main_window.progress_bar.setObjectName("progressBar")
    progress_layout.addWidget(main_window.progress_bar)
    
    control_layout.addLayout(progress_layout)
    control_layout.addStretch()
    layout.addWidget(control_group)
    
    # --- 运行日志 ---
    log_group = QGroupBox("运行日志")
    log_layout = QVBoxLayout(log_group)
    log_layout.setContentsMargins(16, 20, 16, 16)
    log_layout.setSpacing(12)
    
    # 日志标题和导出按钮
    log_header_layout = QHBoxLayout()
    log_header_layout.setSpacing(12)
    log_label = QLabel("实时日志输出")
    log_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
    log_header_layout.addWidget(log_label)
    log_header_layout.addStretch()
    
    main_window.export_log_button = QPushButton("导出日志")
    main_window.export_log_button.clicked.connect(main_window.export_log)
    log_header_layout.addWidget(main_window.export_log_button)
    
    log_layout.addLayout(log_header_layout)
    
    main_window.log_text = QTextEdit()
    main_window.log_text.setReadOnly(True)
    main_window.log_text.setObjectName("log_text")
    main_window.log_text.setMinimumHeight(300)
    log_layout.addWidget(main_window.log_text)
    
    layout.addWidget(log_group)
    
    return run_widget
