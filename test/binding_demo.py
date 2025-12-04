#!/usr/bin/env python3
"""
PyQt双向绑定演示脚本
展示如何在不使用databind库的情况下实现数据双向绑定
"""

import sys
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
)


class DataModel(QObject):
    """数据模型类"""
    data_changed = pyqtSignal(str)  # 数据变化信号
    
    def __init__(self, initial_value: str = ""):
        super().__init__()
        self._data = initial_value
    
    @property
    def data(self) -> str:
        return self._data
    
    @data.setter
    def data(self, value: str):
        if self._data != value:
            self._data = value
            self.data_changed.emit(value)


class BindingDemo(QWidget):
    """双向绑定演示窗口"""
    
    def __init__(self):
        super().__init__()
        self.model = DataModel("初始数据")
        self.init_ui()
        self.setup_bindings()
        self.setWindowTitle("PyQt双向绑定演示")
        self.resize(400, 200)
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 输入框1
        input1_layout = QHBoxLayout()
        input1_layout.addWidget(QLabel("输入框1:"))
        self.input1 = QLineEdit()
        input1_layout.addWidget(self.input1)
        layout.addLayout(input1_layout)
        
        # 输入框2
        input2_layout = QHBoxLayout()
        input2_layout.addWidget(QLabel("输入框2:"))
        self.input2 = QLineEdit()
        input2_layout.addWidget(self.input2)
        layout.addLayout(input2_layout)
        
        # 显示标签
        self.display_label = QLabel()
        layout.addWidget(self.display_label)
        
        # 按钮
        self.show_data_btn = QPushButton("显示当前数据")
        layout.addWidget(self.show_data_btn)
        
        # 初始化显示
        self.update_display()
    
    def setup_bindings(self):
        """设置双向绑定"""
        # 模型到视图
        self.model.data_changed.connect(self.input1.setText)
        self.model.data_changed.connect(self.input2.setText)
        self.model.data_changed.connect(self.update_display)
        
        # 视图到模型
        self.input1.textChanged.connect(lambda text: setattr(self.model, 'data', text))
        self.input2.textChanged.connect(lambda text: setattr(self.model, 'data', text))
        
        # 按钮事件
        self.show_data_btn.clicked.connect(self.show_current_data)
        
        # 初始化视图值
        self.input1.setText(self.model.data)
        self.input2.setText(self.model.data)
    
    def update_display(self):
        """更新显示标签"""
        self.display_label.setText(f"当前数据: {self.model.data}")
    
    def show_current_data(self):
        """显示当前数据"""
        QMessageBox.information(
            self, 
            "当前数据", 
            f"模型中的数据是: {self.model.data}\n"
            f"输入框1的内容是: {self.input1.text()}\n"
            f"输入框2的内容是: {self.input2.text()}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BindingDemo()
    window.show()
    sys.exit(app.exec())