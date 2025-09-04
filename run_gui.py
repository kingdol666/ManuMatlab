#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
纵拉预热仿真系统启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行GUI应用程序
from simulation_gui import SimulationGUI
from PyQt6.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = SimulationGUI()
    gui.show()
    sys.exit(app.exec())