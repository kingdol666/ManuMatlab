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
import matplotlib
import matplotlib.pyplot as plt
from simulation_gui import SimulationGUI
from PyQt6.QtWidgets import QApplication

def setup_matplotlib_chinese_font():
    """设置matplotlib以支持中文显示"""
    try:
        # A more robust way to set font
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        
        # Force rebuild font cache
        # from matplotlib.font_manager import _rebuild
        # _rebuild()
    except Exception as e:
        print(f"设置中文字体失败: {e}")

if __name__ == '__main__':
    setup_matplotlib_chinese_font()
    app = QApplication(sys.argv)
    gui = SimulationGUI()
    gui.show()
    sys.exit(app.exec())
