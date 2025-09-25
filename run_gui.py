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
from matplotlib import font_manager
from simulation_gui import SimulationGUI
from PyQt6.QtWidgets import QApplication

def get_available_chinese_fonts():
    """获取系统中可用的中文字体"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 
                     'Arial Unicode MS', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
    available_fonts = []
    
    try:
        # 获取系统中所有字体
        font_list = font_manager.findSystemFonts()
        
        for font_path in font_list:
            try:
                font_prop = font_manager.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                
                # 检查字体名称是否在中文字体列表中
                for chinese_font in chinese_fonts:
                    if chinese_font.lower() in font_name.lower() or font_name.lower() in chinese_font.lower():
                        if chinese_font not in available_fonts:
                            available_fonts.append(chinese_font)
                        break
            except:
                continue
    except Exception as e:
        print(f"检测字体时出错: {e}")
    
    # 如果没有找到中文字体，添加一些备用字体
    if not available_fonts:
        available_fonts = ['DejaVu Sans', 'Arial', 'Helvetica']
        print("警告：未找到中文字体，将使用默认字体")
    
    return available_fonts

def setup_matplotlib_chinese_font():
    """设置matplotlib以支持中文显示"""
    try:
        # 获取可用的中文字体
        available_fonts = get_available_chinese_fonts()
        print(f"可用的中文字体: {available_fonts}")
        
        # A more robust way to set font
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = available_fonts
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        
        # 设置备用字体
        matplotlib.rcParams['font.fantasy'] = available_fonts[:2]  # 使用前两个字体
        matplotlib.rcParams['font.monospace'] = available_fonts[:2]
        
        print(f"成功设置中文字体: {available_fonts[0] if available_fonts else '默认字体'}")
        
        # 强制重建字体缓存
        try:
            font_manager._rebuild()
            print("字体缓存已重建")
        except Exception as e:
            print(f"重建字体缓存失败: {e}")
            
    except Exception as e:
        print(f"设置中文字体失败: {e}")

if __name__ == '__main__':
    setup_matplotlib_chinese_font()
    app = QApplication(sys.argv)
    gui = SimulationGUI()
    gui.show()
    sys.exit(app.exec())
