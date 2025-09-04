import os
import markdown
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

def create_help_tab():
    """创建帮助选项卡"""
    help_widget = QWidget()
    layout = QVBoxLayout(help_widget)
    
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    
    try:
        # 构建README.md的绝对路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        readme_path = os.path.join(script_dir, "README.md")
        
        with open(readme_path, "r", encoding="utf-8") as f:
            help_text = f.read()
        html_help_text = markdown.markdown(help_text)
        text_edit.setHtml(html_help_text)
    except FileNotFoundError:
        text_edit.setText("帮助文档 (README.md) 未找到。")
    except Exception as e:
        text_edit.setText(f"加载帮助文档时发生错误：{e}")
        
    layout.addWidget(text_edit)
    return help_widget
