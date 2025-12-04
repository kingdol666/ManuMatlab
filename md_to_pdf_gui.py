import os
import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QFileDialog, 
                            QProgressBar, QTextEdit, QComboBox, QMessageBox,
                            QGroupBox, QGridLayout, QCheckBox, QSpinBox, QStyleFactory)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon

class MarkdownConverter(QThread):
    """Markdown转PDF的后台线程"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    conversion_completed = pyqtSignal(bool, str)
    
    def __init__(self, md_path, pdf_path, method):
        super().__init__()
        self.md_path = md_path
        self.pdf_path = pdf_path
        self.method = method
        
    def run(self):
        try:
            self.status_updated.emit("正在读取Markdown文件...")
            self.progress_updated.emit(10)
            
            # 读取Markdown文件
            if not os.path.exists(self.md_path):
                self.conversion_completed.emit(False, "找不到Markdown文件")
                return
                
            with open(self.md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            self.status_updated.emit("正在转换为HTML...")
            self.progress_updated.emit(30)
            
            # 转换为HTML
            import markdown
            html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
            
            # 添加CSS样式
            css_style = """
            body {
                font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
                line-height: 1.6;
                margin: 2cm;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #333;
                margin-top: 1.5em;
                margin-bottom: 0.8em;
            }
            h1 {
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            h2 {
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            }
            code {
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: Consolas, Monaco, monospace;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            blockquote {
                border-left: 4px solid #ddd;
                padding-left: 16px;
                color: #666;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 1em;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            """
            
            # 创建完整的HTML文档
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Markdown转PDF</title>
                <style>
                    {css_style}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            self.status_updated.emit("正在生成PDF...")
            self.progress_updated.emit(60)
            
            # 根据选择的方法进行转换
            if self.method == "pdfkit":
                self.convert_with_pdfkit(full_html)
            elif self.method == "weasyprint":
                self.convert_with_weasyprint(full_html)
            elif self.method == "md2pdf":
                self.convert_with_md2pdf()
            elif self.method == "reportlab":
                self.convert_with_reportlab()
            elif self.method == "html_only":
                self.save_html_only(full_html)
            else:
                self.conversion_completed.emit(False, "未知的转换方法")
                
        except Exception as e:
            self.conversion_completed.emit(False, f"转换失败: {str(e)}")
    
    def convert_with_pdfkit(self, html_content):
        try:
            import pdfkit
            
            options = {
                'page-size': 'A4',
                'margin-top': '2cm',
                'margin-right': '2cm',
                'margin-bottom': '2cm',
                'margin-left': '2cm',
                'encoding': "UTF-8",
                'no-outline': None
            }
            
            self.status_updated.emit("正在使用pdfkit生成PDF...")
            self.progress_updated.emit(80)
            
            pdfkit.from_string(html_content, self.pdf_path, options=options)
            self.progress_updated.emit(100)
            self.conversion_completed.emit(True, f"PDF已成功生成: {self.pdf_path}")
            
        except Exception as e:
            self.conversion_completed.emit(False, f"pdfkit转换失败: {str(e)}")
    
    def convert_with_weasyprint(self, html_content):
        try:
            from weasyprint import HTML
            
            self.status_updated.emit("正在使用WeasyPrint生成PDF...")
            self.progress_updated.emit(80)
            
            HTML(string=html_content).write_pdf(self.pdf_path)
            self.progress_updated.emit(100)
            self.conversion_completed.emit(True, f"PDF已成功生成: {self.pdf_path}")
            
        except Exception as e:
            self.conversion_completed.emit(False, f"WeasyPrint转换失败: {str(e)}")
    
    def convert_with_md2pdf(self):
        try:
            import md2pdf
            
            self.status_updated.emit("正在使用md2pdf直接生成PDF...")
            self.progress_updated.emit(80)
            
            # 使用md2pdf直接转换
            md2pdf.convert(
                md_file_path=self.md_path,
                pdf_file_path=self.pdf_path,
                css=None,  # 使用默认CSS
                base_url=None
            )
            
            self.progress_updated.emit(100)
            self.conversion_completed.emit(True, f"PDF已成功生成: {self.pdf_path}")
            
        except ImportError:
            self.conversion_completed.emit(False, "md2pdf库未安装，请先安装: pip install md2pdf")
        except Exception as e:
            self.conversion_completed.emit(False, f"md2pdf转换失败: {str(e)}")
    
    def convert_with_reportlab(self):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.enums import TA_JUSTIFY
            import markdown
            
            self.status_updated.emit("正在使用ReportLab生成PDF...")
            self.progress_updated.emit(80)
            
            # 尝试注册中文字体
            try:
                # 尝试使用系统中文字体
                pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))
                font_name = 'SimSun'
            except:
                try:
                    pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))
                    font_name = 'SimHei'
                except:
                    font_name = 'Helvetica'  # 默认字体
            
            # 创建PDF文档
            doc = SimpleDocTemplate(self.pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 创建支持中文的样式
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=18,
                spaceAfter=30,
                alignment=1  # 居中
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontName=font_name,
                fontSize=14,
                spaceAfter=12
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=10,
                spaceAfter=12,
                alignment=TA_JUSTIFY
            )
            
            # 读取Markdown文件
            with open(self.md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # 转换Markdown为HTML
            html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
            
            # 简单处理HTML标签，提取文本内容
            import re
            
            # 按行处理内容
            lines = html_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.2 * inch))
                    continue
                
                # 处理标题
                if line.startswith('<h1>'):
                    title = re.sub(r'<[^>]+>', '', line)
                    story.append(Paragraph(title, title_style))
                elif line.startswith('<h2>'):
                    heading = re.sub(r'<[^>]+>', '', line)
                    story.append(Paragraph(heading, heading_style))
                elif line.startswith('<h3>') or line.startswith('<h4>') or line.startswith('<h5>') or line.startswith('<h6>'):
                    heading = re.sub(r'<[^>]+>', '', line)
                    story.append(Paragraph(heading, heading_style))
                else:
                    # 处理普通段落
                    text = re.sub(r'<[^>]+>', '', line)
                    if text:
                        story.append(Paragraph(text, body_style))
            
            # 生成PDF
            doc.build(story)
            
            self.progress_updated.emit(100)
            self.conversion_completed.emit(True, f"PDF已成功生成: {self.pdf_path}")
            
        except ImportError:
            self.conversion_completed.emit(False, "ReportLab库未安装，请先安装: pip install reportlab")
        except Exception as e:
            self.conversion_completed.emit(False, f"ReportLab转换失败: {str(e)}")

    def save_html_only(self, html_content):
        try:
            html_path = self.pdf_path.replace('.pdf', '.html')
            
            self.status_updated.emit("正在保存HTML文件...")
            self.progress_updated.emit(80)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.progress_updated.emit(100)
            self.conversion_completed.emit(True, f"HTML文件已生成: {html_path}\n您可以在浏览器中打开此文件，然后使用打印功能保存为PDF。")
            
        except Exception as e:
            self.conversion_completed.emit(False, f"保存HTML文件失败: {str(e)}")


class MarkdownToPDFConverter(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.md_path = ""
        self.pdf_path = ""
        self.converter_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Markdown转PDF转换器")
        self.setGeometry(100, 100, 800, 600)
        
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QGridLayout()
        
        # Markdown文件选择
        self.md_path_label = QLabel("未选择Markdown文件")
        self.md_path_label.setWordWrap(True)
        md_button = QPushButton("选择Markdown文件")
        md_button.clicked.connect(self.select_md_file)
        
        file_layout.addWidget(QLabel("Markdown文件:"), 0, 0)
        file_layout.addWidget(self.md_path_label, 0, 1)
        file_layout.addWidget(md_button, 0, 2)
        
        # PDF输出路径选择
        self.pdf_path_label = QLabel("未设置输出路径")
        self.pdf_path_label.setWordWrap(True)
        pdf_button = QPushButton("设置输出路径")
        pdf_button.clicked.connect(self.select_pdf_path)
        
        file_layout.addWidget(QLabel("输出路径:"), 1, 0)
        file_layout.addWidget(self.pdf_path_label, 1, 1)
        file_layout.addWidget(pdf_button, 1, 2)
        
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # 转换选项区域
        options_group = QGroupBox("转换选项")
        options_layout = QGridLayout()
        
        # 转换方法选择
        options_layout.addWidget(QLabel("转换方法:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["reportlab", "md2pdf", "pdfkit", "weasyprint", "html_only"])
        self.method_combo.setCurrentIndex(0)  # 默认选择reportlab
        self.method_combo.currentIndexChanged.connect(self.update_method_description)
        options_layout.addWidget(self.method_combo, 0, 1)
        
        # 方法描述
        self.method_description = QLabel("生成HTML文件，可在浏览器中打开并打印为PDF")
        self.method_description.setWordWrap(True)
        self.method_description.setStyleSheet("color: #666; font-style: italic;")
        options_layout.addWidget(self.method_description, 1, 0, 1, 2)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # 转换按钮
        self.convert_button = QPushButton("开始转换")
        self.convert_button.clicked.connect(self.start_conversion)
        self.convert_button.setEnabled(False)
        main_layout.addWidget(self.convert_button)
        
        # 进度条
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        
        # 状态显示
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
        
        # 日志显示
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        main_layout.addWidget(self.log_text)
        
        # 初始化方法描述
        self.update_method_description()
    
    def select_md_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Markdown文件", "", "Markdown文件 (*.md);;所有文件 (*)"
        )
        if file_path:
            self.md_path = file_path
            self.md_path_label.setText(file_path)
            
            # 自动设置输出路径
            if not self.pdf_path:
                self.pdf_path = os.path.splitext(file_path)[0] + ".pdf"
                self.pdf_path_label.setText(self.pdf_path)
            
            self.check_conversion_ready()
    
    def select_pdf_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "设置输出路径", self.pdf_path or "output.pdf", "PDF文件 (*.pdf);;所有文件 (*)"
        )
        if file_path:
            self.pdf_path = file_path
            self.pdf_path_label.setText(file_path)
            self.check_conversion_ready()
    
    def check_conversion_ready(self):
        self.convert_button.setEnabled(bool(self.md_path and self.pdf_path))
    
    def update_method_description(self):
        method = self.method_combo.currentText()
        descriptions = {
            "reportlab": "使用ReportLab直接生成PDF（推荐，支持中文，无需额外依赖）",
            "md2pdf": "使用md2pdf直接生成PDF（需要额外系统依赖）",
            "pdfkit": "使用pdfkit直接生成PDF（需要安装wkhtmltopdf）",
            "weasyprint": "使用WeasyPrint直接生成PDF（需要安装WeasyPrint）",
            "html_only": "生成HTML文件，可在浏览器中打开并打印为PDF（无需额外依赖）"
        }
        self.method_description.setText(descriptions.get(method, ""))
    
    def start_conversion(self):
        if not self.md_path or not self.pdf_path:
            QMessageBox.warning(self, "警告", "请选择Markdown文件和设置输出路径")
            return
        
        # 禁用转换按钮
        self.convert_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # 创建转换线程
        method = self.method_combo.currentText()
        self.converter_thread = MarkdownConverter(self.md_path, self.pdf_path, method)
        self.converter_thread.progress_updated.connect(self.progress_bar.setValue)
        self.converter_thread.status_updated.connect(self.status_label.setText)
        self.converter_thread.conversion_completed.connect(self.conversion_finished)
        
        # 启动线程
        self.converter_thread.start()
    
    def conversion_finished(self, success, message):
        # 重新启用转换按钮
        self.convert_button.setEnabled(True)
        
        # 显示结果
        self.log_text.append(message)
        
        if success:
            QMessageBox.information(self, "成功", message)
            # 询问是否打开文件
            if self.method_combo.currentText() == "html_only":
                html_path = self.pdf_path.replace('.pdf', '.html')
                reply = QMessageBox.question(
                    self, "打开文件", "是否在浏览器中打开生成的HTML文件？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    os.startfile(html_path)
            else:
                reply = QMessageBox.question(
                    self, "打开文件", "是否打开生成的PDF文件？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    try:
                        os.startfile(self.pdf_path)
                    except Exception as e:
                        QMessageBox.warning(self, "警告", f"无法打开PDF文件: {str(e)}\n您可以手动打开文件: {self.pdf_path}")
        else:
            QMessageBox.critical(self, "错误", message)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用程序图标和字体
    if hasattr(QStyleFactory, 'keys'):
        available_styles = QStyleFactory.keys()
        if 'WindowsVista' in available_styles:
            app.setStyle('WindowsVista')
    
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = MarkdownToPDFConverter()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()