import sys
from PyQt6.QtCore import Qt, pyqtSignal, QAbstractTableModel, QModelIndex, QObject
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTableView, QPushButton, QMessageBox, QLineEdit, QSpinBox, QLabel, QDialog
)
from typing import List, Any


# --------------------------
# Model层：数据模型（纯数据）
# --------------------------
class User(QObject):
    """用户数据模型（姓名+年龄）"""
    name_changed = pyqtSignal(str)
    age_changed = pyqtSignal(int)
    
    def __init__(self, name: str = "", age: int = 0, parent=None):
        super().__init__(parent)
        self._name = name
        self._age = age
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str):
        if self._name != value:
            self._name = value
            self.name_changed.emit(value)
    
    @property
    def age(self) -> int:
        return self._age
    
    @age.setter
    def age(self, value: int):
        if self._age != value:
            self._age = value
            self.age_changed.emit(value)


class UserList(QObject):
    """用户模型集合"""
    data_changed = pyqtSignal()  # 数据变化信号
    
    def __init__(self, users: List[User] = None, parent=None):
        super().__init__(parent)
        self._users = users if users is not None else []
    
    def append(self, user: User):
        self._users.append(user)
        self.data_changed.emit()
    
    def __getitem__(self, index: int) -> User:
        return self._users[index]
    
    def __delitem__(self, index: int):
        del self._users[index]
        self.data_changed.emit()
    
    def __len__(self) -> int:
        return len(self._users)
    
    def get_all(self) -> List[User]:
        return self._users.copy()


# --------------------------
# TableModel：表格数据模型
# --------------------------
class UserTableModel(QAbstractTableModel):
    """用户表格数据模型"""
    def __init__(self, user_list: UserList, parent=None):
        super().__init__(parent)
        self._user_list = user_list
        self._headers = ["姓名", "年龄"]
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._user_list)
    
    def columnCount(self, parent=QModelIndex()):
        return 2
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        row = index.row()
        col = index.column()
        
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            user = self._user_list[row]
            if col == 0:
                return user.name
            elif col == 1:
                return str(user.age)
        
        return None
    
    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        
        row = index.row()
        col = index.column()
        user = self._user_list[row]
        
        try:
            if col == 0:
                user.name = value
            elif col == 1:
                user.age = int(value)
            
            self.dataChanged.emit(index, index)
            return True
        except ValueError:
            return False
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._headers[section]
        return None
    
    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable


# --------------------------
# Controller层：ModelManager（核心控制器）
# --------------------------
class ModelManager(QObject):
    model_updated = pyqtSignal()  # 模型变化时通知View刷新

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_list = UserList([
            User(name="张三", age=20),
            User(name="李四", age=25),
            User(name="王五", age=30)
        ])
        self.model = User()  # 临时模型（与输入框绑定）
        
        # 连接用户列表数据变化信号
        self.user_list.data_changed.connect(self.model_updated.emit)

    def add_user(self):
        """基于临时模型添加新用户"""
        if not self.model.name.strip():
            QMessageBox.warning(None, "输入错误", "姓名不能为空！")
            return
        
        new_user = User(
            name=self.model.name.strip(),
            age=self.model.age
        )
        self.user_list.append(new_user)
        
        # 重置临时模型（自动清空输入框）
        self.model.name = ""
        self.model.age = 0
        
        self.model_updated.emit()

    def delete_user(self, index: int):
        """删除指定索引用户"""
        if 0 <= index < len(self.user_list):
            del self.user_list[index]
            self.model_updated.emit()
        else:
            QMessageBox.warning(None, "错误", "无效的用户索引")

    def update_user(self, index: int, name: str = None, age: int = None):
        """更新用户属性"""
        if 0 <= index < len(self.user_list):
            user = self.user_list[index]
            if name is not None:
                user.name = name
            if age is not None:
                user.age = age
            self.model_updated.emit()
        else:
            QMessageBox.warning(None, "错误", "无效的用户索引")

    def get_user_count(self) -> int:
        return len(self.user_list)


# --------------------------
# 修改用户数据的对话框
# --------------------------
class UserEditDialog(QDialog):
    def __init__(self, user, parent=None):
        super().__init__(parent)
        self.user = user
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("修改用户信息")
        self.setModal(True)
        self.resize(300, 150)
        
        layout = QVBoxLayout(self)
        
        # 姓名输入
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("姓名："))
        self.name_edit = QLineEdit(self.user.name)
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)
        
        # 年龄输入
        age_layout = QHBoxLayout()
        age_layout.addWidget(QLabel("年龄："))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        self.age_spin.setValue(self.user.age)
        age_layout.addWidget(self.age_spin)
        layout.addLayout(age_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # 连接信号
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    
    def get_user_data(self):
        """获取修改后的用户数据"""
        return self.name_edit.text().strip(), self.age_spin.value()


# --------------------------
# View层：MainWindow（UI展示与交互）
# --------------------------
class MainWindow(QWidget):
    def __init__(self, controller: ModelManager):
        super().__init__()
        self.controller = controller
        self.init_ui()
        self.bind_signals()
        self.setWindowTitle("PyQt6 MVC：使用PyQt原生信号实现双向绑定")
        self.resize(600, 350)

    def init_ui(self):
        """初始化UI组件"""
        main_layout = QVBoxLayout(self)

        # 输入框区域（与临时模型绑定）
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("姓名："))
        self.name_edit = QLineEdit()
        input_layout.addWidget(self.name_edit)

        input_layout.addWidget(QLabel("年龄："))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        input_layout.addWidget(self.age_spin)
        main_layout.addLayout(input_layout)

        # 表格区域
        self.table_view = QTableView()
        self.table_view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)  # 禁用表格直接编辑
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)  # 设置整行选择
        main_layout.addWidget(self.table_view)

        # 按钮区域
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加用户（基于输入框）")
        self.del_btn = QPushButton("删除选中用户")
        self.edit_btn = QPushButton("修改选中用户")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.del_btn)
        btn_layout.addWidget(self.edit_btn)
        main_layout.addLayout(btn_layout)

        # 初始化表格模型
        self.init_table_model()

    def init_table_model(self):
        """绑定表格与用户列表"""
        self.table_model = UserTableModel(self.controller.user_list)
        self.table_view.setModel(self.table_model)

    def bind_signals(self):
        """绑定信号槽：输入框与模型、按钮与控制器"""
        # 1. 输入框与临时模型双向绑定（使用PyQt原生信号）
        # 模型到视图
        self.controller.model.name_changed.connect(self.name_edit.setText)
        self.controller.model.age_changed.connect(self.age_spin.setValue)
        
        # 视图到模型
        self.name_edit.textChanged.connect(lambda text: setattr(self.controller.model, 'name', text))
        self.age_spin.valueChanged.connect(lambda value: setattr(self.controller.model, 'age', value))
        
        # 初始化视图值
        self.name_edit.setText(self.controller.model.name)
        self.age_spin.setValue(self.controller.model.age)

        # 2. 按钮操作转发给控制器
        self.add_btn.clicked.connect(self.controller.add_user)
        self.del_btn.clicked.connect(self.on_delete_clicked)
        self.edit_btn.clicked.connect(self.on_edit_clicked)

        # 3. 接收控制器的刷新通知
        self.controller.model_updated.connect(self.refresh_view)

    def on_delete_clicked(self):
        """删除选中行"""
        selected_indexes = self.table_view.selectedIndexes()
        if selected_indexes:
            row = selected_indexes[0].row()
            self.controller.delete_user(row)
        else:
            QMessageBox.information(self, "提示", "请先选中一行")
    
    def on_edit_clicked(self):
        """修改选中行"""
        selected_indexes = self.table_view.selectedIndexes()
        if selected_indexes:
            row = selected_indexes[0].row()
            user = self.controller.user_list[row]
            
            # 创建并显示修改对话框
            dialog = UserEditDialog(user, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # 获取修改后的数据
                name, age = dialog.get_user_data()
                # 调用控制器的更新方法
                self.controller.update_user(row, name=name, age=age)
        else:
            QMessageBox.information(self, "提示", "请先选中一行")

    def refresh_view(self):
        """刷新视图"""
        self.init_table_model()
        self.setWindowTitle(f"PyQt6 MVC：当前用户数 {self.controller.get_user_count()}")


# --------------------------
# 程序入口
# --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = ModelManager()
    window = MainWindow(controller)
    window.show()
    sys.exit(app.exec())