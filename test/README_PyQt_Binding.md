# PyQt原生双向绑定实现

本项目展示了如何使用PyQt6的原生功能实现数据双向绑定，替代了原有的databind库。

## 文件说明

- `qttest.py` - 原始使用databind库的实现
- `qttest_modified.py` - 使用PyQt原生信号槽机制实现的版本
- `qttest_pyqt_native.py` - 另一个PyQt原生实现方案
- `qttest_advanced_binding.py` - 高级双向绑定实现
- `test_pyqt_binding.py` - 测试脚本

## 实现原理

### 1. 数据模型

使用继承自`QObject`的类作为数据模型，并定义信号：

```python
class User(QObject):
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
```

### 2. 双向绑定

通过信号槽机制实现双向绑定：

```python
# 模型到视图
self.controller.model.name_changed.connect(self.name_edit.setText)

# 视图到模型
self.name_edit.textChanged.connect(lambda text: setattr(self.controller.model, 'name', text))
```

### 3. MVC架构

- **Model**: 数据模型类（User、UserList）
- **View**: UI界面类（MainWindow）
- **Controller**: 业务逻辑类（ModelManager）

## 运行方法

```bash
# 运行GUI应用程序
python qttest_modified.py

# 运行测试脚本
python test_pyqt_binding.py
```

## 优势

1. **无需额外依赖**：只使用PyQt6原生功能
2. **完全兼容**：与PyQt6生态系统无缝集成
3. **灵活可扩展**：可以根据需要自定义绑定逻辑
4. **性能优异**：直接使用PyQt信号槽机制，性能更好

## 注意事项

1. 所有需要使用信号的类必须继承自`QObject`
2. 信号必须在类级别定义，不能在实例方法中定义
3. 使用lambda函数连接信号时要注意闭包问题