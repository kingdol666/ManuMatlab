from PyQt6.QtWidgets import QStyledItemDelegate, QComboBox
from logic.Models import RollDirection, ScriptType

class ModelDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.script_types = [ScriptType.HEATING, ScriptType.COOLING]
        self.roll_directions = [RollDirection.INITIAL, RollDirection.FORWARD, RollDirection.REVERSE]
        # 为辊温提供一组预设值

    def createEditor(self, parent, option, index):
        # 为脚本类型、滚动方向和辊温创建编辑器
        if index.column() == 1:
            editor = QComboBox(parent)
            editor.addItems(self.script_types)
            return editor
        elif index.column() == 2:
            editor = QComboBox(parent)
            editor.addItems(self.roll_directions)
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        value = index.model().data(index)
        if isinstance(editor, QComboBox):
            editor.setCurrentText(str(value))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            value = editor.currentText()
            model.setData(index, value)
        else:
            super().setModelData(editor, model, index)
