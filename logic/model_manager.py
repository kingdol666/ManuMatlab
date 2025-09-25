import json
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt6.QtCore import Qt
from .Models import ScriptType, RollDirection, SimulationModel

class ModelManager:
    def __init__(self, gui):
        self.gui = gui
        self.simulation_models = []
        self.last_heating_t_gunwen = 350.0
        self.last_heating_roll_direction = RollDirection.FORWARD

    def add_model(self):
        """添加新的仿真模型"""
        is_first_model = len(self.simulation_models) == 0
        
        if is_first_model:
            script_type = ScriptType.HEATING
            roll_direction = RollDirection.INITIAL
            t_gunwen = self.gui.t_gunwen_input.value()
            t_up = self.gui.t_up_input.value()
            self.gui.script_type_combo.setCurrentText(script_type)
            self.gui.roll_direction_combo.setCurrentText(roll_direction)
            self.gui.update_log("已添加首个模型，自动设置为升温初始辊模型")
        else:
            script_type = self.gui.script_type_combo.currentText()
            t_up = self.gui.t_up_input.value()
            
            if script_type == ScriptType.COOLING:
                roll_direction = RollDirection.INITIAL
                if self.last_heating_t_gunwen is not None:
                    t_gunwen = self.last_heating_t_gunwen
                    self.gui.update_log(f"降温模型使用上一次升温的辊温: {t_gunwen:.4f} K")
                else:
                    QMessageBox.warning(self.gui, "警告", "必须先添加一个升温模型，才能添加降温模型。")
                    return
            else:
                roll_direction = self.gui.roll_direction_combo.currentText()
                t_gunwen = self.gui.t_gunwen_input.value()
                self.last_heating_roll_direction = roll_direction
                self.last_heating_t_gunwen = t_gunwen
        
        model = SimulationModel(script_type, roll_direction, t_gunwen, t_up)
        self.simulation_models.append(model)
        self._update_model_table()
        
        model_id = len(self.simulation_models)
        roll_direction_display = roll_direction if script_type == ScriptType.HEATING else "/"
        self.gui.update_log(f"已添加模型 #{model_id}: {script_type}, {roll_direction_display}")
        self.gui.visualization_manager.draw_model_schematic()

    def delete_model(self):
        """删除选中的仿真模型"""
        if self.gui.visualization_manager.model.visualization_active:
            self.gui.visualization_manager.stop_visualization()
            
        selected_ranges = self.gui.model_table.selectedRanges()
        if not selected_ranges:
            QMessageBox.warning(self.gui, "警告", "请先选择要删除的模型")
            return

        rows_to_delete = set()
        for r in selected_ranges:
            for i in range(r.topRow(), r.bottomRow() + 1):
                rows_to_delete.add(i)

        if not rows_to_delete:
            return

        # 从后往前删除，避免索引错误
        for row in sorted(list(rows_to_delete), reverse=True):
            model_id_to_delete = int(self.gui.model_table.item(row, 0).text())
            self.simulation_models.pop(model_id_to_delete - 1)
            self.gui.update_log(f"已删除模型 #{model_id_to_delete}")

        self._update_model_table()
        self.gui.visualization_manager.draw_model_schematic()
    
    def delete_all_models(self):
        """删除所有仿真模型"""
        if self.gui.visualization_manager.model.visualization_active:
            self.gui.visualization_manager.stop_visualization()

        if not self.simulation_models:
            QMessageBox.information(self.gui, "提示", "当前没有模型可删除。")
            return
        reply = QMessageBox.question(
            self.gui, "确认", "确定要删除所有模型吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.simulation_models.clear()
        self.gui.model_table.setRowCount(0)
        self.gui.update_log("已删除所有模型")
        self.gui.visualization_manager.draw_model_schematic()

    def save_model_sequence(self):
        """将当前模型序列保存为 JSON 文件"""
        if not self.simulation_models:
            QMessageBox.information(self.gui, "提示", "当前没有模型可保存。")
            return

        file_path, _ = QFileDialog.getSaveFileName(self.gui, "保存模型序列", "models.json", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            data = [self._model_to_dict(m) for m in self.simulation_models]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self.gui, "成功", f"模型序列已保存到 {file_path}")
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"保存失败: {e}")

    def load_model_sequence(self):
        """从 JSON 文件加载模型序列"""
        file_path, _ = QFileDialog.getOpenFileName(self.gui, "加载模型序列", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
            
            if not isinstance(data_list, list):
                raise ValueError("JSON 文件内容必须是一个列表。")

            new_models = []
            for i, item in enumerate(data_list):
                if not isinstance(item, dict):
                    raise ValueError(f"模型 #{i+1} 的格式不正确，应为字典。")
                model = self._dict_to_model(item)
                new_models.append(model)

            self.simulation_models = new_models
            self._update_model_table()
            QMessageBox.information(self.gui, "成功", f"成功加载 {len(self.simulation_models)} 个模型")
            self.gui.visualization_manager.draw_model_schematic()

        except json.JSONDecodeError:
            QMessageBox.critical(self.gui, "错误", "加载失败：JSON 文件格式错误。")
        except ValueError as e:
            QMessageBox.critical(self.gui, "错误", f"加载失败：{e}")
        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"加载时发生未知错误: {e}")

    def on_model_table_cell_changed(self, row, column):
        """处理模型表格单元格的编辑事件"""
        self.gui.model_table.blockSignals(True)

        model_id = int(self.gui.model_table.item(row, 0).text())
        model_index = model_id - 1
        model = self.simulation_models[model_index]
        item = self.gui.model_table.item(row, column)
        new_value = item.text()

        try:
            if column == 1:
                if new_value not in [ScriptType.HEATING, ScriptType.COOLING]:
                    raise ValueError("无效的脚本类型")
                model.script_type = new_value
            elif column == 2:
                if new_value not in [RollDirection.INITIAL, RollDirection.FORWARD, RollDirection.REVERSE, "/"]:
                    raise ValueError("无效的滚动方向")
                model.roll_direction = new_value if new_value != "/" else RollDirection.INITIAL
            elif column == 3:
                model.T_GunWen = float(new_value) if new_value != "/" else 315.1500
            elif column == 4:
                model.t_up = float(new_value)
            
            self.gui.update_log(f"模型 #{model_id} 的参数已更新")
            self._update_model_table()

        except ValueError as e:
            QMessageBox.warning(self.gui, "输入无效", f"输入的值 '{new_value}' 无效: {e}")
            self._update_model_table()
        except TypeError:
            QMessageBox.warning(self.gui, "输入无效", f"输入的值 '{new_value}' 必须是数字。")
            self._update_model_table()
        
        self.gui.model_table.blockSignals(False)

    def on_script_type_changed(self, script_type):
        """当脚本类型改变时，启用或禁用相关参数。"""
        is_cooling = (script_type == ScriptType.COOLING)
        
        if self.gui.roll_direction_combo.isEnabled() and is_cooling:
            self.last_heating_roll_direction = self.gui.roll_direction_combo.currentText()
            self.last_heating_t_gunwen = self.gui.t_gunwen_input.value()

        self.gui.roll_direction_combo.setEnabled(not is_cooling)
        self.gui.t_gunwen_input.setEnabled(not is_cooling)
        
        if is_cooling:
            self.gui.roll_direction_combo.setCurrentIndex(-1)
            self.gui.t_gunwen_input.clear()
        else:
            self.gui.roll_direction_combo.setCurrentText(self.last_heating_roll_direction)
            self.gui.t_gunwen_input.setValue(self.last_heating_t_gunwen)

    def _update_model_table(self):
        """用当前模型列表刷新模型表格"""
        self.gui.model_table.setRowCount(0)
        for idx, model in enumerate(self.simulation_models, start=1):
            row_pos = self.gui.model_table.rowCount()
            self.gui.model_table.insertRow(row_pos)
            
            id_item = QTableWidgetItem(str(idx))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.gui.model_table.setItem(row_pos, 0, id_item)
            
            self.gui.model_table.setItem(row_pos, 1, QTableWidgetItem(model.script_type))
            
            roll_direction_display = model.roll_direction if model.script_type == ScriptType.HEATING else "/"
            t_gunwen_display = f"{model.T_GunWen:.4f}" if model.script_type == ScriptType.HEATING else "/"
            
            self.gui.model_table.setItem(row_pos, 2, QTableWidgetItem(roll_direction_display))
            self.gui.model_table.setItem(row_pos, 3, QTableWidgetItem(t_gunwen_display))
            self.gui.model_table.setItem(row_pos, 4, QTableWidgetItem(f"{model.t_up:.4f}"))

    def _model_to_dict(self, model: SimulationModel) -> dict:
        """将 SimulationModel 转为可序列化字典"""
        return {
            "script_type": model.script_type,
            "roll_direction": model.roll_direction,
            "T_GunWen": model.T_GunWen,
            "t_up": model.t_up
        }

    def _dict_to_model(self, data: dict) -> SimulationModel:
        """由字典恢复 SimulationModel"""
        return SimulationModel(
            data.get("script_type", ScriptType.HEATING),
            data.get("roll_direction", RollDirection.INITIAL),
            float(data.get("T_GunWen", 315.15)),
            float(data.get("t_up", 1.38))
        )
