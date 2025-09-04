import os
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np

from Models import ScriptType, RollDirection
from run_matlab_simulation import run_matlab_script, load_config, start_shared_engine, stop_shared_engine

class InterruptedError(Exception):
    pass

class SimulationThread(QThread):
    log_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)
    simulation_finished = pyqtSignal(list)
    error_occurred = pyqtSignal(dict)
    model_completed = pyqtSignal(dict)
    
    def __init__(self, simulation_models, parent=None):
        super().__init__(parent)
        self.simulation_models = simulation_models
        self.last_T1 = None
        self.simulation_step_results = []
        self._is_running = True

    def run(self):
        try:
            self.log_updated.emit("开始运行仿真...")
            
            matlab_engine = start_shared_engine()
            if not self._is_running or matlab_engine is None:
                raise InterruptedError("Engine start was terminated.")

            config = load_config()
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp_matlab")
            os.makedirs(temp_dir, exist_ok=True)
            
            total_models = len(self.simulation_models)
            for i, model in enumerate(self.simulation_models):
                if not self._is_running:
                    self.log_updated.emit("仿真已被用户终止。")
                    break
                
                self.log_updated.emit(f"正在运行模型 #{i+1}: {model.script_type}, {model.roll_direction}")
                self.progress_updated.emit(i + 1, total_models)
                
                folder = ""
                if model.script_type == ScriptType.HEATING:
                    if model.roll_direction == RollDirection.INITIAL: folder = "升温1"
                    elif model.roll_direction == RollDirection.FORWARD: folder = "升温5"
                    elif model.roll_direction == RollDirection.REVERSE: folder = "升温3"
                else:
                    folder = "冷却2"
                
                mesh_path = f"e:/MatlabSpace/纵拉预热/{folder}/shijian_rechuandao_mesh.m"
                main_path = f"e:/MatlabSpace/纵拉预热/{folder}/shijian_rechuandao_main.m"

                input_vars = {'T_GunWen_Input': model.T_GunWen, 't_up_input': model.t_up}
                if self.last_T1 is not None and i > 0:
                    input_vars['T1'] = self.last_T1
                
                run_matlab_script(mesh_path, [], config['mesh_params'], input_vars)
                main_output = run_matlab_script(main_path, ['T', 'T_ave', 'Time', 'JXYV', 'BN2', 'T1', 't', 'Ns'], config['main_params'], input_vars)
                
                if main_output:
                    self.log_updated.emit(f"模型 #{i+1} 运行成功")
                    if 'T' in main_output and main_output['T'] is not None:
                        T_array = np.array(main_output['T'])
                        self.last_T1 = T_array[:, -1].reshape(-1, 1) if T_array.ndim == 2 and T_array.shape[1] > 0 else T_array.tolist()
                    
                    step_result = {'model_id': i + 1, 'model_type': model.script_type, 'roll_direction': model.roll_direction, 'output': main_output}
                    self.simulation_step_results.append(step_result)
                    self.model_completed.emit(step_result)
                else:
                    self.log_updated.emit(f"模型 #{i+1} 运行失败: 无法获取输出变量")

        except InterruptedError:
             self.log_updated.emit("仿真在启动过程中被终止。")
        except Exception as e:
            error_info = {"type": "GeneralError", "message": f"仿真过程中发生错误: {str(e)}", "solution": "请检查模型配置或程序文件完整性。"}
            self.error_occurred.emit(error_info)
        finally:
            stop_shared_engine()
            self.simulation_finished.emit(self.simulation_step_results)

    def stop(self):
        self._is_running = False
