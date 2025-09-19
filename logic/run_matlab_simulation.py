import os
import json
import matlab.engine
import numpy as np
from config.paths import get_config_file
import threading
import time

shared_matlab_engine = None
_next_matlab_engine = None
_engine_lock = threading.Lock()

def _start_engine_task():
    """在后台线程中启动MATLAB引擎的任务。"""
    global _next_matlab_engine
    print("正在后台启动备用MATLAB引擎...")
    try:
        engine = matlab.engine.start_matlab()
        with _engine_lock:
            _next_matlab_engine = engine
        print("备用MATLAB引擎已就绪。")
    except Exception as e:
        print(f"启动备用MATLAB引擎失败: {e}")

def start_shared_engine():
    """启动并返回一个共享的MATLAB引擎实例，并预启动下一个引擎"""
    global shared_matlab_engine, _next_matlab_engine
    with _engine_lock:
        if shared_matlab_engine is None:
            print("正在启动初始MATLAB引擎...")
            try:
                # 首次启动是同步的，以确保主引擎立即可用
                shared_matlab_engine = matlab.engine.start_matlab()
                print("初始MATLAB引擎已启动。")
                # 立即开始准备第一个备用引擎
                if _next_matlab_engine is None:
                    threading.Thread(target=_start_engine_task, daemon=True).start()
            except Exception as e:
                print(f"启动MATLAB引擎失败: {e}")
                shared_matlab_engine = None
    return shared_matlab_engine

def restart_shared_engine():
    """
    执行健壮的引擎重启。
    优先尝试热插拔以减少延迟，如果备用引擎未在超时时间内就绪，则执行冷重启。
    """
    global shared_matlab_engine, _next_matlab_engine
    
    print("正在执行MATLAB引擎重启...")
    
    # 1. 尝试在超时时间内等待备用引擎
    timeout = 30  # 等待30秒
    start_time = time.time()
    hot_swap_successful = False
    while time.time() - start_time < timeout:
        with _engine_lock:
            if _next_matlab_engine is not None:
                # --- 执行热插拔 ---
                print("备用引擎已就绪，执行热插拔...")
                old_engine = shared_matlab_engine
                shared_matlab_engine = _next_matlab_engine
                _next_matlab_engine = None
                print("引擎热插拔完成，新引擎已激活。")
                
                # 立即开始准备下一个备用引擎
                threading.Thread(target=_start_engine_task, daemon=True).start()

                # 在后台停止旧引擎
                def stop_old_engine(engine_to_stop):
                    if engine_to_stop:
                        print("正在后台停止旧的MATLAB引擎...")
                        try:
                            engine_to_stop.eval('close all;', nargout=0)
                            engine_to_stop.quit()
                            print("旧的MATLAB引擎已停止。")
                        except Exception as e:
                            print(f"停止旧MATLAB引擎时出错: {e}")
                
                threading.Thread(target=stop_old_engine, args=(old_engine,), daemon=True).start()
                hot_swap_successful = True
                break
        time.sleep(0.5)

    # 2. 如果热插拔失败，则执行冷重启
    if not hot_swap_successful:
        print("警告：热插拔超时，备用引擎未就绪。正在执行冷重启...")
        with _engine_lock:
            # 强制停止所有现有引擎
            if shared_matlab_engine:
                try:
                    shared_matlab_engine.quit()
                    print("已停止当前主引擎。")
                except Exception as e:
                    print(f"停止主引擎时出错（可能已崩溃）: {e}")
                shared_matlab_engine = None
            
            if _next_matlab_engine:
                try:
                    _next_matlab_engine.quit()
                    print("已停止当前备用引擎。")
                except Exception as e:
                    print(f"停止备用引擎时出错: {e}")
                _next_matlab_engine = None

        # 同步启动一个新的主引擎
        print("正在同步启动新的主引擎...")
        start_shared_engine() # 这将创建一个新的主引擎并启动一个新的备用引擎任务
        if shared_matlab_engine:
            print("冷重启完成，新的主引擎已激活。")
        else:
            print("错误：冷重启后未能启动新的主引擎！")

def stop_shared_engine():
    """停止所有共享的MATLAB引擎（活动的和备用的）"""
    global shared_matlab_engine, _next_matlab_engine
    with _engine_lock:
        engines_to_stop = []
        if shared_matlab_engine:
            engines_to_stop.append(shared_matlab_engine)
            shared_matlab_engine = None
        if _next_matlab_engine:
            engines_to_stop.append(_next_matlab_engine)
            _next_matlab_engine = None

    for i, eng in enumerate(engines_to_stop):
        print(f"正在停止引擎 {i+1}/{len(engines_to_stop)}...")
        try:
            eng.quit()
        except Exception as e:
            print(f"停止引擎时出错: {e}")

    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            if 'matlab' in proc.info['name'].lower():
                print(f"Found lingering MATLAB process {proc.info['pid']}. Terminating...")
                proc.kill()
    except ImportError:
        print("Warning: psutil is not installed. Cannot forcefully terminate MATLAB processes.")
    except Exception as e:
        print(f"An error occurred while trying to terminate MATLAB processes: {e}")

def stop_all_matlab_engines():
    stop_shared_engine()

def _execute_with_timeout(eng, script_name, timeout_seconds=600):
    """Executes a MATLAB script with a timeout and handles recovery."""
    future = eng.eval(script_name, nargout=0, background=True)
    try:
        future.result(timeout=timeout_seconds)
        print(f"Script '{script_name}' executed successfully.")
    except matlab.engine.TimeoutError:
        print(f"MATLAB script '{script_name}' timed out. Forcefully restarting engine...")
        future.cancel()
        # Force kill all MATLAB processes and restart
        stop_all_matlab_engines()
        start_shared_engine()
        raise TimeoutError(f"MATLAB script '{script_name}' timed out and engine was forcefully restarted.")
    except Exception as e:
        # Handle other potential exceptions from MATLAB execution
        raise e

def clear_matlab_engine_workspace():
    """Clears variables and closes all figures in the shared MATLAB engine."""
    eng = start_shared_engine()
    if eng:
        try:
            # print("Clearing MATLAB workspace and closing figures...")
            eng.eval('clearvars', nargout=0)
            eng.eval('close all;', nargout=0)
        except Exception as e:
            print(f"Error during MATLAB workspace cleanup: {e}")

def load_config():
    """从config.json加载配置"""
    config_path = get_config_file()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "mesh_params": {"H": 0.00082, "L": 0.00082, "Nx": 20, "Ny": 20},
            "main_params": {
                "k": 0.2, "midu": 1238.0, "Cv": 1450.0, "q": 0.0, "alpha": 25.0,
                "alpha1": 25.0, "T_kongqi": 293.15, "t": 0.0, "dt": 0.1
            }
        }

def run_matlab_script(script_path, output_vars, config, input_vars=None):
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"MATLAB script not found: {script_path}")
    
    eng = start_shared_engine()
    if eng is None:
        raise ConnectionError("MATLAB engine is not available.")

    script_dir = os.path.dirname(script_path)
    try:
        eng.addpath(script_dir, nargout=0)
        # Workspace cleaning is now handled externally by clear_matlab_engine_workspace()
        
        for key, value in config.items():
            eng.workspace[key] = float(value)
        
        if input_vars:
            for var_name, var_value in input_vars.items():
                if isinstance(var_value, np.ndarray):
                    matlab_value = matlab.double(var_value.tolist())
                else:
                    matlab_value = var_value
                eng.workspace[var_name] = matlab_value
        
        script_name = os.path.basename(script_path).replace('.m', '')
        _execute_with_timeout(eng, script_name)
        
        results = {}
        for var in output_vars:
            try:
                results[var] = np.array(eng.workspace[var])
            except Exception as e:
                print(f"Warning: Failed to retrieve variable '{var}': {str(e)}")
                results[var] = None
        # Workspace cleaning is now handled externally by clear_matlab_engine_workspace()
        return results
    except TimeoutError as e:
        # Timeout is already handled in _execute_with_timeout, just propagate the error
        print(f"Timeout error caught in run_matlab_script: {e}")
        raise
    except Exception as e:
        print(f"MATLAB 脚本 {script_path} 运行时发生非超时严重错误: {e}")
        print("错误已捕获，正在尝试重启MATLAB引擎以恢复...")
        restart_shared_engine()
        print("MATLAB引擎重启完成。")
        raise  # 重新引发异常，以便上层调用者知道本次执行失败
    finally:
        # 确保在每次调用后都移除路径，防止MATLAB搜索路径无限增长导致变慢
        if eng and script_dir:
            eng.rmpath(script_dir, nargout=0)
