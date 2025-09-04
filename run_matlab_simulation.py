import os
import json
import matlab.engine
import numpy as np
from config.paths import get_config_file

shared_matlab_engine = None

def start_shared_engine():
    """启动并返回一个共享的MATLAB引擎实例"""
    global shared_matlab_engine
    if shared_matlab_engine is None:
        try:
            print("Starting shared MATLAB engine...")
            shared_matlab_engine = matlab.engine.start_matlab()
        except Exception as e:
            print(f"Failed to start MATLAB engine: {e}")
            shared_matlab_engine = None
    return shared_matlab_engine

def stop_shared_engine():
    """停止共享的MATLAB引擎"""
    global shared_matlab_engine
    if shared_matlab_engine:
        print("Stopping shared MATLAB engine...")
        try:
            shared_matlab_engine.quit()
        except Exception as e:
            print(f"Error stopping shared MATLAB engine: {e}")
        finally:
            shared_matlab_engine = None
    
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

    try:
        script_dir = os.path.dirname(script_path)
        eng.addpath(script_dir, nargout=0)
        eng.clear('all', nargout=0)
        
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
        eng.eval(script_name, nargout=0)
        
        results = {}
        for var in output_vars:
            try:
                results[var] = np.array(eng.workspace[var])
            except Exception as e:
                print(f"Warning: Failed to retrieve variable '{var}': {str(e)}")
                results[var] = None
        return results
    except Exception as e:
        print(f"An error occurred while running MATLAB script {script_path}: {e}")
        raise
