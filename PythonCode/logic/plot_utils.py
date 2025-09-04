import numpy as np

def get_combined_contour_data(simulation_step_results):
    """
    处理仿真结果以生成组合等高线图的数据。

    Args:
        simulation_step_results (list): 仿真步骤结果字典的列表。

    Returns:
        tuple: 一个元组，包含:
            - plot_data (list): 一个字典列表，每个字典包含用于绘图的 'xxxx', 'yyyy', 'tttt'。
            - all_t_min (float): 所有步骤中的最低温度。
            - all_t_max (float): 所有步骤中的最高温度。
            - boundary_times (list): 用于绘制边界线的累积时间列表。
    """
    all_t_min, all_t_max = float('inf'), float('-inf')
    valid_steps_for_temp_range = []

    # 第一次遍历：从所有有效步骤中找到全局温度范围
    for step_result in simulation_step_results:
        output = step_result.get('output', {})
        if all(k in output for k in ['t', 'JXYV', 'BN2', 'T']):
            try:
                T_data = np.array(output['T'])
                if T_data.size > 0:
                    all_t_min = min(all_t_min, np.nanmin(T_data))
                    all_t_max = max(all_t_max, np.nanmax(T_data))
                    valid_steps_for_temp_range.append(step_result)
            except Exception:
                pass  # 忽略导致错误的步骤

    if not valid_steps_for_temp_range:
        return [], 0, 0, []

    plot_data = []
    boundary_times = []
    cumulative_time = 0

    # 第二次遍历：为每个有效步骤生成绘图坐标
    for step_result in valid_steps_for_temp_range:
        output = step_result['output']
        try:
            t_duration = float(np.array(output['t']).item())
            JXYV, BN2, T_data = np.array(output['JXYV']), np.array(output['BN2']), np.array(output['T'])
            
            BN2_int = BN2.astype(int).flatten() - 1
            valid_indices = BN2_int[(BN2_int >= 0) & (BN2_int < JXYV.shape[0]) & (BN2_int < T_data.shape[0])]
            
            if valid_indices.size > 0:
                num_time_steps = T_data.shape[1]
                time_array = np.linspace(0, t_duration, num_time_steps) + cumulative_time
                y_coords = JXYV[valid_indices, 1]
                
                xxxx, yyyy = np.meshgrid(time_array, y_coords)
                tttt = T_data[valid_indices, :]
                plot_data.append({'xxxx': xxxx, 'yyyy': yyyy, 'tttt': tttt})

            cumulative_time += t_duration
            boundary_times.append(cumulative_time)
        except Exception:
            try:
                t_duration = float(np.array(output['t']).item())
                cumulative_time += t_duration
                boundary_times.append(cumulative_time)
            except Exception:
                pass

    return plot_data, all_t_min, all_t_max, boundary_times
