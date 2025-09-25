# Models.py - 存储MATLAB仿真过程中的可变变量参数和实体参数

# 实体参数
class ScriptType:
    HEATING = "升温"  # 升温脚本类型
    COOLING = "降温"  # 降温脚本类型

class RollDirection:
    INITIAL = "初始辊"  # 初始滚动方向
    FORWARD = "正向辊"  # 正向滚动方向
    REVERSE = "逆向辊"  # 逆向滚动方向

# 仿真模型类 - 整合实体参数和可变参数
class SimulationModel:
    def __init__(self, script_type=ScriptType.HEATING, roll_direction=RollDirection.INITIAL, 
                 T_GunWen=315.15, t_up=1.38):
        """
        初始化仿真模型
        
        参数:
            script_type (str): 脚本类型，默认为升温
            roll_direction (str): 滚动方向，默认为初始滚
            T_GunWen (float): 辊温，默认值为315.15
            t_up (float): 时间上限，默认值为1.38
        """
        # 实体参数
        self.script_type = script_type
        self.roll_direction = roll_direction
        
        # 可变参数
        self.T_GunWen = T_GunWen  # 辊温
        self.t_up = t_up          # 时间上限
    
    # 实体参数方法
    def set_script_type(self, script_type):
        """
        设置脚本类型
        
        参数:
            script_type (str): 脚本类型（升温/降温）
        """
        self.script_type = script_type
        print(f"脚本类型已设置为: {self.script_type}")
    
    def set_roll_direction(self, roll_direction):
        """
        设置滚动方向
        
        参数:
            roll_direction (str): 滚动方向（初始滚/正向滚/逆向滚）
        """
        self.roll_direction = roll_direction
        print(f"滚动方向已设置为: {self.roll_direction}")
    
    # 可变参数方法
    def update_T_GunWen(self, new_value):
        """
        更新辊温值
        
        参数:
            new_value (float): 新的辊温值
        """
        self.T_GunWen = new_value
        print(f"辊温已更新为: {self.T_GunWen}")
    
    def update_t_up(self, new_value):
        """
        更新时间上限
        
        参数:
            new_value (float): 新的时间上限
        """
        self.t_up = new_value
        print(f"时间上限已更新为: {self.t_up}")
    
    def get_config_info(self):
        """
        获取完整的配置信息，包括物理参数
        
        返回:
            dict: 包含所有配置信息的字典
        """
        from Config import K, MIDU, CV, Q, ALPHA, T_KONGQI, DT
        return {
            "脚本类型": self.script_type,
            "滚动方向": self.roll_direction,
            "导热系数": K,
            "密度": MIDU,
            "比热容": CV,
            "热流密度": Q,
            "对流换热系数": ALPHA,
            "空气温度": T_KONGQI,
            "时间步长": DT,
            "辊温": self.T_GunWen,
            "时间上限": self.t_up
        }
    
    def get_variable_parameters(self):
        """
        获取当前所有可变参数
        
        返回:
            dict: 包含当前所有可变参数的字典
        """
        return {
            "辊温": self.T_GunWen,
            "时间上限": self.t_up
        }
    
    def get_entity_parameters(self):
        """
        获取当前所有实体参数
        
        返回:
            dict: 包含当前所有实体参数的字典
        """
        return {
            "脚本类型": self.script_type,
            "滚动方向": self.roll_direction
        }

# 创建默认仿真模型实例
simulation_model = SimulationModel()

# 为了向后兼容，保留旧的变量名
variable_params = simulation_model
simulation_config = simulation_model