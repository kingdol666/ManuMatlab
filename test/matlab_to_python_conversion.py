import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 设置中文字体支持
try:
    # 尝试使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告：无法设置中文字体，可能无法正确显示中文标签")

def plot_contour_data(t, Ns, JXYV, BN2, T):
    """
    将MATLAB脚本转换为Python函数，用于绘制等高线图
    
    参数:
    t -- 时间参数
    Ns -- 第一个维度的大小
    JXYV -- 二维数组，包含位置信息
    BN2 -- 索引数组，用于选择特定数据点
    T -- 温度数据数组
    
    返回:
    fig -- matplotlib图形对象
    """
    # 计算TT，相当于MATLAB中的TT=round(t*10)
    TT = int(round(t * 10))
    
    # 确保Ns是整数
    Ns_int = int(Ns)
    
    # 创建JXYV2数组，相当于MATLAB中的JXYV2=zeros(Ns,2*TT)
    JXYV2 = np.zeros((Ns_int, 2 * TT))
    
    # 填充JXYV2数组
    for i in range(0, 2 * TT, 2):  # i从0开始，步长为2，相当于MATLAB的i=1:2:2*TT
        JXYV2[:, i] = JXYV[:, 0] + (i) / 2 - 0.5  # 注意Python索引从0开始，所以(i-1)变为i
        if i+1 < 2 * TT:  # 确保不越界
            JXYV2[:, i+1] = JXYV[:, 1]
    
    # 计算位置变量
    weizi = 2 * TT
    
    # 确保BN2是整数数组，用于索引
    BN2_int = BN2.astype(int)
    
    # 提取xxxx和yyyy数据，相当于MATLAB中的xxxx和yyyy
    xxxx = JXYV2[BN2_int, :weizi:2] * 0.1  # 选择奇数列，步长为2
    yyyy = JXYV2[BN2_int, 1:weizi+1:2]  # 选择偶数列，步长为2
    
    # 确保xxxx和yyyy是2维数组
    if xxxx.ndim == 1:
        xxxx = xxxx.reshape(-1, 1)
    elif xxxx.ndim > 2:
        # 如果是3维或更高维，降维到2D
        xxxx = xxxx.reshape(xxxx.shape[0], -1)
    
    if yyyy.ndim == 1:
        yyyy = yyyy.reshape(-1, 1)
    elif yyyy.ndim > 2:
        # 如果是3维或更高维，降维到2D
        yyyy = yyyy.reshape(yyyy.shape[0], -1)
    
    # 提取温度数据
    tttt = T[BN2_int, :(weizi+1)//2]  # 注意Python中整数除法使用//
    
    # 确保tttt是2维数组
    if tttt.ndim > 2:
        # 如果是3维或更高维，降维到2D
        tttt = tttt.reshape(tttt.shape[0], -1)
    elif tttt.ndim < 2:
        # 如果是1维或0维，重塑为2维
        tttt = tttt.reshape(-1, 1) if tttt.size > 0 else np.array([[]])
    
    # 检查tttt的形状，确保满足contourf的要求
    if tttt.shape[1] < 2:
        # 如果只有一列，复制该列以创建至少两列
        tttt = np.hstack([tttt, tttt])
    
    # 检查xxxx和yyyy的形状，确保与tttt兼容
    if xxxx.shape[1] < 2:
        # 如果只有一列，复制该列以创建至少两列
        xxxx = np.hstack([xxxx, xxxx])
    if yyyy.shape[1] < 2:
        # 如果只有一列，复制该列以创建至少两列
        yyyy = np.hstack([yyyy, yyyy])
    
    # 创建等高线图
    fig, ax = plt.subplots()
    contour = ax.contourf(xxxx, yyyy, tttt, 20, linestyles='none')  # 'linestyle','none' 去掉轮廓线
    fig.colorbar(contour, ax=ax)
    
    # 添加标签
    ax.set_xlabel('Time/s')
    ax.set_ylabel('PBAT Thickness (m)')
    
    # 设置标题
    ax.set_title('Temperature Distribution')
    
    return fig

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    t = 1.0  # 示例时间值
    Ns = 10  # 示例Ns值
    JXYV = np.random.rand(Ns, 2)  # 随机生成JXYV数据
    BN2 = np.array([0, 2, 4, 6, 8])  # 示例索引数组
    T = np.random.rand(Ns, 10)  # 随机生成温度数据
    
    # 调用函数并显示图形
    fig = plot_contour_data(t, Ns, JXYV, BN2, T)
    plt.show()