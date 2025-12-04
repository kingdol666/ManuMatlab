#!/usr/bin/env python3
"""
检查TensorFlow是否能够使用GPU的脚本
"""

import tensorflow as tf
import sys

def check_tensorflow_gpu():
    """检查TensorFlow GPU支持情况"""
    print("=" * 60)
    print("TensorFlow GPU支持检查")
    print("=" * 60)
    
    # 打印TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")
    
    # 检查可用的GPU设备
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n检测到 {len(gpus)} 个GPU设备:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            
        # 检查GPU是否可用
        try:
            # 尝试设置GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\nGPU内存增长设置成功")
        except RuntimeError as e:
            print(f"\nGPU内存增长设置失败: {e}")
            
        # 检查CUDA和cuDNN版本
        print("\nCUDA和cuDNN信息:")
        print(f"CUDA可用: {tf.test.is_built_with_cuda()}")
        
        # 尝试在GPU上执行简单操作
        print("\n尝试在GPU上执行简单操作...")
        try:
            with tf.device('/GPU:0' if gpus else '/CPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("GPU操作成功!")
                print(f"结果:\n{c.numpy()}")
        except Exception as e:
            print(f"GPU操作失败: {e}")
    else:
        print("\n未检测到GPU设备")
        print("TensorFlow将使用CPU运行")
    
    # 检查其他系统信息
    print("\n其他系统信息:")
    print(f"Python版本: {sys.version}")
    
    # 检查是否有其他GPU库
    try:
        import platform
        print(f"操作系统: {platform.system()} {platform.release()}")
    except:
        pass
    
    try:
        from tensorflow.python.platform import build_info
        print(f"TensorFlow构建信息:")
        print(f"  CUDA版本: {build_info.build_info.get('cuda_version', '未知')}")
        print(f"  cuDNN版本: {build_info.build_info.get('cudnn_version', '未知')}")
    except:
        print("无法获取TensorFlow构建信息")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_tensorflow_gpu()