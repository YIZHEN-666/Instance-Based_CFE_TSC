import tensorflow as tf

print("="*40)
print("TensorFlow 版本:", tf.__version__)
print("="*40)

devices = tf.config.list_physical_devices()
print("所有设备：")
for dev in devices:
    print(" -", dev)

gpus = tf.config.list_physical_devices('GPU')
print("\nGPU 检测结果：")
if gpus:
    for gpu in gpus:
        print(" -", gpu)
else:
    print("未检测到 GPU")
print("="*40)
