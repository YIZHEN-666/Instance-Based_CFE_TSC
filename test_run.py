# test_run.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tslearn.datasets import UCR_UEA_datasets
import numpy as np

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# 取一个最小数据集
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("ECG200")
# tslearn 返回 (n, t, 1)；简单归一化
X_train = X_train.astype("float32") / (np.max(X_train) + 1e-8)
X_test  = X_test.astype("float32") / (np.max(X_test) + 1e-8)

# 极简 1D-CNN（类似 FCN 的最小骨架）
model = models.Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Conv1D(64, 8, padding="same", activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(len(np.unique(y_train)), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1, validation_data=(X_test, y_test))
print("Done.")
