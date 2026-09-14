"""Decompõe a partida a fria do TensorFlow em três fases, para separar o que é
custo de importar a biblioteca (Python) do que é custo de arquitetura (GPU).

Uso: frio_decomposto_tf.py
"""
import time, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

t0 = time.perf_counter()
import tensorflow as tf
import numpy as np
t_import = time.perf_counter() - t0

t1 = time.perf_counter()
CLASSES = 10
d, h, lote = 512, 1024, 128
camadas = [tf.keras.layers.Input(shape=(d,)),
           tf.keras.layers.Dense(h, activation="relu"),
           tf.keras.layers.Dense(h, activation="relu"),
           tf.keras.layers.Dense(CLASSES)]
m = tf.keras.Sequential(camadas)
opt = tf.keras.optimizers.Adam(1e-3)
i = np.arange(lote, dtype=np.float32)[:, None]
j = np.arange(d, dtype=np.float32)[None, :]
x = tf.constant(np.sin(i * 0.01 + j * 0.03).astype(np.float32))
y = tf.constant((np.arange(lote) % CLASSES).astype(np.int64))
t_montagem = time.perf_counter() - t1

t2 = time.perf_counter()
@tf.function
def passo(x, y):
    with tf.GradientTape() as tape:
        perda = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(y, m(x, training=True)))
    g = tape.gradient(perda, m.trainable_variables)
    opt.apply_gradients(zip(g, m.trainable_variables))
    return perda
passo(x, y).numpy()
t_primeiro_passo = time.perf_counter() - t2

print("import_ms=%.1f montagem_modelo_ms=%.1f primeiro_passo_ms=%.1f" %
      (t_import * 1e3, t_montagem * 1e3, t_primeiro_passo * 1e3))
