"""Carga pesada no TensorFlow. Uso: bench_grande_tf.py [N D H LOTE] (threads via TF_THREADS)"""
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
a = [int(v) for v in sys.argv[1:5]] if len(sys.argv) >= 5 else []
N, D, H, LOTE = (a + [4096, 128, 512, 256])[:4] if a else (4096, 128, 512, 256)
CLASSES, EPOCAS = 10, 3

import tensorflow as tf
import numpy as np

threads = int(os.environ.get("TF_THREADS", "0"))
if threads:
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
if os.environ.get("TF_FORCE_CPU"):
    tf.config.set_visible_devices([], "GPU")
tf.keras.utils.set_random_seed(7)

i = np.arange(N, dtype=np.float32)[:, None]
j = np.arange(D, dtype=np.float32)[None, :]
x = np.sin(i * 0.01 + j * 0.03).astype(np.float32)
y = (np.arange(N) % CLASSES).astype(np.int64)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(H, activation="relu"),
    tf.keras.layers.Dense(H, activation="relu"),
    tf.keras.layers.Dense(CLASSES, activation=None),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x[:LOTE], y[:LOTE], epochs=1, batch_size=LOTE, verbose=0)  # aquece

inicio = time.perf_counter()
h = model.fit(x, y, epochs=EPOCAS, batch_size=LOTE, shuffle=False, verbose=0)
decorrido = time.perf_counter() - inicio

macs = D * H + H * H + H * CLASSES
gflop = macs * 3 * 2 * N * EPOCAS / 1e9
disp = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print("motor=tensorflow %s  dispositivo=%s  N=%d D=%d H=%d lote=%d" % (tf.__version__, disp, N, D, H, LOTE))
print("parametros=%d" % model.count_params())
print("treino_s=%.3f" % decorrido)
print("gflops=%.2f" % (gflop / decorrido))
