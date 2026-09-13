import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf, numpy as np
tf32 = sys.argv[1] == "on"
tf.config.experimental.enable_tensor_float_32_execution(tf32)
N, D, H, CLASSES, LOTE, EPOCAS = 8192, 2048, 4096, 10, 1024, 3
i = np.arange(N, dtype=np.float32)[:, None]; j = np.arange(D, dtype=np.float32)[None, :]
x = np.sin(i*0.01 + j*0.03).astype(np.float32); y = (np.arange(N) % CLASSES).astype(np.int64)
m = tf.keras.Sequential([tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(H, activation="relu"), tf.keras.layers.Dense(H, activation="relu"),
    tf.keras.layers.Dense(CLASSES)])
m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
m.fit(x[:LOTE], y[:LOTE], epochs=1, batch_size=LOTE, verbose=0)
t0 = time.perf_counter(); m.fit(x, y, epochs=EPOCAS, batch_size=LOTE, shuffle=False, verbose=0)
dt = time.perf_counter() - t0
macs = D*H + H*H + H*CLASSES
print("tf32=%s  treino_s=%.3f  gflops=%.1f" % (tf32, dt, macs*3*2*N*EPOCAS/1e9/dt))
