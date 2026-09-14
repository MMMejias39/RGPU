"""Roda exatamente `passos` passos de treino no TensorFlow, para medir energia
por fora com `rqubit::examples::medir_externo` e isolar o custo fixo de
import/traçado por diferença entre duas contagens de passos — a mesma técnica
já usada em `benchmarks/quantum/README.md`.

Uso: energia_tf.py <lote> <passos>
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

lote = int(sys.argv[1]) if len(sys.argv) > 1 else 512
passos = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
d, h, CLASSES = 512, 1024, 10

i = np.arange(lote, dtype=np.float32)[:, None]
j = np.arange(d, dtype=np.float32)[None, :]
x = tf.constant(np.sin(i * 0.01 + j * 0.03).astype(np.float32))
y = tf.constant((np.arange(lote) % CLASSES).astype(np.int64))

m = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d,)),
    tf.keras.layers.Dense(h, activation="relu"),
    tf.keras.layers.Dense(h, activation="relu"),
    tf.keras.layers.Dense(CLASSES),
])
opt = tf.keras.optimizers.Adam(1e-3)

@tf.function
def passo(x, y):
    with tf.GradientTape() as tape:
        perda = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(y, m(x, training=True)))
    g = tape.gradient(perda, m.trainable_variables)
    opt.apply_gradients(zip(g, m.trainable_variables))
    return perda

for _ in range(10):
    passo(x, y)

t0 = time.perf_counter()
for _ in range(passos):
    r = passo(x, y)
r.numpy()
dt = time.perf_counter() - t0

flop = (d * h + h * h + h * CLASSES) * 3 * 2 * lote
print("motor=tensorflow|lote=%d|passos=%d|ms=%.4f|gflops=%.1f"
      % (lote, passos, dt / passos * 1e3, flop * passos / dt / 1e9))
