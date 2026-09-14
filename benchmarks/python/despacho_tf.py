"""Isola o custo de despacho Python/tf.function do custo de computação real,
no lote pequeno onde o overhead fixo pesa mais. Complementa
`frio_decomposto_tf.py`: aquele decompõe a partida a fria, este decompõe o
passo de regime permanente.

Uso: despacho_tf.py <lote>
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

lote = int(sys.argv[1]) if len(sys.argv) > 1 else 1
d, h, CLASSES = 512, 1024, 10
reps = 200


def cron(f, reps):
    f()
    t0 = time.perf_counter()
    for _ in range(reps):
        r = f()
    if hasattr(r, "numpy"):
        r.numpy()
    return (time.perf_counter() - t0) / reps


i = np.arange(lote, dtype=np.float32)[:, None]
j = np.arange(d, dtype=np.float32)[None, :]
x = tf.constant(np.sin(i * 0.01 + j * 0.03).astype(np.float32))
y = tf.constant((np.arange(lote) % CLASSES).astype(np.int64))

# --- passo real: forward + backward + Adam, como suite_tf.py -------------
m = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d,)),
    tf.keras.layers.Dense(h, activation="relu"),
    tf.keras.layers.Dense(h, activation="relu"),
    tf.keras.layers.Dense(CLASSES),
])
opt = tf.keras.optimizers.Adam(1e-3)

@tf.function
def passo_real(x, y):
    with tf.GradientTape() as tape:
        perda = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(y, m(x, training=True)))
    g = tape.gradient(perda, m.trainable_variables)
    opt.apply_gradients(zip(g, m.trainable_variables))
    return perda

t_real = cron(lambda: passo_real(x, y), reps)

# --- passo trivial: mesma assinatura de chamada, quase nenhum cálculo ----
@tf.function
def passo_trivial(x, y):
    return tf.reduce_sum(x) * 0.0 + tf.cast(y[0], tf.float32)

t_trivial = cron(lambda: passo_trivial(x, y), reps)

print("teste=despacho|lote=%d|real_ms=%.4f|trivial_ms=%.4f|despacho_pct=%.1f"
      % (lote, t_real * 1e3, t_trivial * 1e3, 100.0 * t_trivial / t_real))
