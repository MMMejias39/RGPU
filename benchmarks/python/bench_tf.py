"""Mesma tarefa, mesma arquitetura, mesmos dados — no TensorFlow padrão."""
import os, sys, time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DIR = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf
import numpy as np

if THREADS:
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(THREADS)

tf.keras.utils.set_random_seed(7)

def carregar(nome):
    d = np.loadtxt(os.path.join(DIR, nome), delimiter=",", dtype=np.float32)
    return d[:, :2], d[:, 2].astype(np.int64)

x_treino, y_treino = carregar("treino.csv")
x_teste, y_teste = carregar("teste.csv")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation="relu", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dense(32, activation="relu", kernel_initializer="glorot_uniform"),
    tf.keras.layers.Dense(3, activation=None, kernel_initializer="glorot_uniform"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Aquece o traçado do tf.function para não contaminar a medição.
model.fit(x_treino[:32], y_treino[:32], epochs=1, batch_size=32, verbose=0)
model.set_weights([w.numpy() if hasattr(w, "numpy") else w for w in model.get_weights()])
tf.keras.utils.set_random_seed(7)
for camada in model.layers:
    for var in camada.weights:
        if "kernel" in var.name:
            var.assign(camada.kernel_initializer(var.shape, dtype=var.dtype))
        else:
            var.assign(tf.zeros_like(var))
model.optimizer.build(model.trainable_variables)

inicio = time.perf_counter()
model.fit(x_treino, y_treino, epochs=60, batch_size=32, shuffle=True, verbose=0)
decorrido = time.perf_counter() - inicio

_, acc_treino = model.evaluate(x_treino, y_treino, verbose=0)
_, acc_teste = model.evaluate(x_teste, y_teste, verbose=0)

print("motor=tensorflow", tf.__version__)
print("threads=%s" % (THREADS or "padrão(todos)"))
print("parametros=%d" % model.count_params())
print("treino_s=%.4f" % decorrido)
print("acc_treino=%.4f" % acc_treino)
print("acc_teste=%.4f" % acc_teste)
