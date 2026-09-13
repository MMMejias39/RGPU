"""Contraparte da suíte em TensorFlow. Uso: suite_tf.py <teste> [cpu|gpu]

Usa `tf.function` cru em vez de `model.fit`, para medir o TF sem o overhead do
laço de treino do Keras — é a comparação mais favorável ao TF.
"""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

teste = sys.argv[1] if len(sys.argv) > 1 else "gemm"
forcar = sys.argv[2] if len(sys.argv) > 2 else "gpu"

import tensorflow as tf
import numpy as np

if forcar == "cpu":
    tf.config.set_visible_devices([], "GPU")
dev = "GPU" if any(d.device_type == "GPU" for d in tf.config.get_visible_devices()) else "CPU"
tf.keras.utils.set_random_seed(7)
CLASSES = 10


def cron(f, reps):
    f()  # aquece o traçado
    t0 = time.perf_counter()
    for _ in range(reps):
        r = f()
    if hasattr(r, "numpy"):
        r.numpy()   # força a sincronização do dispositivo
    return (time.perf_counter() - t0) / reps


def dados(n, d):
    i = np.arange(n, dtype=np.float32)[:, None]
    j = np.arange(d, dtype=np.float32)[None, :]
    return np.sin(i * 0.01 + j * 0.03).astype(np.float32)


def modelo(dims):
    camadas = [tf.keras.layers.Input(shape=(dims[0],))]
    for k, u in enumerate(dims[1:]):
        act = None if k == len(dims) - 2 else "relu"
        camadas.append(tf.keras.layers.Dense(u, activation=act))
    return tf.keras.Sequential(camadas)


def treinador(m, opt):
    @tf.function
    def passo(x, y):
        with tf.GradientTape() as tape:
            perda = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(y, m(x, training=True)))
        g = tape.gradient(perda, m.trainable_variables)
        opt.apply_gradients(zip(g, m.trainable_variables))
        return perda
    return passo


if teste == "gemm":
    casos = [("quadrado-2048", 2048, 2048, 2048), ("quadrado-512", 512, 512, 512),
             ("alto-fino", 8192, 64, 64), ("largo", 64, 8192, 64),
             ("K-dominante", 64, 64, 8192), ("lote-mlp", 1024, 2048, 1024)]
    for nome, m_, n_, k_ in casos:
        a = tf.constant(dados(m_, k_)); b = tf.constant(dados(k_, n_))
        mm = tf.function(lambda a=a, b=b: tf.matmul(a, b))
        t = cron(mm, 20)
        gflop = 2.0 * m_ * n_ * k_ / 1e9
        print("teste=gemm|dev=%s|caso=%s|ms=%.4f|gflops=%.1f" % (dev, nome, t * 1e3, gflop / t))

elif teste == "batch":
    d, h = 512, 1024
    for lote in [1, 8, 32, 128, 512, 2048]:
        m = modelo([d, h, h, CLASSES])
        opt = tf.keras.optimizers.Adam(1e-3)
        x = tf.constant(dados(lote, d))
        y = tf.constant((np.arange(lote) % CLASSES).astype(np.int64))
        passo = treinador(m, opt)
        t = cron(lambda: passo(x, y), 20 if lote < 512 else 5)
        print("teste=batch|dev=%s|lote=%d|ms=%.4f|amostras_s=%.0f" % (dev, lote, t * 1e3, lote / t))

elif teste == "arch":
    lote = 256
    casos = [("raso-largo", [512, 2048, 1024, CLASSES]),
             ("medio", [512, 1024, 1024, 1024, 1024, CLASSES]),
             ("profundo-estreito", [512] + [576] * 12 + [CLASSES])]
    for nome, dims in casos:
        m = modelo(dims)
        opt = tf.keras.optimizers.Adam(1e-3)
        x = tf.constant(dados(lote, dims[0]))
        y = tf.constant((np.arange(lote) % CLASSES).astype(np.int64))
        passo = treinador(m, opt)
        t = cron(lambda: passo(x, y), 20)
        print("teste=arch|dev=%s|caso=%s|camadas=%d|parametros=%d|ms=%.4f"
              % (dev, nome, len(dims) - 1, m.count_params(), t * 1e3))

elif teste == "infer":
    d, h = 512, 1024
    for lote in [1, 32, 512, 4096]:
        m = modelo([d, h, h, CLASSES])
        x = tf.constant(dados(lote, d))
        f = tf.function(lambda x=x: m(x, training=False))
        t = cron(f, 20)
        print("teste=infer|dev=%s|lote=%d|ms=%.4f|amostras_s=%.0f" % (dev, lote, t * 1e3, lote / t))

elif teste == "frio":
    # Já estamos dentro do processo; o custo de import é medido por fora.
    d, h, lote = 512, 1024, 128
    m = modelo([d, h, h, CLASSES])
    opt = tf.keras.optimizers.Adam(1e-3)
    x = tf.constant(dados(lote, d))
    y = tf.constant((np.arange(lote) % CLASSES).astype(np.int64))
    t0 = time.perf_counter()
    treinador(m, opt)(x, y).numpy()
    print("teste=frio|dev=%s|primeiro_passo_ms=%.1f" % (dev, (time.perf_counter() - t0) * 1e3))
