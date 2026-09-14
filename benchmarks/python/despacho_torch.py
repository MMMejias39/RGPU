"""Equivalente a `despacho_tf.py`, para PyTorch: isola o custo de despacho
Python/eager do custo de computação real, no mesmo lote.

Uso: despacho_torch.py <lote>
"""
import sys, time
import torch
import torch.nn as nn

lote = int(sys.argv[1]) if len(sys.argv) > 1 else 1
d, h, CLASSES = 512, 1024, 10
reps = 200
dev = torch.device("cuda")


def cron(f, reps):
    f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


x = torch.randn(lote, d, device=dev)
y = (torch.arange(lote, device=dev) % CLASSES)

m = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, CLASSES)).to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-3)
xent = nn.CrossEntropyLoss()


def passo_real():
    opt.zero_grad(set_to_none=True)
    l = xent(m(x), y)
    l.backward()
    opt.step()
    return l


t_real = cron(passo_real, reps)


def passo_trivial():
    # Mesmo estilo de despacho do passo real — sem sincronizar a cada
    # chamada — só que quase sem cálculo.
    return x.sum() * 0.0 + y[0].float()


t_trivial = cron(passo_trivial, reps)

print("teste=despacho|lote=%d|real_ms=%.4f|trivial_ms=%.4f|despacho_pct=%.1f"
      % (lote, t_real * 1e3, t_trivial * 1e3, 100.0 * t_trivial / t_real))
