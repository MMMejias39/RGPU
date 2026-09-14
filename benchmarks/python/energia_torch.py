"""Equivalente a `energia_tf.py`, para PyTorch.

Uso: energia_torch.py <lote> <passos>
"""
import sys, time
import torch
import torch.nn as nn

lote = int(sys.argv[1]) if len(sys.argv) > 1 else 512
passos = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
d, h, CLASSES = 512, 1024, 10
dev = torch.device("cuda")

x = torch.randn(lote, d, device=dev)
y = torch.arange(lote, device=dev) % CLASSES

m = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, CLASSES)).to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-3)
xent = nn.CrossEntropyLoss()

for _ in range(10):
    opt.zero_grad(set_to_none=True)
    xent(m(x), y).backward()
    opt.step()
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(passos):
    opt.zero_grad(set_to_none=True)
    l = xent(m(x), y)
    l.backward()
    opt.step()
torch.cuda.synchronize()
dt = time.perf_counter() - t0

flop = (d * h + h * h + h * CLASSES) * 3 * 2 * lote
print("motor=torch-cuda|lote=%d|passos=%d|ms=%.4f|gflops=%.1f"
      % (lote, passos, dt / passos * 1e3, flop * passos / dt / 1e9))
