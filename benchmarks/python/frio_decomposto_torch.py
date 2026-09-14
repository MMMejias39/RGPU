"""Decompõe a partida a fria do PyTorch em três fases, para separar o que é
custo de importar a biblioteca (Python) do que é custo de arquitetura (GPU).

Uso: frio_decomposto_torch.py
"""
import time

t0 = time.perf_counter()
import torch
import torch.nn as nn
t_import = time.perf_counter() - t0

t1 = time.perf_counter()
d = torch.device("cuda")
m = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10)).to(d)
o = torch.optim.Adam(m.parameters(), 1e-3)
x = torch.randn(128, 512, device=d)
y = torch.arange(128, device=d) % 10
torch.cuda.synchronize()
t_montagem = time.perf_counter() - t1

t2 = time.perf_counter()
l = nn.CrossEntropyLoss()(m(x), y)
l.backward()
o.step()
torch.cuda.synchronize()
t_primeiro_passo = time.perf_counter() - t2

print("import_ms=%.1f montagem_modelo_ms=%.1f primeiro_passo_ms=%.1f" %
      (t_import * 1e3, t_montagem * 1e3, t_primeiro_passo * 1e3))
