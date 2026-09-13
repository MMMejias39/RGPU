"""Mesma carga, em PyTorch. Uso: bench_torch.py [cpu|cuda]"""
import sys, time
import torch
import torch.nn as nn

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda")
if dev.type == "cpu":
    torch.set_num_threads(torch.get_num_threads())
torch.manual_seed(7)

D, H, C = 512, 1024, 10

for lote in [1, 8, 32, 128, 512, 2048]:
    i = torch.arange(lote, dtype=torch.float32).unsqueeze(1)
    j = torch.arange(D, dtype=torch.float32).unsqueeze(0)
    x = torch.sin(i * 0.01 + j * 0.03).to(dev)
    y = (torch.arange(lote) % C).to(dev)

    model = nn.Sequential(
        nn.Linear(D, H), nn.ReLU(),
        nn.Linear(H, H), nn.ReLU(),
        nn.Linear(H, C),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    perda = nn.CrossEntropyLoss()

    def passo():
        opt.zero_grad(set_to_none=True)
        l = perda(model(x), y)
        l.backward()
        opt.step()

    for _ in range(10):
        passo()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    passos = 200 if dev.type == "cuda" else (20 if lote < 512 else 5)
    t0 = time.perf_counter()
    for _ in range(passos):
        passo()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / passos

    flop = (D * H + H * H + H * C) * 3 * 2 * lote
    print("motor=torch-%s|lote=%d|ms=%.4f|gflops=%.1f" % (dev.type, lote, dt * 1e3, flop / dt / 1e9))
