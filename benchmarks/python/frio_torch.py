import time, torch, torch.nn as nn
d = torch.device("cuda")
m = nn.Sequential(nn.Linear(512,1024), nn.ReLU(), nn.Linear(1024,1024), nn.ReLU(), nn.Linear(1024,10)).to(d)
o = torch.optim.Adam(m.parameters(), 1e-3)
x = torch.randn(128,512, device=d); y = torch.arange(128, device=d) % 10
t0=time.perf_counter()
l = nn.CrossEntropyLoss()(m(x), y); l.backward(); o.step(); torch.cuda.synchronize()
print("primeiro_passo_ms=%.1f" % ((time.perf_counter()-t0)*1e3))
