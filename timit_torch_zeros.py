import timeit
import torch
import pandas as pd


setup = """\
import torch
"""

def video_tensor(k):
    a = torch.zeros(k, 3, 112, 112)

def for_loop_nostack(k):
    test = []
    for i in range(k):
        test.append(torch.zeros(3, 112, 112))
    a = torch.stack(test)

def for_loop_stack(k):
    test = []
    for i in range(k):
        test.append(torch.zeros(3, 112, 112))
    a = torch.stack(test)

methods, ks, times = [], [], []
for method in ["video_tensor", "for_loop_nostack", "for_loop_stack"]:
    for k in [5, 10, 50, 100, 500, 1000, 5000, 7000, 10000]:
        t = timeit.timeit(f"{method}({k})", setup=setup, globals=globals(), number=100)
        methods.append(method)
        times.append(t)
        ks.append(k)

df = pd.DataFrame({"method": methods, "k": ks, "time":times})
df.to_csv("torch_zeros_speed.csv")