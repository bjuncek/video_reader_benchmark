import timeit
import torch
import pandas as pd
NUBMER_TRIALS=100

setup = """\
import torch
torch.set_num_threads(1)
"""

def single_tensor(k):
    a = torch.ones(k, 3, 112, 112)


def for_loop_nostack(k):
    test = []
    for i in range(k):
        test.append(torch.ones(3, 112, 112))


def for_loop_stack(k):
    test = []
    for i in range(k):
        test.append(torch.ones(3, 112, 112))
    a = torch.stack(test)


methods, ks, times = [], [], []
for method in ["single_tensor", "for_loop_nostack", "for_loop_stack"]:
    for k in [5, 10, 50, 100, 500, 1000, 5000]:
        t = timeit.timeit(f"{method}({k})", setup=setup, globals=globals(), number=NUBMER_TRIALS)
        methods.append(method)
        times.append(t/NUBMER_TRIALS)
        ks.append(k)

df = pd.DataFrame({"method": methods, "k": ks, "time":times})
df.to_csv("torch_ones_speed.csv")