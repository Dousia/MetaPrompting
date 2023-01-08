import torch
from pytorch_memlab import LineProfiler

def inner():
    torch.nn.Linear(100, 100).cuda()

@profile
def outer():
    linear = torch.nn.Linear(100, 100).cuda()
    linear2 = torch.nn.Linear(100, 100).cuda()
    inner()

# # with LineProfiler(outer, inner) as prof:
# #     outer()
# prof.display()

outer()
