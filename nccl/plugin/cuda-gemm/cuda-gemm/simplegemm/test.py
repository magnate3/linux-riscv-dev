import torch

torch.ops.load_library("gemm.so")

torch.set_default_device("cuda")

m, n, k = 128, 256, 64

a = torch.arange(m * k).reshape(m, k).bfloat16()
b = torch.eye(k, n).bfloat16().T.contiguous().T
c = torch.ops.gemm.pingpong(a, b)
cref = torch.mm(a, b)

print(b.size(), b.stride())
print(a)
print(b)
print(cref)
print(c)

print(torch.allclose(c, cref, atol=0, rtol=0))
