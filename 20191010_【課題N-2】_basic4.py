import torch

"""
演算
"""

torch.manual_seed(0)
a = torch.randn(size=(2, 3))
b = torch.randn(size=(3,))
c = torch.randn(size=(3, 4))

print(a)
print(a.size())

print(b)
print(b.size())

# 内積
norm_b = torch.dot(b, b)
print(norm_b)

# 行内積その1 ( ちなみに行列とベクトルの積は '.mv()' )
# ca = torch.mm(c, a)  # RuntimeError: size mismatch, m1: [3 x 4], m2: [2 x 3]
aa = torch.mm(a.t(), a)
print(aa)
print(aa.size())

# 行内積その2
aa = torch.mm(a.t(), a)
print(aa)
print(aa.size())

# 転置
print(a.size(), a.t().size())
print(b.size(), b.t().size())
print(c.size(), c.t().size())
