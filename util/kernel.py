import math
import torch
import numpy as np

def sanitise(sigma):
    return sigma.clamp(min=-1,max=1)

def increment_kernel(sigma):
    new_sigma = (1-sigma**2).sqrt()
    new_sigma += sigma*(math.pi - sigma.acos())
    new_sigma /= math.pi
    return sanitise(new_sigma)

# TODO: rewrite with cholesky_solve and Frobenius norm trace
def complexity(sigma, c, samples):
    n = sigma.shape[0]
    device = sigma.device
    id = torch.eye(n, device=device)

    assert ( sigma == sigma.t() ).all()
    u = torch.cholesky(sigma)
    inv = torch.cholesky_inverse(u)
    assert (torch.mm(sigma, inv) - id).abs().max() < 1e-03

    nth_root_det = u.diag().pow(2/n).prod()
    inv_trace = inv.diag().sum()
    inv_proj = torch.dot(c, torch.mv(inv, c))

    formula_1 = n/5.0 + nth_root_det * ( (0.5 - 1/math.pi)*inv_trace + inv_proj/math.pi )

    mat = nth_root_det*inv - id

    z = torch.randn((n,samples), device=device).abs()
    cz = torch.mul(c.unsqueeze(1), z)
    contribs = -0.5*(cz * torch.matmul(mat, cz)).sum(dim=0)

    formula_0 = n*math.log(2) + math.log(samples) - torch.logsumexp(contribs, dim=0)

    return formula_0.item(), formula_1.item()

def invert_bound(x):
    return 1-math.exp(-x)