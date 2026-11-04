"""Batched Newton solution of F(x, *params) = 0 with derivatives by the implicit function theorem."""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.func import jvp, vmap

Residual = Callable[..., torch.Tensor]


class NewtonFailed(RuntimeError):
    pass


def jacobian(residual: Residual, x: torch.Tensor, params: Sequence[torch.Tensor]) -> torch.Tensor:
    """dF/dx, (..., M, M), by forward-mode AD (one batched jvp per unknown, vectorised with vmap)."""
    m = x.shape[-1]
    eye = torch.eye(m, dtype=x.dtype, device=x.device)
    tangents = eye.reshape((m,) + (1,) * (x.dim() - 1) + (m,)).expand((m,) + x.shape)
    col = vmap(lambda v: jvp(lambda y: residual(y, *params), (x,), (v,))[1])(tangents)
    return torch.movedim(col, 0, -1)


def newton(residual: Residual, x0: torch.Tensor, params: Sequence[torch.Tensor], *, tol: float = 1e-11,
           maxiter: int = 60, max_step: float = 10.0, linearize: Callable | None = None) -> torch.Tensor:
    """Root x* of residual(x, *params) (x: (..., M); residual elementwise over the batch dims).

    Newton steps rescaled so that max|dx| <= max_step; converged when max|F| < tol for every batch element.
    linearize(x, params): optional residual-like function with the same Jacobian at x, cheaper to
    differentiate. The returned x* carries gradients w.r.t. params through dx*/dtheta = -(dF/dx)^-1 dF/dtheta
    (reverse and forward mode); the iteration itself is not traced.
    """
    params = tuple(params)
    with torch.no_grad():
        plain = tuple(p.detach() for p in params)
        x = x0.detach().clone()
        for it in range(maxiter + 1):
            f = residual(x, *plain)
            if not torch.isfinite(f).all():
                raise NewtonFailed(f"non-finite residual at iteration {it}")
            err = f.abs().amax(-1)
            if bool((err < tol).all()):
                break
            if it == maxiter:
                raise NewtonFailed(f"no convergence after {maxiter} iterations: {int((err >= tol).sum())} "
                                   f"of {err.numel()} systems, max |F| = {float(err.max()):.3e}")
            r = residual if linearize is None else linearize(x, plain)
            dx = torch.linalg.solve(jacobian(r, x, plain), -f.unsqueeze(-1)).squeeze(-1)
            x = x + dx * torch.clamp(max_step / dx.abs().amax(-1, keepdim=True), max=1.0)
    return _ImplicitRoot.apply(residual, x, *params)


class _ImplicitRoot(torch.autograd.Function):
    """Identity on the converged root x*; backward/jvp by the implicit function theorem."""

    @staticmethod
    def forward(residual, x, *params):
        return x.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        residual, x, *params = inputs
        ctx.residual = residual
        ctx.save_for_backward(x, *params)
        ctx.save_for_forward(x, *params)

    @staticmethod
    def backward(ctx, gx):
        x, *params = ctx.saved_tensors
        need = ctx.needs_input_grad[2:]
        with torch.enable_grad():
            leaf = [p.detach().requires_grad_(n) for p, n in zip(params, need)]
            jt = jacobian(ctx.residual, x.detach(), [p.detach() for p in params]).transpose(-1, -2)
            lam = torch.linalg.solve(jt, gx.unsqueeze(-1)).squeeze(-1)
            f = ctx.residual(x.detach(), *leaf)
            wanted = [p for p, n in zip(leaf, need) if n]
            g = iter(torch.autograd.grad(f, wanted, grad_outputs=-lam, allow_unused=True)) if wanted else iter(())
        return (None, None) + tuple(next(g) if n else None for n in need)

    @staticmethod
    def jvp(ctx, _, dx_unused, *dparams):
        x, *params = ctx.saved_tensors
        tang = tuple(torch.zeros_like(p) if d is None else d for p, d in zip(params, dparams))
        _, df = jvp(lambda *p: ctx.residual(x, *p), tuple(params), tang)
        return -torch.linalg.solve(jacobian(ctx.residual, x, params), df.unsqueeze(-1)).squeeze(-1)
