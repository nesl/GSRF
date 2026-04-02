
import torch
import math


# ---- Fourier-Legendre expansion (FLE) ----
def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


# normalization coefficients for associated Legendre polynomials
def _build_norm_table(deg):
    norm = {}
    for l in range(deg + 1):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            norm[(l, m)] = math.sqrt(
                (2 * l + 1) / (4.0 * math.pi)
                * _factorial(l - abs_m) / _factorial(l + abs_m)
            )
    return norm


# batched associated Legendre polynomial computation
def _associated_legendre_batch(deg, z):
    plm = [[None] * (l + 1) for l in range(deg + 1)]
    plm[0][0] = torch.ones_like(z)

    if deg == 0:
        return plm

    sin_theta = torch.sqrt(torch.clamp(1.0 - z * z, min=1e-12))

    for m in range(1, deg + 1):
        plm[m][m] = -(2 * m - 1) * sin_theta * plm[m - 1][m - 1]

    for m in range(0, deg):
        plm[m + 1][m] = z * (2 * m + 1) * plm[m][m]

    for m in range(0, deg + 1):
        for l in range(m + 2, deg + 1):
            plm[l][m] = ((2 * l - 1) * z * plm[l - 1][m]
                         - (l + m - 1) * plm[l - 2][m]) / (l - m)

    return plm


# evaluate complex FLE: coefficients x basis -> (Re, Im)
def eval_fle(deg, coeffs, dirs):
    assert deg <= 10 and deg >= 0
    n_coeffs = (deg + 1) ** 2
    assert coeffs.shape[-1] >= n_coeffs

    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    alpha = torch.atan2(y, x)
    cos_theta = z

    plm = _associated_legendre_batch(deg, cos_theta)
    norm = _build_norm_table(deg)

    basis_re_list = []
    basis_im_list = []

    for l in range(deg + 1):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            bp = norm[(l, m)] * plm[l][abs_m]
            if m < 0:
                bp = bp * ((-1) ** abs_m)

            cos_ma = torch.cos(m * alpha)
            sin_ma = torch.sin(m * alpha)

            basis_re_list.append(cos_ma * bp)
            basis_im_list.append(sin_ma * bp)

    B_re = torch.stack(basis_re_list, dim=-1)
    B_im = torch.stack(basis_im_list, dim=-1)

    a = coeffs[..., 0, :n_coeffs]
    b = coeffs[..., 1, :n_coeffs]

    result_re = (a * B_re - b * B_im).sum(dim=-1)
    result_im = (a * B_im + b * B_re).sum(dim=-1)

    return result_re, result_im
