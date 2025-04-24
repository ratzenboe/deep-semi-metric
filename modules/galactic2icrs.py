import torch
import math


def cartesianGalactic_to_ICRS_torch(x):
    """
    PyTorch version of cartesianGalactic_to_ICRS.

    Parameters
    ----------
    data : array-like or torch.Tensor, shape (..., 6)
        Last dimension must be [X, Y, Z, U, V, W].

    Returns
    -------
    out : torch.Tensor, shape (..., 6)
        Last dimension is [ra, dec, parallax, pmra, pmdec, rv].
    """
    # --- 0) Prep --------------------------------------------------------
    data = torch.as_tensor(x)
    # # work in double precision for best match to NumPy
    # data = data.to(torch.float64)
    device = data.device
    dtype = data.dtype
    *lead, six = data.shape
    assert six == 6, "last dimension must be 6 (X,Y,Z,U,V,W)"

    # split into pos, vel
    pos = data[..., :3]  # (..., 3)
    vel = data[..., 3:]  # (..., 3)

    # constants
    k = 4.740470446
    # rotation from ICRS→Galactic (Hipparcos convention)
    T = torch.tensor([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ], dtype=dtype, device=device, requires_grad=False)
    iT = torch.inverse(T)  # (3×3)

    # --- 1) Transform positions ----------------------------------------
    # Galactic → ICRS
    coords_icrs = torch.matmul(pos, iT.T)  # (..., 3)
    x, y, z = coords_icrs.unbind(-1)  # each (...,)

    # distances & parallax
    dist = torch.linalg.norm(coords_icrs, dim=-1)  # (...)
    parallax = 1000.0 / dist  # mas

    # RA, Dec [deg]
    ra = torch.atan2(y, x) * (180.0 / math.pi)
    ra = torch.remainder(ra, 360.0)
    dec = torch.atan2(z, torch.sqrt(x * x + y * y)) * (180.0 / math.pi)

    # --- 2) Build spherical‐basis matrix A for each star ---------------
    deg2rad = math.pi / 180.0
    ra_rad = ra * deg2rad
    dec_rad = dec * deg2rad

    cos_ra, sin_ra = torch.cos(ra_rad), torch.sin(ra_rad)
    cos_dec, sin_dec = torch.cos(dec_rad), torch.sin(dec_rad)

    # A[..., :, 0] = unit‐vector along r
    # A[..., :, 1] = unit‐vector along +α
    # A[..., :, 2] = unit‐vector along +δ
    A = torch.zeros(*lead, 3, 3, dtype=dtype, device=device, requires_grad=False)
    A[..., 0, 0] = cos_ra * cos_dec
    A[..., 1, 0] = sin_ra * cos_dec
    A[..., 2, 0] = sin_dec

    A[..., 0, 1] = -sin_ra
    A[..., 1, 1] = cos_ra

    A[..., 0, 2] = -cos_ra * sin_dec
    A[..., 1, 2] = -sin_ra * sin_dec
    A[..., 2, 2] = cos_dec

    # --- 3) Transform velocities --------------------------------------
    # Combine: vel_sph = (A^T @ iT) @ vel
    M = torch.matmul(A.transpose(-2, -1), iT)  # (..., 3, 3)
    vel_sph = torch.matmul(M, vel.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    rv = vel_sph[..., 0]
    mu_ra_v = vel_sph[..., 1]
    mu_dm_v = vel_sph[..., 2]

    # convert tangential km/s → mas/yr
    pmra = mu_ra_v * parallax / k
    pmdec = mu_dm_v * parallax / k

    # --- 4) Pack outputs ---------------------------------------------
    out = torch.stack([ra, dec, parallax, pmra, pmdec, rv], dim=-1)
    return out