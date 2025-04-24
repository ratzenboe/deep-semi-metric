import torch
import numpy as np
import math


def pairwise_min_velocity_diff_torch(x, kappa=0.004743717361):
    """
    Computes the minimal 3D velocity difference between every pair of stars,
    given their astrometric parameters, using pure PyTorch.

    Parameters
    ----------
    features : torch.Tensor of shape (N, D) or (B, N, D) where D is either 5 or 6 dimensional
        Last dimension = [ra, dec, parallax, pmra, pmdec, (rv)]  -> rv currently ignored
        ra, dec in degrees; parallax in mas; pmra/pmdec in mas/yr; rv in km/s.
    kappa : float
        Tangential‐velocity conversion factor (default 0.004743717361).

    Returns
    -------
    diff : torch.Tensor of shape (N, N) or (B, N, N)
        diff[i,j] = minimal velocity difference (km/s) between star i and j.
    """
    # add batch dim if needed
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (B, N, 6)
        squeeze_batch = True
    else:
        squeeze_batch = False
    B, N, D = x.shape
    # unpack
    if D==6:
        ra, dec, parallax, pmra, pmdec, _ = x.unbind(-1)  # shapes (B,N)
    elif D==5:
        ra, dec, parallax, pmra, pmdec = x.unbind(-1)  # shapes (B,N)
    else:
        raise ValueError('D=6 or 5')
    # Compute distance and reduced distances
    dist = 1000.0 / parallax
    rd   = kappa * dist
    # trig helpers
    rad     = math.pi / 180.0
    cos_ra  = torch.cos(ra  * rad)
    sin_ra  = torch.sin(ra  * rad)
    cos_dec = torch.cos(dec * rad)
    sin_dec = torch.sin(dec * rad)
    # we only need its rows for T_mtx
    t00, t01, t02 = -0.0548755604, -0.8734370902, -0.4838350155
    t10, t11, t12 = +0.4941094279, -0.4448296300, +0.7469822445
    t20, t21, t22 = -0.8676661490, -0.1980763734, +0.4559837762

    # compute each element T1..T9 (shape B,N)
    T1 = t00*cos_ra*cos_dec + t01*sin_ra*cos_dec + t02*sin_dec
    T2 = -t00*sin_ra          + t01*cos_ra
    T3 = -t00*cos_ra*sin_dec  - t01*sin_ra*sin_dec + t02*cos_dec
    T4 = t10*cos_ra*cos_dec + t11*sin_ra*cos_dec + t12*sin_dec
    T5 = -t10*sin_ra          + t11*cos_ra
    T6 = -t10*cos_ra*sin_dec  - t11*sin_ra*sin_dec + t12*cos_dec
    T7 = t20*cos_ra*cos_dec + t21*sin_ra*cos_dec + t22*sin_dec
    T8 = -t20*sin_ra          + t21*cos_ra
    T9 = -t20*cos_ra*sin_dec  - t21*sin_ra*sin_dec + t22*cos_dec

    # stack into (B, N, 3, 3)
    row1 = torch.stack([T1, T2, T3], dim=-1)
    row2 = torch.stack([T4, T5, T6], dim=-1)
    row3 = torch.stack([T7, T8, T9], dim=-1)
    T_mtx = torch.stack([row1, row2, row3], dim=-2)

    # constant vector c = (B, N, 3)
    c_u = T_mtx[...,0,1]*pmra*rd + T_mtx[...,0,2]*pmdec*rd
    c_v = T_mtx[...,1,1]*pmra*rd + T_mtx[...,1,2]*pmdec*rd
    c_w = T_mtx[...,2,1]*pmra*rd + T_mtx[...,2,2]*pmdec*rd
    c = torch.stack([c_u, c_v, c_w], dim=-1)  # (B,N,3)

    # pairwise differences c_i - c_j => (B, N, N, 3)
    cd = c[:,:,None,:] - c[:,None,:,:]
    ci, cf, ci2 = cd.unbind(-1)  # three (B,N,N) tensors

    # extract T_mtx elements along velocity axis
    a0 = T_mtx[...,0,0]  # (B,N)
    d0 = T_mtx[...,1,0]
    g0 = T_mtx[...,2,0]

    # broadcast for pairs => (B,N,N)
    a = a0.unsqueeze(2); b = a0.unsqueeze(1)
    d = d0.unsqueeze(2); e = d0.unsqueeze(1)
    g = g0.unsqueeze(2); h = g0.unsqueeze(1)

    # compute optimal vr_m, vr_n numerators/denominator
    x_num = (
        a*b*e*cf + a*b*h*ci2
        - a*ci*e**2 - a*ci*h**2
        - b**2 * d*cf - b**2 * g*ci2
        + b*ci*d*e + b*ci*g*h
        + d*e*h*ci2 - d*cf*h**2 - e**2 * g*ci2 + e*cf*g*h
    )
    x_den = (
        a**2*e**2 + a**2*h**2
        - 2*a*b*d*e - 2*a*b*g*h
        + b**2*d**2 + b**2*g**2
        + d**2*h**2 - 2*d*e*g*h + e**2*g**2
    )
    y_num = (
        a**2 * e*cf + a**2 * h*ci2
        - a*b * d*cf - a*b * g*ci2
        - a*ci * d*e - a*ci * g*h
        + b*ci * d**2 + b*ci * g**2
        + d**2*h*ci2 - d*e*g*ci2 - d*cf*g*h + e*cf*g**2
    )
    # For stable division
    eps = torch.finfo(x_den.dtype).eps
    x_den = x_den.clamp(min=eps)
    vr_m = x_num / x_den
    vr_n = y_num / x_den

    # minimal squared norm
    norm2 = (
        (a*vr_m - b*vr_n + ci)**2
      + (d*vr_m - e*vr_n + cf)**2
      + (g*vr_m - h*vr_n + ci2)**2
    )
    # For stable sqrt
    norm2 = torch.clamp(norm2, min=0.0)
    # safe sqrt with epsilon
    diff = torch.sqrt(norm2 + eps)
    # diff = torch.sqrt(norm2)  # (B,N,N)
    # zero diagonal out-of-place (must not be in-place, therefore use multiplication)
    mask = 1.0 - torch.eye(N, device=diff.device, dtype=diff.dtype)
    if diff.dim()==3:
        mask = mask.unsqueeze(0)
    diff = diff * mask

    # squeeze batch if necessary
    if squeeze_batch:
        diff = diff.squeeze(0)
    # scrub any lingering NaNs
    diff = torch.nan_to_num(diff, nan=0.0)
    return diff


# plain numpy code
T = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ]
)
kappa = 0.004743717361

def pairwise_min_velocity_diff(ra, dec, pmra, pmdec, dist):
    """
    Computes, for every pair (i, j), the minimal possible 3D velocity difference
    after each star's radial velocity is chosen to minimize |UVW(i) - UVW(j)|.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination in degrees (length N).
    pmra, pmdec : array_like
        Proper motions in mas/yr (length N).
    dist : array_like
        Distances in parsecs (length N).

    Returns
    -------
    diff_matrix : np.ndarray
        An (N x N) array where diff_matrix[i, j] is the minimal velocity difference
        (in km/s) between star i and star j.
    """
    # Number of stars
    N = len(ra)
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    pmra = np.asarray(pmra)
    pmdec = np.asarray(pmdec)
    dist = np.asarray(dist)

    # -- 1) Construct the T_mtx (N, 3, 3) as in VrOpt.create_trafo_matrix() --

    cos_ra = np.cos(np.radians(ra))
    sin_ra = np.sin(np.radians(ra))
    cos_dec = np.cos(np.radians(dec))
    sin_dec = np.sin(np.radians(dec))

    # "T1" through "T9" for each star
    T1 = T[0, 0] * cos_ra * cos_dec + T[0, 1] * sin_ra * cos_dec + T[0, 2] * sin_dec
    T2 = -T[0, 0] * sin_ra + T[0, 1] * cos_ra
    T3 = -T[0, 0] * cos_ra * sin_dec - T[0, 1] * sin_ra * sin_dec + T[0, 2] * cos_dec
    T4 = T[1, 0] * cos_ra * cos_dec + T[1, 1] * sin_ra * cos_dec + T[1, 2] * sin_dec
    T5 = -T[1, 0] * sin_ra + T[1, 1] * cos_ra
    T6 = -T[1, 0] * cos_ra * sin_dec - T[1, 1] * sin_ra * sin_dec + T[1, 2] * cos_dec
    T7 = T[2, 0] * cos_ra * cos_dec + T[2, 1] * sin_ra * cos_dec + T[2, 2] * sin_dec
    T8 = -T[2, 0] * sin_ra + T[2, 1] * cos_ra
    T9 = -T[2, 0] * cos_ra * sin_dec - T[2, 1] * sin_ra * sin_dec + T[2, 2] * cos_dec

    # Shape (N,3,3)
    # We'll store them as [ [T1, T2, T3], [T4, T5, T6], [T7, T8, T9] ], then transpose(2,0,1)
    T_total = np.array(
        [
            [T1, T2, T3],
            [T4, T5, T6],
            [T7, T8, T9]
        ]
    )  # shape: (3,3,N)
    # Now we swap axes so that it becomes (N,3,3)
    T_mtx = T_total.transpose(2, 0, 1)

    # -- 2) Construct the "constant" vector c_vec (N,3) as in VrOpt.uvw_const_vector() --
    # reduced_dist = kappa * dist
    rd = kappa * dist

    # T2, T3 etc. come from T_mtx
    T2v = T_mtx[:, 0, 1]
    T3v = T_mtx[:, 0, 2]
    T5v = T_mtx[:, 1, 1]
    T6v = T_mtx[:, 1, 2]
    T8v = T_mtx[:, 2, 1]
    T9v = T_mtx[:, 2, 2]

    c_u = T2v * pmra * rd + T3v * pmdec * rd
    c_v = T5v * pmra * rd + T6v * pmdec * rd
    c_w = T8v * pmra * rd + T9v * pmdec * rd
    const_vec = np.column_stack([c_u, c_v, c_w])  # shape (N,3)

    # -- 3) Build the NxN matrix of minimal velocity differences --
    diff_matrix = np.zeros((N, N), dtype=float)

    # Get lower triangle indices
    m, n = np.tril_indices(N, -1)

    # Differences of the constant vectors
    c, f, i = (const_vec[m, :] - const_vec[n, :]).T
    # Denote elements of the (U_m - U_n)**2 + (V_m - V_n)**2 + (W_m - W_n)**2 vector
    # norm**2 = (a*vr_m - b*vr_n + c)**2 + (d*vr_m - e*vr_n + f)**2 + (g*vr_m - h*vr_n + i)**2
    a = T_mtx[:, 0, 0][m]
    b = T_mtx[:, 0, 0][n]
    d = T_mtx[:, 1, 0][m]
    e = T_mtx[:, 1, 0][n]
    g = T_mtx[:, 2, 0][m]
    h = T_mtx[:, 2, 0][n]
    # Compute optimal radial velocity of star m
    x_num = (
        a*b*e*f + a*b*h*i - a*c*e**2 - a*c*h**2 - b**2 * d*f -
        b**2 *g*i + b*c*d*e + b*c*g*h + d*e*h*i - d*f*h**2 - e**2 *g*i + e*f*g*h
    )
    x_den = (
        a**2 * e**2 + a**2 * h**2 - 2*a*b*d*e - 2*a*b*g*h +
        b**2 * d**2 + b**2 * g**2 + d**2 * h**2 - 2*d*e*g*h + e**2 * g**2
    )
    vr_m = x_num / x_den
    # Compute optimal radial velocity of star n
    y_num = (
        a**2 * e*f + a**2 * h*i - a*b*d*f - a*b*g*i - a*c*d*e - a*c*g*h + b*c*d**2 +
        b*c*g**2 + d**2 * h*i - d*e*g*i - d*f*g*h + e*f*g**2
    )
    y_den = (
        a**2 * e**2 + a**2 * h**2 - 2*a*b*d*e - 2*a*b*g*h +
        b**2 * d**2 + b**2 * g**2 + d**2 * h**2 - 2*d*e*g*h + e**2 * g**2
    )
    vr_n = y_num / y_den
    # Compute the minimal difference of the pairwise 3D velocity vectors
    norm2 = (a*vr_m - b*vr_n + c)**2 + (d*vr_m - e*vr_n + f)**2 + (g*vr_m - h*vr_n + i)**2

    # Fill in the upper triangle
    diff_matrix[m, n] = np.sqrt(norm2)
    diff_matrix[n, m] = np.sqrt(norm2)

    return diff_matrix