import numpy as np


# Gagne's helper functions
ra_pol = 192.8595
dec_pol = 27.12825
# Initiate some secondary variables
sin_dec_pol = np.sin(np.radians(dec_pol))
cos_dec_pol = np.cos(np.radians(dec_pol))

# J2000.0 Galactic latitude gb of the Celestial North pole (dec=90 degrees) from Carrol and Ostlie
l_north = 122.932

# Galactic Coordinates matrix
TGAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ]
)

# Initiate some global constants
# 1 AU/yr to km/s divided by 1000
kappa = 0.004743717361

# A very small number used for numerical stability
tiny_number = 1e-318


def equatorial_galactic(ra, dec):
    """Transforms equatorial coordinates (ra,dec) to Galactic coordinates (gl,gb). All inputs must be numpy arrays of the same dimension

    param ra: Right ascension (degrees)
    param dec: Declination (degrees)
    output (gl,gb): Tuple containing Galactic longitude and latitude (degrees)
    """
    # Check for parameter consistency
    num_stars = np.size(ra)
    if np.size(dec) != num_stars:
        raise ValueError(
            "The dimensions ra and dec do not agree. They must all be numpy arrays of the same length."
        )
    # Compute intermediate quantities
    ra_m_ra_pol = ra - ra_pol
    sin_ra = np.sin(np.radians(ra_m_ra_pol))
    cos_ra = np.cos(np.radians(ra_m_ra_pol))
    sin_dec = np.sin(np.radians(dec))
    cos_dec = np.cos(np.radians(dec))
    # Compute Galactic latitude
    gamma = sin_dec_pol * sin_dec + cos_dec_pol * cos_dec * cos_ra
    gb = np.degrees(np.arcsin(gamma))
    # Compute Galactic longitude
    x1 = cos_dec * sin_ra
    x2 = (sin_dec - sin_dec_pol * gamma) / cos_dec_pol
    gl = l_north - np.degrees(np.arctan2(x1, x2))
    # gl = (gl+360.) % 360.
    gl = np.mod(gl, 360.0)  # might be better
    # Return Galactic coordinates tuple
    return gl, gb


def ICRS_to_CartesianGalactic(
    ra,
    dec,
    pmra,
    pmdec,
    rv,
    dist
):
    """
    Transforms equatorial coordinates (ra,dec), proper motion (pmra,pmdec), radial velocity and distance to space velocities UVW. All inputs must be numpy arrays of the same dimension.

    param ra: Right ascension (degrees)
    param dec: Declination (degrees)
    param pmra: Proper motion in right ascension (milliarcsecond per year). 	Must include the cos(delta) term
    param pmdec: Proper motion in declination (milliarcsecond per year)
    param rv: Radial velocity (kilometers per second)
    param dist: Distance (parsec)
    param ra_error: Error on right ascension (degrees)
    param dec_error: Error on declination (degrees)
    param pmra_error: Error on proper motion in right ascension (milliarcsecond per year)
    param pmdec_error: Error on proper motion in declination (milliarcsecond per year)
    param rv_error: Error on radial velocity (kilometers per second)
    param dist_error: Error on distance (parsec)

    output (U,V,W): Tuple containing Space velocities UVW (kilometers per second)
    output (U,V,W,EU,EV,EW): Tuple containing Space velocities UVW and their measurement errors, used if any measurement errors are given as inputs (kilometers per second)
    """
    # Compute elements of the T matrix
    cos_ra = np.cos(np.radians(ra))
    cos_dec = np.cos(np.radians(dec))
    sin_ra = np.sin(np.radians(ra))
    sin_dec = np.sin(np.radians(dec))
    # Compute Galactic coordinates
    gl, gb = equatorial_galactic(ra, dec)
    cos_gl = np.cos(np.radians(gl))
    cos_gb = np.cos(np.radians(gb))
    sin_gl = np.sin(np.radians(gl))
    sin_gb = np.sin(np.radians(gb))

    # Compute elements of the T matrix
    T1 = (
        TGAL[0, 0] * cos_ra * cos_dec
        + TGAL[0, 1] * sin_ra * cos_dec
        + TGAL[0, 2] * sin_dec
    )
    T2 = -TGAL[0, 0] * sin_ra + TGAL[0, 1] * cos_ra
    T3 = (
        -TGAL[0, 0] * cos_ra * sin_dec
        - TGAL[0, 1] * sin_ra * sin_dec
        + TGAL[0, 2] * cos_dec
    )
    T4 = (
        TGAL[1, 0] * cos_ra * cos_dec
        + TGAL[1, 1] * sin_ra * cos_dec
        + TGAL[1, 2] * sin_dec
    )
    T5 = -TGAL[1, 0] * sin_ra + TGAL[1, 1] * cos_ra
    T6 = (
        -TGAL[1, 0] * cos_ra * sin_dec
        - TGAL[1, 1] * sin_ra * sin_dec
        + TGAL[1, 2] * cos_dec
    )
    T7 = (
        TGAL[2, 0] * cos_ra * cos_dec
        + TGAL[2, 1] * sin_ra * cos_dec
        + TGAL[2, 2] * sin_dec
    )
    T8 = -TGAL[2, 0] * sin_ra + TGAL[2, 1] * cos_ra
    T9 = (
        -TGAL[2, 0] * cos_ra * sin_dec
        - TGAL[2, 1] * sin_ra * sin_dec
        + TGAL[2, 2] * cos_dec
    )

    # Calculate XYZ
    X = cos_gb * cos_gl * dist
    Y = cos_gb * sin_gl * dist
    Z = sin_gb * dist
    # Compute UVW
    reduced_dist = kappa * dist
    U = T1 * rv + T2 * pmra * reduced_dist + T3 * pmdec * reduced_dist
    V = T4 * rv + T5 * pmra * reduced_dist + T6 * pmdec * reduced_dist
    W = T7 * rv + T8 * pmra * reduced_dist + T9 * pmdec * reduced_dist

    return X, Y, Z, U, V, W


