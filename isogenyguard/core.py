"""
Core module for ECDSA parameter mapping and private key recovery.

Implements:
- Theorem 1: Bijective mapping (r, s, z) ↔ (u_r, u_z)
- Theorem 9: Private key recovery via special points (k=0)
"""

def rsz_to_uruz(r, s, z, n):
    """
    Maps ECDSA signature components (r, s, z) to transformed parameters (u_r, u_z).
    
    According to Theorem 1:
        u_r = r * s⁻¹ mod n
        u_z = z * s⁻¹ mod n
    
    Args:
        r (int): x-coordinate of point R = kG
        s (int): Second component of signature
        z (int): Hash of the message
        n (int): Order of the elliptic curve group
    
    Returns:
        tuple: (u_r, u_z)
    """
    s_inv = pow(s, -1, n)
    ur = (r * s_inv) % n
    uz = (z * s_inv) % n
    return ur, uz


def uruz_to_rsz(ur, uz, d, n):
    """
    Recovers signature components (r, s, z) from (u_r, u_z) using private key d.
    
    Uses:
        k = u_z + u_r * d mod n
        R = kG → r = x(R)
        s and z can be derived if r is known
    
    Args:
        ur (int): Transformed parameter u_r
        uz (int): Transformed parameter u_z
        d (int): Private key
        n (int): Group order
    
    Returns:
        dict: {'k': k, 'r': r} — r must be computed via elliptic curve arithmetic
    """
    k = (uz + ur * d) % n
    # r = x(k * G) — requires curve arithmetic (not implemented here)
    return {'k': k}


def recover_private_key(special_points, n):
    """
    Recovers private key d from special points where k = 0.
    
    Theorem 9: d ≡ -(u_z^(r+1) - u_z^(r)) mod n
    Special points lie on the line u_z = -u_r * d mod n.
    
    Args:
        special_points (list): List of (u_r, u_z) tuples, sorted by u_r
        n (int): Group order
    
    Returns:
        int or None: Recovered private key d, or None if insufficient data
    """
    if len(special_points) < 2:
        return None

    diffs = []
    # Sort by u_r to ensure consecutive rows
    sorted_points = sorted(special_points, key=lambda x: x[0])

    for i in range(len(sorted_points) - 1):
        ur_curr, uz_curr = sorted_points[i]
        ur_next, uz_next = sorted_points[i + 1]
        # Check if rows are consecutive
        if ur_next == ur_curr + 1:
            # d ≡ -(Δu_z) mod n
            d = (uz_curr - uz_next) % n
            diffs.append(d)

    if not diffs:
        return None

    # Return the most frequent d (mode)
    return max(set(diffs), key=diffs.count)
