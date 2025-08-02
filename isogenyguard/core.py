import numpy as np
from .utils import modular_inverse, calculate_finite_difference

def recover_private_key(ur_values, uz_values, r_values, n):
    """
    Recovers the private key d through special point analysis
    Implementation of Theorem 9 from the research paper
    
    :param ur_values: List of u_r values from signatures
    :param uz_values: List of u_z values from signatures
    :param r_values: List of R_x values from signatures
    :param n: Group order
    :return: Recovered private key d
    """
    # Calculate gradients using finite differences
    d_r_d_uz = calculate_finite_difference(r_values, uz_values)
    d_r_d_ur = calculate_finite_difference(r_values, ur_values)
    
    # Apply formula from Theorem 5: d = -(∂r/∂u_z) * (∂r/∂u_r)^-1 mod n
    d_estimates = []
    for i in range(len(d_r_d_uz)):
        if d_r_d_ur[i] != 0:
            d = (-d_r_d_uz[i] * modular_inverse(d_r_d_ur[i], n)) % n
            d_estimates.append(d)
    
    # Return the modal value (most frequently occurring)
    if d_estimates:
        return max(set(d_estimates), key=d_estimates.count)
    return None

def check_special_points(ur_values, uz_values, n):
    """
    Checks for special points in the data
    
    :param ur_values: List of u_r values
    :param uz_values: List of u_z values
    :param n: Group order
    :return: List of special point indices
    """
    special_points = []
    for i in range(1, len(ur_values)):
        # Check special point condition: u_z ≡ -u_r * d mod n
        # For adjacent points: u_z^(r+1) - u_z^(r) ≡ -d mod n
        delta_ur = ur_values[i] - ur_values[i-1]
        delta_uz = uz_values[i] - uz_values[i-1]
        
        if delta_ur != 0:
            d_candidate = (-delta_uz * modular_inverse(delta_ur, n)) % n
            # Verify d_candidate is integer and within [0, n-1]
            if 0 <= d_candidate < n:
                special_points.append(i)
                
    return special_points
