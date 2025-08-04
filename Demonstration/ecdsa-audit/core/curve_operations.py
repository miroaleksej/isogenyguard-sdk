"""
Module for elliptic curve operations.
Based on the coincurve library for maximum performance with secp256k1.
"""

import coincurve
from typing import Tuple, List, Optional
import numpy as np

class CurveOperations:
    def __init__(self, curve_name: str = "secp256k1"):
        """Initialize curve operations"""
        if curve_name == "secp256k1":
            self.curve_name = "secp256k1"
            self.G = coincurve.GENERATOR
            self.n = coincurve.SECP256K1_ORDER
        else:
            # Support for other curves can be added here
            raise NotImplementedError("Only secp256k1 is supported currently")
    
    def compute_R(self, u_r: int, u_z: int, Q: coincurve.PublicKey) -> coincurve.Point:
        """
        Computes point R = u_r * Q + u_z * G
        
        Args:
            u_r: vertical coordinate
            u_z: horizontal coordinate
            Q: public key
            
        Returns:
            Point R on the elliptic curve
        """
        term1 = Q.multiply(u_r)
        term2 = self.G.multiply(u_z)
        return term1.combine([term2])
    
    def get_Rx(self, u_r: int, u_z: int, Q: coincurve.PublicKey) -> int:
        """
        Computes x-coordinate of point R
        
        Args:
            u_r: vertical coordinate
            u_z: horizontal coordinate
            Q: public key
            
        Returns:
            x-coordinate of point R modulo n
        """
        R = self.compute_R(u_r, u_z, Q)
        return R.point.x() % self.n
    
    def compute_subregion(self, 
                         Q: coincurve.PublicKey, 
                         u_r_min: int, u_r_max: int,
                         u_z_min: int, u_z_max: int) -> List[List[int]]:
        """
        Computes a subregion of the Rₓ table
        
        Args:
            Q: public key
            u_r_min, u_r_max: vertical coordinate range
            u_z_min, u_z_max: horizontal coordinate range
            
        Returns:
            2D list with x-coordinates of points
        """
        subregion = []
        for u_r in range(u_r_min, u_r_max):
            row = []
            for u_z in range(u_z_min, u_z_max):
                row.append(self.get_Rx(u_r, u_z, Q))
            subregion.append(row)
        return subregion
    
    def get_special_point(self, u_r: int, d: Optional[int] = None) -> int:
        """
        Calculates special point u_z* for given u_r
        
        Args:
            u_r: vertical coordinate
            d: private key (optional, not required for actual computations)
            
        Returns:
            Special point u_z* where symmetry occurs
        """
        # For secp256k1, we don't know d, but the formula is u_z* = -u_r * d mod n
        # This is used for theoretical analysis
        if d is not None:
            return (-u_r * d) % self.n
        return None
    
    def get_mirror_point(self, u_r: int, u_z: int, d: Optional[int] = None) -> int:
        """
        Calculates mirror point for given (u_r, u_z)
        
        Args:
            u_r: vertical coordinate
            u_z: horizontal coordinate
            d: private key (optional)
            
        Returns:
            Mirror point u_z' where R_x is the same
        """
        if d is not None:
            return (-u_z - 2 * u_r * d) % self.n
        return None
