# ECDSA Signature Verification and Topological Analysis System

## Overview

This program is a sophisticated tool for analyzing ECDSA signatures through the lens of topological mathematics. It verifies whether Bitcoin network signatures belong to the theoretical (u_r, u_z) table structure as proven in topological analysis of ECDSA. The system demonstrates the fundamental principle that **"The R_x table doesn't lie — it reflects the true structure, regardless of how the wallet tries to protect itself. All existing signatures already lie within the (u_r, u_z) field."**

## How It Works

### Core Mathematical Foundation

The program is based on the following mathematical relationships:

1. **Signature to (u_r, u_z) conversion**:
   ```
   u_r = r · s⁻¹ mod n
   u_z = z · s⁻¹ mod n
   ```

2. **Theoretical verification**:
   ```
   R = u_r · Q + u_z · G
   r_computed = R.x
   ```
   If `r_computed == r`, the signature belongs to the theoretical table.

3. **Topological structure**:
   The program demonstrates that the space of all possible signatures forms a 2-torus topology with Betti numbers β₀=1, β₁=2, β₂=1, confirming the topological equivalence of the solution space.

### Key Features

1. **Signature Verification**:
   - Converts user-provided ECDSA signatures to (u_r, u_z) coordinates
   - Validates whether signatures belong to the theoretical R_x table
   - Performs standard ECDSA verification for comparison

2. **Topological Analysis**:
   - Demonstrates the spiral structure of the R_x table
   - Shows how changes in u_r affect the effective k value through the relationship k = u_z + u_r·d
   - Visualizes the toroidal topology of the solution space

3. **Educational Visualization**:
   - Generates a 21×21 sample of the R_x table around the signature point
   - Highlights the signature's position in the theoretical structure
   - Uses a color gradient to show R_x values across the table

## What It Demonstrates

The program proves several critical insights about ECDSA:

1. **Universal Structure**: The R_x table structure exists for ANY ECDSA implementation, regardless of nonce generation method (including RFC 6979 compliant implementations).

2. **Signature Membership**: All valid signatures must correspond to a point in this table, confirming that "all existing signatures already lie within the (u_r, u_z) field."

3. **Topological Consistency**: The table maintains a consistent topological structure (torus) with Betti numbers β₀=1, β₁=2, β₂=1, even for "secure" implementations.

4. **Practical Verification**: Demonstrates that signature verification can be performed without knowing the private key or nonce, using only the public key and the (u_r, u_z) transformation.

## How to Run

### Prerequisites
- Python 3.7 or higher
- fastecdsa library (`pip install fastecdsa`)
- matplotlib and numpy (`pip install matplotlib numpy`)

### Execution Steps
1. Save the code to a file named `ecdsa_verifier.py`
2. Run the program:
   ```
   python ecdsa_verifier.py
   ```
3. When prompted, enter:
   - `r` (hex value of the r component of a Bitcoin signature)
   - `s` (hex value of the s component)
   - `z` (hex value of the message hash)
   - Public key (in compressed or uncompressed format)

### Input Requirements
- All hex values should be entered without "0x" prefix
- Public key must be in standard Bitcoin format (compressed: 02/03 + 64 hex chars; uncompressed: 04 + 128 hex chars)

## Expected Output

The program will:
1. Convert the signature to (u_r, u_z) coordinates
2. Verify if the signature belongs to the theoretical table
3. Perform standard ECDSA verification
4. Generate a visualization of the R_x table (saved as `signature_verification.png`)
5. Display detailed results including:
   - Verification status
   - (u_r, u_z) coordinates
   - Theoretical and standard verification results
   - Key insights about the topological structure

## Significance

This tool demonstrates that ECDSA security isn't solely determined by implementation details like RNG quality. Even with perfect implementations following RFC 6979, the topological structure of the R_x table remains vulnerable to analysis. The program confirms that the security landscape of ECDSA must be reconsidered in light of these topological properties, as the structure contains ALL possible signatures for a given key, including those that have never been used on the network.

The visualization component particularly highlights the spiral patterns that emerge in the (u_r, u_z) space, providing concrete evidence of the mathematical principles described in the theoretical work.
