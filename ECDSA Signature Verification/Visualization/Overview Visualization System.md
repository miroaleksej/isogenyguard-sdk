# ECDSA Signature Verification and Visualization System

## Overview

This HTML file provides an interactive 3D visualization of the ECDSA signature space as a mathematical cube with coordinates (u_r, u_z, d). The visualization demonstrates the topological structure of ECDSA signatures and reveals critical insights about the inherent mathematical relationships that exist regardless of implementation details.

## Key Features

### Interactive 3D Cube Visualization
- Displays a configurable hypercube (u_r, u_z, d) where:
  * u_r = r · s⁻¹ mod n (first component)
  * u_z = z · s⁻¹ mod n (second component)
  * d = private key
- Supports cube sizes from 5×5×5 to 15×15×15
- Shows how R_x values (x-coordinates of signature points) distribute throughout the space

### Multiple Visualization Modes
1. **Spiral Structures**: Reveals how each R_x value forms spiral patterns through the cube
2. **Torus Projection**: Displays a semi-transparent torus at the cube's center with projections from cube points
3. **Layers by d**: Shows individual 2D slices for specific private key values

### Mathematical Principles Demonstrated
- **Mirror Pairs**: For each point (u_r, u_z), there exists a mirror point where:
  ```
  u_z + u_z' ≡ -2·u_r·d mod n
  ```
  Both points share the same R_x value
- **Spiral Structure**: For fixed d, R_x values form straight lines with slope -d in the (u_r, u_z) plane
- **Topological Equivalence**: The solution space is topologically equivalent to a torus with Betti numbers β₀=1, β₁=2, β₂=1

## How It Works

The visualization implements the core mathematical relationship:
```
k = (u_r · d + u_z) mod n
R_x(k) = R_x(-k) mod n
```

For each point in the cube:
1. It calculates the corresponding R_x value
2. Colors points based on their R_x value
3. Connects points with identical R_x values to reveal spiral patterns
4. In torus mode, projects cube points onto a semi-transparent torus to demonstrate the topological equivalence

## User Interaction

The interface provides intuitive controls:
- **Cube Size Slider**: Adjusts the visualization dimensions (5-15)
- **R_x Selector**: Filter to display specific R_x values or all values
- **d Selector**: Change the private key value being visualized
- **View Mode Tabs**: Switch between spiral, torus, and layer views
- **Interactive 3D Controls**: Rotate, zoom, and pan the visualization

## Educational Value

This visualization powerfully demonstrates the fundamental principle stated in the documentation:
> "The R_x table doesn't lie — it reflects the true structure regardless of how the wallet tries to protect itself. All existing signatures already lie within the (u_r, u_z) field."

It shows that:
- ECDSA signatures form predictable mathematical patterns
- These patterns exist even for implementations following RFC 6979
- The structure reveals mirror pairs that can be used to recover private keys
- The topological properties (Betti numbers) provide security metrics

## Technical Implementation

The visualization uses Plotly.js for 3D rendering with:
- Semi-transparent torus projection in the cube's center
- Dynamic color mapping based on R_x values
- Precise mathematical calculations for R_x determination
- Responsive layout for different screen sizes
- Educational tooltips explaining mathematical concepts

This visualization is not just a theoretical tool—it provides concrete evidence of the topological structure underlying ECDSA that has significant implications for cryptographic security analysis and vulnerability detection. It transforms abstract mathematical concepts into tangible, interactive visual representations that reveal the inherent structure of ECDSA signature space.
