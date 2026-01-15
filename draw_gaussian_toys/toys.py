# import sys
# import os
# import importlib.util

# # Get paths
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# beam_size_file = os.path.join(parent_dir, 'twiss_plots', 'calculate_emittance', 'beam_size.py')

# # Load the module directly
# spec = importlib.util.spec_from_file_location("beam_size", beam_size_file)
# beam_size = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(beam_size)

# # Now you can use the function
# read_outx_with_headers = beam_size.read_outx_with_headers

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Path to the data file
# data_file = os.path.join(parent_dir, 'twiss_IR_v09.outx')

# def calculate_beam_size(betx, bety, ex, ey, gamma=None):
#     """
#     Calculate beam sizes from Twiss parameters and emittances.
#     Uses geometric emittance (not normalized).
    
#     Parameters:
#     -----------
#     betx : float or array
#         Beta function in x [m]
#     bety : float or array
#         Beta function in y [m]
#     ex : float
#         Geometric emittance in x [m·rad] (from TWISS header)
#     ey : float
#         Geometric emittance in y [m·rad] (from TWISS header)
#     gamma : float, optional
#         Lorentz gamma factor (for reference only)
    
#     Returns:
#     --------
#     sigma_x, sigma_y : beam sizes in x and y [m]
    
#     Note:
#     -----
#     The TWISS file EX and EY are already geometric emittances.
#     Beam size formula: σ = sqrt(β * ε_geometric)
#     """
#     sigma_x = np.sqrt(betx * ex)
#     sigma_y = np.sqrt(bety * ey)
#     return sigma_x, sigma_y

# def generate_toys(params, df, n_toys=1000, location='IP1', seed=None):
#     """
#     Generate toy particles using Gaussian distributions.
    
#     Parameters:
#     -----------
#     params : dict
#         Header parameters from TWISS file
#     df : DataFrame
#         TWISS table data
#     n_toys : int
#         Number of toy particles to generate
#     location : str
#         Element name where to generate particles (default: 'IP1')
#     seed : int, optional
#         Random seed for reproducibility
    
#     Returns:
#     --------
#     toys : dict
#         Dictionary with arrays of toy particle coordinates
    
#     Note:
#     -----
#     Uses geometric emittances EX and EY directly from TWISS file.
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     # Get emittances from header (these are geometric emittances)
#     ex = params['EX']
#     ey = params['EY']
#     gamma = params['GAMMA']
    
#     # Print beam parameters
#     print(f"\nBeam parameters:")
#     print(f"  GAMMA = {gamma:.6e}")
#     print(f"  EX (geometric) = {ex:.6e} m·rad")
#     print(f"  EY (geometric) = {ey:.6e} m·rad")
#     print(f"  Normalized εₙₓ = {gamma * ex:.6e} m·rad")
#     print(f"  Normalized εₙᵧ = {gamma * ey:.6e} m·rad")
    
#     # Get Twiss parameters at specified location
#     loc_data = df[df['NAME'] == location].iloc[0]
    
#     betx = loc_data['BETX']
#     bety = loc_data['BETY']
#     alfx = loc_data['ALFX']
#     alfy = loc_data['ALFY']
#     dx = loc_data['DX']
#     dy = loc_data['DY']
#     dpx = loc_data['DPX']
#     dpy = loc_data['DPY']
    
#     # Calculate beam sizes (RMS) using geometric emittances
#     sigma_x, sigma_y = calculate_beam_size(betx, bety, ex, ey, gamma)
    
#     print(f"\nTwiss parameters at {location}:")
#     print(f"  βx = {betx:.6f} m")
#     print(f"  βy = {bety:.6f} m")
#     print(f"  αx = {alfx:.6f}")
#     print(f"  αy = {alfy:.6f}")
#     print(f"  σx = {sigma_x*1e6:.3f} μm")
#     print(f"  σy = {sigma_y*1e6:.3f} μm")
    
#     # Calculate angular divergences
#     gamx = (1 + alfx**2) / betx
#     gamy = (1 + alfy**2) / bety
#     sigma_px = np.sqrt(gamx * ex)
#     sigma_py = np.sqrt(gamy * ey)
    
#     # Momentum spread
#     sige = params['SIGE']
    
#     # Generate uncorrelated Gaussian samples for momentum deviation
#     delta = np.random.normal(0, sige, n_toys)
    
#     # Generate correlated phase space coordinates (x, px) and (y, py)
#     # Using the correlation from alpha parameter
    
#     # Generate independent Gaussian variables
#     u_x = np.random.normal(0, 1, n_toys)
#     u_px = np.random.normal(0, 1, n_toys)
#     u_y = np.random.normal(0, 1, n_toys)
#     u_py = np.random.normal(0, 1, n_toys)
    
#     # Apply correlations from alpha
#     x_betatron = sigma_x * u_x
#     px_betatron = -alfx / betx * sigma_x * u_x + sigma_px * u_px
    
#     y_betatron = sigma_y * u_y
#     py_betatron = -alfy / bety * sigma_y * u_y + sigma_py * u_py
    
#     # Add dispersion contribution
#     x = x_betatron + dx * delta
#     px = px_betatron + dpx * delta
#     y = y_betatron + dy * delta
#     py = py_betatron + dpy * delta
    
#     toys = {
#         'x': x,
#         'px': px,
#         'y': y,
#         'py': py,
#         'delta': delta,
#         's': loc_data['S'],
#         'location': location
#     }
    
#     return toys

# def plot_toys(toys, save_fig=None):
#     """
#     Plot toy particle distributions.
    
#     Parameters:
#     -----------
#     toys : dict
#         Dictionary from generate_toys()
#     save_fig : str, optional
#         Filename to save figure
#     """
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     fig.suptitle(f'Toy Particle Distributions at {toys["location"]} (s={toys["s"]:.2f} m)', 
#                  fontsize=14, fontweight='bold')
    
#     # x-px phase space
#     axes[0, 0].scatter(toys['x']*1e3, toys['px']*1e3, alpha=0.3, s=1)
#     axes[0, 0].set_xlabel('x [mm]')
#     axes[0, 0].set_ylabel("x' [mrad]")
#     axes[0, 0].set_title('Horizontal Phase Space')
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # y-py phase space
#     axes[0, 1].scatter(toys['y']*1e3, toys['py']*1e3, alpha=0.3, s=1)
#     axes[0, 1].set_xlabel('y [mm]')
#     axes[0, 1].set_ylabel("y' [mrad]")
#     axes[0, 1].set_title('Vertical Phase Space')
#     axes[0, 1].grid(True, alpha=0.3)
    
#     # x-y real space
#     axes[0, 2].scatter(toys['x']*1e3, toys['y']*1e3, alpha=0.3, s=1)
#     axes[0, 2].set_xlabel('x [mm]')
#     axes[0, 2].set_ylabel('y [mm]')
#     axes[0, 2].set_title('Real Space')
#     axes[0, 2].grid(True, alpha=0.3)
#     axes[0, 2].axis('equal')
    
#     # x histogram
#     axes[1, 0].hist(toys['x']*1e6, bins=50, alpha=0.7, edgecolor='black')
#     axes[1, 0].set_xlabel('x [μm]')
#     axes[1, 0].set_ylabel('Count')
#     axes[1, 0].set_title(f'x Distribution (σ={np.std(toys["x"])*1e6:.2f} μm)')
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # y histogram
#     axes[1, 1].hist(toys['y']*1e6, bins=50, alpha=0.7, edgecolor='black')
#     axes[1, 1].set_xlabel('y [μm]')
#     axes[1, 1].set_ylabel('Count')
#     axes[1, 1].set_title(f'y Distribution (σ={np.std(toys["y"])*1e6:.2f} μm)')
#     axes[1, 1].grid(True, alpha=0.3)
    
#     # delta histogram
#     axes[1, 2].hist(toys['delta']*1e3, bins=50, alpha=0.7, edgecolor='black')
#     axes[1, 2].set_xlabel('δp/p [×10⁻³]')
#     axes[1, 2].set_ylabel('Count')
#     axes[1, 2].set_title(f'Momentum Spread (σ={np.std(toys["delta"])*1e3:.2f}×10⁻³)')
#     axes[1, 2].grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_fig:
#         plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    
#     plt.show()

# def print_toy_statistics(toys):
#     """Print statistics of toy distributions."""
#     print(f"\n{'='*60}")
#     print(f"Toy Statistics at {toys['location']} (s={toys['s']:.3f} m)")
#     print(f"{'='*60}")
#     print(f"\nNumber of particles: {len(toys['x'])}")
#     print(f"\nHorizontal:")
#     print(f"  σ_x  = {np.std(toys['x'])*1e6:.3f} μm")
#     print(f"  σ_px = {np.std(toys['px'])*1e6:.3f} μrad")
#     print(f"\nVertical:")
#     print(f"  σ_y  = {np.std(toys['y'])*1e6:.3f} μm")
#     print(f"  σ_py = {np.std(toys['py'])*1e6:.3f} μrad")
#     print(f"\nMomentum:")
#     print(f"  σ_δ  = {np.std(toys['delta'])*1e3:.3f} ×10⁻³")
#     print(f"{'='*60}\n")

# if __name__ == '__main__':
#     # Read TWISS file
#     params, df = read_outx_with_headers('twiss_IR_v09.outx')
    
#     # Generate toys at IP1 (interaction point)
#     print("Generating toy particles at IP1...")
#     toys_ip = generate_toys(params, df, n_toys=10000, location='IP1', seed=42)
    
#     # Print statistics
#     print_toy_statistics(toys_ip)
    
#     # Plot distributions
#     plot_toys(toys_ip, save_fig='toys_ip1.png')
    
#     # Example: Generate toys at another location
#     print("\nGenerating toy particles at IQF1...")
#     toys_qf1 = generate_toys(params, df, n_toys=10000, location='IQF1', seed=42)
#     print_toy_statistics(toys_qf1)
#     plot_toys(toys_qf1, save_fig='toys_iqf1.png')

# import xtrack as xt
# import xpart
# import numpy as np

# # Parse the TWISS file
# def parse_madx_twiss(filename):
#     """Parse a MADX TWISS output file (.outx)"""
#     elements = []
    
#     with open(filename, 'r') as f:
#         lines = f.readlines()
    
#     # Find the start of the data (after the * line with column names)
#     data_start = None
#     for i, line in enumerate(lines):
#         if line.startswith('* NAME'):
#             data_start = i + 2  # Data starts 2 lines after column headers
#             break
    
#     if data_start is None:
#         raise ValueError("Could not find data in TWISS file")
    
#     # Parse each element
#     for line in lines[data_start:]:
#         if not line.strip() or line.startswith('@') or line.startswith('*') or line.startswith('$'):
#             continue
        
#         parts = line.split()
#         if len(parts) < 3:
#             continue
        
#         name = parts[0].strip('"')
#         keyword = parts[1].strip('"')
#         s = float(parts[2])
#         length = float(parts[3])
        
#         # Get additional parameters based on element type
#         angle = float(parts[6]) if len(parts) > 6 else 0.0
#         k1l = float(parts[10]) if len(parts) > 10 else 0.0
        
#         elements.append({
#             'name': name,
#             'keyword': keyword,
#             's': s,
#             'length': length,
#             'angle': angle,
#             'k1l': k1l
#         })
    
#     return elements

# # Load and parse the TWISS file
# elements = parse_madx_twiss('twiss_IR_v09.outx')

# # Build xtrack line from elements
# line_elements = []
# element_names = []

# for elem in elements:
#     name = elem['name']
#     keyword = elem['keyword']
#     length = elem['length']
    
#     if keyword == 'DRIFT':
#         line_elements.append(xt.Drift(length=length))
#         element_names.append(name)
    
#     elif keyword == 'QUADRUPOLE':
#         k1 = elem['k1l'] / length if length > 0 else 0.0
#         line_elements.append(xt.Quadrupole(length=length, k1=k1))
#         element_names.append(name)
    
#     elif keyword == 'SBEND':
#         angle = elem['angle']
#         line_elements.append(xt.Bend(length=length, k0=angle/length if length > 0 else 0.0))
#         element_names.append(name)
    
#     elif keyword == 'MARKER':
#         line_elements.append(xt.Marker())
#         element_names.append(name)

# # Create the line
# line = xt.Line(elements=line_elements, element_names=element_names)

# # Set reference particle (from TWISS file header: POSMUON at 5 TeV)
# line.particle_ref = xt.Particles(
#     mass0=0.10565837550000000e9,  # eV
#     q0=1.0,
#     p0c=4999.99999888363072387e9  # eV
# )

# # Build tracker
# line.build_tracker()

# # Generate particles
# particles = xpart.Particles(
#     p0c=5000e9,
#     x=[1e-3, 2e-3, 3e-3],
#     px=[1e-6, 2e-6, 3e-6],
# )

# # Track particles
# line.track(particles, num_turns=100)

# print("Tracking complete!")
# print(f"Final x positions: {particles.x}")

import xtrack as xt
import xpart
import numpy as np
import matplotlib.pyplot as plt

# Parse the TWISS file
def parse_madx_twiss(filename):
    """Parse a MADX TWISS output file (.outx)"""
    elements = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the data (after the * line with column names)
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('* NAME'):
            data_start = i + 2  # Data starts 2 lines after column headers
            break
    
    if data_start is None:
        raise ValueError("Could not find data in TWISS file")
    
    # Parse each element
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('@') or line.startswith('*') or line.startswith('$'):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        name = parts[0].strip('"')
        keyword = parts[1].strip('"')
        s = float(parts[2])
        length = float(parts[3])
        
        # Get additional parameters based on element type
        angle = float(parts[6]) if len(parts) > 6 else 0.0
        k1l = float(parts[10]) if len(parts) > 10 else 0.0
        
        elements.append({
            'name': name,
            'keyword': keyword,
            's': s,
            'length': length,
            'angle': angle,
            'k1l': k1l
        })
    
    return elements

# Load and parse the TWISS file
elements = parse_madx_twiss('twiss_IR_v09.outx')

# Build xtrack line from elements
line_elements = []
element_names = []

for elem in elements:
    name = elem['name']
    keyword = elem['keyword']
    length = elem['length']
    
    if keyword == 'DRIFT':
        line_elements.append(xt.Drift(length=length))
        element_names.append(name)
    
    elif keyword == 'QUADRUPOLE':
        k1 = elem['k1l'] / length if length > 0 else 0.0
        line_elements.append(xt.Quadrupole(length=length, k1=k1))
        element_names.append(name)
    
    elif keyword == 'SBEND':
        angle = elem['angle']
        line_elements.append(xt.Bend(length=length, k0=angle/length if length > 0 else 0.0))
        element_names.append(name)
    
    elif keyword == 'MARKER':
        line_elements.append(xt.Marker())
        element_names.append(name)

# Create the line
line = xt.Line(elements=line_elements, element_names=element_names)

# Set reference particle (from TWISS file header: POSMUON at 5 TeV)
line.particle_ref = xt.Particles(
    mass0=0.10565837550000000e9,  # eV
    q0=1.0,
    p0c=4999.99999888363072387e9  # eV
)

# Build tracker
line.build_tracker()

# Generate Gaussian distributed particles
n_particles = 1000  # Number of particles

particles = xpart.Particles(
    p0c=5000e9,
    x=np.random.normal(0, 1e-3, n_particles),      # Gaussian x with sigma=1mm
    px=np.random.normal(0, 1e-6, n_particles),     # Gaussian px
    y=np.random.normal(0, 1e-3, n_particles),      # Gaussian y with sigma=1mm
    py=np.random.normal(0, 1e-6, n_particles),     # Gaussian py
    zeta=np.random.normal(0, 0.01, n_particles),   # Gaussian z
    delta=np.random.normal(0, 1e-4, n_particles),  # Gaussian momentum deviation
)

# Track particles
line.track(particles, num_turns=100)

# Analyze results
print(f"Particles lost: {np.sum(particles.state <= 0)}")
print(f"Particles surviving: {np.sum(particles.state > 0)}")
print(f"Final x RMS: {np.std(particles.x[particles.state > 0]):.6e} m")

# Plot the distribution
# Filter for surviving particles only
alive = particles.state > 0

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(particles.x[alive], particles.px[alive], s=1, alpha=0.5)
plt.xlabel('x [m]')
plt.ylabel('px [rad]')
plt.title('x-px phase space')

plt.subplot(132)
plt.scatter(particles.y[alive], particles.py[alive], s=1, alpha=0.5)
plt.xlabel('y [m]')
plt.ylabel('py [rad]')
plt.title('y-py phase space')

plt.subplot(133)
plt.hist(particles.x[alive], bins=50)
plt.xlabel('x [m]')
plt.ylabel('Count')
plt.title('x distribution')

plt.tight_layout()
plt.show()