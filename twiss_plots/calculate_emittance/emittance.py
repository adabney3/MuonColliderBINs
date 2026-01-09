#plotting emittance vs s is not necessarily needed as emittance should be conserved
#just wanted to test is emittance is conserved and play with phase space equations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beta_and_alpha import read_outx_file, plot_two_columns, plot_multiple_columns
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from beam_size import read_outx_with_headers, calculate_and_plot_beam_sizes
from phase_advance import compute_phase_advance

def calculate_particle_coordinates(df, params):
    """
    Calculate particle coordinates x(s) and x'(s)
    
    x(s) = sqrt(epsilon) * sqrt(beta(s)) * cos(psi(s))
    x'(s) = -sqrt(epsilon) * (alpha(s) * cos(psi(s)) + sin(psi(s))) / sqrt(beta(s))
    
    """
    epsilon_x = params.get('EX', 0)
    epsilon_y = params.get('EY', 0)
    
    # Calculate for horizontal plane
    x_coords = []
    xp_coords = []
    
    for i, row in df.iterrows():
        beta_x = row['BETX']
        alpha_x = row['ALFX']
        psi_x = row['PSIX']
        
        # Calculate x and x' for each particle at this s position
        x_at_s = np.sqrt(epsilon_x) * np.sqrt(beta_x) * np.cos(psi_x)
        xp_at_s = -np.sqrt(epsilon_x) * (alpha_x * np.cos(psi_x) + 
                                          np.sin(psi_x)) / np.sqrt(beta_x)
        
        x_coords.append(x_at_s)
        xp_coords.append(xp_at_s)
    
    # Calculate for vertical plane
    y_coords = []
    yp_coords = []
    
    for i, row in df.iterrows():
        beta_y = row['BETY']
        alpha_y = row['ALFY']
        psi_y = row['PSIY']
        
        y_at_s = np.sqrt(epsilon_y) * np.sqrt(beta_y) * np.cos(psi_y)
        yp_at_s = -np.sqrt(epsilon_y) * (alpha_y * np.cos(psi_y) + 
                                          np.sin(psi_y)) / np.sqrt(beta_y)
        
        y_coords.append(y_at_s)
        yp_coords.append(yp_at_s)
    
    return {
        'x': np.array(x_coords),
        'xp': np.array(xp_coords),
        'y': np.array(y_coords),
        'yp': np.array(yp_coords),
    }

def calculate_emittance_vs_s(df, params):
    """
    Calculate emittance at each s position to verify it's conserved.
    
    For each particle: ε = γ(s)*x²(s) + 2*α(s)*x(s)*x'(s) + β(s)*x'²(s)
    where γ = (1 + α²)/β
    
    Returns the mean emittance at each s position (should be constant = EX or EY)
    """
    # Get particle coordinates
    coords = calculate_particle_coordinates(df, params)
    
    # Calculate emittance for horizontal plane at each s
    epsilon_x_at_s = []
    epsilon_y_at_s = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        # Horizontal plane
        beta_x = row['BETX']
        alpha_x = row['ALFX']
        gamma_x = (1 + alpha_x**2) / beta_x
        
        # Get all particle coordinates at this s
        x = coords['x'][i, :]
        xp = coords['xp'][i, :]
        
        # Calculate emittance for each particle
        eps_x = gamma_x * x**2 + 2 * alpha_x * x * xp + beta_x * xp**2
        
        # Store mean emittance at this s
        epsilon_x_at_s.append(np.mean(eps_x))
        
        # Vertical plane
        beta_y = row['BETY']
        alpha_y = row['ALFY']
        gamma_y = (1 + alpha_y**2) / beta_y
        
        y = coords['y'][i, :]
        yp = coords['yp'][i, :]
        
        eps_y = gamma_y * y**2 + 2 * alpha_y * y * yp + beta_y * yp**2
        epsilon_y_at_s.append(np.mean(eps_y))
    
    # Add to dataframe
    df['EPSILON_X_CALC'] = epsilon_x_at_s
    df['EPSILON_Y_CALC'] = epsilon_y_at_s
    
    return df

def plot_emittance_vs_s(df, params):
    """Plot calculated emittance vs s to verify conservation"""
    
    epsilon_x_header = params.get('EX', 0)
    epsilon_y_header = params.get('EY', 0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Horizontal emittance
    ax1.plot(df['S'], df['EPSILON_X_CALC'] * 1e6, 'b-', linewidth=2, label='Calculated εₓ')
    ax1.axhline(y=epsilon_x_header * 1e6, color='r', linestyle='--', linewidth=2, 
                label=f'Header value εₓ = {epsilon_x_header*1e6:.3f} μm')
    ax1.set_ylabel('Horizontal Emittance εₓ [μm]')
    ax1.set_title('Emittance Conservation Check')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Vertical emittance
    ax2.plot(df['S'], df['EPSILON_Y_CALC'] * 1e6, 'b-', linewidth=2, label='Calculated εᵧ')
    ax2.axhline(y=epsilon_y_header * 1e6, color='r', linestyle='--', linewidth=2,
                label=f'Header value εᵧ = {epsilon_y_header*1e6:.3f} μm')
    ax2.set_xlabel('s [m]')
    ax2.set_ylabel('Vertical Emittance εᵧ [μm]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

# def plot_emittance_evolution(df, params, filename='twiss_IR_v09.outx'):
#     """Plot emittance ellipses at different s positions"""
    
#     # Select a few positions to plot
#     n_plots = 6
#     indices = np.linspace(0, len(df)-1, n_plots, dtype=int)
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()
    
#     epsilon_x = params.get('EX', 0)
    
#     # Generate particle coordinates
#     coords = calculate_particle_coordinates(df, params, n_particles=100)
    
#     for idx, ax_idx in enumerate(indices):
#         row = df.iloc[ax_idx]
#         s_pos = row['S']
#         name = row['NAME']
        
#         # Get coordinates at this position
#         x = coords['x'][ax_idx, :]
#         xp = coords['xp'][ax_idx, :]
        
#         # Plot particles
#         axes[idx].plot(x * 1e3, xp * 1e3, 'b.', markersize=3, alpha=0.6)
#         axes[idx].set_xlabel('x [mm]')
#         axes[idx].set_ylabel("x' [mrad]")
#         axes[idx].set_title(f's = {s_pos:.2f} m\n{name}')
#         axes[idx].grid(True, alpha=0.3)
#         axes[idx].axis('equal')
        
#         # Add emittance ellipse
#         beta_x = row['BETX']
#         alpha_x = row['ALFX']
#         gamma_x = (1 + alpha_x**2) / beta_x
        
#         # Ellipse parameters
#         width = 2 * np.sqrt(epsilon_x * beta_x) * 1e3  # mm
#         height = 2 * np.sqrt(epsilon_x * gamma_x) * 1e3  # mrad
        
#         # Rotation angle
#         if beta_x != gamma_x:
#             angle = np.degrees(0.5 * np.arctan2(-2*alpha_x, gamma_x - beta_x))
#         else:
#             angle = 0
        
#         ellipse = Ellipse((0, 0), width, height, angle=angle,
#                          fill=False, edgecolor='red', linewidth=2)
#         axes[idx].add_patch(ellipse)
        
#         # Add text with Twiss parameters
#         text = f'β={beta_x:.1f} m\nα={alpha_x:.2f}\nε={epsilon_x*1e6:.2f} μm'
#         axes[idx].text(0.05, 0.95, text, transform=axes[idx].transAxes,
#                       verticalalignment='top', fontsize=8,
#                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    filename = 'twiss_IR_v09.outx'
    
    params, df = read_outx_with_headers(filename)
    
    df = compute_phase_advance(df)
    
    df = calculate_emittance_vs_s(df, params)
    
    plot_emittance_vs_s(df, params)
    
    #plot_emittance_evolution(df, params, filename)