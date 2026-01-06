import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beam_size import read_outx_with_headers

def compute_phase_advance(df):
    """
    Compute phase advance (psi) in both planes
    Phase advance: ψ = ∫ ds/β
    """
    # Initialize phase advance arrays
    psix = np.zeros(len(df))
    psiy = np.zeros(len(df))
    
    # Compute phase advance by numerical integration
    for i in range(1, len(df)):
        # Distance between elements
        ds = df['S'].iloc[i] - df['S'].iloc[i-1]
        
        # Average beta function between two points
        betx_avg = (df['BETX'].iloc[i] + df['BETX'].iloc[i-1]) / 2
        bety_avg = (df['BETY'].iloc[i] + df['BETY'].iloc[i-1]) / 2
        
        # Integrate: Δψ = Δs / β_avg
        psix[i] = psix[i-1] + ds / betx_avg
        psiy[i] = psiy[i-1] + ds / bety_avg
    
    # Convert to degrees and modulo 360
    psix_deg = np.mod(np.degrees(psix), 360)
    psiy_deg = np.mod(np.degrees(psiy), 360)
    
    # Add to dataframe
    df['PSIX'] = psix
    df['PSIY'] = psiy
    df['PSIX_DEG'] = psix_deg
    df['PSIY_DEG'] = psiy_deg
    
    return df

def plot_results(df):
    """Plot beta functions and phase advance"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Beta functions
    axes[0, 0].plot(df['S'], df['BETX'], 'b-', label='βx')
    axes[0, 0].set_xlabel('s [m]')
    axes[0, 0].set_ylabel('βx [m]')
    axes[0, 0].set_title('Horizontal Beta Function')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].plot(df['S'], df['BETY'], 'r-', label='βy')
    axes[0, 1].set_xlabel('s [m]')
    axes[0, 1].set_ylabel('βy [m]')
    axes[0, 1].set_title('Vertical Beta Function')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Phase advance in radians
    axes[1, 0].plot(df['S'], df['PSIX'], 'b-', label='ψx')
    axes[1, 0].set_xlabel('s [m]')
    axes[1, 0].set_ylabel('ψx [rad]')
    axes[1, 0].set_title('Horizontal Phase Advance')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].plot(df['S'], df['PSIY'], 'r-', label='ψy')
    axes[1, 1].set_xlabel('s [m]')
    axes[1, 1].set_ylabel('ψy [rad]')
    axes[1, 1].set_title('Vertical Phase Advance')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('phase_advance.png', dpi=300)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Parse the TWISS file using the existing function
    print("Reading TWISS data...")
    params, df = read_outx_with_headers('twiss_IR_v09.outx')
    
    # Display some header parameters
    print("\nTWISS Parameters:")
    print(f"  Particle: {params.get('PARTICLE', 'N/A')}")
    print(f"  Energy: {params.get('ENERGY', 'N/A')} GeV")
    print(f"  Q1 (from header): {params.get('Q1', 'N/A')}")
    print(f"  Q2 (from header): {params.get('Q2', 'N/A')}")
    print(f"  Sequence length: {params.get('LENGTH', 'N/A')} m")
    
    # Compute phase advance
    print("\nComputing phase advance...")
    df = compute_phase_advance(df)
    
    # Display results at key locations
    print("\nPhase advance at key locations:")
    print("="*80)
    print(f"{'Element':<20} {'S [m]':<12} {'ψx [rad]':<12} {'ψx [°]':<12} {'ψy [rad]':<12} {'ψy [°]':<12}")
    print("="*80)
    
    key_elements = ['IR1$START', 'IP1', 'IR1$END']
    for elem in key_elements:
        row = df[df['NAME'] == elem]
        if not row.empty:
            idx = row.index[0]
            print(f"{elem:<20} {df['S'].iloc[idx]:>11.3f} "
                  f"{df['PSIX'].iloc[idx]:>11.6f} {df['PSIX_DEG'].iloc[idx]:>11.3f} "
                  f"{df['PSIY'].iloc[idx]:>11.6f} {df['PSIY_DEG'].iloc[idx]:>11.3f}")
    
    # Total phase advance
    print("\n" + "="*80)
    print(f"Total horizontal phase advance: {df['PSIX'].iloc[-1]:.6f} rad = {df['PSIX_DEG'].iloc[-1]:.3f}°")
    print(f"Total vertical phase advance:   {df['PSIY'].iloc[-1]:.6f} rad = {df['PSIY_DEG'].iloc[-1]:.3f}°")
    print(f"Horizontal tune (Qx):           {df['PSIX'].iloc[-1]/(2*np.pi):.6f}")
    print(f"Vertical tune (Qy):             {df['PSIY'].iloc[-1]/(2*np.pi):.6f}")
    
    plot_results(df)