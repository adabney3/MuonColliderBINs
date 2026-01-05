from twiss_plots import read_outx_file, plot_two_columns, plot_multiple_columns
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_outx_with_headers(filename):
    """
    Read MAD-X TWISS output file including header parameters.
    
    Returns:
        tuple: (params_dict, dataframe)
            - params_dict: Dictionary of header parameters from @ lines
            - dataframe: Pandas DataFrame with table data
    """
    params = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract header parameters (@ lines)
    for line in lines:
        if line.startswith('@'):
            parts = line.split()
            if len(parts) >= 3:
                param_name = parts[1]
                param_value = parts[3]
                
                # Try to convert to float if possible
                try:
                    param_value = float(param_value)
                except ValueError:
                    # Keep as string if not a number
                    param_value = param_value.strip('"')
                
                params[param_name] = param_value
    
    # Read the table data (skip @ and * and $ lines)
    data_lines = []
    column_names = None
    
    for line in lines:
        if line.startswith('*'):
            # Column names line
            column_names = line.strip().split()[1:]  # Skip the '*'
        elif line.startswith('$'):
            # Skip format line
            continue
        elif line.startswith('@'):
            # Skip header lines
            continue
        elif line.strip() and not line.startswith('*') and not line.startswith('$'):
            # Data line
            data_lines.append(line.strip())
    
    # Parse data lines
    data = []
    for line in data_lines:
        # Split by whitespace, handling quoted strings
        parts = []
        current = []
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char.isspace() and not in_quotes:
                if current:
                    parts.append(''.join(current))
                    current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        data.append(parts)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=column_names)
    
    # Convert numeric columns
    for col in df.columns:
        if col != 'NAME' and col != 'KEYWORD':
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    
    return params, df

# Read the file with headers
params, df = read_outx_with_headers('twiss_IR_v09.outx')

if df is not None:
    # Get parameters from header
    gamma = params['GAMMA']
    ex = params['EX']
    ey = params['EY']
    
    print(f"Beam parameters from header:")
    print(f"  GAMMA = {gamma:.6e}")
    print(f"  EX = {ex:.6e} m·rad")
    print(f"  EY = {ey:.6e} m·rad")
    
    # Calculate normalized emittance
    norm_ex = gamma * ex
    norm_ey = gamma * ey
    
    print(f"\nNormalized emittances:")
    print(f"  εₙₓ = {norm_ex:.6e} m·rad")
    print(f"  εₙᵧ = {norm_ey:.6e} m·rad")
    
    # Calculate beam size: σ = sqrt(β * ε)
    df['SIGMA_X'] = np.sqrt(df['BETX'] * ex)
    df['SIGMA_Y'] = np.sqrt(df['BETY'] * ey)
    
    # Convert to micrometers for easier reading
    df['SIGMA_X_UM'] = df['SIGMA_X'] * 1e6  # m to μm
    df['SIGMA_Y_UM'] = df['SIGMA_Y'] * 1e6  # m to μm
    
    # Add normalized emittance columns
    df['NORM_EX'] = norm_ex
    df['NORM_EY'] = norm_ey
    
    # Print beam sizes at key locations
    print("\nBeam sizes at key locations:")
    print(df[['NAME', 'S', 'BETX', 'BETY', 'SIGMA_X_UM', 'SIGMA_Y_UM']].head(10))
    
    # Find minimum beam sizes (at IP)
    min_x_idx = df['SIGMA_X'].idxmin()
    min_y_idx = df['SIGMA_Y'].idxmin()
    
    print(f"\nMinimum beam sizes:")
    print(f"  At {df.loc[min_x_idx, 'NAME']}: σₓ = {df.loc[min_x_idx, 'SIGMA_X_UM']:.3f} μm")
    print(f"  At {df.loc[min_y_idx, 'NAME']}: σᵧ = {df.loc[min_y_idx, 'SIGMA_Y_UM']:.3f} μm")
    
    # Plot horizontal beam size
    plot_two_columns(df, 'S', 'SIGMA_X_UM', title='Horizontal Beam Size σₓ [μm] vs S [m]')
    
    # Plot vertical beam size
    plot_two_columns(df, 'S', 'SIGMA_Y_UM', title='Vertical Beam Size σᵧ [μm] vs S [m]')
    
    # Plot both beam sizes together
    plot_multiple_columns(df, 'S', ['SIGMA_X_UM', 'SIGMA_Y_UM'], 
                          title='Beam Sizes along IR1')