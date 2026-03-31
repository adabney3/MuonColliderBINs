import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

def get_twiss_parameters(filename):
    with open(filename, 'r') as f:
            lines = f.readlines()
    
    header_line = None
    data_start = None
    for i, line in enumerate(lines):
            if line.startswith('*'):
                header_line = line
            elif line.startswith('$'):
                data_start = i + 1
                break
    
    col_names = header_line.strip().replace('*', '').split()

    data = pd.read_csv(filename, 
                          #delim_whitespace=True,
                          sep=r'\s+',
                          skiprows=data_start,
                          names=col_names,
                          header=None)
    
    for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str).str.strip('"')
    
    numeric_cols = [col for col in data.columns if col not in ['NAME', 'KEYWORD']]
    for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

df = get_twiss_parameters('twiss_IR_v09.outx')

def phase_ellipse(beta, alpha, emittance, n = 300):
      gamma = (1 + alpha**2) / beta
      phi = np.linspace(0, 2 * np.pi, n)
      u = np.sqrt(emittance * beta) * np.cos(phi)
      pu = -(alpha / np.sqrt(beta)) * np.cos(phi) - np.sqrt(gamma) * np.sin(phi)
      pu *= np.sqrt(emittance)
      return u, pu

def plot_single_ellipse(df, s_value, emittance=5.283e-10, plane='both'):
    df_phys = df[df['BETX'] > 0].copy()
    row = df_phys.iloc[(df_phys['S'] - s_value).abs().argmin()]

    planes = ['x', 'y'] if plane == 'both' else [plane]
    fig, axes = plt.subplots(1, len(planes), figsize=(6 * len(planes), 5))
    if len(planes) == 1:
        axes = [axes]

    for ax, pl in zip(axes, planes):
        beta_col = 'BETX' if pl == 'x' else 'BETY'
        alpha_col = 'ALFX' if pl == 'x' else 'ALFY'

        beta  = row[beta_col]
        alpha = row[alpha_col]
        u, pu = phase_ellipse(beta, alpha, emittance)

        ax.plot(u, pu, lw=2, color='steelblue')
        ax.fill(u, pu, alpha=0.15, color='steelblue')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')

        # Pad limits so ellipse doesn't touch the edges
        u_pad  = (u.max()  - u.min())  * 0.1
        pu_pad = (pu.max() - pu.min()) * 0.1
        ax.set_xlim(u.min()  - u_pad,  u.max()  + u_pad)
        ax.set_ylim(pu.min() - pu_pad, pu.max() + pu_pad)

        ax.set_xlabel('x (m)' if pl == 'x' else 'y (m)')
        ax.set_ylabel("x' (rad)" if pl == 'x' else "y' (rad)")
        ax.set_title(f"{pl}-plane  |  {row['NAME']}  |  s = {row['S']:.2f} m")
        ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')  # consistent sci notation
    fig.tight_layout()
    return fig

fig = plot_single_ellipse(df, s_value=5.0, emittance=5.283e-10, plane='both')
fig.savefig('ellipse.png', dpi=150, bbox_inches='tight')