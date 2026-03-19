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

def plot_ellipses_at_s(df, s_values=None, emittance=None, plane='x'):
    df_phys = df[df['BETX']>0].copy()

    if s_values is not None:
            rows = [df_phys.iloc[(df_phys['S'] - s).abs().argmin()] for s in s_values]
    else:
            rows = [df_phys.iloc[i] for i in range(len(df_phys))]

    planes = ['x', 'y'] if plane == 'both' else [plane]
    fig, axes = plt.subplots(1, len(planes), figsize = (6 * len(planes), 5))
    if len(planes) == 1:
          axes = [axes]
    
    cmap = plt.cm.viridis
    s_min = df_phys['S'].min()
    s_max = df_phys['S'].max()

    for ax, pl in zip(axes, planes):
        beta_col = 'BETX' if pl == 'x' else 'BETY'
        alpha_col = 'ALFX' if pl == 'x' else 'ALFY'
        coord = 'x (m)' if pl == 'x' else 'y (m)'
        mom = "x' (rad)" if pl == 'x' else "y' (rad)"

        for row in rows:
            beta = row[beta_col]
            alpha = row[alpha_col]
            eps = emittance if emittance is not None else 5.283e-10

            color = cmap((row['S'] - s_min) / (s_max - s_min))
            u, pu = phase_ellipse(beta, alpha, eps)
            ax.plot(u, pu, color = color, lw = 1.2, alpha = 0.75)

        sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = s_min, vmax = s_max))

        sm.set_array([])
        plt.colorbar(sm, ax = ax, label = 's (m)')

        ax.axhline(0, color = 'gray', lw = 0.5)
        ax.axvline(0, color = 'gray', lw = 0.5)
        ax.set_xlabel(coord)
        ax.set_ylabel(mom)
        ax.set_title(f'{pl}-plane phase ellipses along IR1')
        ax.set_aspect('equal', 'datalim')

    fig.tight_layout()
    return fig

def animate_ellipses(df, emittance = None, interval = 120, plane = 'both'):
    df_phys = df[(df['BETX']>0) & (df['BETY']>0)].reset_index(drop = True)
    planes = ['x', 'y'] if plane == 'both' else [plane]
    eps = emittance if emittance is not None else 5.283e-10

    fig, axes = plt.subplots(1, len(planes), figsize = (6 * len(planes), 5))
    if len(planes) == 1:
            axes = [axes]
    
    lines, fills = [], []
    for ax, pl in zip(axes, planes):
          line, = ax.plot([], [], lw = 2)
          fill = ax.fill([], [], alpha = 0.15)[0]
          lines.append(line)
          fills.append(fill)
          ax.axhline(0, color = 'gray', lw = 0.5)
          ax.axvline(0, color = 'gray', lw = 0.5)
          ax.set_xlabel('x (m)' if pl == 'x' else 'y (m)')
          ax.set_ylabel("x' (rad)" if pl == 'x' else "y' (rad)")
    
    title = fig.suptitle('')

    for ax, pl in zip(axes, planes):
        beta_col = 'BETX' if pl == 'x' else 'BETY'
        alpha_col = 'ALFX' if pl == 'x' else 'ALFY'
        all_u, all_pu = [], []
        for _, row in df_phys.iterrows():
            u, pu = phase_ellipse(row[beta_col], row[alpha_col], eps)
            all_u.extend(u); all_pu.extend(pu)
        pad = 0.1
        umax = max(abs(v) for v in all_u) * (1 + pad)
        pumax = max(abs(v) for v in all_pu) * (1 + pad)
        ax.set_xlim(-umax, umax)
        ax.set_ylim(-pumax, pumax)
        ax.set_aspect('equal', 'datalim')
    
    def update(frame):
        row = df_phys.iloc[frame]
        for ax, line, fill, pl in zip(axes, lines, fills, planes):
              beta_col = 'BETX' if pl == 'x' else 'BETY'
              alpha_col = 'ALFX' if pl == 'x' else 'ALFY'
              u, pu = phase_ellipse(row[beta_col], row[alpha_col], eps)
              line.set_data(u, pu)
              fill.set_xy(np.column_stack([u, pu]))
        title.set_text(f"{row['NAME']} s = {row['S']:.2f} m")
        return lines + fills + [title]
    
    ani = animation.FuncAnimation(fig, update, frames = len(df_phys), interval = interval, blit = True)
    fig.tight_layout()
    return ani

df = get_twiss_parameters('twiss_IR_v09.outx')

fig = plot_ellipses_at_s(df, emittance = 5.283e-10, plane = 'both')
plt.show()

ani = animate_ellipses(df, emittance = 5.283e-10, interval = 120, plane = 'both')
plt.show()
ani.save('phase_ellipses.gif', writer = 'pillow', fps = 8)
