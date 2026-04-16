import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from one_ellipse import get_twiss_parameters

df = get_twiss_parameters('../twiss_IR_v09.outx')

def interpolate_twiss(df, param, s_query):
    return np.interp(s_query, df['S'].values, df[param].values)

s_fine = np.linspace(df['S'].min(), df['S'].max(), 300)

params = {
    'BETX': interpolate_twiss(df, 'BETX', s_fine),
    'BETY': interpolate_twiss(df, 'BETY', s_fine),
    'ALFX': interpolate_twiss(df, 'ALFX', s_fine),
    'ALFY': interpolate_twiss(df, 'ALFY', s_fine),
}

print("s range:", s_fine[0], "to", s_fine[-1])
print("BETX range:", params['BETX'].min(), "to", params['BETX'].max())
print("BETY range:", params['BETY'].min(), "to", params['BETY'].max())

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(s_fine, params['BETX'], label='BETX interp', color='steelblue')
axes[0].plot(s_fine, params['BETY'], label='BETY interp', color='tomato')
axes[0].scatter(df['S'], df['BETX'], color='steelblue', s=20, zorder=5, label='BETX original')
axes[0].scatter(df['S'], df['BETY'], color='tomato', s=20, zorder=5, label='BETY original')
axes[0].set_ylabel('Beta function (m)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(s_fine, params['ALFX'], label='ALFX interp', color='steelblue')
axes[1].plot(s_fine, params['ALFY'], label='ALFY interp', color='tomato')
axes[1].scatter(df['S'], df['ALFX'], color='steelblue', s=20, zorder=5, label='ALFX original')
axes[1].scatter(df['S'], df['ALFY'], color='tomato', s=20, zorder=5, label='ALFY original')
axes[1].set_ylabel('Alpha functions (m)')
axes[1].set_xlabel('s (m)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('interpolation_check.png')
plt.show()