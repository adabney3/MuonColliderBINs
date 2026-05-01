import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from one_ellipse import get_twiss_parameters, phase_ellipse
from test_interpolation import interpolate_twiss
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# def animate(i):
#     plt.clf()
#     s = s_fine[i]
    
#     beta_x  = params['BETX'][i]
#     alpha_x = params['ALFX'][i]
#     beta_y  = params['BETY'][i]
#     alpha_y = params['ALFY'][i]

#     ux, pux = phase_ellipse(beta_x, alpha_x, emittance)
#     uy, puy = phase_ellipse(beta_y, alpha_y, emittance)

#     ax_x = plt.subplot(1, 2, 1)
#     ax_x.plot(ux, pux, lw=2, color='steelblue')
#     ax_x.fill(ux, pux, alpha=0.15, color='steelblue')
#     ax_x.axhline(0, color='gray', lw=0.5, ls='--')
#     ax_x.axvline(0, color='gray', lw=0.5, ls='--')
#     ax_x.set_xlabel('x (m)')
#     ax_x.set_ylabel("x' (rad)")
#     ax_x.set_title(f'x-plane  |  s = {s:.2f} m')
#     ax_x.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')

#     ax_y = plt.subplot(1, 2, 2)
#     ax_y.plot(uy, puy, lw=2, color='steelblue')
#     ax_y.fill(uy, puy, alpha=0.15, color='steelblue')
#     ax_y.axhline(0, color='gray', lw=0.5, ls='--')
#     ax_y.axvline(0, color='gray', lw=0.5, ls='--')
#     ax_y.set_xlabel('y (m)')
#     ax_y.set_ylabel("y' (rad)")
#     ax_y.set_title(f'y-plane  |  s = {s:.2f} m')
#     ax_y.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')

#     plt.tight_layout()

df = get_twiss_parameters('../twiss_IR_v09.outx')
df_phys = df[df['BETX'] > 0].copy()  # drop markers with BETX=0

s_fine = np.linspace(df_phys['S'].min(), df_phys['S'].max(), 300)

params = {
    'BETX': interpolate_twiss(df_phys, 'BETX', s_fine),
    'BETY': interpolate_twiss(df_phys, 'BETY', s_fine),
    'ALFX': interpolate_twiss(df_phys, 'ALFX', s_fine),
    'ALFY': interpolate_twiss(df_phys, 'ALFY', s_fine),
}
emittance = 5.283e-10

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_x, ax_y = axes 

all_ux, all_pux, all_uy, all_puy = [], [], [], []
for i in range(len(s_fine)):
    ux, pux = phase_ellipse(params['BETX'][i], params['ALFX'][i], emittance)
    uy, puy = phase_ellipse(params['BETY'][i], params['ALFY'][i], emittance)
    all_ux.extend(ux); all_pux.extend(pux)
    all_uy.extend(uy); all_puy.extend(puy)

xlim_x = (min(all_ux),  max(all_ux))
ylim_x = (min(all_pux), max(all_pux))
xlim_y = (min(all_uy),  max(all_uy))
ylim_y = (min(all_puy), max(all_puy))

ax_x.set_xlim(xlim_x)
ax_x.set_ylim(ylim_x)
ax_x.set_autoscale_on(False)

ax_y.set_xlim(xlim_y)
ax_y.set_ylim(ylim_y)
ax_y.set_autoscale_on(False)

plt.tight_layout()

def animate(i):
    ax_x.cla() 
    ax_y.cla()

    s = s_fine[i]

    ux, pux = phase_ellipse(params['BETX'][i], params['ALFX'][i], emittance)
    uy, puy = phase_ellipse(params['BETY'][i], params['ALFY'][i], emittance)

    ax_x.plot(ux, pux, lw=2, color='steelblue')
    ax_x.fill(ux, pux, alpha=0.15, color='steelblue')
    ax_x.axhline(0, color='gray', lw=0.5, ls='--')
    ax_x.axvline(0, color='gray', lw=0.5, ls='--')
    ax_x.set_xlabel('x (m)')
    ax_x.set_ylabel("x' (rad)")
    ax_x.set_title(f'x-plane  |  s = {s:.2f} m')
    ax_x.set_xlim(xlim_x)
    ax_x.set_ylim(ylim_x)
    ax_x.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')

    ax_y.plot(uy, puy, lw=2, color='steelblue')
    ax_y.fill(uy, puy, alpha=0.15, color='steelblue')
    ax_y.axhline(0, color='gray', lw=0.5, ls='--')
    ax_y.axvline(0, color='gray', lw=0.5, ls='--')
    ax_y.set_xlabel('y (m)')
    ax_y.set_ylabel("y' (rad)")
    ax_y.set_title(f'y-plane  |  s = {s:.2f} m')
    ax_y.set_xlim(xlim_y)
    ax_y.set_ylim(ylim_y)
    ax_y.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')

    plt.tight_layout()

ani = FuncAnimation(fig, animate, frames=len(s_fine), interval=100, repeat=True)
ani.save('phase_space_animation.gif', writer='pillow')
writer = FFMpegWriter(fps=10, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
ani.save('phase_space_animation.mp4', writer=writer)
print('saved!')
