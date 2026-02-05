from beta_and_alpha import read_outx_file, plot_single_column, plot_two_columns, plot_multiple_columns
import pandas as pd
import matplotlib.pyplot as plt
import math

df = read_outx_file('twiss_IR_v09.outx')

plot_two_columns(df, 'S', 'DX', 'Dispersion X vs S')


plot_two_columns(df, 'S', 'DY', 'Dispersion Y vs S')


plot_multiple_columns(df, 'S', ['DX', 'DY'], 'Dispersion vs S')
