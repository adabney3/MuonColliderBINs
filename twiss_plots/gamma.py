from beta_and_alpha import read_outx_file, plot_single_column, plot_two_columns, plot_multiple_columns
import pandas as pd
import matplotlib.pyplot as plt
import math

df = read_outx_file('twiss_IR_v09.outx')

if df is not None:

    #calculate gamma
    df['GAMMAX_CALC'] = (1 + df['ALFX']**2) / df['BETX']
    df['GAMMAY_CALC'] = (1 + df['ALFY']**2) / df['BETY']

#plot calculated gamma functions
plot_two_columns(df, 'S', 'GAMMAX_CALC', 'Calculated Gamma X vs S')
plot_two_columns(df, 'S', 'GAMMAY_CALC', 'Calculated Gamma Y vs S')
plot_multiple_columns(df, 'S', ['GAMMAX_CALC', 'GAMMAY_CALC'], 'Calculated Gamma Functions vs S')