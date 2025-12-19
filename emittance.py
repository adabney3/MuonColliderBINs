from twiss_plots import read_outx_file, plot_two_columns, plot_multiple_columns
import math

df = read_outx_file('twiss_IR_v09.outx')

if df is not None:
    print("CHECKING X AND Y VALUES:")
    print(f"X column - Min: {df['X'].min()}, Max: {df['X'].max()}, Mean: {df['X'].mean()}")
    print(f"Y column - Min: {df['Y'].min()}, Max: {df['Y'].max()}, Mean: {df['Y'].mean()}")
    print(f"\nFirst 10 X values: {df['X'].head(10).tolist()}")
    print(f"First 10 Y values: {df['Y'].head(10).tolist()}")
    
    #horizontal emittance at each point: EX = X^2 / BETX
    df['EX_calc'] = (df['X']**2) / df['BETX']
    
    #vertical emittance at each point: EY = Y^2 * BETY
    df['EY_calc'] = (df['Y']**2) * df['BETY']
    
    print("CALCULATED EMITTANCES (at each S position):")
    print(f"EX_calc statistics (m·rad):")
    print(df['EX_calc'].describe())
    print(f"\nEY_calc statistics (m·rad):")
    print(df['EY_calc'].describe())
    
    #some sample calculations
    print("SAMPLE CALCULATIONS (first 5 points):")
    print(df[['S', 'X', 'BETX', 'EX_calc', 'Y', 'BETY', 'EY_calc']].head())

#plot calculated emittances
plot_two_columns(df, 'S', 'EX_calc', 'Horizontal Emittance vs S')
plot_two_columns(df, 'S', 'EY_calc', 'Vertical Emittance vs S')
plot_multiple_columns(df, 'S', ['EX_calc', 'EY_calc'], 'Emittances vs S')