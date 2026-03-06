import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#I chose the smallest values of beta and alpha from the twiss file (line 73) because it produced the best ellipse
#I tested with other values as well but larger beta and alpha values produced straight lines, not ellipses

beta  =  0.00149999999999999 #semi-major axis
#beta = 17057.11360944676198415 #testing with first values of beta and alpha in file; however, these values gave me a straight line not ellipse
#beta = 409827.87423663376830518
#beta = 722573.20524224289692938
alpha =  0.00000000001248069 #alpha relates to tilt
#alpha = -784.32981445063455794
#alpha = 29970.80558916701193084
#alpha = -7126.45070737356400059
gamma = (1 + alpha**2) / beta  #semi-minor axis
x = 0
y = 0
epsilon = 0.00000000052830000  #emmittance (EX) found in header of file

theta   = np.linspace(0, 2 * np.pi, 500)
x       =  np.sqrt(epsilon * beta) * np.cos(theta)
x_prime = -np.sqrt(epsilon / beta) * (alpha * np.cos(theta) + np.sin(theta))

plt.figure(figsize=(8, 6))
plt.plot(x, x_prime, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel("x [m]")
plt.ylabel("x' [rad]")
plt.title(f"Phase Space Ellipse (β={beta:.4f} m, α={alpha:.4e})")
plt.tight_layout()
plt.show()