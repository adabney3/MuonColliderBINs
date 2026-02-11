#in this file, I first parse the MADX .outx file using the function parse_madx_twiss
#the contents of the file are then sorted into two lists (line_elements and element_names) which are parameters for the line class from xtrack
#the reference particle is set according to the information in the header of the .outx file
#then creates an object (particles) from the Particles class from xpart, with Gaussian distributed initial conditions
#the track class tracks the particles through the line for a specified number of turns

import xtrack as xt
import xpart
import numpy as np
import matplotlib.pyplot as plt

def parse_madx_twiss(filename):
    """Parse a MADX TWISS output file (.outx)"""
    elements = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('* NAME'):
            data_start = i + 2  # Data starts 2 lines after column headers
            break
    
    if data_start is None:
        raise ValueError("Could not find data in TWISS file")
    
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('@') or line.startswith('*') or line.startswith('$'):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        name = parts[0].strip('"')
        keyword = parts[1].strip('"')
        s = float(parts[2])
        length = float(parts[3])
        
        angle = float(parts[6]) if len(parts) > 6 else 0.0
        k1l = float(parts[10]) if len(parts) > 10 else 0.0
        
        elements.append({
            'name': name,
            'keyword': keyword,
            's': s,
            'length': length,
            'angle': angle,
            'k1l': k1l
        })
    
    return elements

elements = parse_madx_twiss('twiss_IR_v09.outx')

# Build xtrack line from elements
line_elements = []
element_names = []

for elem in elements:
    name = elem['name']
    keyword = elem['keyword']
    length = elem['length']
    
    if keyword == 'DRIFT':
        line_elements.append(xt.Drift(length=length))
        element_names.append(name)
    
    elif keyword == 'QUADRUPOLE':
        k1 = elem['k1l'] / length if length > 0 else 0.0
        line_elements.append(xt.Quadrupole(length=length, k1=k1))
        element_names.append(name)
    
    elif keyword == 'SBEND':
        angle = elem['angle']
        line_elements.append(xt.Bend(length=length, k0=angle/length if length > 0 else 0.0))
        element_names.append(name)
    
    elif keyword == 'MARKER':
        line_elements.append(xt.Marker())
        element_names.append(name)

line = xt.Line(elements=line_elements, element_names=element_names)

line.particle_ref = xt.Particles(
    mass0=0.10565837550000000e9,  # eV
    q0=1.0,
    p0c=4999.99999888363072387e9  # eV
)

line.build_tracker()

# Generate Gaussian distributed particles
n_particles = 1000  # Number of particles

# particles = xpart.Particles(
#     p0c=5000e9,
#     x=np.random.normal(0, 1e-3, n_particles),      # Gaussian x with sigma=1mm
#     px=np.random.normal(0, 1e-6, n_particles),     # Gaussian px
#     y=np.random.normal(0, 1e-3, n_particles),      # Gaussian y with sigma=1mm
#     py=np.random.normal(0, 1e-6, n_particles),     # Gaussian py
#     zeta=np.random.normal(0, 0.01, n_particles),   # Gaussian z
#     delta=np.random.normal(0, 1e-4, n_particles),  # Gaussian momentum deviation
# )

# generate one six-dimensional gaussian
particles = xpart.generate_matched_gaussian_bunch(
    num_particles = n_particles,
    nemitt_x= 'EX',
    nemitt_y= 'EY',
    sigma_z =  'SIGT',
    line = line,
)

# Track particles
line.track(particles, num_turns=10)

print(f"Particles lost: {np.sum(particles.state <= 0)}")
print(f"Particles surviving: {np.sum(particles.state > 0)}")
print(f"Final x RMS: {np.std(particles.x[particles.state > 0]):.6e} m")

alive = particles.state > 0

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(particles.x[alive], particles.px[alive], s=1, alpha=0.5)
plt.xlabel('x [m]')
plt.ylabel('px [rad]')
plt.title('x-px phase space')

plt.subplot(132)
plt.scatter(particles.y[alive], particles.py[alive], s=1, alpha=0.5)
plt.xlabel('y [m]')
plt.ylabel('py [rad]')
plt.title('y-py phase space')

plt.subplot(133)
plt.hist(particles.x[alive], bins=50)
plt.xlabel('x [m]')
plt.ylabel('Count')
plt.title('x distribution')

plt.tight_layout()
plt.show()