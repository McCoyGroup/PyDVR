'''1D Colbert and Miller DVR

Note that we only supply a kinetic energy and a grid as the default in the DVR
class will handle everything else

http://xbeams.chem.yale.edu/~batista/v572/ColbertMiller.pdf
'''

import numpy as np, math

def grid(domain=(-5, 5), divs=10, **kw):
    '''Calculates the grid'''
    rmin=domain[0]; rmax=domain[1];
    inc=(rmax-rmin)/(divs-1)

    return [rmin+i*inc for i in range(divs)]

def kinetic_energy(grid=None, **kw):
    '''Calculates the KE as reported by CM'''
    dx=grid[1]-grid[0]
    divs=len(grid)
    ke=np.empty((divs, divs))

    hb=kw['hb'] if 'hb' in kw else 1
    m=kw['mass'] if 'mass' in kw else 1

    coeff=(hb**2)/(2*m*(dx**2))

    for i in range(divs):
        for j in range(divs):
            if i==j:
                ke[i, j]=(-1**(i-j))*coeff*(math.pi**2)/3
            else:
                ke[i, j]=(-1**(i-j))*coeff*(2)/((i-j)**2)

    return ke
