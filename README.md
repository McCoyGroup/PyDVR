# PyDVR
A sample, class-based, python DVR implementation.

This implementation also employs the use of the plotting utilities in McUtils and wavefunction class in Psience. 
All of which are available elsewhere in the McCoy Group github. In addition to these packages, PyDVR also extensively uses numpy, 
scipy, and a little matplotlib (for now).    

Currently PyDVR only handles nD DVR as described by Colbert and Miller (see References for paper and algorithm), but is
structured in such a way that it can be extended to other algorithms. 

This implementation has a lot of features for interpreting and visualizing the results that are still in progress. 
To get started, if you wanted to run a 1D DVR using a harmonic oscillator potential function:
```python
from PyDVR.DVR import * 
def ho(grid, k=1):
    return k / 2 * np.power(grid, 2)
    
dvr_1D = DVR("ColbertMiller1D")
res = dvr_1D.run(potential_function=ho, mass=1, divs=10, domain=(-4, 4), num_wfns=5)
```
Here we have set our potential function to a simple harmonic oscillator, set the ```mass``` of our system to 5, 
created a grid with 10 ```divs``` or grid points, our grid has a ```domain``` of (-4, 4) and our results object will hold 
the five lowest energy wavefunctions and energies (```num_wfns```) NOTE: all values are treated as if they are in ATOMIC UNITS.  

To move past 1D, the base call changes according to the following:
```python
from PyDVR.DVR import * 
def ho_2D(grid, k1=1, k2=1):
    return k1 / 2 * np.power(grid[:, 0], 2) + k2 / 2 * np.power(grid[:, 1], 2)
dvr_2D = DVR("ColbertMillerND")
res = dvr_2D.run(potential_function=ho_2D, divs=(10, 10), domain=((-4, 4), (-4, 4)), num_wfns=5)
``` 
Notice that for every dimension added, ```divs``` will need another ```int``` value, and domain will take another ```tuple```.

After making this call, the variable ```res``` is a subclass of the DVR that holds all the results in a way that is easy to access.
Values currently held are: 
```python
grid = res.grid
# nd array of grid points determined by domain and divs. 
kinetic_energy = res.kinetic_energy
# nd array of the kinetic energy operator.
potential_energy = res.potential_energy
# nd array of the potential energy operator (full matrix).
potential_energy = res.potential_energy.diagonal()
# potential energy values, only matrix diagonal. 
wavefunction_class = res.wavefunctions
# wavefunction_class is its own subclass as defined by Wavefun in Psience. 
wavefunctions = wavefunction_class.data
# wavefunctions are oriented column-wise in the nd array stored here.
energies = wavefunction_class.energies
# nd array of energies for the provided number of wavefunctions. 
```

