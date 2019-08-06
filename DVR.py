import os


class DVR:

    '''This is a manager class for working with DVRs

It uses files from which it loads the spec data and methods
Currently all defaults are for 1D but the ND extension shouldn't be bad
'''

    loaded_DVRs = {}  # for storing DVRs loaded from file
    dvr_dir = os.path.join(os.path.dirname(__file__), "Classes")

    def __init__(self, dvr_file="ColbertMiller1D", **kwargs):

        self.params = kwargs  # these are the global parameters passed to all methods
        if dvr_file is None:
            self._dvr = None
        else:
            self._dvr = self.load_dvr(dvr_file)

    @classmethod
    def dvr_file(self, dvr):
        '''Locates the DVR file for dvr'''

        if os.path.exists(dvr):
            dvr_file = dvr
        else:
            dvr_file = os.path.join(self.dvr_dir, dvr+".py")

        if not os.path.exists(dvr_file):
            raise DVRException("couldn't load DVR "+dvr)

        return dvr_file

    @classmethod
    def _load_env(cls, file, env):
        '''Fills in necessary parameters in an env from a module

        :param env:
        :type env:
        :return:
        :rtype:
        '''

        cls.loaded_DVRs[file] = env
        for k in ('grid', 'kinetic_energy', 'potential_energy', 'hamiltonian', 'wavefunctions'):
            if 'grid' not in env:
                raise DVRException("{}.{}: DVR class '{}' didn't export property '{}' (in file {})".format(
                    cls.__name__,
                    'load_dvr',
                    env['class'],
                    k,
                    env['file']
                ))

        return env

    @classmethod
    def load_dvr(self, dvr):
        '''Loads a DVR from file into the DVR class'''

        dvr_file = self.dvr_file(dvr)
        if dvr_file not in self.loaded_DVRs:
            # defaults and parameters passed as global
            dvr_class = os.path.splitext(os.path.basename(dvr_file))[0]
            _load_env = {
                'DVR': self,
                'class': dvr_class,
                'file': dvr_file,
                'potential_energy': self._potential_energy,
                'hamiltonian': self._hamiltonian,
                'wavefunctions': self._wavefunctions
                }

            import sys

            dvr_dir = os.path.dirname(dvr_file)
            sys.path.insert(0, os.path.dirname(dvr_dir))
            sys.path.insert(0, dvr_dir)
            try:
                exec("from .Classes.{} import *".format(dvr_class), globals(), _load_env)
            except ImportError:
                exec("from .Classes.{} import *".format(dvr_class), globals(), _load_env)
            finally:  # want to preserve error handling but still close the file
                sys.path.pop(0)

            load = self._load_env(dvr_file, _load_env)

        else:

            load = self.loaded_DVRs[dvr_file]

        return load

    def _dvr_prop(self, prop_name):
        '''Gets a DVR property'''

        if prop_name in self.params:
            prop = self.params[prop_name]
        elif prop_name in self._dvr:
            prop = self._dvr[prop_name]
            if callable(prop):
                prop = prop(**self.params)
        else:
            raise DVRException("no property "+prop_name)

        return prop

    def domain(self):
        """Computes the domain for the DVR"""
        return self._dvr_prop('domain')

    def divs(self):
        """Computes the divisions for the DVR"""
        return self._dvr_prop('divs')

    def grid(self):
        """Computes the grid for the DVR"""
        return self._dvr_prop('grid')

    def kinetic_energy(self):
        """Computes the kinetic_energy for the DVR"""
        return self._dvr_prop('kinetic_energy')

    def potential_energy(self):
        """Computes the potential_energy for the DVR"""
        return self._dvr_prop('potential_energy')

    def hamiltonian(self):
        """Computes the hamiltonian for the DVR"""
        return self._dvr_prop('hamiltonian')

    def wavefunctions(self):
        """Computes the wavefunctions for the DVR"""
        return self._dvr_prop('wavefunctions')

    def _run(self):
        from Psience.Wavefun import Wavefunctions
        from .Wavefunctions import DVRWavefunctions

        def get_res():
            res_class = self.params['results_class'] if 'results_class' in self.params else self.Results
            return res_class(parent=self, **self.params)
        grid = self.grid()
        self.params['grid'] = grid
        if self.params['result'] == 'grid':
            return get_res()

        pe = self.potential_energy()
        self.params['potential_energy'] = pe
        if self.params['result'] == 'potential_energy':
            return get_res()

        ke = self.kinetic_energy()
        self.params['kinetic_energy'] = ke
        if self.params['result'] == 'kinetic_energy':
            return get_res()

        h = self.hamiltonian()
        self.params['hamiltonian'] = h
        if self.params['result'] == 'hamiltonian':
            return get_res()

        wf = self.wavefunctions()
        self.params['wavefunctions'] = DVRWavefunctions(*wf, grid=grid)
        if self.params['result'] == 'wavefunctions':
            return get_res()
        return get_res()

    def run(self, **runpars):
        """ Runs the DVR. Resets state after the run"""

        par = self.params.copy()
        try:
            if 'result' not in self.params:
                self.params['result'] = 'wavefunctions'
            self.params.update(runpars)
            res = self._run()
        finally:
            self.params = par

        return res

    @staticmethod
    def _potential_energy(**pars):
        """ A default ND potential implementation for reuse"""
        import numpy as np

        if 'potential_function' in pars:
            # explicit potential function passed; map over coords
            pf=pars['potential_function']
            if isinstance(pf, str):
                # these mostly just exist as a few simple test cases
                if pf == 'harmonic_oscillator':
                    k = pars['k'] if 'k' in pars else 1
                    re = pars['re'] if 're' in pars else 0
                    pf = lambda x, k=k, re=re: 1/2*k*(x-re)**2
                elif pf == 'morse_oscillator':
                    de = pars['De'] if 'De' in pars else 10
                    a = pars['alpha'] if 'alpha' in pars else 1
                    re = pars['re'] if 're' in pars else 0
                    pf = lambda x, a=a, de=de, re=re: de*(1-np.exp((-a*(x-re)))**2)
                else:
                    raise DVRException("unknown potential "+pf)

            grid = pars['grid']
            dim = len(grid.shape)
            if dim > 1:
                import scipy.sparse as sp
                from functools import reduce
                from operator import mul

                npts = reduce(mul, grid.shape[:-1], 1)
                grid = np.reshape(grid, (npts, grid.shape[-1]))
                pot = sp.diags([pf(grid)], [0])
            else:
                pot = np.diag(pf(grid))
        elif 'potential_values' in pars:
            # array of potential values at coords passed
            pot = np.diag(pars['potential_values'])
        elif 'potential_grid' in pars:
            # TODO: extend to include ND, scipy.griddata
            import scipy.interpolate as interp
            import scipy.sparse as sp

            grid = pars['grid']
            dim = len(grid.shape)
            if dim > 1:
                dim -= 1
                from functools import reduce
                from operator import mul

                npts = reduce(mul, grid.shape[:-1], 1)
                grid = np.reshape(grid, (npts, grid.shape[-1]))

            if dim == 1:
                interpolator = lambda g1, g2: interp.interp1d(g1[:, 0], g1[:, 1], kind='cubic')(g2)
            else:
                def interpolator(g, g2):
                    # g is an np.ndarray of potential points and values
                    # g2 is the set of grid points to interpolate them over

                    shape_dim = len(g.shape)
                    if shape_dim == 2:
                        points = g[:, :-1]
                        vals = g[:, -1]
                        return interp.griddata(points, vals, g2)
                    else:
                        # assuming regular structured grid
                        mesh = g.transpose(np.roll(np.arange(len(g.shape)), 1))
                        points = tuple(np.unique(x) for x in mesh[:-1])
                        vals = mesh[-1]
                        return interp.interpn(points, vals, g2)
            wtf = np.nan_to_num(interpolator(pars['potential_grid'], grid))
            pot = sp.diags([wtf], [0])
        else:
            raise DVRException("couldn't construct potential matrix")

        return pot

    @staticmethod
    def _hamiltonian(**pars):
        """A default hamiltonian implementation for reuse"""
        return pars['kinetic_energy']+pars['potential_energy']

    @staticmethod
    def _wavefunctions(**pars):
        """A default wavefunction implementation for reuse"""
        import numpy as np

        return np.linalg.eigh(pars['hamiltonian'])

    class Results:
        """
        A subclass that can wrap all of the DVR run parameters and results into a clean interface for reuse and extension
        """
        def __init__(self,
                     grid=None,
                     kinetic_energy=None,
                     potential_energy=None,
                     hamiltonian=None,
                     wavefunctions=None,
                     parent=None,
                     **opts
                     ):

            self.parent = None,
            self.grid = grid
            self.kinetic_energy = kinetic_energy
            self.potential_energy = potential_energy
            self.parent = parent
            self.wavefunctions = wavefunctions
            self.opts = opts

        @property
        def dimension(self):
            dim = len(self.grid.shape)
            if dim > 1:
                dim -= 1
            return dim

        def plot_potential(self, plot_class=None, plot_units=None, **opts):
            from McUtils.Plots import Plot, Plot3D
            import numpy as np

            # get the grid for plotting
            MEHSH = self.grid
            unrolly_polly_OLLY = np.roll(np.arange(len(MEHSH.shape)), 1)
            mesh = MEHSH.transpose(unrolly_polly_OLLY)

            if plot_class is None:
                dim = self.dimension
                if dim == 1:
                    plot_class = Plot
                elif dim == 2:
                    plot_class = Plot3D
                else:
                    raise DVRException("{}.{}: don't know how to plot {} dimensional potential".format(
                        type(self).__name__,
                        'plot',
                        dim
                    ))
            if plot_units is 'wavenumbers':
                pot = self.potential_energy.diagonal()
                pot = (pot - min(pot))*219474.6
                pot[pot > 15000] = 15000
            else:
                pot = self.potential_energy.diagonal()

            return plot_class(*mesh, pot.reshape(mesh[0].shape), **opts)


class ResultsInterpreter(DVR.Results):
    """A subclass of results to do some quick analysis..."""
    def __init__(self, **results):
        super().__init__(**results)
        MEHSH = self.grid
        unrolly_polly_OLLY = np.roll(np.arange(len(MEHSH.shape)), 1)
        mesh = MEHSH.transpose(unrolly_polly_OLLY)
        self.x = mesh[0].flatten()
        self.y = mesh[1].flatten()
        poo = self.potential_energy.diagonal()
        poo = poo.reshape(mesh[0].shape).T
        self.potential_energy_vector = poo.flatten()

    def print_energies(self):
        poo = self.potential_energy_vector
        e = self.wavefunctions.energies
        e = (e - min(poo))*219474.6
        return e

    def plot_pot_cuts(self, coordinate, num_to_plot):
        vals = np.column_stack((self.x, self.y, self.potential_energy_vector))
        if coordinate == 'x':
            xvals = np.unique(self.x)
            slices = [vals[self.x == xv] for xv in xvals]
            for x in range(*num_to_plot):  # not correct implementation.. plots never stop looping
                for slip in slices:
                    plt.plot(slip[:, 1], slip[:, 2], 'o')
                    plt.show()
        elif coordinate == "y":
            yvals = np.unique(self.y)
            slyces = [vals[self.y == yv] for yv in yvals]
            for y in range(*num_to_plot):  # not correct implementation.. plots never stop looping
                for slyp in slyces:
                    plt.plot(slyp[:, 1], slyp[:, 2], 'o')
                    plt.show()
        else:
            print("I do not know that coordinate.")

class DVRException(Exception):
    """An Exception in a DVR """
    pass
