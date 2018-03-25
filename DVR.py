import os

class DVR:

    '''This is a manager class for working with DVRs

It uses files from which it loads the spec data and methods
Currently all defaults are for 1D but the ND extension shouldn't be bad
'''

    loaded_DVRs={} # for storing DVRs loaded from file
    dvr_dir=os.path.join(os.path.dirname(__file__), "Classes")

    def __init__(self, dvr_file="ColbertMiller1D", **kwargs):

        self.params=kwargs #these are the global parameters passed to all methods
        if dvr_file is None:
            self._dvr=None
        else:
            self._dvr=self.load_dvr(dvr_file)

    @classmethod
    def dvr_file(self, dvr):
        '''Locates the DVR file for dvr'''

        dvr_file=dvr
        if os.path.exists(dvr):
            dvr_file=dvr
        else:
            dvr_file=os.path.join(self.dvr_dir, dvr+".py")

        if not os.path.exists(dvr_file):
            raise DVRException("couldn't load DVR "+dvr)

        return dvr_file

    @classmethod
    def load_dvr(self, dvr):
        '''Loads a DVR from file into the DVR class'''

        dvr_file=self.dvr_file(dvr)
        if not dvr_file in self.loaded_DVRs:
            #defaults and parameters passed as global
            _load_env={
                'DVR':self,
                'potential_energy':self._potential_energy,
                'hamiltonian':self._hamiltonian,
                'wavefunctions':self._wavefunctions
                }
            try: #for python2 compatibility
                handle=open(dvr_file)
                exec(handle.read(), _load_env, _load_env) #load DVR code
                DVR.loaded_DVRs[dvr_file]=_load_env;
            finally: # want to preserve error handling but still close the file
                try:
                    handle.close()
                except:
                    pass

        return DVR.loaded_DVRs[dvr_file]

    def _dvr_prop(self, prop_name):
        '''Gets a DVR property'''

        prop=None;
        if prop_name in self.params:
            prop=self.params[prop_name]
        elif prop_name in self._dvr:
            prop=self._dvr[prop_name]
            if callable(prop):
                prop=prop(**self.params)
        else:
            raise DVRException("no property "+prop_name)

        return prop

    def domain(self):
        '''Computes the domain for the DVR'''
        return self._dvr_prop('domain')
    def divs(self):
        '''Computes the divisions for the DVR'''
        return self._dvr_prop('divs')
    def grid(self):
        '''Computes the grid for the DVR'''
        return self._dvr_prop('grid')
    def kinetic_energy(self):
        '''Computes the kinetic_energy for the DVR'''
        return self._dvr_prop('kinetic_energy')
    def potential_energy(self):
        '''Computes the potential_energy for the DVR'''
        return self._dvr_prop('potential_energy')
    def hamiltonian(self):
        '''Computes the hamiltonian for the DVR'''
        return self._dvr_prop('hamiltonian')
    def wavefunctions(self):
        '''Computes the wavefunctions for the DVR'''
        return self._dvr_prop('wavefunctions')

    def _run(self):

        grid=self.grid()
        self.params['grid']=grid
        if self.params['return']=='grid':
            return grid

        pe=self.potential_energy()
        self.params['potential_energy']=pe
        if self.params['return']=='potential_energy':
            return pe

        ke=self.kinetic_energy()
        self.params['kinetic_energy']=ke
        if self.params['return']=='kinetic_energy':
            return ke

        h=self.hamiltonian()
        self.params['hamiltonian']=h
        if self.params['return']=='hamiltonian':
            return ke

        wf=self.wavefunctions()
        return wf

    def run(self, **runpars):
        ''' Runs the DVR. Resets state after the run '''

        par=self.params.copy()
        self.params['return']='wavefunctions'
        self.params.update(runpars)
        res=self._run()
        self.parms=par

        return res

    @staticmethod
    def _potential_energy(**pars):
        ''' A default 1D potential implementation for reuse'''
        import numpy as np

        if 'potential_function' in pars:
            # explicit potenial function passed; map over coords
            pf=pars['potential_function']
            if isinstance(pf, str):
                # these mostly just exist as a few simple test cases
                if pf=='harmonic_oscillator':
                    k=pars['k'] if 'k' in pars else 1
                    re=pars['re'] if 're' in pars else 0
                    pf=lambda x,k=k,re=re:1/2*k*(x-re)**2
                elif pf=='morse_oscillator':
                    import math
                    de=pars['De'] if 'De' in pars else 10
                    a=pars['alpha'] if 'alpha' in parse else 1
                    re=pars['re'] if 're' in pars else 0
                    pf=lambda x,a=a,de=de,re=re: de*((1-math.e^(-a*(x-re)))^2)
                else:
                    raise DVRException("unknown potential "+pf)
            pot=np.diag([pf(x) for x in pars['grid']])
        elif 'potential_values' in pars:
            # array of potential values at coords passed
            pot=np.diag(pars['potential_values'])
        elif 'potential_grid' in pars:
            ## TODO: extend to ND include
            import scipy.interpolate as interp
            cspline=interp.CubicSpline(
                #abcissa
                [gp[0] for gp in pars['potential_grid']],
                #interpolated values
                [gp[1] for gp in pars['potential_grid']]
                )
            np.diag([cspline(x) for x in pars['grid']])
        else:
            raise DVRException("couldn't construct potential matrix")

        return pot

    @staticmethod
    def _hamiltonian(**pars):
        '''A default hamiltonian implementation for reuse'''
        return pars['kinetic_energy']+pars['potential_energy']

    @staticmethod
    def _wavefunctions(**pars):
        '''A default wavefunction implementation for reuse'''
        import numpy as np

        return np.linalg.eig(pars['hamiltonian'])


class DVRException(Exception):
    '''An Exception in a DVR '''
    pass
