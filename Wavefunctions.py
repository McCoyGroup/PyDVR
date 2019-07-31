"""
Provides a DVRWavefunction class that inherits from the base Psience wavefunction
"""

from Psience.Wavefun import Wavefunction

class DVRWavefunction(Wavefunction):
    def plot(self, figure = None):
        if self.data['dimension'] == 1:
            self.data['grid']