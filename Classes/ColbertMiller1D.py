'''1D Colbert and Miller DVR

Note that we only supply a kinetic energy and a grid as the default in the DVR
class will handle everything else

http://xbeams.chem.yale.edu/~batista/v572/ColbertMiller.pdf
'''

import numpy as np, math

def grid_neginfinf(domain=None, divs=None, **kw):
    """
    Provides the Colbert-Miller DVR grid for the [-inf, inf] range
    :param domain:
    :type domain:
    :param divs:
    :type divs:
    :param flavor:
    :type flavor:
    :param kw:
    :type kw:
    :return:
    :rtype:
    """

    return domain[0] + (domain[1] - domain[0]) * np.arange(1, divs)/(divs+1)

def kinetic_energy_neginfinf(grid=None, mass=1, hb=1, **kw):
    '''Computes the kinetic energy for the grid'''

    dx=grid[1]-grid[0] # recomputed here simply to decouple the calling from dvr_grid
    divs=len(grid)
    ke=np.empty((divs, divs))

    coeff=(hb**2)/(2 * mass * (dx ** 2))
    # compute the band values for the first row
    b_val_0 = coeff*(math.pi**2)/3
    col_rng = np.arange(1, divs+1) # the column indices -- also what will be used for computing the off diagonal bands
    row_rng = np.arange(0, divs) # the row indices -- computed once and sliced
    b_vals = coeff * ((-1)**col_rng) * 2 / (col_rng**2)

    for i in range(divs):
        if i == 0:
            np.fill_diagonal(ke, b_val_0)
        else:
            col_inds = col_rng[i-1:-1]#+(i-1)
            row_inds = row_rng[:-i]
            ke[row_inds, col_inds] = b_vals[i-1]
            ke[col_inds, row_inds] = b_vals[i-1]

    return ke

def real_momentum_neginfinf(grid=None, hb=1, **kw):
    raise NotImplementedError("don't have an expression for the momentum on the [-inf, inf] range")

def grid_02pi(domain=None, divs=None, **kw):
    """
    Provides the Colbert-Miller 1D grid for the [0, 2Pi] range
    :param domain:
    :type domain:
    :param divs:
    :type divs:
    :param kw:
    :type kw:
    :return:
    :rtype:
    """

    if divs % 2 != 1:
        raise ValueError('number of DVR points must go as (2N + 1), i.e. it must be odd')

    return domain[0] + (domain[1] - domain[0]) * np.arange(1, divs+1)/divs

def kinetic_energy_02pi(grid=None, mass=1, hb=1, **kw):
    """
    Colbert-Miller kinetic energy for the [0, 2pi] range
    :param grid:
    :type grid:
    :param mass:
    :type mass:
    :param hb:
    :type hb:
    :param kw:
    :type kw:
    :return:
    :rtype:
    """


    coeff = hb**2/(2 * mass)
    divs = len(grid)
    n = (divs - 1)//2
    ke=np.empty((divs, divs))
    np.fill_diagonal(ke, coeff*n * (n+1) / 3)

    col_rng = np.arange(1, divs + 1)  # the column indices -- also what will be used for computing the off diagonal bands
    row_rng = np.arange(0, divs)  # the row indices -- computed once and sliced
    for i in range(1, divs):
        col_inds = col_rng[i - 1:-1]  # +(i-1)
        row_inds = row_rng[:-i]
        val = coeff*(-1)**(i) * np.cos(i * np.pi / divs) / (2 * np.sin(i * np.pi / divs)**2)
        ke[row_inds, col_inds] = val
        ke[col_inds, row_inds] = val

    return ke

def real_momentum_02pi(grid=None, hb=1, **kw):
    """
    Provides the real part of the momentum for the [0, 2pi] range
    :param grid:
    :type grid:
    :param hb:
    :type hb:
    :param kw:
    :type kw:
    :return:
    :rtype:
    """

    divs = len(grid)
    p = np.zeros((divs, divs))

    col_rng = np.arange(1, divs + 1)  # the column indices -- also what will be used for computing the off diagonal bands
    row_rng = np.arange(0, divs)  # the row indices -- computed once and sliced
    for i in range(1, divs):
        col_inds = col_rng[i - 1:-1]  # +(i-1)
        row_inds = row_rng[:-i]
        val = hb/2 * (-1)**(i) / np.sin(i * np.pi / divs)
        p[row_inds, col_inds] = val
        p[col_inds, row_inds] = val

    return p


# def get_kinE(self):
#     # final KE consists of three parts: T_j,j', G(tau), and d^2G/dtau^2
#     from FourierExpansions import calc_curves
#     Tmat = self.get_T()  # nxn
#     G_tau = calc_curves(self.grid, self.GmatCoeffs, function="fourier")
#     # dG2_tau = self.calc_Gderivs()
#     # G2_mat = np.diag(dG2_tau)  # project gmat out to diagonal like potential
#     # calculate KE matrix
#     kinE = np.zeros((self.nPts, self.nPts))
#     for j in range(len(self.grid)):
#         for j_prime in range(j+1):
#             kinE[j, j_prime] = (1/2)*(Tmat[j, j_prime]*(G_tau[j]+G_tau[j_prime]))
#             kinE[j_prime, j] = (1/2)*(Tmat[j, j_prime]*(G_tau[j]+G_tau[j_prime]))
#     return kinE

flavor_grid_map = {
    '[-inf,inf]': grid_neginfinf,
    '[0,2pi]': grid_02pi
}
def get_flavor(key):
    """
    :param key:
    :type key: str
    :return:
    :rtype:
    """
    return key.lower().replace(" ", "").replace(")", "]").replace("(", "[")
def grid(domain=None, divs=None, flavor='[-inf,inf]', **kw):
    return flavor_grid_map[get_flavor(flavor)](
        domain=domain,
        divs=divs,
        flavor=flavor,
        **kw
    )

flavor_ke_map = {
    '[-inf,inf]': kinetic_energy_neginfinf,
    '[0,2pi]': kinetic_energy_02pi
}
def kinetic_energy(grid=None, mass=1, hb=1, g=None, g_deriv=None, flavor='[-inf,inf]', **kw):
    if g is not None:
        mass = 1 / 2 # to get rid of the 1/2m
    ke_1D = flavor_ke_map[get_flavor(flavor)](
        grid,
        mass=mass,
        hb=hb,
        **kw
    )

    if g is not None:
        if g_deriv is None:
            raise ValueError("if a function for `g` is supplied, also need a function, `g_deriv` for the second derivative of `g`")
        # add the average value of `g` across the grid points
        try:
            iter(g)
        except TypeError:
            g_vals = g(grid)
        else:
            g_vals = np.asanyarray(g)

        try:
            iter(g_deriv)
        except TypeError:
            g_deriv_vals = g_deriv(grid)
        else:
            g_deriv_vals = np.asanyarray(g_deriv)

        g_vals = 1/2*(g_vals[:, np.newaxis] + g_vals[np.newaxis, :])
        g_deriv_vals = (hb**2)/2*np.diag(g_deriv_vals)
        ke_1D = ke_1D*g_vals + g_deriv_vals

    return ke_1D

flavor_mom_map = {
    '[-inf,inf]': real_momentum_neginfinf,
    '[0,2pi]': real_momentum_02pi
}
def real_momentum(grid=None, hb=1, flavor='[-inf,inf]', **kw):
    return flavor_mom_map[get_flavor(flavor)](
        grid,
        hb=hb,
        **kw
    )