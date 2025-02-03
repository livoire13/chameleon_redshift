# -*- coding: utf-8 -*-
"""
Python code used in the article
"Testing screened scalar-tensor theories of gravity with atomic clocks"
by Hugo Lévy (author of the code) and Jean-Philippe Uzan.

This script was used to produce Fig. 4, where we study the influence of the
atom's nucleus on the chameleon field value at the location of the electron
cloud (i.e. a few Bohr radii away from the nucleus).

WARNING: you need to install the femtoscope package to be able to run this
script, see https://github.com/onera/femtoscope

"""

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

from femtoscope.physics.physical_problems import Chameleon
from femtoscope.inout.meshfactory import generate_1d_mesh_from_array
from femtoscope.misc.analytical import plot_alpha_map
from femtoscope.misc.unit_conversion import compute_phi_0
from femtoscope.misc.unit_conversion import (mass_to_nat, density_to_nat,
                                             nat_to_length)
from femtoscope.misc.constants import M_PL


#%% Functions

def compton_wavelength(Lambda, beta, npot, rho):
    rho = density_to_nat(rho)
    lambda_c = sqrt((mass_to_nat(M_PL) * npot * Lambda ** (npot+4)
                      / (beta * rho)) ** ((npot+2) / (npot+1))
                    / (npot * (npot+1) * Lambda ** (npot+4)))
    return nat_to_length(lambda_c)


def compton_wavelength_dimensionless(alpha, npot, rho):
    return sqrt(alpha * rho ** (-(npot+2)/(npot+1)) / (npot+1))


def plot_parameter_space(M_param=True, isos='default'):
    if isinstance(isos, str) and isos == 'default':
        isos = [1, 1e-10, 1e-20, 1e-30, 1e-40]
    plot_alpha_map(list(Lambda_bounds), list(beta_bounds), npot,
                   M_param=M_param, iso_alphas=list(isos), figsize=(6, 5))


def plot_compton_wavelength(densities, npot=1, M_param=True, isos=None):
    beta_array = np.logspace(
        np.log10(beta_bounds[0]), np.log10(beta_bounds[1]), 100)
    Lambda_array = np.flip(np.logspace(
        np.log10(Lambda_bounds[0]), np.log10(Lambda_bounds[1]), 100))
    bs, ls = np.meshgrid(beta_array, Lambda_array, sparse=True)
    labels = ["{:.1e} kg/m3".format(rho) for rho in densities]
    xticks = [1, 1e5, 1e10, 1e15]
    yticks = 10**(np.arange(np.log10(Lambda_bounds[0]),
                  np.log10(Lambda_bounds[1])+1))
    if M_param:
        xtick_labels = [r'$1$', r'$10^{-5}$', r'$10^{-10}$', r'$10^{-15}$']
    else:
        xtick_labels = ['1', r'$10^{5}$', r'$10^{10}$', r'$10^{15}$']
    ytick_labels = ['', r'$10^{-6}$', '',
                    r'$10^{-4}$', '', r'$10^{-2}$', '', r'$1$', '']

    fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=len(densities))
    for kk, ax in enumerate(axs):
        rho = densities[kk]
        cwl = compton_wavelength(ls, bs, npot, rho)
        cf = ax.contourf(beta_array, Lambda_array, np.log10(cwl), levels=500)
        if isos is not None:
            ax.contour(beta_array, Lambda_array, np.log10(cwl), levels=isos)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(xticks, labels=xtick_labels)
        ax.set_yticks(yticks, labels=ytick_labels)
        ax.tick_params(axis='both', direction='in', color='white',
                        bottom=True, top=True, right=True, left=True)
        ax.set_ylabel(r'$\Lambda \, \mathrm{[eV]}$', fontsize=15)
        ax.set_title(labels[kk])
        if M_param:
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel(r'$M/M_{\mathrm{Pl}}$', fontsize=15)
        else:
            ax.set_xlabel(r'$\beta$', fontsize=15)
        for c in cf.collections:
            c.set_rasterized(True)
        fig.colorbar(cf, ax=ax)
    plt.tight_layout()


def redshift_dimensionless(Lambda, beta, npot, phi1, phi2):
    """
    Compute the redshift from the dimensionless scalar field values in the
    region where the atoms are being 'interrogated'.

    Parameters
    ----------
    Lambda : float or np.ndarray
        Energy scale [eV].
    beta : float or np.ndarray
        Coupling constant [none].
    npot : int
        Exponent [none].
    phi1 : float
        Scalar field value 1 [none].
    phi2 : float
        Scalar field value 2 [none].

    Returns
    -------
    redshift : float or np.ndarray
        Scalar contribution to the redshift.

    """

    phi0 = compute_phi_0(Lambda, beta, npot, rho0)
    redshift = beta/mass_to_nat(M_PL) * phi0 * abs(phi1 - phi2)
    return redshift


# %% Parameters

# Meta parameters
fem_order = 2
npot = 1
L0 = 1  # 1 meter (characteristic length scale)
rho0 = 1  # 1 kg / m^3 (characteristic density)
beta_bounds = np.array([1e-10, 1e18])
Lambda_bounds = np.array([1e-7, 1e1])

# Positions [m]
wall_position = 1
alu_position = 5e-2
nucleus_position = 2e-15
electron_position = 1e-10

# Thickness [m]
wall_thickness = 1e-2
alu_thickness = 3e-4

# Densities [kg / m^3]
rho_vac = 0 # inside the vacuum chamber
rho_wall = 8e3 # stainless steel
rho_alu = 2.7e3 # aluminium
rho_nucleus = 2.5e17 # atomic nucleus

rho_min = rho_vac
phi_max = np.nan


def mesh_adaptative(with_foil=False):
    xx = np.logspace(-16, -3, int(1e4)) - 1e-16
    if with_foil:
        yy = np.delete(
            np.linspace(1e-4, alu_position + alu_thickness, int(5e4)), 0)
    else:
        yy = np.delete(
            np.linspace(1e-3, wall_position + wall_thickness, int(5e4)), 0)
    xx = np.concatenate((xx, yy))
    return generate_1d_mesh_from_array(xx)
    

def chameleon_profile(alpha, with_foil=False, with_nucleus=False):

    rho_max = rho_alu if with_foil else rho_wall
    phi_min = rho_max ** (-1 / (npot + 1))
    param_dict = {
        'alpha': alpha, 'npot': npot, 'rho_min': None, 'rho_max': rho_max
    }

    # Mesh
    pre_mesh = mesh_adaptative(with_foil=with_foil)

    dim_func_entities = []

    def right_boundary(coors, domain=None):
        return [np.argmax(coors.squeeze())]
    
    dim_func_entities.append((0, right_boundary, 0))

    def nucleus_region(coors, domain=None):
        return np.where(coors.squeeze() <= nucleus_position)[0]
    dim_func_entities.append((1, nucleus_region, 300))

    def vacuum_region(coors, domain=None):
        cc = coors.squeeze()
        if with_foil:
            return np.where((cc < alu_position) & (cc > nucleus_position))[0]
        else:
            return np.where((cc < wall_position) & (cc > nucleus_position))[0]
    dim_func_entities.append((1, vacuum_region, 301))

    def wall_region(coors, domain=None):
        if with_foil:
            return np.where(coors.squeeze() >= alu_position)[0]
        else:
            return np.where(coors.squeeze() >= wall_position)[0]
    dim_func_entities.append((1, wall_region, 302))

    density_dict = {
        ('subomega', 300): rho_nucleus if with_nucleus else rho_vac,
        ('subomega', 301): rho_vac,
        ('subomega', 302): rho_alu if with_foil else rho_wall,
    }

    partial_args_dict_int = {
        'dim': 1,
        'name': 'wf_int',
        'pre_mesh': pre_mesh,
        'fem_order': fem_order,
        'dim_func_entities': dim_func_entities,
        'pre_ebc_dict': {('vertex', 0): rho_max ** (-1 / (npot + 1))}
    }

    chameleon = Chameleon(param_dict, 1, coorsys='polar')
    chameleon.set_wf_int(partial_args_dict_int, density_dict)
    chameleon.set_wf_residual(partial_args_dict_int, density_dict)
    guess_dict = {
        'int': phi_min*np.ones(chameleon.wf_int.field.coors.shape[0]),
        'ext': None
        }
    chameleon.set_default_solver(guess=guess_dict)
    chameleon.set_default_monitor(150)

    solver = chameleon.default_solver
    solver.solve(verbose=True)
    wf_int = solver.wf_int
    rr = wf_int.field.coors.squeeze()
    sol = solver.sol_int
    res = chameleon.default_monitor.criteria['ResidualVector'].value

    return rr, sol, res


def plot_profile(rr, sol, res, show=True):
    fig, axs = plt.subplots(figsize=(12, 6), nrows=2, ncols=1, sharex=True)
    axs[0].scatter(rr, sol, s=0.5, color='k')
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r"$\phi$ [dimensionless]", fontsize=14)
    axs[1].scatter(rr, abs(res), s=0.5, color='red')
    axs[1].set_yscale('log')
    axs[1].set_ylabel("residual [log scale]", fontsize=14)
    axs[1].set_xlabel(r"$r$ [dimensionless]", fontsize=14)
    plt.subplots_adjust(hspace=0)
    if show:
        plt.show()
        print("phi max = {:.1e}".format(np.max(sol)))
        print("phi eff = {:.1e}".format(phi_max))
    else:
        return fig


#%% Ratio with vs without the nucleus : COMPUTATIONS

plt.ioff()
alphas = 10. ** (np.arange(-42, 11, 1))
phi_nuc_list = []
phi_vac_list = []
ratio_list = []

for alpha in alphas:
    print("alpha = {:.1e}".format(alpha))
    rr_nuc, sol_nuc, res_nuc = chameleon_profile(alpha, with_foil=True, with_nucleus=True)
    rr_vac, sol_vac, res_vac = chameleon_profile(alpha, with_foil=True, with_nucleus=False)
    phi_nuc_list.append(sol_nuc)
    phi_vac_list.append(sol_vac)
    idx_electron = np.argmin(abs(rr_nuc - electron_position))
    ratio_list.append(sol_nuc[idx_electron] / sol_vac[idx_electron])
    
    
#%% Ratio with vs without the nucleus : PLOT

logalphas = np.log10(np.array(alphas))
# ratio_list = [0.9999798920235878, 0.9999799458054026, 0.9999799358903803,
#               0.9999800865409681, 0.9999801502232972, 0.9999795991437206,
#               0.9999800985284893, 0.9999759721511224, 0.9999710205942078,
#               0.9997152734499256, 0.9999833049137253, 0.9999332316684297,
#               0.9999797059092864, 0.9999791961714534, 0.9999775319556465,
#               0.9999724677736938, 0.9999570284213108, 0.999911154981067,
#               0.9997862646582827, 0.9994903799408382, 0.9988973556816518,
#               0.9981160164727035, 0.9986191818869443, 0.9996754181135706,
#               0.9999300367055161, 0.9999849268353187, 0.9999967525851574,
#               0.9999993003656765, 0.9999998492682645, 0.9999999675258534,
#               0.9999999930036563, 0.9999999984926831, 0.9999999996752582,
#               0.9999999999300366, 0.9999999999849286, 0.9999999999967534,
#               0.9999999999992996, 0.9999999999998478, 0.9999999999999682,
#               0.9999999999999925, 0.999999999999998, 0.9999999999999998,
#               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

if type(ratio_list == list):
    ratio_list = np.array(ratio_list)

xticks = [-40, -30, -20, -10, 0, 10]
xtick_labels = [r'$10^{-40}$', r'$10^{-30}$', r'$10^{-20}$', r'$10^{-10}$',
                r'$10^{0}$', r'$10^{10}$']
yticks = [0, 0.5e-3, 1e-3, 1.5e-3]

plt.figure(figsize=(3.5, 3))
plt.plot(logalphas, abs(1-ratio_list), color='k')
plt.axvline(x=-17, c='red', linestyle='dashed')
plt.ylabel(r"$\left| 1 - \text{ratio} \right|$", fontsize=15)
plt.xlabel(r"$\alpha$", fontsize=15)
plt.tick_params(axis='both', direction='in', top=True, bottom=True, left=True,
                right=True)
plt.xticks(xticks, xtick_labels)
ax = plt.gca()
ax.tick_params(axis='x', which='major', pad=5)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()
