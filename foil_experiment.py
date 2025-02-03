# -*- coding: utf-8 -*-
"""
Python code used in the article
"Testing screened scalar-tensor theories of gravity with atomic clocks"
by Hugo Lévy (author of the code) and Jean-Philippe Uzan.

This script was used to produce Fig. 3, showing potential constraints on the
chameleon parameter space resulting from the experimental design involving a
foil surrounding some of the atoms. Details about the assumptions used for
this numerical study can be found in Sec. V A 3.

WARNING: you need to install the femtoscope package to be able to run this
script, see https://github.com/onera/femtoscope

"""

from pathlib import Path
import pickle

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

from femtoscope.physics.physical_problems import Chameleon
from femtoscope.misc.unit_conversion import compute_Lambda, compute_phi_0
from femtoscope.inout.meshfactory import generate_uniform_1d_mesh
from femtoscope.misc.analytical import plot_alpha_map
from femtoscope.misc.unit_conversion import (mass_to_nat, density_to_nat,
                                             nat_to_length)
from femtoscope.misc.constants import M_PL
from femtoscope import TMP_DIR


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


# %% Parameters of the study

# Meta parameters
fem_order = 2
npot = 1
L0 = 1  # 1 meter (characteristic length scale)
rho0 = 1  # 1 kg / m^3 (characteristic density)
beta_bounds = np.array([1e-10, 1e18])
Lambda_bounds = np.array([1e-7, 1e1])

# Positions [m]
wall_position = 1
alu_position = 1e-2

# Thickness [m]
wall_thickness = 1e-2
alu_thickness = 3e-4

# Densities [kg / m^3]
rho_uhv = 1e-10  # ultra-high vacuum
rho_xhv = 1e-15  # extremely-high vacuum
rho_vac = rho_xhv  # inside the vacuum chamber
rho_wall = 8e3  # stainless steel
rho_alu = 2.7e3  # aluminium

rho_min = min(rho_vac, rho_wall, rho_alu)
rho_max = max(rho_vac, rho_wall, rho_alu)
phi_min = rho_max ** (-1 / (npot + 1))
phi_max = rho_min ** (-1 / (npot + 1))


def chameleon_profile(alpha, with_foil=False):
    """
    Compute the chameleon field profile inside the vacuum chamber for a given
    dimensionless parameter `alpha` [see Eq. (33)], with or without the
    aluminium foil surrounding the cloud of atoms.
    
    Parameters
    ----------
    alpha : float
        Dimensionless parameter given by Eq. (33) and governing the dynamics of
        chameleon field. Relevant values range from 1e-40 to 1e+10.
    with_foil : bool
        Whether the foil should be added in the numerical domain.
        The default is False.
    """

    rho_max = rho_alu if with_foil else rho_wall
    param_dict = {
        'alpha': alpha, 'npot': npot, 'rho_min': rho_vac, 'rho_max': rho_max
    }

    # Mesh
    if with_foil:
        pre_mesh = generate_uniform_1d_mesh(0, alu_position + alu_thickness, 1e5)
    else:
        pre_mesh = generate_uniform_1d_mesh(0, wall_position + wall_thickness, 1e5)

    dim_func_entities = []

    def right_boundary(coors, domain=None):
        return [np.argmax(coors.squeeze())]
    dim_func_entities.append((0, right_boundary, 0))

    def inside_region(coors, domain=None):
        if with_foil:
            return np.where(coors.squeeze() < alu_position)[0]
        else:
            return np.where(coors.squeeze() < wall_position)[0]
    dim_func_entities.append((1, inside_region, 300))

    def wall_region(coors, domain=None):
        if with_foil:
            return np.where(coors.squeeze() >= alu_position)[0]
        else:
            return np.where(coors.squeeze() >= wall_position)[0]
    dim_func_entities.append((1, wall_region, 301))

    density_dict = {
        ('subomega', 300): rho_vac,
        ('subomega', 301): rho_alu if with_foil else rho_wall
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
    chameleon.set_default_solver()
    chameleon.set_default_monitor(30)

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


#%% Redshift in parameter space : COMPUTATIONS

plt.ioff()
alphas = 10. ** (np.arange(-42, 11, 1))
phi_foil_list = []
phi_wall_list = []

for alpha in alphas:
    print("alpha = {:.1e}".format(alpha))
    rr_foil, sol_foil, res_foil = chameleon_profile(alpha, with_foil=True)
    rr_wall, sol_wall, res_wall = chameleon_profile(alpha, with_foil=False)
    phi_foil_list.append(sol_foil[0])
    phi_wall_list.append(sol_wall[0])
    fig_foil = plot_profile(rr_foil, sol_foil, res_foil, show=False)
    fig_wall = plot_profile(rr_wall, sol_wall, res_wall, show=False)
    name = 'alpha_{}'.format(int(np.log10(alpha)))
    pathname_foil = Path(TMP_DIR / 'foil' / name).with_suffix('.png')
    pathname_wall = Path(TMP_DIR / 'wall' / name).with_suffix('.png')
    fig_foil.savefig(pathname_foil, dpi=300)
    fig_wall.savefig(pathname_wall, dpi=300)

#%% Redshift in parameter space : POST-PROCESS

betas = []
Lambdas = []
zs = []
beta_array = np.logspace(-1, 18, 100)
for kk, alpha in enumerate(alphas):
    Lambda_array = compute_Lambda(beta_array, alpha, npot, L0, rho0)
    phi1 = phi_foil_list[kk]
    phi2 = phi_wall_list[kk]
    z_array = redshift_dimensionless(Lambda_array, beta_array, npot, phi1, phi2)
    betas.append(beta_array)
    Lambdas.append(Lambda_array)
    zs.append(z_array)

# Compute redshift
betas_aux = np.array(betas).flatten()
Lambdas_aux = np.array(Lambdas).flatten()
zs_aux = np.array(zs).flatten()

idx_del = np.where((Lambdas_aux < Lambda_bounds[0]/20)
                   | (Lambdas_aux > 20*Lambda_bounds[1]))[0]
betas = np.delete(betas_aux, idx_del)
Lambdas = np.delete(Lambdas_aux, idx_del)
zs = np.delete(zs_aux, idx_del)

data = np.stack((betas, Lambdas, zs)).T
pathname = Path(TMP_DIR / 'data1D.pkl')
with open(pathname, 'wb') as f:
    pickle.dump(data, f)


#%% Redshift in parameter space : PLOT

reload = False
def reload_data():
    pathname = Path(TMP_DIR / 'data1D_saved.pkl')
    with open(pathname, 'rb') as f:
        data = pickle.load(f)
    return data[:, 0], data[:, 1], data[:, 2]

if reload:
    betas, Lambdas, zs = reload_data()

M_param = True

# Uncertainty on redshift measurement
clk1 = -15  # relative redshift error with pair of clocks 1
clk2 = -20  # relative redshift error with pair of clocks 2

xticks = [1, 5, 10, 15]
yticks = np.arange(np.log10(Lambda_bounds[0]), np.log10(Lambda_bounds[1])+1)
cbar_ticks = [0, -10, -20, -30, -40]

if M_param:
    xtick_labels = [r'$1$', r'$10^{-5}$', r'$10^{-10}$', r'$10^{-15}$']
else:
    xtick_labels = ['1', r'$10^{5}$', r'$10^{10}$', r'$10^{15}$']
ytick_labels = ['', r'$10^{-6}$', '',
                r'$10^{-4}$', '', r'$10^{-2}$', '', r'$1$', '']
cbar_tick_labels = [r'$1$', r'$10^{-10}$', r'$10^{-20}$', r'$10^{-30}$',
                    r'$10^{-40}$']

fig, ax = plt.subplots(figsize=(5, 4), nrows=1, ncols=1)
tcf = ax.tricontourf(np.log10(betas), np.log10(Lambdas), np.log10(zs), levels=500)
ax.tricontour(np.log10(betas), np.log10(Lambdas), np.log10(zs),
              levels=[-20, -15], colors=['red', 'darkorange'],
              linestyles='dashed')
ax.set_ylim(np.log10(Lambda_bounds))
ax.set_xticks(xticks, labels=xtick_labels)
ax.set_yticks(yticks, labels=ytick_labels)
ax.tick_params(axis='both', direction='in', color='white',
                bottom=True, top=True, right=True, left=True)
ax.set_ylabel(r'$\Lambda \, \mathrm{[eV]}$', fontsize=15)
for c in tcf.collections:
    c.set_rasterized(True)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
    vmin=-42, vmax=0))
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', aspect=15)
cbar.set_label(r'$\displaystyle \frac{\beta}{M_{\mathrm{Pl}}} \left|\phi_2 - '
               r'\phi_1\right|$', rotation=270, labelpad=40, fontsize=13)
cbar.ax.plot(([0, 1]), [clk2, clk2], c='red', linewidth=1.5, linestyle='dashed')
cbar.ax.plot(([0, 1]), [clk1, clk1], c='darkorange', linewidth=1.5,
             linestyle='dashed')
cbar.set_ticks(ticks=cbar_ticks, labels=cbar_tick_labels)

if M_param:
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$M/M_{\mathrm{Pl}}$', fontsize=15)
else:
    ax.set_xlabel(r'$\beta$', fontsize=15)
plt.tight_layout()
plt.show()


