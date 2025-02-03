# -*- coding: utf-8 -*-
"""
Python code used in the article
"Testing screened scalar-tensor theories of gravity with atomic clocks"
by Hugo Lévy (author of the code) and Jean-Philippe Uzan.

This script was used to produce Figs. 1 and 9, and to check the condition
given by Eq. (35).

WARNING: you need to install the femtoscope package to be able to run this
script, see https://github.com/onera/femtoscope

"""

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

from femtoscope.misc.constants import M_PL
from femtoscope.misc.unit_conversion import (mass_to_nat, nat_to_density,
                                             density_to_nat, nat_to_length,
                                             length_to_nat)

#%% Functions

def rho_limit(Lambda, beta, npot):
    """

    Parameters
    ----------
    Lambda : float or np.ndarray
        Energy scale -- eV.
    beta : float or np.ndarray
        Coupling parameter -- none.
    npot : int
        Exponent parameter -- none.

    Returns
    -------
    float or np.ndarray
        Density validity bound -- kg/m^3.

    """
    rho_min = npot * Lambda ** (npot+4) * (beta / mass_to_nat(M_PL)) ** npot
    return nat_to_density(rho_min)


def phi_eff(Lambda, beta, npot, rho):
    """

    Parameters
    ----------
    Lambda : float or np.ndarray
        Energy scale -- eV.
    beta : float or np.ndarray
        Coupling parameter -- none.
    npot : int
        Exponent parameter -- none.
    rho : float
        density -- kg/m^3.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    rho = density_to_nat(rho)
    return (mass_to_nat(M_PL) * npot * Lambda ** (npot + 4) \
            / (beta*rho)) ** (1 / (npot + 1))

def scalar_redshift(rho1, rho2, Lambda, beta, npot):
    """

    Parameters
    ----------
    rho1 : float
        density 1 -- kg/m^3.
    rho2 : float
        density 2 -- kg/m^3.
    Lambda : float or np.ndarray
        Energy scale -- eV.
    beta : float or np.ndarray
        Coupling parameter -- none.
    npot : int
        Exponent parameter -- none.

    Returns
    -------
    z_scalar : float or np.ndarray
        Scalar contribution to the redshift -- none.

    """
    phi1 = phi_eff(Lambda, beta, npot, rho1)
    phi2 = phi_eff(Lambda, beta, npot, rho2)
    z_scalar = abs(beta * (phi1 - phi2) / mass_to_nat(M_PL))
    return z_scalar


def compton_wavelength(Lambda, beta, npot, rho):
    rho = density_to_nat(rho)
    lambda_c = sqrt( (mass_to_nat(M_PL) * npot * Lambda ** (npot+4) \
                      / (beta * rho)) ** ((npot+2) / (npot+1)) \
                    / ( npot * (npot+1) * Lambda ** (npot+4) ) )
    return nat_to_length(lambda_c)


#%% Parameters

# The densities are reported in Table II of the article
rho_water = 1e3 # kg/m^3
rho_air = 1.225 # kg/m^3
rho_lead = 11.4e3 # kg/m^3
rho_vac = 1e-10 # kg/m^3
rho_xhv = 1e-15  # kg/m^3
rho_ipm = 1e-20 # kg/m^3
rho_pairs = [(rho_water, rho_air), (rho_lead, rho_vac), (rho_air, rho_ipm)]

# Chameleon parameters (n, beta, Lambda)
npot = 1  # chameleon exponent parameter
beta_bounds = np.array([1e-1, 1e18])
Lambda_bounds = np.array([1e-7, 1e1])

Npts = 100

# Uncertainty on redshift measurement
clk1 = -15  # relative redshift error with pair of clocks 1
clk2 = -20  # relative redshift error with pair of clocks 2


#%% Redshift computations

beta_array = np.logspace(
    np.log10(beta_bounds[0]), np.log10(beta_bounds[1]), Npts)
Lambda_array = np.flip(np.logspace(
    np.log10(Lambda_bounds[0]), np.log10(Lambda_bounds[1]), Npts))

bb, ll = np.meshgrid(beta_array, Lambda_array)
bs, ls = np.meshgrid(beta_array, Lambda_array, sparse=True)

redshifts = []
for densities in rho_pairs:
    rho1, rho2 = densities
    redshifts.append(scalar_redshift(rho1, rho2, ls, bs, npot))


#%% Fig. 1 -- Optimal constraints

M_param = True  # use M instead of beta for the x-axis of the parameter space

z_min = min([np.min(z) for z in redshifts])
z_max = max([np.max(z) for z in redshifts])

titles = ('Water / Air', 'Lead / UHV', 'Air / Orbit')

xticks = [1, 1e5, 1e10, 1e15]
if M_param:
    xtick_labels = [r'$1$', r'$10^{-5}$', r'$10^{-10}$', r'$10^{-15}$']
else:
    xtick_labels = ['1', r'$10^{5}$', r'$10^{10}$', r'$10^{15}$']
yticks = 10**(np.arange(np.log10(Lambda_bounds[0]),
                        np.log10(Lambda_bounds[1])+1))
ytick_labels = ['', r'$10^{-6}$', '', r'$10^{-4}$',
                '', r'$10^{-2}$', '', r'$1$', '']
cticks = [0, -10, -20, -30, -40]
cticks_labels = [r'$1$', r'$10^{-10}$', r'$10^{-20}$',
                 r'$10^{-30}$', r'$10^{-40}$']

fig, axs = plt.subplots(figsize=(11, 3), nrows=1, ncols=len(redshifts))

for kk, ax in enumerate(axs):
    rho2 = rho_pairs[kk][1]
    cwl = compton_wavelength(ls, bs, npot, rho2)
    cf = ax.contourf(beta_array, Lambda_array, np.log10(redshifts[kk]),
                     levels=500, vmin=np.log10(z_min), vmax=np.log10(z_max))
    ax.contour(beta_array, Lambda_array, np.log10(redshifts[kk]),
               levels=[clk2, clk1], vmin=np.log10(z_min),
               vmax=np.log10(z_max), colors=['red', 'darkorange'],
               linestyles='dashed')
    ax.contour(beta_array, Lambda_array, np.log10(cwl), levels=[0],
               colors='silver', linestyles='dotted')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(xticks, labels=xtick_labels)
    ax.set_yticks(yticks, labels=ytick_labels)
    ax.tick_params(axis='both', direction='in', color='white', bottom=True,
                   top=True, right=True, left=True)
    ax.minorticks_off()
    ax.set_title(titles[kk], fontsize=17)
    ax.set_ylabel(r'$\Lambda \, \mathrm{[eV]}$', fontsize=15)
    if M_param:
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xlabel(r'$M/M_{\mathrm{Pl}}$', fontsize=15)
    else:
        ax.set_xlabel(r'$\beta$', fontsize=15)
    for c in cf.collections:
        c.set_rasterized(True)

plt.subplots_adjust(wspace=0.5)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
    vmin=-40, vmax=1))
cbar = fig.colorbar(sm, ax=axs, orientation='vertical', aspect=10)
cbar.set_label(r'$\displaystyle \frac{\beta}{M_{\mathrm{Pl}}} \left|\phi_2 - '
               r'\phi_1\right|$',
               rotation=270, labelpad=40, fontsize=13)
cbar.ax.set_yticks(cticks, cticks_labels)
cbar.ax.plot(([0, 1]), [clk2, clk2], c='red', linewidth=1.5,
             linestyle='dashed')
cbar.ax.plot(([0, 1]), [clk1, clk1], c='darkorange', linewidth=1.5,
             linestyle='dashed')
plt.show()


#%% Verification of the condition given by Eq. (35)

limit_density = rho_limit(ls, bs, npot)
rho_min = np.min(limit_density)
rho_max = np.max(limit_density)

fig, ax = plt.subplots(figsize=(5, 4), nrows=1, ncols=1)
ax.contourf(beta_array, Lambda_array, np.log10(limit_density), levels=500)
ax.set_xscale('log')
ax.set_yscale('log')
ax.minorticks_off()
ax.set_xticks(xticks, labels=xtick_labels)
ax.set_yticks(yticks, labels=ytick_labels)
ax.tick_params(axis='both', direction='in', color='white', bottom=True,
                top=True, right=True, left=True)
ax.set_ylabel(r'$\Lambda \, \mathrm{[eV]}$', fontsize=15)
if M_param:
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$M/M_{\mathrm{Pl}}$', fontsize=15)
else:
    ax.set_xlabel(r'$\beta$', fontsize=15)
for c in cf.collections:
    c.set_rasterized(True)

plt.subplots_adjust(wspace=0.5)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
    vmin=np.log10(rho_min), vmax=np.log10(rho_max)))
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', aspect=10)
cbar.set_label(r'$\log_{10} \rho_{\lim}$',
               rotation=270, labelpad=20, fontsize=15)
plt.tight_layout()
plt.show()


#%% Compton wavelength

cwl_levels = [-18, -7]  # iso-Compton-wavelength to be contoured
cwl = compton_wavelength(ls, bs, npot, rho_water)  # medium = water

cwl_min = np.min(cwl)
cwl_max = np.max(cwl)

fig, ax = plt.subplots(figsize=(5, 4), nrows=1, ncols=1)
ax.contourf(beta_array, Lambda_array, np.log10(cwl), levels=500)
ax.contour(beta_array, Lambda_array, np.log10(cwl), levels=cwl_levels)
ax.set_xscale('log')
ax.set_yscale('log')
ax.minorticks_off()
ax.set_xticks(xticks, labels=xtick_labels)
ax.set_yticks(yticks, labels=ytick_labels)
ax.tick_params(axis='both', direction='in', color='white', bottom=True,
                top=True, right=True, left=True)
ax.set_ylabel(r'$\Lambda \, \mathrm{[eV]}$', fontsize=15)
if M_param:
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$M/M_{\mathrm{Pl}}$', fontsize=15)
else:
    ax.set_xlabel(r'$\beta$', fontsize=15)
for c in cf.collections:
    c.set_rasterized(True)
ax.set_title("Compton wavelength", fontsize=17)

plt.subplots_adjust(wspace=0.5)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
    vmin=np.log10(cwl_min), vmax=np.log10(cwl_max)))
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', aspect=10)
cbar.set_label(r'$\log_{10} \lambda_c$',
               rotation=270, labelpad=20, fontsize=15)
plt.tight_layout()
plt.show()


#%% Fig. 9 -- Optimal constraints with 1m-radius boxes

M_param = True
lc = 1  # one-meter boxes (radius)

rho1 = density_to_nat(1e5)
lc = length_to_nat(lc)
Mp = mass_to_nat(M_PL)
n = npot

def beta_star(rho2, eps, lc):
    rho2 = density_to_nat(rho2)
    return Mp*n**(1/(n + 2))*(eps**((n + 2)/(2*n + 8))*lc**(n/(n + 4)) \
            *rho2**(n*(n + 2)/(2*(n + 1)*(n + 4)))*(n + 1)**(n/(2*n + 8)) \
            /(n**(1/(n + 4))*(rho2**(-1/(n + 1)) - 1/rho1**(1/(n + 1))) \
            **((n + 2)/(2*n + 8))))**((n + 4)/(n + 2))/(lc**((2*n + 2) \
            /(n + 2))*rho2*(n + 1)**((n + 1)/(n + 2)))

def Lambda_star(rho2, eps, lc):
    rho2 = density_to_nat(rho2)
    return eps**((n + 2)/(2*n + 8))*lc**(n/(n + 4))*rho2**(n*(n + 2) \
            /(2*(n + 1)*(n + 4)))*(n + 1)**(n/(2*n + 8))/(n**(1/(n + 4)) \
            *(rho2**(-1/(n + 1)) - 1/rho1**(1/(n + 1)))**((n + 2)/(2*n + 8)))


rho_array = np.logspace(3, -32, 500)

colors = ['darkorange', 'red']
labels = [r"10^{-15}", r"10^{-20}"]
xticks = [1, 1e5, 1e10, 1e15]
yticks = 10**(np.arange(np.log10(Lambda_bounds[0]), np.log10(Lambda_bounds[1])+1))
if M_param:
    xtick_labels = [r'$1$', r'$10^{-5}$', r'$10^{-10}$', r'$10^{-15}$']
else:
    xtick_labels = ['1', r'$10^{5}$', r'$10^{10}$', r'$10^{15}$']

fig, ax = plt.subplots(figsize=(3.75, 4), nrows=1, ncols=1)
for kk, eps in enumerate([1e-15, 1e-20]):
    beta_cwl = beta_star(rho_array, eps, lc)
    Lambda_cwl = Lambda_star(rho_array, eps, lc)
    ax.plot(beta_cwl, Lambda_cwl, linestyle='--', color=colors[kk], linewidth=2,
            gapcolor='silver', label=r'$z_{\phi} = %s$' % labels[kk])
ax.set_xlim(beta_bounds)
ax.set_ylim(Lambda_bounds)
ax.set_xscale('log')
ax.set_yscale('log')
ax.minorticks_off()
ax.set_xticks(xticks, labels=xtick_labels)
ax.set_yticks(yticks, labels=ytick_labels)
ax.tick_params(axis='both', direction='in', color='black', bottom=True,
               top=True, right=True, left=True)
ax.set_ylabel(r'$\Lambda \, \mathrm{[eV]}$', fontsize=15)
if M_param:
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$M/M_{\mathrm{Pl}}$', fontsize=15)
ax.legend(loc='lower right')
plt.tight_layout()
