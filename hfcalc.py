#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################
## Project: Minimal-Basis Restricted Hartree-Fock
## Script purpose: Demonstrate the concepts introduced in Chapter 3 in Modern Quantum Chemistry by Szabo and Ostlund.
## Date: 4/8/2018
## Author: Abdulrahman (Abdul) Aldossary, UC-Berkeley Student
##################################################

import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt


# The steps below follow the SCF procedure described in page 146 of Szabo and Ostlund
# step 1
def STO3G(zeta):
    alpha = np.zeros((nbasis, N_STO))
    D = np.zeros((nbasis, N_STO))
    # get tabulated numbers
    for i in range(nbasis):
        alpha[i] = [0.109818, 0.405771, 2.22766]
        D[i] = [0.444635, 0.535328, 0.154329]
    # renormalize as necessary
    for i in range(nbasis):
        alpha[i] *= zeta[i] ** 2.0
        for j in range(N_STO):
            D[i, j] *= (2.0 * alpha[i, j] / np.pi) ** 0.75
    return D, alpha


# step 2
def integrals(D, alpha, R, Z):
    # the following analytical integrals can be found in the appendix of the book
    def sintegral(a, b, RA, RB):
        RARB = np.linalg.norm(RA - RB)
        return (np.pi / (a + b)) ** 1.5 * np.exp(-a * b / (a + b) * RARB ** 2.0)

    def tintegral(a, b, RA, RB):
        RARB = np.linalg.norm(RA - RB)
        return a * b / (a + b) * (3.0 - 2.0 * a * b / (a + b) * RARB ** 2.0) * (
                (np.pi / (a + b)) ** 1.5 * np.exp(-a * b / (a + b) * RARB ** 2.0))

    def fnot(t):
        if t == 0:
            return 1.0
        return 0.5 * (np.pi / t) ** 0.5 * erf(t ** 0.5)

    def vintegral(a, b, RA, RB, Z, RC):
        RARB = np.linalg.norm(RA - RB)
        RP = (a * RA + b * RB) / (a + b)
        RPRC = np.linalg.norm(RP - RC)
        return -2.0 * np.pi / (a + b) * Z * np.exp(-a * b / (a + b) * RARB ** 2.0) * fnot((a + b) * RPRC ** 2.0)

    def twointegral(a, b, g, d, RA, RB, RC, RD):
        RARB = np.linalg.norm(RA - RB)
        RCRD = np.linalg.norm(RC - RD)
        RP = (a * RA + b * RB) / (a + b)
        RQ = (g * RC + d * RD) / (g + d)
        RPRQ = np.linalg.norm(RP - RQ)
        return 2.0 * np.pi ** 2.5 / ((a + b) * (g + d) * (a + b + g + d) ** 0.5) * np.exp(
            -a * b / (a + b) * RARB ** 2.0 - g * d / (g + d) * RCRD ** 2.0) * (
                   fnot((a + b) * (g + d) / (a + b + g + d) * RPRQ ** 2.0))

    S = np.zeros((nbasis, nbasis))
    T = np.zeros((nbasis, nbasis))
    V = np.zeros((nbasis, nbasis))
    twoints = np.zeros((nbasis, nbasis, nbasis, nbasis))
    for mu in range(nbasis):
        for nu in range(nbasis):
            for mu_i in range(N_STO):
                for nu_i in range(N_STO):
                    S[mu, nu] += D[mu, mu_i] * D[nu, nu_i] * sintegral(alpha[mu, mu_i], alpha[nu, nu_i], R[mu], R[nu])
                    T[mu, nu] += D[mu, mu_i] * D[nu, nu_i] * tintegral(alpha[mu, mu_i], alpha[nu, nu_i], R[mu], R[nu])
                    for z in range(len(Z)):
                        V[mu, nu] += D[mu, mu_i] * D[nu, nu_i] * (
                            vintegral(alpha[mu, mu_i], alpha[nu, nu_i], R[mu], R[nu], Z[z], R[z]))
                    for lamb in range(nbasis):
                        for sigm in range(nbasis):
                            for lamb_i in range(N_STO):
                                for sigm_i in range(N_STO):
                                    # mu,nu,lamb,sigm correspond to A, B, C, D,  respectively.
                                    # They correspond to k, l, o, p for the alpha parameters, respectively.
                                    twoints[mu, nu, lamb, sigm] += D[mu, mu_i] * D[nu, nu_i] * D[lamb, lamb_i] * D[
                                        sigm, sigm_i] * twointegral(alpha[mu, mu_i], alpha[nu, nu_i], alpha[lamb, lamb_i],
                                                               alpha[sigm, sigm_i], R[mu],
                                                               R[nu], R[lamb], R[sigm])
    Hcore = T + V
    return S, T, V, Hcore, twoints


# steps 3-4
def findX(S):
    sval, U = np.linalg.eig(S)
    # X = np.matmul(U,np.linalg.inv(s**0.5)) # canonical orthonogalization
    X = np.matmul(U, np.diag(sval ** (-0.5)))  # symmetric orthogonalization
    Xdag = X.T
    return X, Xdag


# steps 5-6
def calcF(P, twoints, Hcore):
    G = np.zeros((nbasis, nbasis))
    # swapped variables so that m, n, l, s correspond to mu, nu, lambda, sigma, respectively.
    for mu in range(nbasis):
        for nu in range(nbasis):
            for lamb in range(nbasis):
                for sigm in range(nbasis):
                    G[mu, nu] += P[lamb, sigm] * (twoints[mu, nu, sigm, lamb] - 0.5 * twoints[mu, lamb, sigm, nu])
    F = Hcore + G
    return G, F


# step 7
def calcFprime(F, X, Xdag):
    Fprime = np.matmul(np.matmul(Xdag, F), X)
    return Fprime


# steps 8-9
def diagF(Fprime, X):
    # epsilon, Cprime = np.linalg.eig(Fprime)
    # C = np.matmul(X, Cprime)
    # return C, epsilon

    U = np.linalg.eig(Fprime)[1]
    Udag = np.transpose(U)
    f = np.matmul(np.matmul(Udag, Fprime), U)
    Cprime = U
    epsilon = f
    C = np.matmul(X, Cprime)
    return C, epsilon


# step 10
def calcP(C):
    # P = 2.0 * np.matmul(C[:, :nelec // 2], C[:, :nelec // 2].T)
    P = np.zeros((nbasis, nbasis))
    for mu in range(nbasis):
        for nu in range(nbasis):
            for i in range(nelec // 2):
                P[mu, nu] += 2.0 * C[mu, i] * C[nu, i]
    return P


# step 11.1
def findDelta(P, Pold):
    # delta = np.linalg.norm(P - Pold)
    delta = 0.0
    for mu in range(nbasis):
        for nu in range(nbasis):
            delta = delta + np.power(P[mu, nu] - Pold[mu, nu], 2.0)
    delta = delta ** 0.5 / 2.0
    return delta


# step 11-12
def RHF(D, alpha, R, Z, imax, convergence):
    S, T, V, Hcore, twoints = integrals(D, alpha, R, Z)
    X, Xdag = findX(S)
    P = np.identity(nbasis)  # as first guess
    # P = np.zeros((nbasis,nbasis))  # as first guess
    i = 0
    delta = 1.0
    while delta > convergence and i < imax:
        Pold = P  # sometimes reassigning confuses pointers. Deep copy instead
        G, F = calcF(P, twoints, Hcore)
        Fprime = calcFprime(F, X, Xdag)
        C, epsilon = diagF(Fprime, X)
        P = calcP(C)
        delta = findDelta(P, Pold)
        i += 1
        if v == 1:
            print('\n trial:', i, '--- delta = ', delta)
            print(' Pold=', Pold)
            print(' F=', F)
            print(' C=', C)
            print(' epsilon=', epsilon)
            print(' P=', P)
        # P = 0.8 * P + 0.2 * Pold # possible way to enforce convergence of SCF

    if delta > convergence and i == imax:
        print("didn't converge")
        # return RHF(dist+1e-10) # sometimes fixes the convergence issue
    # E0 = 0.5 * np.trace(np.dot(P, (Hcore + F)))
    E0 = 0.0
    for mu in range(nbasis):
        for nu in range(nbasis):
            E0 += 0.5 * P[nu, mu] * (Hcore[mu, nu] + F[mu, nu])
    Enuc = 0.0
    for i in range(nnuc):
        for j in range(nnuc):
            if i > j:
                Enuc += Z[i] * Z[j] / (np.linalg.norm(R[i] - R[j]))
    Etot = Enuc + E0

    # del D, alpha, S, T, V, X, Xdag, P, E0
    print('Energy at %1.3f a.u. is %1.5f a.u.' % (dist, Etot))
    return Etot


### Specify inputs:

# Choose v = 1 for write out all intermediate matrices
v = 0

# need to specify below:
# Z: vector of nuclear charges
# zeta: basis exponents
# nelec: number of electrons (has to be even, since this is RHF)

# Choose q = 0 for H2, 1 for HeH+, 2 for He2, 3 for He2^2+
q = 0
if q == 0:  # H2
    Z = np.array([1.0, 1.0])
    zeta = np.array([1.24, 1.24])
    nelec = 2
elif q == 1:  # HeH+
    Z = np.array([2.0, 1.0])
    zeta = np.array([2.0925, 1.24])
    nelec = 2
elif q == 2:  # He2
    Z = np.array([2.0, 2.0])
    zeta = np.array([2.0925, 2.0925])
    nelec = 4
elif q == 3:  # He_2^2+
    Z = np.array([2.0, 2.0])
    zeta = np.array([2.0925, 2.0925])
    nelec = 2
else:  # default is H2
    Z = np.array([1.0, 1.0])
    zeta = np.array([1.24, 1.24])
    nelec = 2

nnuc = len(Z)  # number of nuclei

nbasis = len(zeta)  # number of basis functions
N_STO = 3  # Specify STO-(N)G basis set for step 1
D, alpha = STO3G(zeta)  # get STO-3G coefficients and exponents

# Specify rigor of SCF
imax = 20
convergence = 1e-6

Rs = np.arange(0.7, 5, 0.1)

Etot = np.array([])
for dist in Rs:
    R = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, dist]])
    Etot = np.append(Etot, RHF(D, alpha, R, Z, imax, convergence))

plt.plot(Rs, Etot)
# plt.xlim(min(Rs),max(Rs))
# plt.ylim(-1,1)
plt.title(r'Internuclear distance Vs. Energy', size=20)
plt.xlabel(r'$|R| a.u.$', size=10)
plt.ylabel(r'Energy $a.u.$', size=10)
# plt.savefig('hfcalc-RvsE.png')
plt.show()
# plt.cla()


# let's do H4 because we can! Uncomment below to do that

"""
# Specify inputs
Z = np.ones(4)
zeta = 1.24*Z
nelec = 4  # number of electrons

nnuc = len(Z)  # number of nuclei
nbasis = len(zeta)  # number of basis functions

N_STO = 3  # Specify STO-(N)G basis set for step 1
D, alpha = STO3G(zeta)

# Specify rigor of SCF
imax = 20 
convergence = 1e-6

Rs = np.arange(1, 5, 0.1)
Etot = np.array([])
for dist in Rs:
    R = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, dist],
                  [0.0, 0.0, 2.*dist],
                  [0.0, 0.0, 3.*dist],])
    Etot = np.append(Etot, RHF(D, alpha, R, Z, imax, convergence))


plt.plot(Rs, Etot)
# plt.xlim(min(Rs),max(Rs))
# plt.ylim(-1,1)
plt.title(r'Internuclear distance Vs. Energy', size=20)
plt.xlabel(r'$|R| a.u.$', size=10)
plt.ylabel(r'Energy $a.u.$', size=10)
# plt.savefig('hfcalc-RvsE.png')
plt.show()
# plt.cla()
"""
