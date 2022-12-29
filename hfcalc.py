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
# Specify inputs
nnuc = 2 # number of nuclei
nele = 2 # number of electrons
N = 3 # Specify STO-(N)G basis set for step 1
# Choose v = 1 for write out all intermediate matrices
v = 1
# Choose q = 0 for H2, 1 for HeH+
q = 1
if q == 0:
	Z = np.array([1.0, 1.0])
	zeta = np.array([1.24, 1.24])
if q == 1:
	Z = np.array([2.0, 1.0])
	zeta = np.array([2.0925, 1.24])
# Specify rigor of SCF
imax = 20
convergence = 1e-6

# The steps below follow the SCF procedure described in page 146 of Szabo and Ostlund
# step 1
def STONG():
	alpha = np.zeros((nele,N)); D = np.zeros((nele,N))
	# get tabulated numbers
	for i in range(nele):
		alpha[i] = [0.109818,0.405771,2.22766]
		D[i] = [0.444635,0.535328,0.154329]
	# renormalize as necessary
	for i in range(nele):
		alpha[i] = zeta[i]**2.0 * alpha[i]
		for j in range(N):
			D[i,j] = D[i,j] * (2.0*alpha[i,j]/(np.pi))**(3.0/4.0)
	return D,alpha
# step 2	
def integrals(D,alpha,R):
	# the following analytical integrals can be found in the appendix of the book
	def sintegral(a,b,RA,RB):
		RARB = np.linalg.norm(RA-RB)
		return ((np.pi/(a+b))**(3.0/2.0)*np.exp(-a*b/(a+b)*RARB**2.0))
		
	def tintegral(a,b,RA,RB):
		RARB = np.linalg.norm(RA-RB)
		return (a*b/(a+b)*(3.0-2.0*a*b/(a+b)*RARB**2.0)*((np.pi/(a+b))**(3.0/2.0)*np.exp(-a*b/(a+b)*RARB**2.0)))
	
	def fnot(t):
		if t == 0:
			return 1.0
		return (0.5*(np.pi/t)**0.5*erf(t**0.5))
		
	def vintegral(a,b,RA,RB,Z,RC):
		RARB = np.linalg.norm(RA-RB)
		RP = (a*RA+b*RB)/(a+b)
		RPRC = np.linalg.norm(RP-RC)
		return (-2.0*np.pi/(a+b)*Z*np.exp(-a*b/(a+b)*RARB**2.0)*fnot((a+b)*RPRC**2.0))
		
	def twointegral(a,b,g,d,RA,RB,RC,RD):
		RARB = np.linalg.norm(RA-RB)
		RCRD = np.linalg.norm(RC-RD)
		RP = (a*RA+b*RB)/(a+b)
		RQ = (g*RC+d*RD)/(g+d)
		RPRQ = np.linalg.norm(RP-RQ)
		return (2.0*np.pi**(5.0/2.0)/((a+b)*(g+d)*(a+b+g+d)**0.5)*np.exp(-a*b/(a+b)*RARB**2.0-g*d/(g+d)*RCRD**2.0)*(fnot((a+b)*(g+d)/(a+b+g+d)*RPRQ**2.0)))
	
	S = np.zeros((nele,nele)); T = np.zeros((nele,nele)); V = np.zeros((nele,nele))
	twoints = np.zeros((nele,nele,nele,nele))
	for i in range(nele):
		for j in range(nele):
			for k in range(N):
				for l in range(N):
					S[i,j] = S[i,j] + D[i,k]*D[j,l]*sintegral(alpha[i,k],alpha[j,l],R[i],R[j])
					T[i,j] = T[i,j] + D[i,k]*D[j,l]*tintegral(alpha[i,k],alpha[j,l],R[i],R[j])
					for z in range(len(Z)):
						V[i,j] = V[i,j] + D[i,k]*D[j,l]*vintegral(alpha[i,k],alpha[j,l],R[i],R[j],Z[z],R[z])
					for m in range(nele):
						for n in range(nele):
							for o in range(N):
								for p in range(N):
									# i,j,m,n correspond to A, B, C, D,  respectively. They correspond to k, l, o, p for the alpha parameters, respectively.
									twoints[i,j,m,n] = twoints[i,j,m,n] + D[i,k]*D[j,l]*D[m,o]*D[n,p]*twointegral(alpha[i,k],alpha[j,l],alpha[m,o],alpha[n,p],R[i],R[j],R[m],R[n])
	Hcore = T + V
	return S,T,V,Hcore,twoints
# steps 3-4
def findX(S):
	sval,U = np.linalg.eig(S)
	Udag = np.transpose(U)
	s = np.matmul(np.matmul(Udag,S),U)
	s[0,1],s[1,0] = 0.0,0.0 # get rid of noise on the diagonals
	# X = np.matmul(U,np.linalg.inv(s**0.5)) # canonical orthonogalization
	X = np.matmul(U,np.diag(sval**(-0.5))) # symmetric orthogonalization
	Xdag = X.transpose()
	P = np.identity(nele) # as first guess
	return X,Xdag,P
# steps 5-7
def calcF(P,twoints,Hcore,X,Xdag):
	G = np.zeros((nele,nele))
	# swapped variables so that m, n, l, s correspond to mu, nu, lambda, sigma, respectively.
	for m in range(nele):
		for n in range(nele):
			for l in range(nele):
				for s in range(nele):
					G[m,n] = G[m,n] + P[l,s] * (twoints[m,n,s,l]-0.5*twoints[m,l,s,n])
	F = Hcore + G
	Fprime = np.matmul(np.matmul(Xdag,F),X)
	return G,F,Fprime
# steps 8-9
def diagF(Fprime,X):
	U = np.linalg.eig(Fprime)[1]
	Udag = np.transpose(U)
	f = np.matmul(np.matmul(Udag,Fprime),U)
	Cprime = U
	epsilon = f
	C = np.matmul(X,Cprime)
	return C,epsilon
# step 10
def calcP(C):
	P = np.zeros((nele,nele))
	for i in range(nele):
		for j in range(nele):
			for a in range(nele//2):
				P[i,j] = P[i,j] + 2.0*C[i,a]*C[j,a]
	return P
# step 11.1
def findDelta(P,Pold):
	delta = 0.0
	for i in range(nele):
		for j in range(nele):
			delta = delta + np.power(P[i,j]-Pold[i,j],2.0)
	delta = delta**0.5/2.0
	return delta
# step 11-12
def RHF(dist):
	R = np.array([[0.0,0.0,0.0],
			[0.0,0.0,dist]])
	D, alpha = STONG()
	S,T,V,Hcore,twoints = integrals(D,alpha,R)
	X,Xdag,P = findX(S)
	i = 0; delta = 1.0
	while delta > convergence and i < imax:
		# Pold = P # avoid reassigning to not confuse pointers. Deep copy instead
		Pold = np.zeros((2,2))
		for k in range(len(P)):
			for l in range(len(P[k])):
				Pold[k,l] = float(P[k,l])
		G, F, Fprime = calcF(P,twoints,Hcore,X,Xdag)
		C, epsilon = diagF(Fprime,X)
		P = calcP(C)
		delta = findDelta(P,Pold)
		i = i + 1
		if v == 1:
			print('\n trial:',i,'--- delta = ',delta)
			print(' Pold=',Pold)
			print(' F=',F); print(' C=',C)
			print(' epsilon=',epsilon)
			print(' P=',P);
		# P = 0.8 * P + 0.2 * Pold # possible way to enforce convergence of SCF
	if delta > convergence and i == imax:
		print("didn't converge")
		# return RHF(dist+1e-10) # sometimes fixes the convergence issue
	E0 = 0.0
	for i in range(nele):
		for j in range(nele):
			E0 = E0 + 0.5*P[j,i]*(Hcore[i,j]+F[i,j])
	Etot = 0.0
	for i in range(nnuc):
		for j in range(nnuc):
			if i > j:
				Etot = Etot + Z[i]*Z[j]/(np.linalg.norm(R[i]-R[j]))
	Etot = Etot + E0
	
	# del D, alpha, S, T, V, X, Xdag, P, E0
	print('Energy at %1.3f a.u. is %1.5f a.u.'%(dist,Etot))
	return Etot

Rs = np.arange(0.5,5,0.01)

Etot = np.array([])
for dist in Rs:
	Etot = np.append(Etot,RHF(dist))

import pylab
pylab.plot(Rs,Etot)
# pylab.xlim(min(Rs),max(Rs))
# pylab.ylim(-1,1)
pylab.title(r'Internuclear distance Vs. Energy',size=20)
pylab.xlabel(r'$|R| a.u.$',size=10)
pylab.ylabel(r'Energy $a.u.$',size=10)
# pylab.savefig('hfcalc-RvsE.png')
pylab.show()
pylab.cla()