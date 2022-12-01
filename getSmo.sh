#!/usr/bin/env python3

# written by Adam Rettig & Abdul Aldossary
# takes two checkfiles, returns MO overlap between their alphas
# takes two more inputs as range (beginning orbital and ending orbital)

import sys
import io
import re
import math
import numpy as np

## parser
from optparse import OptionParser
import os 

def ParseInput(ArgsIn):
    '''Parse command line options using outparse.'''
    UseMsg='''usage: %prog [checkfile1] [checkfile2] [start] [end] [options]
    Gets the MO overlap between two different checkfiles, also (optional) can specify a submat to get
    <S>_pq with start: first row & col (Python indexing) and end row & col (Python indexing).
    It assumes checkfiles to have the same S matrix, i.e. same geometry and basis set.
    --AA
    '''
    parser=OptionParser(usage=UseMsg)
    parser.add_option('--debug', dest='debug', action='store_true', help='Turns on debug printing in the script.')
    parser.add_option('--spin1', dest='spin1', action='store', type='str', default='a', help='Specify which spin block to consider for file1 (default: a)')
    parser.add_option('--spin2', dest='spin2', action='store', type='str', default='a', help='Specify which spin block to consider for file2 (default: a)')
    parser.add_option('--bra', dest='bra', action='store', type='int', help='Specify a specific row to get overlap with (Python indexing) (Note: if start end is specified, this would be extra specifying)')
    parser.add_option('--ket', dest='ket', action='store', type='int', help='Specify a specific column to get overlap with (Python indexing) (Note: if start end is specified, this would be extra specifying)')
    parser.add_option('--overlap', dest='overlap', action='store_true', help='Prints the overlap matrix')
    parser.add_option('--svals', dest='svals', action='store_true', help='Prints the overlap matrix singular values from SVD')
    parser.add_option('--minsval', dest='minsval', action='store_true', help='Prints the overlap matrix minimum singular value from the SVD')

    options, args = parser.parse_args(ArgsIn)
    if len(args) < 2:
        parser.print_help()
        sys.exit(1)
    else:
        return options, args


# grab S matrix
def getS(text):
    overlapLinePattern = re.compile(r"Overlap Matrix.*")
    overlapMatch = overlapLinePattern.search(text)
    if(overlapMatch == None):
        return np.zeros(0)
    overlapLine = overlapMatch.group(0)
    overlapStart = overlapMatch.start(0)
    overlapN = int(overlapLine.split()[-1])

    trimText = text[overlapStart:].split('\n', 1)[1]
    overlapElemsStr = trimText.split()[:overlapN]
    overlapElems = np.array([float(i) for i in overlapElemsStr])

    nRows = int(np.floor(np.sqrt(overlapN*2.0)))
    Sinds = np.tril_indices(nRows)

    SLower = np.zeros((nRows, nRows))
    SLower[Sinds] = overlapElems
    SUpper = np.triu(SLower.T, 1)
    S = SLower + SUpper

    return S


# grab alpha MO coeffs
def getAlpha(text):
    alphaLinePattern = re.compile(r"Alpha MO coefficients.*")
    alphaMatch = alphaLinePattern.search(text)
    if(alphaMatch == None):
        return np.zeros(0)
    alphaLine = alphaMatch.group(0)
    alphaStart = alphaMatch.start(0)
    alphaN = int(alphaLine.split()[-1])

    trimText = text[alphaStart:].split('\n', 1)[1]
    alphaElemsStr = trimText.split()[:alphaN]
    alphaElems = np.array([float(i) for i in alphaElemsStr])
    alpha = alphaElems.reshape(int(np.sqrt(alphaN)), int(np.sqrt(alphaN))).T
    return alpha

# get how many alpha electrons
def getnoccAlpha(text):
    nAlphaLinePattern = re.compile(r"Number of alpha electrons.*")
    nAlphaMatch = nAlphaLinePattern.search(text)
    nAlphaLine = nAlphaMatch.group(0)
    nAlpha = int(nAlphaLine.split()[-1])
    return nAlpha


# grab beta MO coeffs
def getBeta(text):
    betaLinePattern = re.compile(r"Beta MO coefficients.*")
    betaMatch = betaLinePattern.search(text)
    if(betaMatch == None):
        return np.zeros(0)
    betaLine = betaMatch.group(0)
    betaStart = betaMatch.start(0)
    betaN = int(betaLine.split()[-1])

    trimText = text[betaStart:].split('\n', 1)[1]
    betaElemsStr = trimText.split()[:betaN]
    betaElems = np.array([float(i) for i in betaElemsStr])
    beta = betaElems.reshape(int(np.sqrt(betaN)), int(np.sqrt(betaN))).T
    return beta

# get how many beta electrons
def getnoccBeta(text):
    nBetaLinePattern = re.compile(r"Number of beta electrons.*")
    nBetaMatch = nBetaLinePattern.search(text)
    nBetaLine = nBetaMatch.group(0)
    nBeta = int(nBetaLine.split()[-1])
    return nBeta

# compute MO overlap
def calcOverlap(S, alpha, beta):
    overlaps = np.dot(alpha.T, np.dot(S, beta))
    return overlaps




options, args = ParseInput(sys.argv)

# dump all inputs for debugging
if options.debug:
    print("these are the args:")
    print(args)
    print("these are the options:")
    print(options)

#see if we were passed real files
for i in [1, 2]:
    if not os.path.exists(args[i]):
        print("specified input file does not exist " + args[i])
        sys.exit(1)

# read files
filenames = []
for i in [1, 2]:
    filenames.append(args[i])
    if options.debug:
        print('Reading input file: ' + args[i])

try:
    beg, end = int(args[3]), int(args[4])        
except: # defaults to whole matrix
    beg, end = 0, -1

if options.debug:
    print('taking submatrix; Smo = Smo[%d:%d,%d:%d]', beg, end, beg, end)


## read in job checkpoint file
orbs = []
for i in range(len(filenames)): # readily supports multiple inputs in the future
    infile = io.open(filenames[i], "r")
    text = infile.read()
    S = getS(text)
    orbs.append(getAlpha(text)[:, beg:end])

# get overlap, crashes if they don't have the right dimensions
overlap = calcOverlap(S, orbs[0], orbs[1])

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

if options.overlap:
    print(overlap)

if options.svals or options.minsval:
    overlapSV = np.linalg.svd(overlap)[1]

if options.svals:
    print(overlapSV)
    
if options.minsval:
    print(overlapSV[-1])
    #print(np.average(overlapSV)) # average


#infileName1, infileName2 = sys.argv[1:3]
##print(infileName1)
##print(infileName2)
#try:
#    beg, end = int(sys.argv[3]), int(sys.argv[4])
#except: # defaults to whole matrix
#    beg, end = 0, -1
#
## read in job checkpoint file
#inFile1 = io.open(infileName1, "r")
#inFile2 = io.open(infileName2, "r")
#text1 = inFile1.read()
#text2 = inFile2.read()
#
## calculate overlap of job
#S = getS(text1)
#alpha1 = getAlpha(text1)[:, beg:end]
#alpha2 = getAlpha(text2)[:, beg:end]
##alpha1 = getBeta(text1)[:, beg:end]
##alpha2 = getBeta(text2)[:, beg:end]
#overlap = calcOverlap(S, alpha1, alpha2)
#
#np.set_printoptions(linewidth=np.inf)
#np.set_printoptions(suppress=True)
#
#print(overlap)
#
#overlapSV = np.linalg.svd(overlap)[1]
#print(overlapSV)
##print(np.average(overlapSV))
#print(overlapSV[-1])
#
