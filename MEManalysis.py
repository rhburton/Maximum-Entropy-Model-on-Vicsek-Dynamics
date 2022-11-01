# Read flock data, execute MEM, plot data
import numpy as np
import ThesisFunctions as tf
import math
import re
import matplotlib.pyplot as plt


# initialize parameters, decide what to plot
plotLogLikeVSn_c = "no"
plotLogLikeVSncGlobal = "yes"
figureIndex = 15 # parameter that keeps track of which figure you're on, start high to be sure
ncMax = 99

#import Vicsek parameters from 1st line of data file
f = open("VicsekData.txt", "r")
fl = f.readlines()
(interactionType, noiseType, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L, framesToEquilibrium, snapsPerFlock, pollInterval, numFlocks) = tf.importVicsekParam(fl[0])
betaVic = 2./sigmaVic # sigmaVic is the standard distribution of the noise
JprodVicsek = JVic*betaVic
print "Jprod for the ENTIRE Vicsek flock is %s at n_cVic = %s"%(JprodVicsek, ncVic)
print ""


###############################################
# Put all flocks into one simulation
snapsPerFlock = snapsPerFlock*numFlocks
numFlocks=1
###############################################



# initialize the data we take for every frame. For quantity X we have list X[i][t]: ith continuous flock
# t'th snapshot within that flock
optimalNcSnapshot = [[0 for i in range(snapsPerFlock)] for j in range(numFlocks)]
optimalJSnapshot = [[0 for i in range(snapsPerFlock)] for j in range(numFlocks)]
detProdTensor = [[0 for i in range(snapsPerFlock)] for j in range(numFlocks)]
ncCorrTensor = [[0 for i in range(snapsPerFlock)] for j in range(numFlocks)]
avMagnetization = [0 for j in range(numFlocks)]

for flockID in range(numFlocks):
	for t in range(snapsPerFlock):
		lineIndex = snapsPerFlock*flockID + t + 1 # grab the correct configuration from file
		#print "lineIndex = %s, flockID = %s/%s, t = %s/%s"%(lineIndex, flockID, numFlocks, t, snapsPerFlock)
		configLine = fl[lineIndex]
		config = tf.importConfig(configLine, N)

		# extract data from a single flock snapshot
		(ncOptimalSnap, Jsnap, detProdTensor[flockID][t], ncCorrTensor[flockID][t], logLikelihoodFigure) = tf.OptimizeN_c(config, N, L, plotLogLikeVSn_c, ncMax)
		
		optimalNcSnapshot[flockID][t] = ncOptimalSnap
		optimalJSnapshot[flockID][t] = Jsnap
		print "\rOptimal J snapshot is %s, Optimal n_c snapshot is %s"%(Jsnap, ncOptimalSnap)
		print avMagnetization[flockID]
		(M, theta_0) = tf.Magnetization(config, N)
		avMagnetization[flockID] += M
	avMagnetization[flockID] = avMagnetization[flockID]/snapsPerFlock
globalJVec, JarithAvVec = tf.globalJ(optimalJSnapshot, numFlocks, snapsPerFlock)
print "Global J Vector: %s"%globalJVec
globalNcVec, ncArithAvVec, highestAvLogLike = tf.globalNC(detProdTensor, globalJVec, ncCorrTensor, N, numFlocks, snapsPerFlock, plotLogLikeVSncGlobal, ncMax)
print "Average Magnetization vector = %s"%avMagnetization
'''
print "Average J Vector = %s\n"%JarithAv
print "Global J Vector = %s\n"%globalJVector
print "Product of all possible determinants = %s \n"%detProdTensor
print "Every single nc correlation = %s"%(ncCorrTensor)
'''
print "Global J Vector = %s\n"%globalJVec
print "Global nc = %s"%globalNcVec

# print arithmetic n_c estimate and its variance
arithAvNc = 0
arithAvNcSq = 0
ncVector = []
for flockID in range(numFlocks):
	for t in range(snapsPerFlock):
		ncVector.append(optimalNcSnapshot[flockID][t])
arithAvNc = np.average(ncVector)
stdDevNc = np.std(ncVector)
#stdDevNc = np.sqrt(arithAvNcSq - arithAvNc**2)
print "Arithmetic <n*> = %s"%arithAvNc
print "Arithmetic <n*> Standard Dev = %s"%stdDevNc
print "LogLike of ng = %s"%highestAvLogLike

plt.show()





