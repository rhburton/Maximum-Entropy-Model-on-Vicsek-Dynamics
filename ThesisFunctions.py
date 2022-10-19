# Function file for polished code
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import os
import pylab as pl
from matplotlib import collections  as mc
import sys
import time
import re

# returns most likely n_c provided that n_c is known to be constant across all snapshots
def globalNC(detProdTensor, globalJVec, ncCorrTensor, N, numFlocks, snapsPerFlock, plotLogLikeVSncGlobal, ncMax):
	globalNcVec = [0 for i in range(numFlocks)]
	ncArithAvVec = [0 for i in range(numFlocks)]

	figureIndx = 69
	for flockID in range(numFlocks):
		# this vector measures the average likelihood over all snapshots for J_g at each desired n_c
		logLikeAverage = [0 for i in range(ncMax)]
		for nc in range(1, ncMax+1):
			for t in range(snapsPerFlock):
				logLike = 0.5*(N-1)*np.log((globalJVec[flockID]/(2*math.pi)))
				logLike += 0.5*np.log(detProdTensor[flockID][t][nc-1])
				logLike += -0.5*globalJVec[flockID]*N*nc*(1-ncCorrTensor[flockID][t][nc-1])
				logLikeAverage[nc-1] += logLike
			logLikeAverage[nc-1] = logLikeAverage[nc-1]/snapsPerFlock

		highestAvLikelihood = logLikeAverage[0]
		for nc in range(1, ncMax+1):
			if logLikeAverage[nc-1] > highestAvLikelihood: 
				globalNcVec[flockID] = nc
				highestAvLikelihood = logLikeAverage[nc-1]
		print "Global n_c = %s"%globalNcVec[flockID]


		#plot log likelihood vs n_c when asked
		if plotLogLikeVSncGlobal == "yes":
			plt.figure(figureIndx)
			n_cVec = [i for i in range(1, ncMax+1)]
			plt.title('Average Log Likelihood vs n_cGlobal')
			plt.ylabel('Log Likelihood (dimensionless)')
			plt.xlabel('n_c')
			plt.plot(n_cVec, logLikeAverage, 'ro')
			#plt.plot(n_cVec, detContribution, 'gv')
			#plt.plot(n_cVec, corrContribution, 'c^')
			plt.pause(0.00001)
			figureIndx +=1 


	return globalNcVec, ncArithAvVec, highestAvLikelihood

# returns the most likely J provided that J is known to be constant across all snapshots
def globalJ(optimalJSnapshot, numFlocks, snapsPerFlock):
	globalJVector = [0 for i in range(numFlocks)]
	JarithmeticAvVec = [0 for i in range(numFlocks)]
	for flock in range(numFlocks):
		inverseSum = 0
		arithmeticAverageSum = 0
		for snap in range(snapsPerFlock):
			inverseSum += 1/optimalJSnapshot[flock][snap]
			arithmeticAverageSum += optimalJSnapshot[flock][snap]
		JarithmeticAvVec[flock] = arithmeticAverageSum/snapsPerFlock
		globalJVector[flock] = snapsPerFlock/inverseSum
	return (globalJVector, JarithmeticAvVec)

# import a flock snapshot
def importConfig(configLine, N):
	config = [[0, 0, 0] for j in range(N)]
	parsedNumbers = re.findall('-?\d+.\d+', configLine)
	for i in range(N):
		config[i][0] = parsedNumbers[3*i + 0]
		config[i][1] = parsedNumbers[3*i + 1]
		config[i][2] = parsedNumbers[3*i + 2]
	# test if the config string is parsed correctly
	if "%s\n"%re.sub('\'', '', str(config)) != configLine: raise Exception('Config import FAILED')
	for i in range(N):
		for j in range(3):
			config[i][j] = float(config[i][j])
	return config

# return the flock parameters in a flock data file, properly typecasted
def importVicsekParam(fileLine1):
	(interactionType, noiseType, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L, framesToEquilibrium, snapshotsPerFlock, pollInterval, numFlocks, BLANK) = re.split(",", fileLine1)
	#print "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"%(interactionType, noiseType, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L, framesToEquilibrium,snapshotsPerFlock, pollInterval,numFlocks, BLANK)
	# typecast correctly
	N = int(N)
	ncVic = int(ncVic)
	sigmaVic = float(sigmaVic)
	eta = float(eta)
	nu = float(nu)
	dtVic = float(dtVic)
	JVic = float(JVic)
	L=int(L)
	framesToEquilibrium = int(framesToEquilibrium)
	snapshotsPerFlock = int(snapshotsPerFlock)
	pollInterval = int(pollInterval)
	numFlocks=int(numFlocks)
	return (interactionType, noiseType, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L, framesToEquilibrium, snapshotsPerFlock, pollInterval, numFlocks)

# plots Vicsek magnetization vs time
def plotVicsekMvsTime(MvsT):
	t = [i for i in range(len(MvsT))]

	plt.figure(13)
	plt.clf()
	plt.draw()
	plt.plot(t, MvsT, 'ro')

	plt.xlabel('Timesteps')
	plt.ylabel('Vicsek Magnetization')
	plt.title('Vicsek Magnetization vs Time')
	axes = plt.gca()
	axes.ticklabel_format(useOffset=False)

	#axes.set_ylim([.8,1.000001])
	#axes.set_ylim(ymax=1.0000001)
	#plt.set_ylim(ymax=1.02)
	#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
	plt.pause(0.001)
	return plt.figure(13)

def checkVicsekNcSize(config, N, ncVic, L):
	# make sure my flock is connected
	(Jij, Sij) = VicsekInteractionMatrices(config, N, ncVic, L)
	(eVal, nullspace) = laplacianEigenvalues(Sij, N, ncVic)
	if nullspace-1 > ncVic: 
		raise Exception('NC VICSEK TOO SMALL: NON-INTERACTING PARTS OF FLOCK')
	logDetProd = 0
	for i in range(nullspace, N):
		logDetProd += np.log(eVal[i])
	return logDetProd



# plots a given Vicsek configuration
def PlotVicsek(config, N, L, keepPlot):
	X = [config[i][0] for i in range(N)]
	Y = [config[j][1] for j in range(N)] 

	spinX = [0 for i in range(N)]
	spinY = [0 for i in range(N)]
	for n in range(N):
		theta = config[n][2]
		spinX[n] = np.cos(theta)
		spinY[n] = np.sin(theta)

	# ERASE IF I WANT TO SHOW NEAREST NEIGHBORS #########
	plt.figure(1)
	plt.clf()
	plt.quiver(X, Y, spinX, spinY)
	plt.title('Vicsek Configuration')
	edge = 0.01
	plt.xlim(-edge, L+edge)
	plt.ylim(-edge, L+edge)

	plt.pause(.05)
	if keepPlot == "yes": return plt.figure(1)
	return


# computes magnetization of a Vicsek flock
def Magnetization(config, N):
	Mx = 0
	My = 0
	for i in config:
		Mx += math.cos(i[2])
		My += math.sin(i[2])
	avMx = Mx/N
	avMy = My/N
	M = math.sqrt(avMx**2 + avMy**2)
	# use arctan2 to give the branch cut associated with the counter-clockwise angle from the x-axis
	theta_0 = np.arctan2(avMy, avMx)
	#print "avMx = %s, avMy = %s"%(avMx, avMy)
	return (M, theta_0)


# takes true distance of Vicsek particles i and j accounting for periodic boundary conditions
def VicsekDistanceSq(config, i, j, L):
	x1 = config[i][0]
	x2 = config[j][0]
	y1 = config[i][1]
	y2 = config[j][1]
	possibleDist = [0 for l in range(9)]
	mod = L

	possibleDist[0] = (x1 - x2)**2 + (y1 - y2)**2
	possibleDist[1] = (x1 - x2 - mod)**2 + (y1 - y2)**2
	possibleDist[2] = (x1 - x2 + mod)**2 + (y1 - y2)**2
	possibleDist[3] = (x1 - x2)**2 + (y1 - y2 - mod)**2
	possibleDist[4] = (x1 - x2)**2 + (y1 - y2 + mod)**2
	possibleDist[5] = (x1 - x2 + mod)**2 + (y1 - y2 + mod)**2
	possibleDist[6] = (x1 - x2 - mod)**2 + (y1 - y2 - mod)**2
	possibleDist[7] = (x1 - x2 + mod)**2 + (y1 - y2 - mod)**2
	possibleDist[8] = (x1 - x2 - mod)**2 + (y1 - y2 + mod)**2

	# sort list to find shortest distance from point i to point j
	possibleDist.sort()
	if possibleDist[0] > (float(L)**2)/2:
		print "BOUNDARY CONDITIONS ARE GOOFING UP i=%s j=%s"%(i, j)
		print "x1=%s y1=%s"%(x1, y1)
		print "x2=%s y2=%s"%(x2, y2)
		print "shortest distance = %s"%(possibleDist[0])

	return possibleDist[0]


# computes matrices J_ij and S_ij from a Vicsek configuration
def VicsekInteractionMatrices(config, N, n_c, L):
	# compute effective nearest "n_c" neighbor interaction matrix
	Jij = [[0 for j in range(N)] for i in range(N)]
	Sij = [[0 for j in range(N)] for i in range(N)]
	# temporary variable for storing relative distances
	rVec = [0 for i in range(N)]
	for i in range(N):
		# find the nearest n_c neighbors to the ith particle
		for j in range(N):
			rVec[j] = VicsekDistanceSq(config, i, j, L)
		# make a sorted copy (and yes, I do have to be this annoyingly explicit with my pointers)
		rSort = [0 for kappa in range(N)]
		for kappa in range(N): rSort[kappa] = rVec[kappa]
		rSort.sort()
		#print rSort
		#print rVec
		#print rSort

		# fill out the interaction matrices
		# index over nearest n_c neighbors, discounting itself
		for k in range(1, n_c + 1):
			for l in range(N):
				if rSort[k] == rVec[l] and rSort[k]!=0:
					Jij[i][l] += 1
					Sij[i][l] += 0.5
					Sij[l][i] += 0.5
	return Jij, Sij

# computes the average correlation up to range n_c
def nnAvCorrelation(config, N, n_c, Jij):
	L=9
	if Jij == 0:
		Jij = VicsekInteractionMatrices(config, N, n_c, L)[0]
	LocalCorr = 0
	for i in range(N):
		for j in range(N):
			LocalCorr += Jij[i][j] * math.cos(config[i][2] - config[j][2])

	LocalCorr = float(LocalCorr)/(N*n_c)
	return LocalCorr

def laplacianEigenvalues(Sij, N, n_c):
	# find sum of each row/column of Sij (important number in our computation)
	SrowSum = [0 for i in range(N)]
	for i in range(N):
		for j in range(N): 
			SrowSum[i] += Sij[i][j]
	# compute Mij (the Laplacian)
	Mij = [[0 for i in range(N)] for j in range(N)]
	for i in range(N):
		for j in range(N):
			Mij[i][j] = -Sij[i][j]
			if i == j: 
				Mij[i][j] += SrowSum[i]
	# compute eigenvalues of Mij: eigh is for symmetric matrices
	(eVal, eVec) = np.linalg.eigh(Mij)
	rank = np.linalg.matrix_rank(Mij)
	nullspace = N-rank

	return (eVal, nullspace)

# computes the log likelihood for a given Vicsek configuration and n_c
# automatically computes S_ij for you
def logLikelihoodFn(config, N, n_c, L):
	(Jij, Sij) = VicsekInteractionMatrices(config, N, n_c, L)
	# find sum of each row/column of Sij (important number in our computation)
	SrowSum = [0 for i in range(N)]
	for i in range(N):
		for j in range(N): 
			SrowSum[i] += Sij[i][j]

	# compute Mij
	Mij = [[0 for i in range(N)] for j in range(N)]
	for i in range(N):
		for j in range(N):
			Mij[i][j] = -Sij[i][j]
			if i == j: 
				Mij[i][j] += SrowSum[i]

	# compute eigenvalues of Mij: eigh is for symmetric matrices
	eVal = np.linalg.eigvalsh(Mij)
	#print "Eigenvalues are %s"%eVal
	rank = np.linalg.matrix_rank(Mij)
	nullspace = N-rank

	#print "\n####### n_c = %s #######"%(n_c)
	#print "Nullspace = %s"%(nullspace)
	# examine exactly which eigenvalues are supposed to correspond to nullvectors
	#for i in range(0, nullspace):
		#print "0-eigenvalue --- %s"%(eVal[i])
	#print "Biggest 0-eigenvalue is %s"%(eVal[nullspace - 1])
	#print "Spectral Gap is %s"%(eVal[nullspace])
	'''if (N - rank) > 1:
		print "\nn_c = %s HAS AN ADDITIONAL %s SYMMETRIES "%(n_c, N - rank - 1)'''
	
	'''
	# check for negative eigenvalues
	negativeEigenvals = 0
	print "Eigenvalues: %s"%(eVal)

	# check for anomalous eigenvalues
	for i in range(len(eVal)):
		if eVal[i] < 0: negativeEigenvals += 1
	if negativeEigenvals > (N - rank):
		print ">>>>>>>>>> THERE ARE/IS %s NEGATIVE EIGENVALUE(s) <<<<<<<<<<"%(negativeEigenvals)
		for k in range(0, N):
			if eVal[k] < 0: print ">>>>>>>>>> a_%s = %s <<<<<<<<<<"%(k, eVal[k])
	'''

	#np.log is base e
	logLike = 0

	# include the reduced determinant term
	logdetMdagger = 0
	determinantProduct = 1
	for i in range(nullspace, N):
		logdetMdagger = logdetMdagger + np.log(eVal[i])
		determinantProduct = determinantProduct*eVal[i]
	logDetError = 100*(logdetMdagger-np.log(determinantProduct))/logdetMdagger
	determinantError = 100*(math.exp(logdetMdagger)-determinantProduct)/(math.exp(logdetMdagger))

	if logDetError > 0.01 or determinantError > 0.0001: 
		print "DETERMINANT ERROR WAS BIG, %s OR LOG DETERMINANT ERROR WAS BIG, %s"%(determinantError,logDetError)
	#print "logdetMdagger = %s"%(logdetMdagger)
	logLike = logdetMdagger
	detContribution = logdetMdagger

	ncCorr = nnAvCorrelation(config, N, n_c, Jij)
	corrContribution = -rank*np.log(n_c*(1- ncCorr))
	logLike = logLike + corrContribution
	
	'''
	# alternate test of log likelihood calculation for stability
	otherLogLike = 0
	for i in range(nullspace, N):
		otherLogLike += np.log(eVal[i]/(n_c*(1- ncCorr)))
	print "log likelihood discrepancy = %s"%str(logLike-otherLogLike)
	'''

	# is the rank term indeed negligible?
	percentRank = 100* np.absolute(logLike - logdetMdagger)/logLike
	#print "The rank term term is %s %% of the log likelihood term"%(percentRank)

	return logLike, detContribution, corrContribution, determinantProduct, ncCorr

# finds optimal value of n_c for a single snapshot by maximizing its log liklihood
def OptimizeN_c(config, N, L, plotLogLikeVSn_c, ncMax):
	#ranges from 0 to N-1, and we just discount the 0th entry
	logLike = [0 for i in range(0, N-1)]
	detContribution = [0 for i in range(0, N-1)]
	corrContribution = [0 for i in range(0, N-1)]
	detProductVec = [0 for i in range(0, N-1)]
	ncCorrVec = [0 for i in range(0, N-1)]

	# time the process
	startTime = time.time()
	for nc in range(1, ncMax+1):
		# show progress of this lengthy calculation
		#percentComplete = str(100.*nc/(N-1))
		#percentComplete = percentComplete[:5] # only look at first 5 digits
		#elapsedTime = time.time() - startTime
		#estTime = (N-1)*elapsedTime/float(nc+1)
		#sys.stdout.write("\rTime Elapsed = %s"%(str(elapsedTime)[:7]))
		#sys.stdout.write("\r%s%%"%(percentComplete))
		#sys.stdout.write("\n estimated time to completion: %s"%(estTime))
		sys.stdout.write("\rtesting %s / %s "%(nc, ncMax))
		sys.stdout.flush()


 		# put log likelihoods for each n_c into a vector
		(logLike[nc-1], detContribution[nc-1], corrContribution[nc-1], detProductVec[nc-1], ncCorrVec[nc-1]) = logLikelihoodFn(config, N, nc, L)

		# plot log likelihood vs n_c when asked
		if plotLogLikeVSn_c == "yes":
			plt.figure(2)
			n_cVec = [i for i in range(1, ncMax+1)]
			#print "n_cVec = %s"%(n_cVec)
			#print "logLikelihood vector = %s"%(logLike)
			plt.title('Log Likelihood vs n_c')
			plt.ylabel('Log Likelihood (dimensionless)')
			plt.xlabel('n_c')
			plt.plot(n_cVec, logLike, 'ro')
			#plt.plot(n_cVec, detContribution, 'gv')
			#plt.plot(n_cVec, corrContribution, 'c^')
			plt.pause(0.001)
			#plt.clf()

	maxLikelihood = logLike[0]
	n_cStar = 1
	for i in range(ncMax):
		if logLike[i] > maxLikelihood:
			maxLikelihood = logLike[i]
			n_cStar = i+1

	# compute optimal Jsnap
	(Jij, Sij) = VicsekInteractionMatrices(config, N, n_cStar, L)
	ncCorrOpt = nnAvCorrelation(config, N, n_cStar, Jij)
	JsnapOpt = (N-1)/(N * n_cStar * (1-ncCorrOpt))

	# plot log likelihood vs n_c when asked
	if plotLogLikeVSn_c == "yes":
		plt.figure(2)
		n_cVec = [i for i in range(1, N)]
		#print "n_cVec = %s"%(n_cVec)
		#print "logLikelihood vector = %s"%(logLike)
		plt.title('Log Likelihood vs n_c')
		plt.ylabel('Log Likelihood (dimensionless)')
		plt.xlabel('n_c')
		plt.plot(n_cVec, logLike, 'ro')
		#plt.plot(n_cVec, detContribution, 'gv')
		#plt.plot(n_cVec, corrContribution, 'c^')
		plt.pause(0.01)
		return (n_cStar, JsnapOpt, detProductVec, ncCorrVec, plt.figure(2))
	return (n_cStar, JsnapOpt, detProductVec, ncCorrVec, "no figure")

# Iterates a Vicsek configuration over the next time step, metric of length r=1
def newConfigVicsekMetric(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L):
	config2 = [[0, 0, 0] for i in range(N)]
	# find average number of effectively interacting neighbors
	averageRadiusLNeighbors = 0

	# use fact that distance is symmetric
	# introduce matrix of relative distance, D
	Dsq = [[0 for i in range(N)] for j in range(N)]

	for n in range(N):
		spinx = 0
		spiny = 0
		# compute the average of all particle's directions (including the nth particle)
		neighbors = 0
		# also includes self-interaction
		for m in range(N):
			if m > n:
				distanceSq = VicsekDistanceSq(config, n, m, L)
				Dsq[m][n] = distanceSq
				Dsq[n][m] = distanceSq
			else: distanceSq = Dsq[m][n]
			# interaction distance is taken to be 1; adjust rho and L to change interaction length
			if distanceSq <= 1:
				if m==n: 
					spinx += np.cos(config[m][2])
					spiny += np.sin(config[m][2])
				else: 
					spinx += np.cos(config[m][2])*dtVic*JVic
					spiny += np.sin(config[m][2])*dtVic*JVic
					#self-interaction does not count towards n_c^EXP
					neighbors += 1 
		averageRadiusLNeighbors += neighbors

		theta = np.arctan2(spiny,spinx)
		# include noise term
		if noiseType == "Uniform":
			theta += dtVic*np.random.uniform(-eta/2., eta/2.)
		elif noiseType == "Gaussian":
			theta += dtVic*np.random.normal(0, sigmaVic)
		else: raise Exception('Invalid noise type')
		config2[n][2] = theta

		# compute new positions for the particles
		dx = nu * np.cos(config2[n][2])*dtVic
		dy = nu * np.sin(config2[n][2])*dtVic
		config2[n][0] = (config[n][0] + dx)%L
		config2[n][1] = (config[n][1] + dy)%L
	averageRadiusLNeighbors = averageRadiusLNeighbors/float(N)
	return config2

# Iterates a Vicsek configuration over the next time step, birds only directly account for info from their own
# neighborhoods; NOT SYMMETRIC
def newConfigVicsekTopo(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L):
	# introduce matrix of relative distance, D
	# The lists are then ordered by size, which quickly gives nearest neighbors
	Dsq = [[0 for i in range(N)] for j in range(N)]
	dSort = [[0 for i in range(N)] for j in range(N)]
	for n in range(N):
		for m in range(N):
			if m>n:
				distanceSq = VicsekDistanceSq(config, n, m, L)
				Dsq[m][n] = distanceSq
				Dsq[n][m] = distanceSq
				dSort[n][m] = distanceSq
				dSort[m][n]= distanceSq

	# now order by length, note dSort[i][0] is just 0
	for n in range(N):
		dSort[n].sort()
	#print "Sorted array is %s"%(dSort[5])

	# identify the sets of n_c neighborhoods
	nbhSet = [[0 for i in range(ncVic)] for j in range(N)]
	for i in range(N):
		for ncIndx in range(ncVic):
			#search for the neighbors
			for k in range(N):
				if Dsq[i][k] == dSort[i][ncIndx+1]:
					nbhSet[i][ncIndx] = k
					break
	#print "Naive topo neighborhood set = %s"%(nbhSet)

	# next update Vicsek positions/velocities (note the asymmetry) 
	# DON'T FORGET ABOUT THE SELF-COUNTING!!!!! not scaled by dt, J_vic
	config2 = [[0, 0, 0] for i in range(N)]
	for i in range(N):
		theta = 0
		# self-count first
		spinx = np.cos(config[i][2])
		spiny = np.sin(config[i][2])
		for ncIndx in range(ncVic):
			# grab the proper neighbors
			nhbr_ncIndx = nbhSet[i][ncIndx]
			spinx += JVic*dtVic*np.cos(config[nhbr_ncIndx][2])
			spiny += JVic*dtVic*np.sin(config[nhbr_ncIndx][2])
		# noiseless theta
		theta = np.arctan2(spiny, spinx)
		# noise term
		# include noise term
		if noiseType == "Uniform":
			theta += dtVic*np.random.uniform(-eta/2., eta/2.)
		elif noiseType == "Gaussian":
			theta += dtVic*np.random.normal(0, sigmaVic)
		else: raise Exception('Invalid noise type')

		config2[i][2] = theta

		# compute new positions for the particles
		dx = nu * np.cos(config2[i][2])*dtVic
		dy = nu * np.sin(config2[i][2])*dtVic
		config2[i][0] = (config[i][0] + dx)%L
		config2[i][1] = (config[i][1] + dy)%L
	return config2

# Iterates a Vicsek configuration over the next time step, 
def newConfigVicsekSymTopoWEIRDFIXVERSION(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L):
	config2 = [[0, 0, 0] for i in range(N)]
	(Jij, nij) = VicsekInteractionMatrices(config, N, ncVic, L) # remember nii = 0

	#compute n_i1 + n_i2 + ... + n_iN for m_ij scaling
	n_i = [0 for i in range(N)]
	for i in range(N):
		sum_ni=0
		for j in range(N):
			sum_ni += nij[i][j]
		n_i[i] = sum_ni
		#print sum_ni
			
	for i in range(N):
		theta = 0
		# self-count first
		spinx = np.cos(config[i][2])
		spiny = np.sin(config[i][2])
		for j in range(N):
			spinx += JVic*dtVic*np.cos(config[j][2]) * (nij[i][j]/np.sqrt(n_i[i]))
			spiny += JVic*dtVic*np.sin(config[j][2]) * (nij[i][j]/np.sqrt(n_i[i]))
		# noiseless theta
		theta = np.arctan2(spiny, spinx)
		# noise term
		if noiseType == "Uniform": theta += dtVic*np.random.uniform(-eta/2., eta/2.)
		elif noiseType == "Gaussian": theta += dtVic*np.random.normal(0, sigmaVic)
		else: raise Exception('Invalid noise type')
		config2[i][2] = theta

		# compute new positions for the particles
		dx = nu * np.cos(config2[i][2])*dtVic
		dy = nu * np.sin(config2[i][2])*dtVic
		config2[i][0] = (config[i][0] + dx)%L
		config2[i][1] = (config[i][1] + dy)%L
	#print "SymTopo neighborhood set = %s"%(nij)
	return config2


def newConfigVicsekSymTopo(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L):
	# Update equation for Vicsek flock, symmetric topological nbh (Sij)
	# ds_i = Arg(s_i + \sum Sij s_j) + noise * dt

	config2 = [[0, 0, 0] for i in range(N)]
	(Jij, Sij) = VicsekInteractionMatrices(config, N, ncVic, L) # remember nii = 0
	
	for i in range(N):
		theta = 0
		# self-count first
		spinx = np.cos(config[i][2])
		spiny = np.sin(config[i][2])
		for j in range(N):
			spinx += JVic*dtVic*np.cos(config[j][2]) * Sij[i][j]
			spiny += JVic*dtVic*np.sin(config[j][2]) * Sij[i][j]
		# noiseless theta
		theta = np.arctan2(spiny, spinx)
		# noise term
		if noiseType == "Uniform": theta += dtVic*np.random.uniform(-eta/2., eta/2.)
		elif noiseType == "Gaussian": theta += dtVic*np.random.normal(0, sigmaVic)
		else: raise Exception('Invalid noise type')
		config2[i][2] = theta

		# compute new positions for the particles
		dx = nu * np.cos(config2[i][2])*dtVic
		dy = nu * np.sin(config2[i][2])*dtVic
		config2[i][0] = (config[i][0] + dx)%L
		config2[i][1] = (config[i][1] + dy)%L
	#print "SymTopo neighborhood set = %s"%(nij)
	return config2

def newConfigVicsekLangevin(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L):
	config2 = [[0, 0, 0] for i in range(N)]
	(Jij, nij) = VicsekInteractionMatrices(config, N, ncVic, L)
	for i in range(N):
		theta = config[i][2]
		for j in range(N):
			theta += -JVic*dtVic*nij[i][j]*np.sin(theta-config[j][2])
		# noise term
		if noiseType == "Uniform": theta += dtVic*np.random.uniform(-eta/2., eta/2.)
		elif noiseType == "Gaussian": theta += np.random.normal(0, dtVic**.5)
		else: raise Exception('Invalid noise type')
		config2[i][2] = theta

		# compute new positions for the particles
		dx = nu * np.cos(config2[i][2])*dtVic
		dy = nu * np.sin(config2[i][2])*dtVic
		config2[i][0] = (config[i][0] + dx)%L
		config2[i][1] = (config[i][1] + dy)%L


	return config2

def newConfigVicsekLangevinOLDVERSION(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L):
	config2 = [[0, 0, 0] for i in range(N)]
	(Jij, nij) = VicsekInteractionMatrices(config, N, ncVic, L)
	for i in range(N):
		theta = config[i][2]
		for j in range(N):
			theta += -JVic*dtVic*nij[i][j]*np.sin(theta-config[j][2])
		# noise term
		if noiseType == "Uniform": theta += dtVic*np.random.uniform(-eta/2., eta/2.)
		elif noiseType == "Gaussian": theta += dtVic*np.random.normal(0, sigmaVic)
		else: raise Exception('Invalid noise type')
		config2[i][2] = theta

		# compute new positions for the particles
		dx = nu * np.cos(config2[i][2])*dtVic
		dy = nu * np.sin(config2[i][2])*dtVic
		config2[i][0] = (config[i][0] + dx)%L
		config2[i][1] = (config[i][1] + dy)%L


	return config2

def newConfigVicsek(interactionType, noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L):

	if interactionType == "metric": 
		config2 = newConfigVicsekMetric(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
	elif interactionType == "topo": 
		config2 = newConfigVicsekTopo(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
	elif interactionType == "symTopo":
		config2 = newConfigVicsekSymTopo(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
	elif interactionType == "langevin":
		config2 = newConfigVicsekLangevin(noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
	else: raise Exception('Invalid interaction type')

	return config2



