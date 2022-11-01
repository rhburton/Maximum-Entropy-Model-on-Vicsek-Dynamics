# Simulate Vicsek flocking, write data to files
import numpy as np
import ThesisFunctions as tf
import math
import matplotlib.pyplot as plt
import sys

fileName = "VicsekData.txt"
f = open(fileName,"w+")

# symTopo is a symmetric topological nearest neighbors rule
interactionType = "symTopo" # POSSIBLE OPTIONS: "metric", "topo", "symTopo"
noiseType = "Uniform" 		# POSSIBLE OPTIONS: "Gaussian", "Uniform"
keepPlot = True
# initialize Vicsek variables (time-continuous)
N=100			# number of particles
L=1 			# system length/width
JVic = 1		# Vicsek interaction strength
dtVic = 1 		# Vicsek timestep
nu= 0.05			# velocity (units of system length per dtVic)

# IF TOPOLOGICAL: 
ncVic = 5		# number of Vicsek interacting neighbors
# SELECT NOISE TYPE
betaVic = 1 	#IF GAUSSIAN NOISE
sigmaVic = 2./betaVic

eta = 0.3   	#IF UNIFORM NOISE (see definition after 3.1 in thesis, units of radians (?))

#config = (particle number, (xpos, ypos, angle))
config = [[np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, 2*math.pi)] for i in range(N)]
config = [[np.random.uniform(0, L), np.random.uniform(0, L), 1] for i in range(N)]

# Determinant Product is zero if the flocks are non-interacting (if so, throw out this run)
if interactionType != "metric":
	logDetProd = tf.checkVicsekNcSize(config, N, ncVic, L)
print "Determinant Product = %s"%logDetProd

# <Phi>_exp from equation 4.14 with the approximation N >> 1
expectedCorrNcV = 1 - 1./(JVic*betaVic*ncVic)
print "MEM expects an n_c nearest neighbor correlation of %s"%expectedCorrNcV



######################################################################
# Parameters governing when flocks are polled
######################################################################
t=0 						# frame number
framesToEquilibrium = 100 	# wait this many frames before beginning draw
snapshotsPerFlock = 1 		# record this many snapshots in one poll 
pollInterval = 50		 	# number of frames between recording intervals
numFlocks = 30      		# number of times we poll for flock data
plotFlockThisOften = 1 		# plot the flock every this many frames

#write initial data to line 1
f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\n"%(interactionType, noiseType, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L, framesToEquilibrium,snapshotsPerFlock, pollInterval, numFlocks))


# Evolve the flock over time
cutoffM = 0.999 		# say flock has thermalized once the magnetization reaches this threshold
cutoffCorrNcV = 0.3 	# say flock has thermalized once this nearest neighbor correlation has been reached

# This first while loop is to reach thermal equilibrium
while True:
	sys.stdout.write("\rPrepping Frame %s"%t)
	sys.stdout.write("\r")
	sys.stdout.flush()
	config = tf.newConfigVicsek(interactionType, noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
	t+=1
	# check in on flock progress every X frames
	if t%plotFlockThisOften ==0: 
		tf.PlotVicsek(config, N, L, keepPlot)
		currentMagnetization = tf.Magnetization(config, N)[0]
		CorrNcV = tf.nnAvCorrelation(config, N, ncVic, 0)
		JBeta_MEM = 1/(ncVic * (1-CorrNcV))
		print "Magnetization = %s, CorrNcV = %s, JBeta_MEM = %s"%(currentMagnetization, CorrNcV, JBeta_MEM)
	# check thermal cutoffs every so often
	if t%framesToEquilibrium==0: 
		CorrNcV = tf.nnAvCorrelation(config, N, ncVic, 0)
		(M, theta0) = tf.Magnetization(config, N)
		# this cutoff is to alert us if the flock has very very high magnetization 
		# (MEM model is unstable in this situation)
		if M > cutoffM:
			print "\rM cutoff reached"
			for j in range(framesToEquilibrium):
				config = tf.newConfigVicsek(interactionType, noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
			break
		if CorrNcV > cutoffCorrNcV:
			print "\rCorrNcV cutoff reached"
			for j in range(framesToEquilibrium):
				config = tf.newConfigVicsek(interactionType, noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
			break

tf.PlotVicsek(config, N, L, keepPlot)
print "Reached Equlibrium"
# record data for numFlocks number of short bursts of consecutive frames
for i in range(numFlocks):
	print "\rTesting flock number %s"%i
	# loop for burst of consecutive frames
	for j in range(snapshotsPerFlock):
		config = tf.newConfigVicsek(interactionType, noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
		#tf.PlotVicsek(config, N, L, keepPlot)
		f.write("%s\n"%(config))
		tf.PlotVicsek(config, N, L, keepPlot)
		# print to consol the CorrNcV and estimated Jbeta info
		CorrNcV = tf.nnAvCorrelation(config, N, ncVic, 0)
		JBeta_MEM = 1/(ncVic * (1-CorrNcV))
		print "CorrN_c = %s, JBeta_MEM = %s"%(tf.nnAvCorrelation(config, N, ncVic, 0), JBeta_MEM)

	#tf.PlotVicsek
	print "flock %s recorded"%(i)
	if i+1 == numFlocks: break
	# this is the wait between bursts
	for j in range(pollInterval):
		sys.stdout.write("\rPoll refreshing %s / %s "%(j, pollInterval))
		sys.stdout.flush()
		config = tf.newConfigVicsek(interactionType, noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
print "\rFinished"
f.close()
plt.show()
