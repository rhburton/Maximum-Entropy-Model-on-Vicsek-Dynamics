# Simulate flocking events, write data to files
import numpy as np
import ThesisFunctions as tf
import math
import matplotlib.pyplot as plt
import sys




fileName = "asdfasdfasdf,10.txt"
f = open(fileName,"w+")

interactionType = "symTopo" # POSSIBLE OPTIONS: "metric", "topo", "symTopo", "langevin"
noiseType = "Uniform" 		# POSSIBLE OPTIONS: "Gaussian", "Uniform"
keepPlot = "yes"
# initialize Vicsek variables (time-continuous)
N=100			# number of particles
L=1 			# system length
JVic = 1		# Vicsek interaction strength
dtVic = 1 		# Vicsek timestep
nu= 0.05			# velocity

# IF TOPOLOGICAL: 
ncVic = 5		# number of Vicsek interacting neighbors
# SELECT NOISE TYPE
betaVic = 1 	#IF GAUSSIAN NOISE
sigmaVic = 2./betaVic

eta = 0.3   	#IF UNIFORM NOISE

#config = (particle#, (xpos, ypos, angle))
config = [[np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, 2*math.pi)] for i in range(N)]
config = [[np.random.uniform(0, L), np.random.uniform(0, L), 1] for i in range(N)]

# make sure the n_c choice doesn't give disjoint flocks
if interactionType != "metric":
	logDetProd = tf.checkVicsekNcSize(config, N, ncVic, L)
print "Determinant Product = %s"%logDetProd

expectedCorrNcV = 1 - 1./(JVic*betaVic*ncVic)
print "MEM expects an NC corr of %s"%expectedCorrNcV



######################################################################
# Parameters governing when flocks are polled
######################################################################
t=0 			# frame number
framesToEquilibrium = 100 # wait this many frames before beginning draw
snapshotsPerFlock = 1 	# record this many snapshots in one poll 
pollInterval = 50 # number of frames between recording intervals
numFlocks = 30      	# number of times we poll for flocks
plotFlockThisOften = 1

#write initial data to line 1
f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\n"%(interactionType, noiseType, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L, framesToEquilibrium,snapshotsPerFlock, pollInterval,numFlocks))


# Evolve the flock over time
# get to equilibrium first
cutoffM = 0.999
cutoffCorrNcV = 0.3
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
	if t%framesToEquilibrium==0: 
		CorrNcV = tf.nnAvCorrelation(config, N, ncVic, 0)
		(M, theta0) = tf.Magnetization(config, N)
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
for i in range(numFlocks):
	print "\rTesting flock number %s"%i
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
	for j in range(pollInterval):
		sys.stdout.write("\rPoll refreshing %s / %s "%(j, pollInterval))
		sys.stdout.flush()
		config = tf.newConfigVicsek(interactionType, noiseType, config, N, ncVic, sigmaVic, eta, nu, dtVic, JVic, L)
print "\rFinished"
f.close()
plt.show()