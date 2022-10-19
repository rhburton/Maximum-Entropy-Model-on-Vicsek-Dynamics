# Plot final data
import matplotlib.pyplot as plt
from scipy import stats

####################################################################################
####################################################################################
# plot (n*, ng) versus nv
nv = [5, 7, 9, 13, 17, 25, 35]
nArith =  [8.13, 20.63, 21.10, 32.37, 39.73, 58.3, 53.43]
stdDev_nArith = [3.68, 24.50, 21.38, 27.10, 27.52, 29.34, 27.11]
ng = [8, 19, 19, 31, 39, 57, 52]


plt.figure(1)
plt.errorbar(nv, nArith, yerr=stdDev_nArith, fmt="o")
plt.plot(nv, ng, 'ro')
# find least sqs linear regression for nGlobal
slope, intercept, r_value, p_value, std_err = stats.linregress(nv, ng)

plt.plot([1, 80], [1, 80], color='black')
plt.plot([0, 40], [intercept, 40*slope+intercept], color='red')
print "SLOPE OF FIGURE(1) LINEAR REGRESSION LINE IS = %s, INTERCEPT = %s"%(slope, intercept)

plt.plot()

plt.xlabel('Interaction Length of Vicsek Flock (nV)')
plt.ylabel('Arithmetic Average of MEM (<n*>, ng)')
plt.title('MEM Interaction Ranges vs Vicsek Interaction Length (nV) ')
axes = plt.gca()
axes.set_xlim([0,90])
axes.set_ylim([0,90])

'''
plt.figure(2)
plt.plot(nv, ng, 'bo')
plt.plot()

plt.xlabel('nV of Vicsek Flock')
plt.ylabel('n_g, Global estimate for nV')
plt.title('Inferred Global Average ng vs Actual nV of Vicsek Flock ')
axes = plt.gca()
axes.set_xlim([0,90])
axes.set_ylim([0,90])
'''

####################################################################################
####################################################################################
# v = 0
eta = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3, 1.6]
ngV0 = [9, 10, 11, 9, 9, 12, 13, 10, 10, 10, 9]
LikeNgV0 = [127.32, 84.82, 59.91, 37.58, 20.38, 3.17, -8.27, -20.28, -32.02, -55.06, -74.09]



# v=0.05
ngV05 = [8, 9, 10, 11, 11, 12, 12, 13, 10, 12, 20]
LikeNgV05 = [125.59, 87.04, 58.26, 36.97, 19.82, 3.18, -9.03, -18.72, -28.19, -53.56, -74.6]

plt.figure(3)
plt.plot(eta, ngV0, 'rd')
plt.plot(eta, ngV05, 'go')

slopeV0, interceptV0, r_valueV0, p_valueV0, std_errV0 = stats.linregress(eta, ngV0)
slopeV05, interceptV05, r_valueV05, p_valueV05, std_errV05 = stats.linregress(eta, ngV05)
print stats.linregress([0, 1], [0, 1])
print "slopeV0 = %s, interceptV0 = %s"%(slopeV0, interceptV0)
print "slopeV05 = %s, interceptV05 = %s"%(slopeV05, interceptV05)
xV0 = [0, 1.8]
yV0 = [interceptV0, 1.8*slopeV0+interceptV0]
xV05 = [0, 1.8]
yV05 = [interceptV05, 1.8*slopeV05+interceptV05]
plt.plot(xV0, yV0, color="red")
plt.plot(xV05, yV05, color="green")

plt.plot()
plt.xlabel('Noise (eta)')
plt.ylabel('Global Interaction Range (ng)')
plt.title('Global Interaction Range (ng) vs Noise (eta)')
axes = plt.gca()
axes.set_xlim([0,1.8])
axes.set_ylim([0,21])

#plt.figure(100)
#slopeV0, interceptV0, r_valueV0, p_valueV0, std_errV0 = stats.linregress(eta, ngV0)
#slopeV05, interceptV05, r_valueV05, p_valueV05, std_errV05 = stats.linregress(eta, ngV05)
#plt.plot([0, 1.8], [interceptV0, 1.8*slopeV0])
#plt.plot([0, 1.8], [interceptV05, 1.8*slopeV05])
#plt.xlabel('noise (eta)')
#plt.ylabel('ng (blue nu=0, red nu=0.05)')
#plt.title('Global interaction range (ng) vs Noise (eta)')
#axes = plt.gca()
#axes.set_xlim([0,1.8])
#axes.set_ylim([0,21])


plt.figure(4)
plt.plot(eta, LikeNgV0, 'ro')
plt.plot(eta, LikeNgV05, 'go')
plt.plot()
plt.xlabel('Noise (eta)')
plt.ylabel('LogLikelihood of ng')
plt.title('Log Likelihood of ng vs Noise (eta)')
axes = plt.gca()
axes.set_xlim([0, 1.8])
axes.set_ylim([-80,130])


####################################################################################
####################################################################################
v = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
Jg = 2.97, 3.33, 4.63, 3.71, 3.38, 2.82, 4.25, 3.67, 3.06, 3.11, 3.64
ng = 12, 11, 8, 10, 11, 12, 9, 11, 12, 12, 11
likeNg = 37.54, 36.28, 36.07, 34.55, 36.19, 35.07, 36.11, 38.11, 36.04, 35.07, 37.88




plt.figure(5)
plt.plot(v, Jg, 'bo')

slopeJgvsV, interceptJgvsV, r_valueV0, p_valueV0, std_errV0 = stats.linregress(v, Jg)
print "slopeJgvsV = %s, interceptJgvsV = %s"%(slopeJgvsV, interceptJgvsV)
xNu = [0, .1]
yJg = [interceptJgvsV, .1*slopeJgvsV+interceptJgvsV]
plt.plot(xNu, yJg, color="blue")

plt.plot()
plt.xlabel('Vicsek Velocity (nu)')
plt.ylabel('MEM Global Interaction Strength (Jg)')
plt.title('MEM Global Interaction Strength (Jg) vs Vicsek Velocity (nu)')
axes = plt.gca()
axes.set_xlim([0, .11])
axes.set_ylim([0,5])

plt.figure(6)
plt.plot(v, ng, 'bo')

slopengvsV, interceptngvsV, r_valueV0, p_valueV0, std_errV0 = stats.linregress(v, ng)
print "slopengvsV = %s, interceptngvsV = %s"%(slopengvsV, interceptngvsV)
xNu = [0, .1]
yng = [interceptngvsV, .1*slopengvsV+interceptngvsV]
plt.plot(xNu, yng, color="blue")

plt.plot()
plt.xlabel('Vicsek Velocity (nu)')
plt.ylabel('MEM Global Interaction Range (ng)')
plt.title('MEM Global Interaction Range (ng) vs Vicsek Velocity (nu)')
axes = plt.gca()
axes.set_xlim([0, .11])
axes.set_ylim([0,13])

plt.figure(7)
plt.plot(v, likeNg, 'bo')
plt.plot()
plt.xlabel('Vicsek Velocity (nu)')
plt.ylabel('LogLikelihood of ng, LogLike(ng)')
plt.title('LogLikelihood of ng vs Vicsek Velocity (nu)')
axes = plt.gca()
#axes.set_xlim([0, 1.8])
axes.set_ylim([0,50])




plt.show()