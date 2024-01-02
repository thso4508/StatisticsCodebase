'Following is a gathering of useful code snippets for statistical caclulations'

## Numpy linear algebra.

'Matrix manipulation and etcetera'

from numpy import linalg as LA
a = np.array([[1, 1j], [-1j, 1]])
eigenvalues, eigenvectors = LA.eig(a)

'''DOCUMENTATION:
https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html'''

'Singular vectors'
import numpy.linalg.svd

a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
U, S, Vh = np.linalg.svd(a, full_matrices=True)

'''DOCUMENTATION:
https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html'''

## Probability 1.

## Distribution functions

'Binomial probability functions'
from scipy.stats import binom
'The binomial distribution is the probability distribution for the number of successes in a sequence of Bernoulli trials'

event = binom('number of tries', 'probability') 

P_N = binom.ppf('x amount of succeses', 'number of tries', 'probability') 
# The Cumulative Distribution Function or CDF is:
#The probability of all outcomes less than or equal to a given value x,
#Graphically, this is the the total area of everything less than or equal to x (**the total area of the left of x*)

P_N = binom.cdf('x amount of succeses', 'number of tries', 'probability') 
#The Probability Point Function or PPF is the inverse of the CDF. 
#Specifically, the PPF returns the exact point where the probability of everything to the left is equal to y.
#This can be thought of as the percentile function since the PPF tells us the value of a given percentile of the data.

P_N = binom.pmf('x amount of succeses', 'number of tries', 'probability')
# PDF / PMF: Probability {Mass/Density} Functions
#The .pmf() and .pdf() functions find the probability of an event at a specific point in the distribution.
#The Probability Mass Function (PMF) -- or .pmf() -- is only defined on discrete distributions where each event has a fixed probability of occurring.
#The Probability Density Function (PDF) -- or .pdf() -- is only defined on continuous distributions where it finds the probability of an event occurring within a window around a specific point.

r = binom.rvs('number of tries' 'probability', size=1000)
#The .rvs() function returns a random sample of the distribution with probability equal to the distribution.binm if something is 80% likely, that value will be sampled 80% of the time.
#In coinflip, we expect more results with 1 (50% occurrence of 1 head) than 0 or 2 (25% occurrence of either zero heads or two heads).

mean, var, skew,  = binom.stats('number of tries', 'probability', moments='mvs')
#The mean, variance, skewness, kurt

'''Good sites binomial...
DOCUMENTATION:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html

https://discovery.cs.illinois.edu/learn/Polling-Confidence-Intervals-and-Hypothesis-Testing/Python-Functions-for-Random-Distributions/
https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Discrete-Random-Variables/The-Binomial-Distribution/The-Binomial-Distribution/index.html'''



'Bernoulli probability distr functions'
from scipy.stats import bernoulli

#The bernoulli is similiar to binom
#Pmf for bernoulli is f(k) = 1-p if k = 0 else p if k = 1

prob = bernoulli.cdf('x amount of succeses', 'number of tries' ,'probability')

'''Good sites bernoulli...
DOCUMENTATION:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html'''

'Normal probability istr functions / Gaussian PDF'
from scipy.stats import norm

#The norm is similiar to binom
#Pmf for bernoulli is f(k) = (e^(-x^2 / 2) / sqrt (2*pi))

vals = norm.pdf(loc='mean', scale='standard deviation')

'''Good sites norm...
DOCUMENTATION:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

https://proclusacademy.com/blog/practical/normal-distribution-python-scipy/'''

#Generators
import random
import sympy

#For a Linear Congruent Generator c and m should be a random prime number
'LCG'
r_list = list(range(size))
r_list[0] = seed
a = random.random()+seed
c = sympy.randprime(1,100)
m = sympy.randprime(1,100)
while c < seed:
        c = sympy.randprime(1,100)
while m < seed:    
        m = sympy.randprime(1,100)
for i in range(1,size):
        r_list[i] = (((i * a) + c) % m)  


#Uniform Distribution
        '''Following will create a uniform distribution based upon a list '''
import numpy as np
max_v = np.maximum(r_list)
min_v = np.minimum(r_list)
treshold = max_v - min_v
for i in range(1,len(r_list)):
        if r_list[i] > treshold:
                r_list[i] = 1
        else:
                r_list[i] = 0