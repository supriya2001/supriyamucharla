import numpy as np
import matplotlib.pyplot as plt
import math as m

import subprocess
import shlex

def ccircle(A,B,C):
  p = np.zeros(2)
  n1 = dirvec(B,A)
  p[0] = 0.5*(np.linalg.norm(A)**2-np.linalg.norm(B)**2)
  n2 = dirvec(C,B)
  p[1] = 0.5*(np.linalg.norm(B)**2-np.linalg.norm(C)**2)
  #Intersection
  N=np.vstack((n1,n2))
  O=np.linalg.inv(N)@p
  r = np.linalg.norm(A -O)
  return O,r

def line_gen(A,B):
  len =10
  x_AB = np.zeros((2,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def dirvec(A,B):
  return B-A

def normvec(A,B):
  return omat@dirvec(A,B)
  
omat = np.array([[0,1],[-1,0]])

A = np.array([-1,-1])
B = np.array([2,2])
C = np.array([-4,2])
len = 100

p = np.zeros(2)

O,r = ccircle(A,B,C)

theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + O).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')

A = np.array([2,2])
B = np.array([5,-1])
C = np.array([8,2])
len = 100

p = np.zeros(2)

O,r = ccircle(A,B,C)

theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + O).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')

O1=np.array([-1,2])
P=np.array([2,2])
O2=2*P-O1
print('the intercept on x axis is')
print(2*((9-(O2[1]**2))**0.5))

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.axis('equal')
plt.show()
