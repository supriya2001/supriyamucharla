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

A = np.array([3,0])
B = np.array([0,3])
C = np.array([-3,0])
len = 100

p = np.zeros(2)

O,r = ccircle(A,B,C)

theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + O).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')
O = np.array([0,0])
P = np.array([4,7])
l1 = np.linalg.norm(P-O) - 3
l2 = np.linalg.norm(P-O) + 3
D = l1*l2
print('the product of length is')
print(D)

len=10
x_OP = np.zeros((2,len))
lam = np.linspace(-0.5,1,len)
for i in range(len):
	temp1=O + lam[i]*(P-O)
	x_OP[:,i]= temp1.T

plt.plot(x_OP[0,:],x_OP[1,:])	

plt.plot(O[0],O[1],'o')
plt.text(O[0],O[1]*1.05,'O')
plt.plot(P[0],P[1],'o')
plt.text(P[0],P[1]*1.02,'P')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.axis('equal')
plt.show()
