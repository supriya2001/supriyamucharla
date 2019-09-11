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

A= np.array([2,3])
B= np.array([4,5])
C= (A+B)/2

n1= dirvec(A,B)
n2= np.array([-1,4])
c1= n1@C
c2= 3

t= np.array([c1,c2])
N = np.vstack([n1,n2])
O = np.linalg.inv(N)@t

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(O[0], O[1], 'o')
plt.text(O[0] * (1 + 0.03), O[1] * (1 - 0.1) , 'O')

C= 2*O-B
len = 100

p = np.zeros(2)

O,r = ccircle(A,B,C)

#Generating all lines
x_AB = line_gen(A,B)
x_BO = line_gen(B,O)
x_OA = line_gen(O,A)
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + O).T

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BO[0,:],x_BO[1,:],label='$BO$')
plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='upper right')
plt.grid() # minor
plt.axis('equal')

print(dirvec(O,A))
print('radius is')
print(np.linalg.norm(A-O))

plt.show()
