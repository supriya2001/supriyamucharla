import numpy as np
import matplotlib.pyplot as plt
import math as m

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

len=100
O = np.array([-2,2])
r = 2
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + O).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle1$')


plt.plot(O[0],O[1],'o')
plt.text(O[0],O[1]*1.05,'O')


a1 = np.array([1.5,0])
a2 = np.array([0,1.2]) 
b1 = np.array([-5,0])
b2 = np.array([-2,2])
c1 = np.array([1,0])
c2 = np.array([5,-3])
d1 = np.array([-0.8,0])
d2 = np.array([0,-2])

len=10

A = np.zeros((2,len))
lam = np.linspace(-2,2,len)
for i in range(len):
	temp1=a1+lam[i]*(a2-a1)
	A[:,i]=temp1.T

plt.plot(A[0,:],A[1,:],label='option A')
plt.plot(a1[0], a1[1], 'o')
plt.plot(a2[0], a2[1], 'o')



B = np.zeros((2,len))
lam = np.linspace(-2,2,len)
for i in range(len):
	temp1=b1+lam[i]*(b2-b1)
	B[:,i]=temp1.T

plt.plot(B[0,:],B[1,:],label='option B')
plt.plot(b1[0], b1[1], 'o')
plt.plot(b2[0], b2[1], 'o')



C = np.zeros((2,len))
lam = np.linspace(-2,2,len)
for i in range(len):
	temp1=c1+lam[i]*(c2-c1)
	C[:,i]=temp1.T

plt.plot(C[0,:],C[1,:],label='option C')
plt.plot(c1[0], c1[1], 'o')
plt.plot(c2[0], c2[1], 'o')



D = np.zeros((2,len))
lam = np.linspace(-2,2,len)
for i in range(len):
	temp1=d1+lam[i]*(d2-d1)
	D[:,i]=temp1.T

plt.plot(D[0,:],D[1,:],label='option D')
plt.plot(a1[0], a1[1], 'o')
plt.plot(a2[0], a2[1], 'o')

plt.legend(loc='best')
plt.grid
plt.show()
