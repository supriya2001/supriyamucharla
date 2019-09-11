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

len=10000
O = np.array([2,-3])
r = 5
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + O).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle1$')
plt.plot(O[0],O[1],'o')
plt.text(O[0],O[1]*1.05,'O')


O = np.array([-3,2])
r = 5*(3**0.5)
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + O).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle2$')
plt.plot(O[0],O[1],'o')
plt.text(O[0],O[1]*1.05,'O')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.axis('equal')
plt.show()
