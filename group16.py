import numpy as np
import matplotlib.pyplot as plt

O = np.array([0,0])

def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return omat@dir_vec(A,B)
  
def line_gen(A,B):
  len =10
  x_AB = np.zeros((2,len))
  lam_1 = np.linspace(-1,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
  
def linegen(A,B):
  len =10
  x_AB = np.zeros((2,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
  

a=-1

n1 = np.array([1,a-1])
n2 = np.array([2,a**2])

N = np.vstack([n1,n2])

q = np.zeros(2)
q[0]=1
q[1]=1
H = np.linalg.inv(N)@q

print (H)







A = np.array([1,0]) 
B = np.array([3,1])
C = np.array([0,1])
D = np.array([1,-1])
P = H



x_AB = line_gen(A,B)
x_CD = linegen(C,D)
x_OP = linegen(O,P)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_OP[0,:],x_OP[1,:],label='$OP$')



plt.plot(A[0], A[1], '.')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], '.')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], '.')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], '.')
plt.text(D[0] * (1 + 0.1), D[1] * (1 - 0.1) , 'D')
plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')
plt.plot(O[0], O[1], 'o')
plt.text(O[0] * (1 + 0.1), O[1] * (1 - 0.1) , 'O')
print ('the distance between the meeting point and the origin is ',(((.6**2)+(.2**2))**.5))





plt.legend(loc='best')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
