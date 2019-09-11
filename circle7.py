import numpy as np
import matplotlib.pyplot as plt
import math as m

P=np.array([1,-1])
n1 = np.array([2,1])
n2 = np.array([1,-1])
t = np.array([3,1])
N = np.vstack([n1,n2])
O = np.linalg.inv(N)@t 

print('the equation of tangent is')
print(P-O,'x =',(P-O)@O)
