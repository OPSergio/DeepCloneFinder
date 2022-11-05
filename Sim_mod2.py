# Importación de librerías
import numpy as np
import random
import pandas as pd
import io
import matplotlib.pyplot as plt

from numpy.lib.function_base import delete
from numpy.ma.core import size
Original= pd.read_csv('C:/Users/solmo/Desktop/Master/TFM/Pseudo_resume.csv', sep= ';')

A= np.array(Original)
Base= np.array(Original)
B=np.array(Original)
C=np.array(Original)
D=np.array(Original)


for i in range(200):
    for k in range (6):
      x= random.randint(0,3)
      y= random.randint(0,199)
      m= random.randint(0,3)
      n= random.randint(0,199)
      z= random.randint(0,3)
      s= random.randint(0,199)
      o= random.randint(0,3)
      p= random.randint(0,199)
      A[:,y]=0
      A[x][y]=1
      B[:,n]=0
      B[m][n]=1
      C[:,s]=0 
      C[z][s]=1
      D[:,p]=0
      D[o][p]=1
    E= (4*A+2*B)/6
    np.savetxt(("Matriz"+ str(i) + ".txt"), E, delimiter= ', ' )
    A=np.array(Original)
    B=np.array(Original)
    C=np.array(Original)
    D=np.array(Original)
print("Fin")
print(E)

    

