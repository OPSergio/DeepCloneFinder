# Importación de librerías
import numpy as np
import random
import pandas as pd
import io
import matplotlib.pyplot as plt

from Sim_mod2 import C


# Simulador de matrices.
## Matriz de probabilidad.
## Nº de Alelos
## Nº de Mutaciones/alelos
x= random.randint(0,3)
y= random.randint(0,9)

randomarray= np.random.randint(2 , size=(4, 10))
print(randomarray)
A = randomarray
A[:,y]= 0
A[x][y]=1

# Bucle para la realización de mutaciones

x= random.randint(0,3)
y= random.randint(0,9)

randomarray= np.random.randint(2 , size=(4, 10))
A= randomarray
A[:,y]= 0
A[x][y]=1

for i in range(5):
  x= random.randint(0,3)
  y= random.randint(0,9)
  A[:,y]= 0
  A[x][y]=1
  B= (A+randomarray)/2 
  print(B)
print("Fin")

# CREACION DATASET
## 50 MATRICES PARA 1,2,3 Y 4 MUTACIONES
### Partimos del gen 16 rRNA de Pseudomona 

#Cargamos matriz original

from numpy.lib.function_base import delete
from numpy.ma.core import size
Original= pd.read_csv('C:/Users/solmo/Desktop/Master/TFM/Pseudo_resume.csv', sep= ';')

A= np.array(Original)
Base= np.array(Original)
print(A)

##Creamos matrices 1 mutacion.
#for i in range(100):
  #x= random.randint(0,3)
  #y= random.randint(0,200)
  #A[:,y]= 0
  #A[x][y]=1
  #B= (A+Base)/2 
  #np.savetxt(("Una_mutacion"+ str(i) + ".txt"), B, delimiter= ', ' )
#print("Fin")
#print(B.shape)

#Creamos matrices 2 mutaciones

#for i in range(100):
  #for j in range(2): 
    #x= random.randint(0,3)
    #y= random.randint(0,199)
    #A[:,y]=0
    #A[x][y]=1
  #B= (A+Base)/2 
  #np.savetxt(("Dos_mutacion"+ str(i) + ".txt"), B, delimiter= ', ' )
#print("Fin")

#Creamos matrices 3 mutaciones

#for i in range(100):
  #for j in range(3): 
    #x= random.randint(0,3)
    #y= random.randint(0,199)
    #A[:,y]=0
    #A[x][y]=1
  #B= (A+Base)/2 
  #np.savetxt(("Tres_mutacion"+ str(i) + ".txt"), B, delimiter= ', ' )
#print("Fin")

### Del mismo modo, modificando los diferentes valores del bucle, obtenemos el resto de datos.

for i in range(50):
  for j in range (3):
    for k in range (1):
      x= random.randint(0,3)
      y= random.randint(0,199)
      A[:,y]=0
      A[x][y]=1
    B = (A+Base)
  C= (B)/(4)
  #np.savetxt(("Tres_Tres"+ str(i) + ".txt"), C, delimiter= ', ' )
print("Fin")
print(C)