# Importación de librerías
import numpy as np
import random
import pandas as pd
import io
import matplotlib.pyplot as plt


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

#Creamos matrices 1 mutacion.
#for i in range(50):
  #x= random.randint(0,3)
  #y= random.randint(0,200)
  #A[:,y]= 0
  #A[x][y]=1
  #B= (A+Base)/2 
  #np.savetxt(("Una_mutacion"+ str(i) + ".txt"), A, delimiter= ', ' )
#print("Fin")

#Creamos matrices 2 mutaciones

for i in range(45):
  for j in range(2): 
    x= random.randint(0,3)
    y= random.randint(0,200)
    A[:,y]= 0
    A[x][y]=1
  B= (A+Base)/2 
  np.savetxt(("Dos_mutacion"+ str(i) + ".txt"), A, delimiter= ', ' )
print("Fin")

