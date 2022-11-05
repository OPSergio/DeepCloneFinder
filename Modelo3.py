import imp
from tkinter import E
from turtle import onclick
import numpy as np
import random
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
from sklearn.svm import OneClassSVM
import tensorflow as tf
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import pickle
import time
from keras.callbacks import TensorBoard
from keras.layers import MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import Convolution2D as Con2D
from keras.optimizers import SGD




### GENERAMOS LAS MATRICES QUE CONTENGAN LOS DATOS PARA TODAS LAS CATEGORIAS

ruta_uno= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/uno"
uno = []

for i in os.listdir (ruta_uno) :
    i= pd.read_csv(os.path.join(ruta_uno,i), header= None)
    uno.append(i)

uno= np.array(uno)
print(uno.shape)

ruta_dos= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/dos"

dos = []

for i in os.listdir (ruta_dos) :
    i= pd.read_csv(os.path.join(ruta_dos,i), header= None)
    dos.append(i)

dos= np.array(dos)
print(dos.shape)

ruta_tres= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/tres"

Tres = []

for i in os.listdir (ruta_tres) :
    i= pd.read_csv(os.path.join(ruta_tres,i), header= None)
    Tres.append(i)

Tres= np.array(Tres)
print(Tres.shape)


ruta_cuatro= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/cuatro"

cuatro = []

for i in os.listdir (ruta_cuatro) :
    i= pd.read_csv(os.path.join(ruta_cuatro,i), header= None)
    cuatro.append(i)

cuatro= np.array(cuatro)
print(cuatro.shape)

ruta_cinco= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/cinco"

cinco = []

for i in os.listdir (ruta_cinco) :
    i= pd.read_csv(os.path.join(ruta_cinco,i), header= None)
    cinco.append(i)

cinco= np.array(cinco)
print(cinco.shape)

ruta_seis= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/seis"

seis = []

for i in os.listdir (ruta_seis) :
    i= pd.read_csv(os.path.join(ruta_seis,i), header= None)
    seis.append(i)

seis= np.array(seis)
print(seis.shape)


ruta_siete= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/siete"

siete = []

for i in os.listdir (ruta_siete) :
    i= pd.read_csv(os.path.join(ruta_siete,i), header= None)
    siete.append(i)

siete= np.array(siete)
print(siete.shape)

ruta_ocho= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/ocho"

ocho = []

for i in os.listdir (ruta_ocho) :
    i= pd.read_csv(os.path.join(ruta_ocho,i), header= None)
    ocho.append(i)

ocho= np.array(ocho)
print(ocho.shape)

ruta_nueve= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/nueve"

nueve = []

for i in os.listdir (ruta_nueve) :
    i= pd.read_csv(os.path.join(ruta_nueve,i), header= None)
    nueve.append(i)

nueve= np.array(nueve)
print(nueve.shape)

ruta_diez= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/diez"

diez = []

for i in os.listdir (ruta_diez) :
    i= pd.read_csv(os.path.join(ruta_diez,i), header= None)
    diez.append(i)

diez= np.array(diez)
print(diez.shape)

ruta_once= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/once"

once = []

for i in os.listdir (ruta_once) :
    i= pd.read_csv(os.path.join(ruta_once,i), header= None)
    once.append(i)

once= np.array(once)
print(once.shape)

ruta_doce= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/doce"

doce = []

for i in os.listdir (ruta_doce) :
    i= pd.read_csv(os.path.join(ruta_doce,i), header= None)
    doce.append(i)

doce= np.array(doce)
print(doce.shape)

ruta_trece= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/trece"

trece = []

for i in os.listdir (ruta_trece) :
    i= pd.read_csv(os.path.join(ruta_trece,i), header= None)
    trece.append(i)

trece= np.array(trece)
print(trece.shape)

ruta_catorce= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/catorce"

catorce = []

for i in os.listdir (ruta_catorce) :
    i= pd.read_csv(os.path.join(ruta_catorce,i), header= None)
    catorce.append(i)

catorce= np.array(catorce)
print(catorce.shape)

ruta_quince= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/quince"

quince = []

for i in os.listdir (ruta_quince) :
    i= pd.read_csv(os.path.join(ruta_quince,i), header= None)
    quince.append(i)

quince= np.array(quince)
print(quince.shape)

ruta_dieciseis= "C:/Users/solmo/Desktop/Master/TFM/Modelo3/dieciseis"

dieciseis = []

for i in os.listdir (ruta_dieciseis) :
    i= pd.read_csv(os.path.join(ruta_dieciseis,i), header= None)
    dieciseis.append(i)

dieciseis= np.array(dieciseis)
print(dieciseis.shape)


r_orig= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/Orig"

orig = []

for i in os.listdir (r_orig) :
    i= pd.read_csv(os.path.join(r_orig,i), header= None)
    orig.append(i)

orig= np.array(orig)
print(orig.shape)

##UNIFICAMOS TODOS LOS DATOS EN UNA UNICA MATRIZ


Datos_modelo= np.concatenate([uno,dos,Tres,cuatro,cinco,seis,siete,ocho,nueve,diez,once,doce,catorce,quince,orig])
print(len(Datos_modelo))
Datos_Modelo= np.array(Datos_modelo)
print(Datos_Modelo.shape)

#ReshapeDatos = Datos_Modelo.reshape(list(Datos_Modelo.shape) + [1])
#print(ReshapeDatos.shape)

###GENERAMOS LAS ETIQUETAS

##Generamos las etiquetas  (Labels) Teniendo en cuenta el orden
L_cero= np.repeat (0,100)
L_uno = np.repeat(1,200)
L_dos = np.repeat(2,200)
L_tres = np.repeat(3,200)
L_cuatro = np.repeat(4,200)
L_cinco= np.repeat (5,200)
L_seis= np.repeat (6,200)
L_siete= np.repeat (7,200)
L_ocho= np.repeat (8,200)
L_nueve= np.repeat (9,200)
L_diez= np.repeat (10,200)
L_once= np.repeat (11,200)
L_doce= np.repeat (12,200)
#L_trece= np.repeat (2,200)
L_catorce= np.repeat (13,200)
L_quince= np.repeat (14,200)
#L_dieciseis= np.repeat (3,200)



labels= np.concatenate([L_uno,L_dos,L_tres,L_cuatro,L_cinco,L_seis,L_siete,L_ocho,L_nueve,L_diez,L_once,L_doce,L_catorce,L_quince,L_cero])
Labels= np.array(labels)
print(len(Labels))


#Separamos datos para entrenamiento y test
Datos_Modelo3D= np.reshape(Datos_Modelo, (Datos_Modelo.shape + (1,)))

datos_entrenamiento, datos_test, etiq_train, etiq_test = train_test_split(Datos_Modelo3D, Labels, test_size= 0.3, random_state= 0)
### Definimos estructura de la red neuronal para el primer modelo


#modelo = Sequential()

#modelo.add(Con2D(32,(3,3), activation='relu', input_shape= (4,200,1)))
#modelo.add(MaxPooling2D(pool_size=(2,2)))

#modelo.add(Flatten())
#modelo.add(Dense(100, activation='sigmoid'))
#modelo.add(Dense(15, activation='softmax'))
#modelo.summary()


modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4,200,1)), 
    tf.keras.layers.Dense(units=256, activation='sigmoid'),
    tf.keras.layers.Dense(units=256, activation='sigmoid'),
    tf.keras.layers.Dense(units=256, activation='sigmoid'),
    tf.keras.layers.Dense(15, activation='softmax')
])



#Compilar el modelo estilo clasificador de imagenes
modelo.compile(
    optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#### Entrenamos la red neuronal


#model.compile(loss="mean_squared_error", optimizer="adam", metrics=["binary_accuracy"])
#tensorboard = TensorBoard(log_dir="logs\{}".format(time.time()))
historial= modelo.fit(datos_entrenamiento, etiq_train, batch_size=32, epochs=200, validation_data= (datos_test, etiq_test))

acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

rango_epocas = range(200)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
plt.legend(loc='lower right')
plt.title('Precisión de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento')
plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas')
plt.legend(loc='upper right')
plt.title('Pérdida de entrenamiento y pruebas')
plt.show()

