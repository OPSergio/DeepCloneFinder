from pyexpat import model
import numpy as np
import random
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import pickle
import time
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator




### GENERAMOS LAS MATRICES QUE CONTENGAN LOS DATOS PARA TODAS LAS CATEGORIAS

ruta_1mut= "C:/Users/solmo/Desktop/Master/TFM/Python/una_mutacion"
Unamutacion = []

for i in os.listdir (ruta_1mut) :
    i= pd.read_csv(os.path.join(ruta_1mut,i), header= None)
    Unamutacion.append(i)

Unamutacion= np.array(Unamutacion)
print(Unamutacion.shape)

ruta_2mut= "C:/Users/solmo/Desktop/Master/TFM/Python/dos_mutaciones"

Dosmutaciones = []

for i in os.listdir (ruta_2mut) :
    i= pd.read_csv(os.path.join(ruta_2mut,i), header= None)
    Dosmutaciones.append(i)

Dosmutaciones= np.array(Dosmutaciones)
print(Dosmutaciones.shape)

ruta_3mut= "C:/Users/solmo/Desktop/Master/TFM/Python/tres_mutaciones"

Tresmutaciones = []

for i in os.listdir (ruta_3mut) :
    i= pd.read_csv(os.path.join(ruta_3mut,i), header= None)
    Tresmutaciones.append(i)

Tresmutaciones= np.array(Tresmutaciones)
print(Tresmutaciones.shape)


r_una_tres= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/uno_tres"

unatres = []

for i in os.listdir (r_una_tres) :
    i= pd.read_csv(os.path.join(r_una_tres,i), header= None)
    unatres.append(i)

unatres= np.array(unatres)
print(unatres.shape)

r_dos_tres= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/dos_tres"

dostres = []

for i in os.listdir (r_dos_tres) :
    i= pd.read_csv(os.path.join(r_dos_tres,i), header= None)
    dostres.append(i)

dostres= np.array(dostres)
print(dostres.shape)

r_tres_tres= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/tres_tres"

trestres = []

for i in os.listdir (r_tres_tres) :
    i= pd.read_csv(os.path.join(r_tres_tres,i), header= None)
    trestres.append(i)

trestres= np.array(trestres)
print(trestres.shape)


r_una_cuatro= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/uno_cuatro"

unacuatro = []

for i in os.listdir (r_una_cuatro) :
    i= pd.read_csv(os.path.join(r_una_cuatro,i), header= None)
    unacuatro.append(i)

unacuatro= np.array(unacuatro)
print(unacuatro.shape)

r_dos_cuatro= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/dos_cuatro"

doscuatro = []

for i in os.listdir (r_dos_cuatro) :
    i= pd.read_csv(os.path.join(r_dos_cuatro,i), header= None)
    doscuatro.append(i)

doscuatro= np.array(doscuatro)
print(doscuatro.shape)

r_tres_cuatro= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/tres_cuatro"

trescuatro = []

for i in os.listdir (r_tres_cuatro) :
    i= pd.read_csv(os.path.join(r_tres_cuatro,i), header= None)
    trescuatro.append(i)

trescuatro= np.array(trescuatro)
print(trescuatro.shape)

r_orig= "C:/Users/solmo/Desktop/Master/TFM/Modelo2/Orig"

orig = []

for i in os.listdir (r_orig) :
    i= pd.read_csv(os.path.join(r_orig,i), header= None)
    orig.append(i)

orig= np.array(orig)
print(orig.shape)

##UNIFICAMOS TODOS LOS DATOS EN UNA UNICA MATRIZ


Datos_modelo= np.concatenate([Unamutacion,Dosmutaciones,Tresmutaciones,unatres,dostres,trestres,unacuatro,doscuatro,trescuatro,orig])
print(len(Datos_modelo))
Datos_Modelo= np.array(Datos_modelo).astype(float)
print(Datos_Modelo.shape)
Datos_Modelo3D= np.reshape(Datos_Modelo, (Datos_Modelo.shape + (1,)))



datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8 , 1.0],
    validation_split=0.2)



###GENERAMOS LAS ETIQUETAS

##Generamos las etiquetas  (Labels) Teniendo en cuenta el orden

Labels_Unam = np.repeat(0,100)
Labels_dos = np.repeat(1,100)
Labels_tres = np.repeat(2,100)
Labels_unotres = np.repeat(3,100)
Labels_dostres= np.repeat (4,100)
Labels_trestres= np.repeat (5,100)
Labels_unocuatro= np.repeat (6,100)
Labels_doscuatro= np.repeat (7,100)
Labels_trescuatro= np.repeat (8,100)
Labels_orig= np.repeat (9,100)

#class_names = ["Caso1", "Caso2", "Caso3", "Caso3","Caso4", "Caso5", "Caso6", "Caso7","Caso8", "Caso9"]

labels= np.concatenate([Labels_Unam,Labels_dos,Labels_tres, Labels_unotres, Labels_dostres, Labels_trestres, Labels_unocuatro, Labels_doscuatro, Labels_trescuatro, Labels_orig])
Labels= np.array(labels)
print(len(Labels))


#Separamos datos para entrenamiento y test




datos_entrenamiento, datos_test, etiq_train, etiq_test = train_test_split(Datos_Modelo3D, Labels, test_size= 0.3, random_state= 0)
### Definimos estructura de la red neuronal para el primer modelo
## Sobre los datos de entrenamiento, generamos modificaciones aleatorias empleando ImageDataGenerator de keras

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8 , 1.0])

datagen.fit(datos_entrenamiento)


modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4,200,1)), 
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#Compilar el modelo estilo clasificador de imagenes
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#### Entrenamos la red neuronal


#model.compile(loss="mean_squared_error", optimizer="adam", metrics=["binary_accuracy"])
#tensorboard = TensorBoard(log_dir="logs\{}".format(time.time()))
historial= modelo.fit(datos_entrenamiento, etiq_train, batch_size=32, epochs=100, validation_data= (datos_test, etiq_test))

acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

rango_epocas = range(100)

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


