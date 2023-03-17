import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Funcion entropia cruzada
def crossentropy(labels, pb):    
    return -np.sum(labels * np.log(pb) + (1 - labels) * np.log(1 - pb)) #Función de entropía cruzada
                                           
#Etiquetas
labels_1= np.random.randint(0,2,10)
label2 = np.array(labels_1)

#Probabilidades
pb1=np.random.rand(10,1)
pb2=np.array(pb1)

print("")
print('Cross Entropy: ',crossentropy(label2,pb2))