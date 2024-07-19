import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images. 
The dataset is divided into five training batches and one test batch, each with 10000 images. 
The test batch contains exactly 1000 randomly-selected images from each class. The training batches 
contain the remaining images in random order, but some training batches may contain more images from 
one class than another. Between them, the training batches contain exactly 5000 images from each class.
"""

# Función para cargar datos
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Arrays para los datos y las etiquetas
train_data = []
train_labels = []

# Cargar los 5 batch de entrenamiento
for i in range(1, 6):
    batch = unpickle(os.path.join('cifar-10-batches-py', f'data_batch_{i}'))
    train_data.append(batch[b'data'])
    train_labels.extend(batch[b'labels'])

# Concatenar los datos cargados de los 5 batch
train_data = np.concatenate(train_data)

# Reestructurar los datos para tener la forma (50000, 3, 32, 32) ----> (#ejemplos, canales RGB, dimensión1, dimensión2)
train_data = train_data.reshape((50000, 3, 32, 32))

# Calcular la media y la desviación estándar para cada canal para luego normalizar los valores de los píxeles
media = np.mean(train_data, axis=(0, 2, 3)) / 255  # ----> Rango valor píxel por canal RGB [0, 255]
desviacion = np.std(train_data, axis=(0, 2, 3)) / 255  # ----> (0, 2, 3) = (#ejemplos, dimensión1, dimensión2)

# Normalizar los datos
train_data = train_data / 255  # ----> Escalar a 0-1
train_data = (train_data - media[:, None, None]) / desviacion[:, None, None]  # ----> Normalizar

# Convertir los datos y etiquetas a tensores de PyTorch
train_data = torch.tensor(train_data).float()  # ----> Convierte a float
train_labels = torch.tensor(train_labels)

# Crear dataloader para el conjunto de entrenamiento
train_dataset = TensorDataset(train_data, train_labels)  # ----> Emparejar datos y etiquetas
trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

print("DataLoader de entrenamiento creado.")

# Definir el modelo de red neuronal (CNN)
class RedN(nn.Module):
    def __init__(self):
        super(RedN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # ----> Capa convolucional 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # ----> Capa convolucional 2
        self.pool = nn.MaxPool2d(2, 2)  # ----> Capa de MaxPooling
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # ----> Capa completamente conectada 1
        self.fc2 = nn.Linear(512, 10)  # ----> Capa completamente conectada 2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # ----> Aplicar ReLu y Maxpooling a Capa convolucional 1
        x = self.pool(F.relu(self.conv2(x)))  # ----> Aplicar ReLu y Maxpooling a Capa convolucional 2
        x = x.view(-1, 64 * 8 * 8)  # ----> Reorganizar
        x = F.relu(self.fc1(x))  # ----> Aplicar ReLu a Capa completamente conectada 1
        x = self.fc2(x)
        return x  # ----> Vector con las probabilidades de las 10 clases

def main():
    modelo = RedN()

    # Arrays para épocas y perdidas para graficar
    xepocas = []
    yperdida = []

    # Definir la función de pérdida y el optimizador
    Fperdida = nn.CrossEntropyLoss()  # ----> Función de pérdida de entropía cruzada
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)  # ----> Ajusta la tasa de aprendizaje y el gradiente de forma adaptiva

    epocas = 10

    for epoca in range(epocas):
        perdidaAcum = 0
        for i, data in enumerate(trainloader, 0):
            imagenes, etiquetas = data

            optimizador.zero_grad()  # ----> Inicializar el gradiente
            salidas = modelo(imagenes)  # ----> Aplicar el modelo
            perdida = Fperdida(salidas, etiquetas)  # ----> Calcular la pérdida
            perdida.backward()  # ----> Retropropagación
            optimizador.step()  # ----> Actualizar los parámetros del modelo

            # Imprimir estadísticas
            perdidaAcum += perdida.item()
            if i % 100 == 99:  # Imprimir cada 100 minibatches
                print(f"[{epoca + 1}, {i + 1}] pérdida: {round(perdidaAcum / 100, 3)}")
                if i + 1 == 500:
                    xepocas.append(epoca + 1)
                    yperdida.append(perdidaAcum/100)
                perdidaAcum = 0

    # Graficar la pérdida respecto a cada época
    plt.plot(xepocas, yperdida)
    plt.grid()
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.savefig("Gráfica de perdida.png")
    plt.show()

    print("Entrenamiento finalizado")

    # Guardar el modelo entrenado
    torch.save(modelo.state_dict(), "red_neuronal.pth")

# Ejecutar el código
if __name__ == "__main__":
    main()