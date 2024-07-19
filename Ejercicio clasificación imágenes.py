import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

# Función para cargar datos
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Cargar los datos de prueba
test_batch = unpickle(os.path.join('cifar-10-batches-py', 'test_batch'))
test_data = test_batch[b'data']
test_labels = test_batch[b'labels']

# Reestructurar los datos para tener la forma (10000, 3, 32, 32)
test_data = test_data.reshape((10000, 3, 32, 32))

# Calcular la media y la desviación estándar para cada canal
media = np.mean(test_data, axis=(0, 2, 3)) / 255
desviacion = np.std(test_data, axis=(0, 2, 3)) / 255

# Normalizar los datos de prueba
test_data = test_data / 255  # ----> Escalar a 0-1
test_data = (test_data - media[:, None, None]) / desviacion[:, None, None]  # ----> Normalizar

# Convertir los datos a tensores
test_data = torch.tensor(test_data).float()
test_labels = torch.tensor(test_labels)

# Crear un dataset y un dataloader para los datos de prueba
test_dataset = TensorDataset(test_data, test_labels)
testloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

print("DataLoader de prueba creado.")

# Definir el modelo de red neuronal igual al de entrenamiento
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
    # Cargar el modelo entrenado en Entrenamiento.py
    modelo = RedN()
    modelo.load_state_dict(torch.load("red_neuronal.pth"))
    modelo.eval()

    correctos = 0  # ----> Contador para predicciones correctas
    total = 0  # ----> Contador de ejemplos evaluados

    with torch.no_grad():
        for data in testloader:
            imagenes, etiquetas = data
            salidas = modelo(imagenes)
            _, prediccion = torch.max(salidas.data, 1)
            total += etiquetas.size(0)
            correctos += (prediccion == etiquetas).sum().item()

    print(f"Precisión en los datos de prueba: {round(100 * correctos / total, 2)}%")

# Ejecutar el código
if __name__ == "__main__":
    main()

