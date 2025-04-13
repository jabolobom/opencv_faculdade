from functions import *
import cv2
from sklearn.cluster import KMeans
from identifyColor import *
from resize import *
from turnToGrayScale import *
import kagglehub
from borders import *

# Baixando o DataSet e colocando o endereço em uma variavel
path = kagglehub.dataset_download("bahadrsametarman/balloon-dataset-from-oidv6")

# Criando o endereço para as imagens
imagesPath = path + '\\BalloonDataset\\test\\images'



# Rodando a função para dividir as imagens pelo valor dominante da escala RGB
images = RunDominantColor(imagesPath)

# Setando os caminhos para salvar as imagens
pathRed = path + '\\Testes\\RGB\\Vermelho'
pathGreen = path + '\\Testes\\RGB\\Verde'
pathBlue = path + '\\Testes\\RGB\\Azul'
pathExcedente = path + '\\Testes\\RGB\\NoDominantValue'

# Salvando as imagens
print(SaveImages(images[0], pathRed))
print(SaveImages(images[1], pathGreen))
print(SaveImages(images[2], pathBlue))
print(SaveImages(images[3], pathExcedente))

# Rodando a função para fazer o UpScale
pathUpScale = path + '\\Testes\\UpScale'
print(SaveImages(RunResize(imagesPath), pathUpScale))

# Rodando a função para fazer o GrayScale
pathGrayScale =  path + '\\Testes\\GrayScale'
print(SaveImages(RunGrayScale(imagesPath), pathGrayScale))

# Rodando a função para detectar as bordas
pathEdges = path + '\\Testes\\DetectEdges'
print(SaveImages(RunDetectEdges(pathGrayScale), pathEdges))