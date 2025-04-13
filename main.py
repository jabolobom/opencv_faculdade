from functions import *
import cv2
from sklearn.cluster import KMeans
from identifyColor import *
from resize import *
from turnToGrayScale import *

# Baixando o DataSet e colocando o endereço em uma variavel
path = kagglehub.dataset_download("bahadrsametarman/balloon-dataset-from-oidv6")

# Criando o endereço para as imagens
imagesPath = path + '\\BalloonDataset\\test\\images'

images = RunDominantColor(imagesPath)

pathRed = path + '\\Testes\\RGB\\Vermelho'
pathGreen = path + '\\Testes\\RGB\\Verde'
pathBlue = path + '\\Testes\\RGB\\Azul'
pathExcedente = path + '\\Testes\\RGB\\NoDominantValue'

print(SaveImages(images[0], pathRed))
print(SaveImages(images[1], pathGreen))
print(SaveImages(images[2], pathBlue))
print(SaveImages(images[3], pathExcedente))

pathUpScale = path + '\\Testes\\UpScale'
print(SaveImages(RunResize(imagesPath), pathUpScale))

pathGrayScale =  path + '\\Testes\\GrayScale'
print(SaveImages(RunGrayScale(imagesPath), pathGrayScale))
