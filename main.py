from functions import *
import cv2
from sklearn.cluster import KMeans
from identifyColor import *
from resize import *
from turnToGrayScale import *
import kagglehub
from borders import *
from thresholding import *
from adjustBrightnessContrast import *
from boudingBoxes import *
from HsvSegmentation import *

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

# Rodando a função para fazer o Thresholding
pathThresholding = path + '\\Testes\\Thresholding'
print(SaveImages(RunThresholding(pathGrayScale), pathThresholding)) 

# Rodando a função para ajustar brilho/contraste - aumenta a claridade
pathAdjusted = path + '\\Testes\\BrightnessContrastAdjusted\\Brighter'
print(SaveImages(RunAdjustBrightnessContrast(imagesPath, alpha=1.5, beta=20, gamma=1.1),pathAdjusted))

# Rodando a função para ajustar brilho/contraste - diminui a claridade
pathAdjusted = path + '\\Testes\\BrightnessContrastAdjusted\\Darker'
print(SaveImages(RunAdjustBrightnessContrast(imagesPath, alpha=0.8, beta=0, gamma=0.3),pathAdjusted))

# Rodando a função para detectar os bounding boxes
pathBoundingBoxes = path + '\\Testes\\BoundingBoxes'
print(SaveImages(RunBoundingBoxes(imagesPath), pathBoundingBoxes))

# Rodando a função de segmentação HSV
lower, upper = np.array([35,50,50]), np.array([85,255,255])
pathHsvSegmentation = path + '\\Testes\\HsvSegmentation'
print(SaveImages(HsvSegmentation(imagesPath, lower, upper), pathHsvSegmentation))