import cv2
import kagglehub
from functions import LoadImages

def RunGrayScale():
    # Baixando o DataSet e colocando o endereço em uma variavel
    path = kagglehub.dataset_download("bahadrsametarman/balloon-dataset-from-oidv6")

    # Criando o endereço para as imagens
    imagesPath = path + '\\BalloonDataset\\test\\images'

    # Carregando as imagens
    images = LoadImages(imagesPath)

    # Criando Array para as imagens com upscale
    grayScaleImages = []

    for img in images:
        grayScaleImages.append(GrayScaleImage(img))

        
    return grayScaleImages

def GrayScaleImage(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
