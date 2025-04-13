import cv2
import kagglehub
from functions import LoadImages

def RunGrayScale(path):
    # Carregando as imagens
    images = LoadImages(path)

    # Criando Array para as imagens com upscale
    grayScaleImages = []

    # Passa por todas as imagens, adicionando no array após o GrayScale
    for img in images:
        grayScaleImages.append(GrayScaleImage(img))

        
    return grayScaleImages

def GrayScaleImage(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
