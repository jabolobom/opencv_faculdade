from unittest import result
import cv2, os
import numpy as np
from functions import *

def HsvSegmentation(imgspath, lower, upper):
    if not isinstance(lower, np.ndarray) or not isinstance(upper, np.ndarray):
        raise TypeError('lower and upper precisam ser np.ndarray')

    imglist = LoadImages(imgspath)

    HsvSegmentationImages = []

    for img in imglist:
        # lê os files dentro do path
        # image = cv2.imread(img)

        # conversão pra hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # blur pra remover ruídos da imagem
        blur = cv2.medianBlur(hsv, 7)

        # cria uma mascara que esconde todos valores fora do range de cor
        mask = cv2.inRange(blur, lower, upper)

        # aplica a mascara
        resultado = cv2.bitwise_and(img, img, mask=mask)

        # pra mostrar o resultado (debug)
        HsvSegmentationImages.append(resultado)
    
    return HsvSegmentationImages
    

