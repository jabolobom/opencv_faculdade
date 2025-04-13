import cv2
from functions import *

def RunDetectEdges(path):

    # Carregando as imagens
    images = LoadImages(path)

    # Criando Array para as imagens com bordas realçadas
    imgsEdges = []

    # Passa por todas as imagens, adicionando no array após o realce
    for img in images:
        imgsEdges.append(detectEdges(img))

    
    return imgsEdges


def detectEdges(image):
    blur_kernel_size=(5, 5)
    threshold1=30
    threshold2=100
    

    # Aplica desfoque para reduzir ruído
    blurred = cv2.GaussianBlur(image, blur_kernel_size, 0)

    # Aplica detecção de borda
    edges = cv2.Canny(blurred, threshold1, threshold2)

    return edges