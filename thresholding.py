import cv2
from functions import *

def RunThresholding(path, metodo='binario'):
    # Carregando as imagens
    images = LoadImages(path)

    # Criando Array para as imagens com upscale
    thresholdingImages = []

    # Passa por todas as imagens, adicionando no array após o Upscale
    for img in images:
        thresholdingImages.append(Thresholding(img, metodo))

    
    return thresholdingImages


def Thresholding(img, metodo='binario', valorThreshold=127, valorMaximo=255, tipoOtsu=False):
    # Verifica se a imagem está em escala de cinza
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define o tipo de thresholding
    if metodo == 'binario':
        tipo = cv2.THRESH_BINARY
    elif metodo == 'binario_inv':
        tipo = cv2.THRESH_BINARY_INV
    elif metodo == 'trunc':
        tipo = cv2.THRESH_TRUNC
    elif metodo == 'tozero':
        tipo = cv2.THRESH_TOZERO
    elif metodo == 'tozero_inv':
        tipo = cv2.THRESH_TOZERO_INV
    else:
        raise ValueError("Método de thresholding não reconhecido")
    
    # Aplica thresholding
    if tipoOtsu:
        tipo += cv2.THRESH_OTSU
        valorThreshold = 0  # Otsu ignora este valor e calcula o threshold automaticamente
    
    _, imagem_tratada = cv2.threshold(img, valorThreshold, valorMaximo, tipo)
    
    return imagem_tratada