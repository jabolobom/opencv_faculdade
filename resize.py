import os
import kagglehub
import cv2
from functions import LoadImages

def RunResize(path):
    # Carregando as imagens
    images = LoadImages(path)

    # Criando Array para as imagens com upscale
    upScaledImages = []

    for img in images:
        upScaledImages.append(UpScaleImage(img))

    
    return upScaledImages

def UpScaleImage(image):
    scaleFactor=2.0
    interpolationMethod='cubic'

    # Valida o input
    if image is None:
        raise ValueError("Input image cannot be None")
    
    # Mapeia os métodos de interpolação para constantes do OpenCV
    interpolationMap = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    # Recebe o método de interpolação
    interp = interpolationMap.get(interpolationMethod.lower(), cv2.INTER_CUBIC)
    
    # Calcula as novas dimensões
    height, width = image.shape[:2]
    newWidth = int(width * scaleFactor)
    newHeight = int(height * scaleFactor)
    
    # Faz o upscaling
    upscaledImg = cv2.resize(
        image, 
        (newWidth, newHeight), 
        interpolation=interp
    )
    
    return upscaledImg