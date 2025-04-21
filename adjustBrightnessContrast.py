import cv2
import numpy as np
from functions import *

def adjust_linear(img, alpha=1.0, beta=0):
    # Ajusta o brilho e contraste
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def adjust_gamma(img, gamma=1.0):
    # Ajusta o gamma
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def RunAdjustBrightnessContrast(images_path, alpha=1.5, beta=30, gamma=1.2):
    # Carregando imagens
    images = LoadImages(images_path)
    adjusted_images = []
    # Passando pelas imagens
    for img in images:
        # Aplica ajuste linear (contraste + brilho)
        linear_adj = adjust_linear(img, alpha, beta)
        # Aplica gamma correction
        final_adj = adjust_gamma(linear_adj, gamma)
        adjusted_images.append(final_adj)
    
    return adjusted_images
