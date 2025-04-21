import cv2
import numpy as np
from functions import *

def adjust_linear(img, alpha=1.0, beta=0):
    """
    Ajuste linear de brilho (beta) e contraste (alpha).
    Formula: output = alpha * img + beta
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def adjust_gamma(img, gamma=1.0):
    """
    Ajuste não-linear (Gamma Correction).
    Formula: output = img^(1/gamma)
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def RunAdjustBrightnessContrast(images_path, alpha=1.5, beta=30, gamma=1.2):
    """
    Aplica ajustes de brilho/contraste a todas as imagens do diretório.
    Parâmetros:
        alpha: Ganho de contraste (1.0 = original, <1.0 reduz, >1.0 aumenta).
        beta: Ganho de brilho (0 = original, >0 aumenta, <0 reduz).
        gamma: Correção gamma (>1 escurece, <1 clareia).
    """
    images = LoadImages(images_path)
    adjusted_images = []
    
    for img in images:
        # Aplica ajuste linear (contraste + brilho)
        linear_adj = adjust_linear(img, alpha, beta)
        # Aplica gamma correction
        final_adj = adjust_gamma(linear_adj, gamma)
        adjusted_images.append(final_adj)
    
    return adjusted_images
