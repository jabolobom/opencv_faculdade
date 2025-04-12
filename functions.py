import os
import cv2

def LoadImages(path):
    images = []
    for fileName in os.listdir(path):
        # Testa se o arquivo é uma imagem
        if fileName.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            
            # Constróis o endereço inteiro para o arquivo
            imgPath = os.path.join(path, fileName)

            # Lê a imagem
            img = cv2.imread(imgPath)

            # Testa se foi carregada corretamente
            if img is not None:
                images.append(img)
            else:
                print(f"Could not load image: {fileName}")

    return images