import os
import kagglehub
import cv2



def RunResize():
    # Baixando o DataSet e colocando o endereço em uma variavel
    path = f'{kagglehub.dataset_download("bahadrsametarman/balloon-dataset-from-oidv6")}' 

    # Criando o endereço para as imagens
    imagesPath = path + '\\BalloonDataset\\test\\images'

    # Carregando as imagens
    images = LoadImages(imagesPath)

    # Criando Array para as imagens com upscale
    upScaledImages = []

    for img in images:
        upScaledImages.append(UpScaleImage(img))

    
    return upScaledImages


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