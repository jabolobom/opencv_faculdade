import cv2
import kagglehub
from sklearn.cluster import KMeans
from functions import *

def RunDominantColor(path):
    # Carregando as imagens
    images = LoadImages(path)

    # Criando Array para as imagens com upscale
    REDimgs = []
    GREENimgs = []
    BLUEimgs = []
    excedenteImgs = []
    
    # Testa qual o valor dominante da escala RGB para cada imagem e coloca em seu array respectivo 
    for img in images:
        imgRGB = dominantColor(img)
        if (imgRGB[0] > imgRGB[1] and imgRGB[0] > imgRGB[2]):
            REDimgs.append(img)
        elif (imgRGB[1] > imgRGB[0] and imgRGB[1] > imgRGB[2]):
            GREENimgs.append(img)
        elif (imgRGB[2] > imgRGB[0] and imgRGB[2] > imgRGB[1]):
            BLUEimgs.append(img)
        else:
            excedenteImgs.append(img)
    
    # Organiza todos os arrays junto
    imgsRGB = [REDimgs, GREENimgs, BLUEimgs, excedenteImgs]
    
    return imgsRGB

# K é a quantidade de clusters dominantes que o algortimo procura,
# ou seja, o 1 cluster de (nesse caso) cor dominante.
def dominantColor(img, k=1):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converte pra rgb

    pixellist = image_rgb.reshape((-1, 3)) # converte a imagem de um array (width, weight, 3) pra um
    # array unidimensional, (-1 = pixel individual resultado de w * h, 3 = valores rgb do pixel)
    # resulta em uma lista bem grande, mas com isso dá pra criar clusters de cores semelhantes

    # cria um gráfico com os valores do pixellist
    # junta os pontos do gráfico em clusters de cores (valores) semelhantes
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixellist)

    # busca o centro do maior cluster, a cor média, a cor dominante
    result = kmeans.cluster_centers_[0]

    return result.astype(int) # resulta em 3 valores RGB
