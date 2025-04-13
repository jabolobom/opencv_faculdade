from functions import *
import cv2
from sklearn.cluster import KMeans

# K é a quantidade de clusters dominantes que o algortimo procura,
# ou seja, o 1 cluster de (nesse caso) cor dominante.
def dominantColor(img, k=1):
    img = cv2.imread(img) # pode remover isso aqui depois

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

print(dominantColor('bliss.jpg')) # teste

# Se comentar essa parte de baixo a função tá funcionando legal, só precisa integrar com as outras
# e apresentar o resultado bonitinho

# Caminho do dataset, da pra usar o kagglehub pra carregar...
dataset_path = None

imgset = LoadImages(dataset_path) # retorna a lista de imagens
# pra cada x imagem, roda a função de análise nela
for x in imgset:
    cordominante = dominantColor(x)
    print(f"Cor dominante = {cordominante}\n")


