import cv2
from functions import *

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def RunBoundingBoxes(path):
    # Carregando as imagens
    images = LoadImages(path)

    # Criando Array para as imagens com bounding boxes
    boudingBoxImgs = []

    # Testa qual o valor dominante da escala RGB para cada imagem e coloca em seu array respectivo 
    for img in images:
        imgBB = boudingFaces(img)

        if imgBB is not None:
            boudingBoxImgs.append(imgBB)

    return boudingBoxImgs

def boudingFaces(img):
    # Detecta os rostos na imagem
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    if len(faces) == 0:
        return None
    
    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, 'Rosto', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return img

