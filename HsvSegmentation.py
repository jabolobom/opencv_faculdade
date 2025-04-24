from unittest import result
import cv2, os
import numpy as np

def HsvSegmentation(imglist, lower, upper, save_path):
    if not isinstance(lower, np.ndarray) or not isinstance(upper, np.ndarray):
        raise TypeError('lower and upper precisam ser np.ndarray')

    for img in imglist:
        # lê os files dentro do path
        image = cv2.imread(img)

        # conversão pra hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # blur pra remover ruídos da imagem
        blur = cv2.medianBlur(hsv, 7)

        # cria uma mascara que esconde todos valores fora do range de cor
        mask = cv2.inRange(blur, lower, upper)

        # aplica a mascara
        resultado = cv2.bitwise_and(image, image, mask=mask)

        result_filename = os.path.join(save_path, os.path.basename(img))
        cv2.imwrite(result_filename, resultado)

        # pra mostrar o resultado (debug)
        # cv2.imshow('resultado', resultado)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# função exemplo
HsvSegmentation(
    ["example.jpg"],
    np.array([35, 50, 50]),   # amarelo
    np.array([85, 255, 255])   # limite do verde
)

