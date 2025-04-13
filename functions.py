import os
import shutil
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

def SaveImages(images, path):
    # Testa se o caminho existe, limpa se existir, ou cria se não
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Pasta '{path}' criada")
    else:
        print(ClearFolder(path))
    
    # Loop para salvar as imagens enumerando elas
    for i, img in enumerate(images):
        cv2.imwrite(f"{path}/imagem_{i}.jpg", img)

    return (f"{len(images)} imagens salvas em '{path}/'")

def ClearFolder(path):

    """
    Remove todos os arquivos e subpastas de uma pasta.
    
    Args:
        path (str): Caminho da pasta a ser limpa.
        excluir_pasta (bool): Se True, remove a própria pasta. Se False (padrão), mantém a pasta vazia.
    """
    if not os.path.exists(path):
        return(f"A pasta '{path}' não existe!")

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Remove arquivos e links simbólicos
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove subpastas e seu conteúdo

    return (f"Conteúdo da pasta '{path}' removido (pasta mantida vazia).")