import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

Matlike = np.ndarray

def imshow(img: Matlike, new_fig: bool = True, title: str = None, color_img: bool = False, blocking: bool = True, 
           colorbar: bool = False, ticks: bool = False):
    """
    Muestra una imagen en una ventana de matplotlib.
    """
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)



def img_reading(name: str) -> np.ndarray:
    """
    Se encarga de cargar la imágen
    """
    PATH = os.getcwd()
    DATA_PATH = os.path.join(PATH, 'data')

    img: np.ndarray =  cv2.imread(os.path.join(DATA_PATH,name))
    return img



def img_procesor(img: Matlike) -> tuple[Matlike, Matlike]:
    """
    Función procesadora de la imágen.
    Se aplica 'Top-Hat' y un umbral  
    """  

    se: Matlike = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))    

    img_top_hat: Matlike = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
    img_top_hat_normalized: Matlike = cv2.normalize(img_top_hat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, img_thresh = cv2.threshold(img_top_hat_normalized, 100, 255, cv2.THRESH_BINARY)
    #_, img_thresh = cv2.threshold(img_top_hat_normalized, 75, 255, cv2.THRESH_BINARY)
    
    return img_top_hat, img_thresh



def filter_components(  stats: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Función para el filtrado de las componentes.
    El criterio de selección es por medio de: 

    - Relación de aspecto -> y = 2x 
    - Area -> Entre 20 y 50 pixeles
    - Posición con respecto a otra componente -> Se buscan aquellas cercanas una de otra con respecto al eje x 
        y aquellas ubicadas dentro de un rango en la coordenada y 
    """
    aspect_min = 1.3
    aspect_max = 5

    area_min = 15  
    area_max = 150

    stats_by_aspect = []
    
    # Filtrado por Relación de Aspecto y Area
    for i, st in enumerate(stats):        
                
        width = st[2]
        height = st[3]
        aspect_ratio =  height / width  if width > 0 else 0

        if aspect_min <= aspect_ratio <= aspect_max and area_min <= st[4] <= area_max:      
            stats_by_aspect.append(st)
            
    letters_stats = []

    for i, st in enumerate(stats_by_aspect):

        # if i == 1 and st[4] > area_min: 
        #     letters_stats.append(stats_by_aspect[i])
        #     continue
        
        for j, _ in enumerate(stats_by_aspect): 
            
            if i != j:                
                delta_x = abs(stats_by_aspect[i][0] - stats_by_aspect[j][0])
                delta_y = abs(stats_by_aspect[i][1] - stats_by_aspect[j][1])

                if delta_x <= 20 and delta_y <= 5: 
                    letters_stats.append(stats_by_aspect[j])
                    letters_stats.append(stats_by_aspect[i])
    

    return letters_stats



def object_detector(img_thresh: Matlike) -> tuple[list[Matlike], Matlike]:
    """ 
    Se detectan los componentes encontrados en la imagen umbralizada.
    Por medio de filter_components() se filtran los componentes pertenecientes a las letras de las patentes
    """
   
    connectivity = 10
    _, _, stats, _ = cv2.connectedComponentsWithStats(img_thresh, connectivity, cv2.CV_32S)  
   
    letters_stats = filter_components(stats)
                   
    return letters_stats


def proccess_plotter(img, img_top_hat, img_thresh, letter_stats, show: bool =False):

    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    
    img_thresh_copy = img_thresh.copy()
    img_thresh_copy = cv2.cvtColor(img_thresh_copy, cv2.COLOR_GRAY2BGR)

    for st in letter_stats:
        cv2.rectangle(img_copy, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=1)
        cv2.rectangle(img_thresh_copy, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=1)

    if show == True: 

        plt.figure(figsize=(16, 9))        
        ax1 = plt.subplot(231); plt.xticks([]), plt.yticks([]), plt.imshow(img, cmap='gray'), plt.title('Imagen Original (escala de grises)'), plt.colorbar(shrink=0.5)
        plt.subplot(232, sharex=ax1, sharey=ax1), plt.imshow(img_top_hat, cmap='gray'), plt.title('Top-Hat'), plt.colorbar(shrink=0.5)
        plt.subplot(233, sharex=ax1, sharey=ax1), plt.imshow(img_thresh, cmap='gray'), plt.title('Umbralado'), plt.colorbar(shrink=0.5)

        plt.subplot(234, sharex=ax1, sharey=ax1), plt.imshow(img_thresh_copy, cmap='gray'), plt.title('Detección de letras'), plt.colorbar(shrink=0.5)
        plt.subplot(235, sharex=ax1, sharey=ax1), plt.imshow(img_copy, cmap='gray'), plt.title('Resultado Final'), plt.colorbar(shrink=0.5)

        plt.tight_layout()                       
        plt.show(block= True)
   
    else: 
        plt.figure(figsize=(16, 9))        
        ax1 = plt.subplot(121); plt.xticks([]), plt.yticks([]), plt.imshow(img, cmap='gray'), plt.title('Imagen Original (escala de grises)'), plt.colorbar(shrink=0.5)
        plt.subplot(122, sharex=ax1, sharey=ax1), plt.imshow(img_copy, cmap='gray'), plt.title('Resultado Final'), plt.colorbar(shrink=0.5)

        plt.tight_layout()                       
        plt.show(block= True)
   


def execute(show_steps: bool = True) -> None:
    
    dir_: list[str] = os.listdir(os.path.join(os.getcwd(), 'data'))
    patentes: list[str] = [dir_ for dir_ in dir_ if dir_.startswith('img')]

    for image_path in patentes:
        img: Matlike = img_reading(image_path)
        img_gray: Matlike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_top_hat, img_thresh = img_procesor(img_gray)

        letter_stats = object_detector(img_thresh)

        proccess_plotter(img_gray, img_top_hat, img_thresh, letter_stats, show_steps)



def show_process():
    try:
        
        respuesta = input("Bienvenido al detector de patentes.\n¿Le gustaría ver el proceso completo? (S/N): ").strip().upper()        
        
        if respuesta not in ["S", "N"]:
            raise ValueError("Debe ingresar 'S' para Sí o 'N' para No.")
        
        return respuesta == "S"
    
    except ValueError as e:
        print(f"Entrada inválida: {e}")
        return show_process()


if __name__ == "__main__":    
    show_steps = show_process()
    execute(show_steps)
