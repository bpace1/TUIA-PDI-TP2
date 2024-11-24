import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

Matlike = np.ndarray

def imshow(img: Matlike, new_fig: bool = True, title: str = None, color_img: bool = False, 
           blocking: bool = True, colorbar: bool = False, ticks: bool = False):
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

def img_reading() -> Matlike:
    """
    Se encarga de ecargar la imágen
    """
    PATH = os.getcwd()
    DATA_PATH = os.path.join(PATH, 'data')

    img: Matlike =  cv2.imread(os.path.join(DATA_PATH,'monedas.jpg'))
    return img

def img_preprocessing(img: Matlike) -> tuple[Matlike, Matlike]:
    """
    Preprocesa la imagen. Devuelve una tupla de arrays con la imagen en escala de grises 
    y la imagen el resultado de aplicar un filtro de blur a la imagen en escala de grises.
    """
    img_gray: Matlike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur: Matlike = cv2.GaussianBlur(img_gray, (3, 3), 0)
    return img_gray, img_blur



def detect_coins(img: Matlike) -> Matlike:
    """
    Detecta monedas en la imagen.
    """
    detected_circles = cv2.HoughCircles(img,  
                    cv2.HOUGH_GRADIENT, 1.2, 90, param1 = 65, 
                param2 = 170, minRadius = 70, maxRadius = 200)     
    return detected_circles

def draw_circles(img: Matlike, detected_circles: Matlike) -> Matlike:
    """
    Dibuja circulos negros de monedas detectadas.
    """
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        
        monedas_with_circles = img.copy()

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            
            cv2.circle(monedas_with_circles, (a, b), r, (0, 0, 0), -1)  # Interior negro

    return monedas_with_circles

def binarize(img: Matlike) -> Matlike:
    """
    binariza la imágen
    """
    _, monedas_negras = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)
    return monedas_negras

def filter_components(stats: Matlike, labels: Matlike, centroids: Matlike, th_area: int = 5) -> tuple[int, Matlike, Matlike, Matlike]:
    """
    Filtra componentes por un área mayor a un valor definido.
    """
    filtered_stats: list[Matlike] = []
    filtered_indices: list[int] = []

    for i, stat in enumerate(stats):
        _, _, _, _, area = stat
        if area >= 5:  
            filtered_stats.append(stat)
            filtered_indices.append(i)
    
    filtered_stats = np.array(filtered_stats)
    
    new_labels = np.zeros_like(labels)
    
    for new_label, old_idx in enumerate(filtered_indices, start=1): 
        new_labels[labels == old_idx] = new_label
    
    new_centroids = centroids[filtered_indices]
    
    new_num_labels = len(filtered_indices) + 1  
    
    return new_num_labels, new_labels, filtered_stats, new_centroids

def coin_classification(img: Matlike ,num_labels: int, labels: Matlike , stats: Matlike, centroids: Matlike) -> tuple[int, int, int]:   
    num_monedas_1: int = 0
    num_monedas_50_cents: int = 0
    num_monedas_10_cents: int = 0

    im_color = img.copy()
    im_color = cv2.resize(im_color, (1366, 768))

    for centroid in centroids:
        cv2.circle(im_color, tuple(np.int32(centroid)), 0, color=(255,255,255), thickness=-1)
    for st in stats:
        if st[4] < 7000:
            cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
            num_monedas_10_cents += 1
        elif st[4] >= 7000 and st[4] < 9000:
            cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,0,255),thickness=2)
            num_monedas_1+= 1
        elif st[4] >= 9000 and st[4] < 12000:
            cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(255,0,0),thickness=2)
            num_monedas_50_cents += 1


    return num_monedas_1, num_monedas_50_cents, num_monedas_10_cents, im_color


#------------------------------------------------------------------------------

def execute(show_steps: bool = False) -> None:
    """
    Ejecución del programa de detección completa.
    """

    img: Matlike = img_reading()
    img_gray, img_blur = img_preprocessing(img)
    
    detected_circles: Matlike = detect_coins(img_blur)
    monedas_with_circles: Matlike = draw_circles(img_blur, detected_circles)
    monedas_blancas: Matlike = binarize(monedas_with_circles)

    monedas_resized: Matlike =  cv2.resize(monedas_blancas, (1366, 768))

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(monedas_resized, 8, cv2.CV_32S)

    num_labels_filtered, labels_filtered , stats_filtered, centroids_filtered  = filter_components(stats, labels, centroids)

    num_monedas_1, num_monedas_50_cents, num_monedas_10_cents, im_color = coin_classification(img, num_labels_filtered, labels_filtered , stats_filtered, centroids_filtered)

    if show_steps:
        imshow(img_gray, title='Imágen en escala de grises')
        imshow(monedas_with_circles, title='Monedas detectadas')
        imshow(monedas_blancas, title='Binarización')
    
    cv2.imshow('Monedas detectadas por valor', im_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'Cantidad de monedas de $1: {num_monedas_1}')
    print(f'Cantidad de monedas de $0.50: {num_monedas_50_cents}')
    print(f'Cantidad de monedas de $0.10: {num_monedas_10_cents}')


if __name__ == '__main__':
    execute(True)




