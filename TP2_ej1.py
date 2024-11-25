import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

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

def procesar_imagen_para_deteccion(img: Matlike) -> np.ndarray:
    """
    Función para preprocesar una imagen con el objetivo de detectar dados y monedas. 
    Realiza una serie de transformaciones en la imagen para mejorar la detección de formas 
    mediante el uso de la representación HSV, filtrado de ruido, y operaciones morfológicas.
    
    Args:
        img (Any): Imagen de entrada en formato BGR.

    Returns:
        np.ndarray: Imagen procesada después de aplicar umbralización y operaciones morfológicas.
    """
    
    imagen_hsv: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    imagen_redimensionada = cv2.resize(imagen_hsv, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    imagen_hsv = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2HSV)
    imagen_hsv = cv2.blur(imagen_hsv, (15, 15))

    imagen_modificada_hsv = imagen_hsv.copy()

    # Ajuste del canal Hue (matiz) con valores dentro del rango [0, 179]
    imagen_modificada_hsv[:, :, 0] = np.clip(imagen_hsv[:, :, 0] + 26, 0, 179)

    # Ajuste del canal Saturation (saturación) dentro del rango [0, 255]
    imagen_modificada_hsv[:, :, 1] = np.clip(cv2.blur(imagen_hsv[:, :, 1] + 0, (5, 5)), 0, 255)

    # Ajuste del canal Value (valor) dentro del rango [0, 255]
    imagen_modificada_hsv[:, :, 2] = np.clip(imagen_hsv[:, :, 2] + 0, 0, 255)

    # Aplicar un ajuste de "lightness" sobre el canal Value
    imagen_modificada_hsv[:, :, 2] = np.clip(cv2.blur(imagen_modificada_hsv[:, :, 2] + 0, (5, 5)), 0, 255)

    imagen_modificada_bgr = cv2.cvtColor(imagen_modificada_hsv, cv2.COLOR_HSV2BGR)

    imagen_gris = cv2.cvtColor(imagen_modificada_bgr, cv2.COLOR_BGR2GRAY)

    _, imagen_umbralizada = cv2.threshold(imagen_gris, 63, 255, cv2.THRESH_BINARY)

    num_etiquetas, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(imagen_umbralizada, connectivity=8)

    imagen_filtrada = np.zeros_like(imagen_umbralizada, dtype=np.uint8)
    
    for i in range(1, num_etiquetas):  # Omite el fondo (etiqueta 0)
        if estadisticas[i, cv2.CC_STAT_AREA] >= 1300:
            imagen_filtrada[etiquetas == i] = 255

    # Aplicar operaciones morfológicas para mejorar la detección
    estructura: np.array = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Cerrar las formas (rellenar espacios pequeños entre los dados y completar monedas)
    imagen_cerrada: np.array = cv2.morphologyEx(imagen_filtrada.copy(), cv2.MORPH_CLOSE, estructura, iterations=9)

    # Abrir las formas (separar figuras que se unieron durante el cierre)
    imagen_abierta: np.array = cv2.morphologyEx(imagen_cerrada.copy(), cv2.MORPH_OPEN, estructura, iterations=9)
    
    imshow(imagen_abierta, title="Imagen Preprocesada")

    return imagen_abierta

def detectar_figuras_por_factor_forma(img: np.array, min_fp: float = 0.06, max_fp: float = 0.08) -> list[np.array]:
    """
    Detecta cuadrados en una imagen utilizando el factor de forma.
    
    Parametros:
        - img (np.array): Imagen en escala de grises.
        - min_fp (float): Valor mínimo del factor de forma para clasificar un contorno como círculo.
        - max_fp (float): Valor máximo del factor de forma para clasificar un contorno como círculo.
    
    Retorna:
        - cuadrados (list[np.array]): Lista de contornos clasificados como cuadrados.
    """
    imagen_dibujar = img.copy()
    
    _, binarized_img = cv2.threshold(img.copy(), 50, 190, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(binarized_img, 50, 150)

    imshow(edges, title="Imagen con Canny")

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cuadrados = []

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        p = cv2.arcLength(contour, True)
        shape = None
        
        if p == 0:
            continue
        
        fp = area / (p**2)

        if not min_fp < fp < max_fp:
            cuadrados.append(contour)
            shape = "cuadrado"
            color = (255,0,0)

        x, y = contour[0,0]
        
        if shape is not None:
            cv2.drawContours(imagen_dibujar, [contour], -1, color, 2)

            cv2.putText(imagen_dibujar, f'Area: {area:.2f}', (x-40, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    imshow(imagen_dibujar, title="Imagen con contornos")

    return cuadrados

def contar_circulos(dados: list[np.array], thresh_img: Matlike) -> tuple[dict[str, int], int]:
    """
    Detecta los puntos en los dados y calcula el puntaje basado en los círculos detectados.
    
    Parametros:
        - dados (list[np.array]): Lista de contornos de los dados (regiones de los dados).
    
    Retorna:
        - dict_dados (dict[str, int]): Diccionario con los puntajes de cada dado.
        - puntaje_total (int): Puntaje total sumando los puntajes de todos los dados.
    """
    dict_dados = {}
    puntaje_total = 0
    dado_n: int = 1
    
    imagen_original = cv2.resize(img_reading(), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    for dado in dados:
        x, y, w, h = cv2.boundingRect(dado)
        area = cv2.contourArea(dado)

        if area < 500:
            continue
        
        roi_dado = imagen_original[y:y+h, x:x+w]
        
        imagen_gris: np.array = cv2.cvtColor(roi_dado, cv2.COLOR_BGR2GRAY)

        imagen_desenfocada: np.array = cv2.GaussianBlur(imagen_gris, (15, 15), 3)
        
        imshow(imagen_desenfocada, title="Imagen ROI dados")

        circles = cv2.HoughCircles(
            imagen_desenfocada,
            cv2.HOUGH_GRADIENT,
            dp=1.3,
            minDist=10,
            param1=10,
            param2=22,
            minRadius=1,
            maxRadius=30
        )

        puntaje_dado = 0

        dibujar = roi_dado.copy()
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                puntaje_dado += 1
                
                cv2.circle(dibujar, (cx, cy), r, (0, 255, 0), 1)
                
        imshow(dibujar, title="Resultados círculos")
        
        dict_dados[f"Dado {dado_n}"] = puntaje_dado
        puntaje_total += puntaje_dado

        dado_n += 1
    
    return dict_dados, puntaje_total

def execute(show_steps: bool = False) -> None:
    """
    Ejecución completa de la detección de monedas y dados, incluyendo el cálculo de puntajes.
    """
    img: Matlike = img_reading()
    img_gray, img_blur = img_preprocessing(img)
    detected_circles: Matlike = detect_coins(img_blur)
    monedas_with_circles: Matlike = draw_circles(img_blur, detected_circles)
    monedas_blancas: Matlike = binarize(monedas_with_circles)
    monedas_resized: Matlike = cv2.resize(monedas_blancas, (1366, 768))


    _, labels, stats, centroids = cv2.connectedComponentsWithStats(monedas_resized, 8, cv2.CV_32S)
    num_labels_filtered, labels_filtered, stats_filtered, centroids_filtered = filter_components(stats, labels, centroids)
    num_monedas_1, num_monedas_50_cents, num_monedas_10_cents, im_color = coin_classification(img, num_labels_filtered, labels_filtered, stats_filtered, centroids_filtered)

    if show_steps:
        imshow(img, title='Imagen')
        imshow(img_gray, title='Escala de Grises')
        imshow(img_blur, title='Blur')
        imshow(monedas_with_circles, title='Círculos Detectados')
        imshow(monedas_blancas, title='Binarización')

    print(f'Cantidad de monedas de $1: {num_monedas_1}')
    print(f'Cantidad de monedas de $0.50: {num_monedas_50_cents}')
    print(f'Cantidad de monedas de $0.10: {num_monedas_10_cents}')

    cv2.imshow('Monedas detectadas por valor', im_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    img_preprocesada = procesar_imagen_para_deteccion(img)
    
    _, thresh_image = cv2.threshold(img_preprocesada, 60, 255, cv2.THRESH_BINARY)
    
    cuadrados = detectar_figuras_por_factor_forma(img_preprocesada, min_fp=0.063, max_fp=0.075)

    dados, puntaje_total = contar_circulos(dados=cuadrados, thresh_img=thresh_image)

    print(f"Cantidad de dados: {len(dados)}")
    print(f"Total: {puntaje_total}")


if __name__ == '__main__':
    execute(True)




