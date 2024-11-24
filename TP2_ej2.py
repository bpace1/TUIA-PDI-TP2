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



def img_procesor(img: Matlike) -> tuple[Matlike, Matlike]:
    """
    Función procesadora de la imágen.
    Se aplica 'Top-Hat' y un umbral  
    """  
    #erosion: Matlike = cv2.erode(img, kernel=np.ones((1,1),np.uint8), iterations=1)
    #blur: Matlike = cv2.GaussianBlur(erosion, (1, 1), 0)

    se: Matlike = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))       
    

    img_top_hat: Matlike = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
    img_top_hat_normalized: Matlike = cv2.normalize(img_top_hat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #_, img_thresh = cv2.threshold(img_top_hat, 80, 255, cv2.THRESH_BINARY)
    _, img_thresh = cv2.threshold(img_top_hat_normalized, 100, 255, cv2.THRESH_BINARY)

    return img_top_hat, img_thresh

def get_coordinates(stats: Matlike, area_min: int = 20, area_max: int = 200, 
                    aspect_min: float = 1.5, aspect_max: float = 2.5) -> list[Matlike]:
    """
    Devuelve las coordenadas de cada bounding box.
    """
    letters_stats_coordinates = []

    for _, st in enumerate(stats, start=1):
    # Filtrado por área
        area = st[4]
        if area_min <= area <= area_max:
            
            # Filtrado por relación de aspecto
            width = st[2]
            height = st[3]
            aspect_ratio =  height / width  if width > 0 else 0 
                
            if aspect_min <= aspect_ratio <= aspect_max:
                letters_stats_coordinates.append(st)
                # cv2.rectangle(im_color, (st[0], st[1]), (st[0] + width, st[1] + height), color=(0, 255, 0), thickness=1)
                # imshow(img=im_color, color_img=True)
    return letters_stats_coordinates

def filter_components(num_labels: int, stats: np.ndarray, labels: np.ndarray,
                       centroids: np.ndarray, x_tolerance: int = 30, y_tolerance: int = 15) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtra componentes cercanos entre sí y alineados horizontalmente, específicos para la detección de patentes.
    
    :param num_labels: Número de etiquetas encontradas.
    :param stats: Estadísticas de cada componente conectado.
    :param labels: Matriz de etiquetas de los componentes.
    :param centroids: Coordenadas de los centroides.
    :param x_tolerance: Tolerancia en el eje X para considerar proximidad.
    :param y_tolerance: Tolerancia en el eje Y para considerar alineación horizontal.
    :return: Nuevos componentes filtrados: número, estadísticas, etiquetas y centroides.
    """
    filtered_indices: list[int] = []
    
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            # Calcular distancia en X e Y entre componentes
            delta_x = abs(centroids[i][0] - centroids[j][0])
            delta_y = abs(centroids[i][1] - centroids[j][1])
            
            # Filtrar por proximidad y alineación horizontal
            if delta_x <= x_tolerance and delta_y <= y_tolerance:
                filtered_indices.append(i)
                filtered_indices.append(j)

    # Eliminar duplicados
    filtered_indices = list(set(filtered_indices))

    # Filtrar stats y centroids por los índices seleccionados
    filtered_stats = stats[filtered_indices]
    filtered_centroids = centroids[filtered_indices]

    # Crear nueva matriz de etiquetas con solo los componentes filtrados
    new_labels = np.zeros_like(labels, dtype=np.int32)
    for new_idx, original_idx in enumerate(filtered_indices):
        new_labels[labels == original_idx] = new_idx + 1

    # Número de etiquetas filtradas
    new_num_labels = len(filtered_indices)

    return new_num_labels, filtered_stats, new_labels, filtered_centroids



def plate_filter(img_thresh: Matlike) -> tuple[list[Matlike], Matlike]:
    """ 
    Función que filtra la imágen umbralizada en busca de las letras pertenecientes a la patente.
    Para ello, se utiliza Connected Components y se filtra con respcto al área y relación de aspecto.

    Área: 40 - 150 pixeles
    
    Relación de aspecto:
        y aprox 20 pix
        x aprox 10 pix  --> y = 2x 
                        --> y/x = 2 (aprox)
    
    Se devuelve una lista con las coordenadas de cada bounding box              
    """
   
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thresh, connectivity, cv2.CV_32S)  

    new_num_labels, filtered_stats, new_labels, filtered_centroids = filter_components(num_labels, stats, labels, centroids)

    # Coloreamos los elementos
    labels = np.uint8(255/num_labels*labels)
    im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
    
    letters_stats_final = get_coordinates(filtered_stats)
     
                
    return letters_stats_final, im_color


def img_reading(name: str) -> np.ndarray:
    """
    Se encarga de ecargar la imágen
    """
    PATH = os.getcwd()
    DATA_PATH = os.path.join(PATH, 'data')

    img: np.ndarray =  cv2.imread(os.path.join(DATA_PATH,name))
    return img

def normalize_coordinates(stats: list[Matlike], img_shape: tuple[int, int]) -> list[dict]:
    """
    Normaliza las coordenadas de los bounding boxes al rango [0, 1] basado en el tamaño de la imagen.
    """
    img_h, img_w = img_shape[:2]
    normalized_stats = []

    for st in stats:
        x, y, w, h, area = st
        norm_x = x / img_w
        norm_y = y / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        normalized_stats.append({
            "x": norm_x, "y": norm_y,
            "width": norm_w, "height": norm_h,
            "area": area / (img_w * img_h)
        })
    return normalized_stats

def execute(show_steps: bool = True) -> None:
    dir: list[str] = os.listdir(os.path.join(os.getcwd(), 'data'))
    patentes: list[str] = [dir for dir in dir if dir.startswith('img')]

    for image_path in patentes:
        img: Matlike = img_reading(image_path)
        img_gray: Matlike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_top_hat, img_thresh = img_procesor(img_gray)

        letter_stats, img_color = plate_filter(img_thresh)

        img_color_rgb: Matlike = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        if show_steps:
            imshow(img_top_hat, title="Imagen procesada con Top-Hat")
            imshow(img_thresh, title="Umbralado")

        for st in letter_stats:
            x, y, w, h = int(st["x"] * img.shape[1]), int(st["y"] * img.shape[0]), \
                         int(st["width"] * img.shape[1]), int(st["height"] * img.shape[0])
            cv2.rectangle(img_color_rgb, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)

        plt.figure()
        plt.imshow(img_color_rgb)
        plt.title("Letras detectadas y normalizadas")
        plt.show()

if __name__ == '__main__':
    execute(True)