import cv2
import numpy as np
import matplotlib.pyplot as plt

Matlike = np.ndarray

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    """
    Función para mostrar las imágenes
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



def img_procesor(img: Matlike) -> Matlike:
    """
    Función procesadora de la imágen.
    Se aplica 'Top-Hat' y un umbral  
    """  
    # Top-Hat
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))       
    img_top_hat = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
    img_top_hat = cv2.normalize(img_top_hat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # imshow(img_top_hat, title = "Imagen procesada con Top-Hat")

    # Umbralado 
    _, img_thresh = cv2.threshold(img_top_hat, 80, 255, cv2.THRESH_BINARY)
    # imshow(img_thresh, title = "Umbralado")

    # Visualización 
    # plt.figure()   
    # ax1 = plt.subplot(131); plt.xticks([]), plt.yticks([]), plt.imshow(img, cmap = 'gray'), plt.title('Imagen Original (escala de grises)'), plt.colorbar(shrink = 0.5)
    # plt.subplot(132,sharex=ax1,sharey=ax1), plt.imshow(img_top_hat, cmap="gray"), plt.title('Top-Hat'), plt.colorbar(shrink = 0.5)
    # plt.subplot(133,sharex=ax1,sharey=ax1), plt.imshow(img_thresh, cmap='gray'), plt.title('Umbralado'), plt.colorbar(shrink = 0.5)
    # plt.show()

    return img_top_hat, img_thresh


def plate_filter(img_thresh: Matlike) -> list:
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

    # Coloreamos los elementos
    labels = np.uint8(255/num_labels*labels)
    #imshow(img=labels)
    im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
    


    area_min, area_max = 40, 150
    aspect_min, aspect_max = 1.5, 2.5  

    letters_stats = []

    for i, st in enumerate(stats):
        
        if i == 0:  # Fondo
            continue
        
        # Filtrado por área
        area = st[4]
        if area_min <= area <= area_max:
            
            # Filtrado por relación de aspecto
            width = st[2]
            height = st[3]
            aspect_ratio =  height / width  if width > 0 else 0 
                  
            if aspect_min <= aspect_ratio <= aspect_max:
                letters_stats.append(st)

                # cv2.rectangle(im_color, (st[0], st[1]), (st[0] + width, st[1] + height), color=(0, 255, 0), thickness=1)
                # imshow(img=im_color, color_img=True)
            
                
    return letters_stats
    
    
    





for i in range(1,13):

    img=cv2.imread(f'./data/img0{i}.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_top_hat, img_thresh = img_procesor(img_gray)

    letter_stats = plate_filter(img_thresh)

    img_color = img.copy()

    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)


    for st in letter_stats:
        cv2.rectangle(img_color_rgb, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=1)



    plt.figure()
    plt.imshow(img_color_rgb)
    plt.show()    


