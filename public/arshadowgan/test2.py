import os
import os.path as osp
import cv2 as cv
import numpy as np
import math
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
node_names = {'input_image':      'placeholder/input_image:0',
              'input_mask':       'placeholder/input_mask:0',
              'output_attention': 'concat_1:0',
              'output_image':     'Tanh:0'}
data_root = osp.join(source_dir, 'data')
output_dir = osp.join(source_dir, 'output')

def read_image(image_path, channels):
    image = cv.imread(image_path, cv.IMREAD_COLOR if channels == 3 else cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
    return image

def kmeans_segmentation(image, nb_classes=6, use_color=False):
    if use_color:
        data = image.reshape((-1, 3)).astype(np.float32)
    else:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        data = gray_image.reshape((-1, 1)).astype(np.float32)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    ret, label, center = cv.kmeans(data, nb_classes, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    segmented_image = label.reshape((image.shape[0], image.shape[1]))
    return segmented_image

def normalize_segments(segmented):
    unique_values = np.unique(segmented)
    max_value = 255
    step = max_value // len(unique_values)
    normalized_image = np.zeros_like(segmented, dtype=np.uint8)
    for i, val in enumerate(unique_values):
        normalized_image[segmented == val] = i * step
    return normalized_image

def analyze_segments_for_shadows(image, labels, nb_classes):
    # Convertendo a imagem para o espaço de cor LAB
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    shadow_mask = np.zeros_like(labels, dtype=np.uint8)

    # Análise de cada segmento
    for i in range(nb_classes):
        segment_mask = (labels == i)
        if np.any(segment_mask):
            segment_lab = lab_image[segment_mask]
            l_channel = segment_lab[:, 0]  # Componente L* que representa a luminosidade
            mean_l = np.mean(l_channel)

            # Considerar sombra se a luminosidade é muito baixa
            if mean_l < 50:  # Limiar de luminosidade para detecção de sombras
                shadow_mask[segment_mask] = 255

    return shadow_mask

def apply_grabcut(image):
    if image is None:
        return None

    # Inicializar a máscara e o modelo de background/foreground
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Retângulo que cobre a maior parte, mas não toda, da imagem
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

    # Aplicar GrabCut
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 15, cv.GC_INIT_WITH_RECT)

    # Transformar a máscara para binário onde o foreground é 1
    # Pixels com 2 e 0 são background, 1 e 3 são foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Criar a imagem de foreground usando a máscara
    foreground = image * mask2[:, :, np.newaxis]

    # A máscara deve ser multiplicada por 255 para visualização correta
    return foreground, mask2 * 255

def combine_masks(segmentation_mask, contour_mask, color = 128):
    # Crie uma cópia da máscara de contornos para manipular
    combined_mask = contour_mask.copy()

    # Onde a máscara de segmentação é proximo de branca, defina a máscara combinada para cinza
    combined_mask[segmentation_mask > 200] = color

    return combined_mask

def extract_object_masks(image):
    # Usar a função threshold para garantir que a imagem está binarizada corretamente
    _, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

        # Aplicar operações morfológicas para limpar a máscara
    if binary_image is not None:
        kernel = np.ones((5, 5), np.uint8)
        # Aplicar erosão para remover ruídos finos
        binary_image = cv.erode(binary_image, kernel, iterations=1)

        # Aplicar dilatação para preencher lacunas
        binary_image = cv.dilate(binary_image, kernel, iterations=1)

        # Aplicar erosão para remover ruídos finos
        #binary_image = cv.erode(binary_image, kernel, iterations=3)

        # Aplicar fechamento para remover ruídos e preencher lacunas
        binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
        
    # Encontrar componentes conectados na imagem binarizada
    num_labels, labels = cv.connectedComponents(binary_image)

    masks = []

    # Loop para processar cada componente identificado (ignorando o fundo que é o label 0)
    for label in range(1, num_labels):
        # Criar uma máscara para o objeto atual
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        masks.append(mask)

    return masks

def find_largest_object(masks):
    max_area = 0
    largest_mask = None

    # Percorrer cada máscara no array
    for mask in masks:
        # Contar os pixels brancos (255)
        current_area = np.count_nonzero(mask == 255)

        # Se a área atual for maior que a área máxima registrada, atualizar
        if current_area > max_area:
            max_area = current_area
            largest_mask = mask



    return largest_mask

def calculate_center_of_mass(image):
    # Converter para escala de cinza se ainda não for
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Encontrar o centro de massa do objeto
    moments = cv.moments(image)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0  # Centro não encontrado, ou área do objeto é zero

    return cx, cy

def create_image_with_center_marks(image, object_center, shadow_center):
    # Converter para BGR para desenhar a linha em vermelho
    output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    # Desenhar o centro de massa do objeto em verde
    cv.circle(output_image, object_center, 5, (0, 255, 0), -1)
    
    # Desenhar o centro de massa da sombra em azul
    cv.circle(output_image, shadow_center, 5, (255, 0, 0), -1)
    
    # Desenhar uma linha entre o centro de massa do objeto e o centro de massa da sombra
    cv.line(output_image, object_center, shadow_center, (0, 0, 255), 2)
    
    return output_image


def calculate_shadow_angle(object_center, shadow_center):
    ox, oy = object_center
    sx, sy = shadow_center
    
    # Calcular o vetor entre o centro do objeto e o centro da sombra
    vector = (sx - ox, sy - oy)
    
    # Calcular o ângulo do vetor
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    
    return angle


def find_closest_shadow(object_mask, shadow_mask):
    # Calcular o centro de massa do objeto
    object_center = calculate_center_of_mass(object_mask)
    object_base = find_base_from_center_of_mass(object_mask, object_center)
    if object_base is None:
        return None
    
    num_labels, shadow_labels = cv.connectedComponents((shadow_mask == 255).astype(np.uint8))

    min_score = float('inf')
    closest_shadow = None

    for label in range(1, num_labels):  # Ignorar o fundo
        shadow_component_mask = (shadow_labels == label).astype(np.uint8) * 255
        shadow_center = calculate_center_of_mass(shadow_component_mask)
        dist = np.sqrt((object_base[0] - shadow_center[0]) ** 2 + (object_base[1] - shadow_center[1]) ** 2)
        area = cv.countNonZero(shadow_component_mask)
        
        # Calcular o score combinando a distância e o tamanho da sombra com penalização exponencial na distância
        adjusted_area = area / (np.exp(dist / 100) + 1)  # Ajustar o peso do tamanho com base na distância
        score = dist / (adjusted_area + 1)  # Adicionando 1 para evitar divisão por zero
        
        if score < min_score:
            min_score = score
            closest_shadow = shadow_component_mask
    
    # Se não encontrar nenhuma sombra, retornar uma máscara vazia
    if closest_shadow is None:
        #print("None closest shadow detected")
        closest_shadow = np.zeros_like(shadow_mask)
    
    return closest_shadow

def combine_object_and_shadow_mask(object_mask, shadow_mask):
    # Encontrar a sombra mais próxima do objeto
    closest_shadow = find_closest_shadow(object_mask, shadow_mask)
    
    # Combinar a máscara do objeto com a sombra mais próxima
    combined_mask = object_mask.copy()
    combined_mask[closest_shadow == 255] = 128  # Definir a sombra como cinza (valor 128)
    
    return combined_mask

def find_base_from_center_of_mass(image, center):
    h, w = image.shape
    cx, cy = center

    # Descer verticalmente a partir do centro de massa até encontrar um pixel preto
    for y in range(cy, h):
        if image[y, cx] == 0:
            return (cx, y - 1)  # Retornar o pixel branco logo acima do pixel preto

    return None

def calculate_proportion(largest_object_mask):
    # Encontrar a bounding box do maior objeto
    x, y, w, h = cv.boundingRect(largest_object_mask)
    
    # Calcular a proporção altura/largura
    proportion = w / h
    return proportion

def main():
    if not osp.exists(data_root):
        print("No data directory")
        exit(0)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    image_list = sorted(os.listdir(osp.join(data_root, 'noshadow')))
    for i in image_list:
        output_new_dir = osp.join(output_dir, osp.splitext(i)[0])
        if not osp.exists(output_new_dir):
            os.makedirs(output_new_dir)
        image_path = osp.join(data_root, 'noshadow', i)
        mask_path = osp.join(data_root, 'mask', i)

        image = read_image(image_path, 3)
        if image is None:
            continue
        
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
        mask = read_image(mask_path, 1)
        inverse_mask = cv.bitwise_not(mask)
        masked_img = cv.bitwise_and(img_gray, inverse_mask)
        img_blur = cv.GaussianBlur(masked_img, (3,3), 0) 
        blured_image = cv.GaussianBlur(image, (5, 5), 0)
        
        # GRABCUT #######################################################################################
        grabCut_image, grabCut_mask = apply_grabcut(blured_image)
        cv.imwrite(osp.join(output_new_dir, f'{i}grabcut_foreground.png'), grabCut_image)
        cv.imwrite(osp.join(output_new_dir, f'{i}grabcut_mask.png'), grabCut_mask)
        
        # Segmentação por k-means ################################################################################################################
        nb_classes = 10
        segmented = kmeans_segmentation(blured_image, nb_classes, use_color=True)
        normalized_segmented = normalize_segments(segmented)
        # Mapa de cores para visualização
        segmented_image = cv.applyColorMap(normalized_segmented.astype(np.uint8), cv.COLORMAP_JET)
        cv.imwrite(osp.join(output_new_dir, f'{i}_kmeans_segmented.png'), segmented_image)
        # Análise de sombras nos segmentos
        segmentation_mask = analyze_segments_for_shadows(blured_image, segmented, nb_classes)
        cv.imwrite(osp.join(output_new_dir, f'{i}_segmentation_mask.png'), segmentation_mask)
        
        # COMBINING MASKS #######################################################################################
        combined_mask = combine_masks(segmentation_mask, grabCut_mask, 128)
        cv.imwrite(osp.join(output_new_dir, f'{i}_combined_mask.png'), combined_mask)
        objectsWithShadowsImage = combine_masks(mask, combined_mask, 0)
        cv.imwrite(osp.join(output_new_dir, f'{i}_objectsWithShadowsImage.png'), objectsWithShadowsImage)
        
        # Pega o maior objeto da imagem ################################################################
        objectsWithoutShadow = combine_masks(segmentation_mask, objectsWithShadowsImage,0)
        objectsWithoutShadow = combine_masks(mask, objectsWithoutShadow, 0)
        cv.imwrite(osp.join(output_new_dir, f'{i}_objectsWithoutShadow.png'), objectsWithoutShadow)
        grabCut_masks = extract_object_masks(objectsWithoutShadow)
        largest_object_mask = find_largest_object(grabCut_masks)
        if largest_object_mask is not None:
            cv.imwrite(osp.join(output_new_dir, f'{i}_largest_object.png'),  largest_object_mask)

        # Pega o centro de massa do maior objeto ##########################################################
        object_center = calculate_center_of_mass(largest_object_mask)

        # Junta maior objeto com sua sombra ############################################################
        mascara_filtrada = combine_masks(mask, segmentation_mask, 0)
        combined_mask2 = combine_object_and_shadow_mask(largest_object_mask, mascara_filtrada)
        cv.imwrite(osp.join(output_new_dir, f'{i}_largest_object+shadow.png'),  combined_mask2)
        
        # Calcular o centro de massa da sombra
        shadow_center = calculate_center_of_mass((combined_mask2 == 128).astype(np.uint8))
        
        # Calcular o ângulo da sombra
        shadow_angle = calculate_shadow_angle(object_center, shadow_center)
        
        # Calcular a proporção do objeto
        proportion = calculate_proportion(largest_object_mask)
        
        if shadow_angle is not None:
            #print(f"Angulo da Sombra: {shadow_angle} graus")
            #print(f"Centro de Massa do Objeto: {object_center}")
            #print(f"Centro de Massa da Sombra: {shadow_center}")
            print(f"{object_center[0]} {object_center[1]} {shadow_center[0]} {shadow_center[1]} {proportion}")
            marked_image = create_image_with_center_marks(combined_mask2, object_center, shadow_center)
            cv.imwrite(osp.join(output_new_dir, f'{i}_largest_object_mass_center.png'),  marked_image)
            cv.imwrite(osp.join("output/Final_Images", f'{i}_final_image.png'), marked_image)
        
        output_image = marked_image


if __name__ == '__main__':
	main()