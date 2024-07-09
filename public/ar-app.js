var renderer, rend;
var emptyObj, vObj, vObjMask, light, origLight, shadowPlane, wPlane, dPlane;
var sphere;
var adjustX, adjustZ;
let position = new THREE.Vector3(); // Para armazenar posição da câmera virtual
let quaternion = new THREE.Quaternion(); // Para armazenar a rotação da câmera virtual
var mouse = new THREE.Vector2();
var emptyPlane;
var ray = new THREE.Raycaster();
var point = new THREE.Vector2();
var loader = new THREE.TextureLoader();

var planeSize = 150.00;
var sPlaneSize = 150.00;
var sPlaneSegments = 300.00;
var vObjHeight = 1.20;
var vObjRatio = 1.00;
var adjustX = 0.00;
var adjustZ = 0.00;
var done = false;

const loaderElement = document.createElement("div");
loaderElement.setAttribute("class", "loader");
loaderElement.setAttribute("id", "loader");
document.body.appendChild(loaderElement);
//const select = document.getElementById("select");
const submitBtn = document.getElementById("submitButton");
const returnBtn = document.getElementById("returnButton");
returnBtn.style.display = "none";
const select2 = document.getElementById("select2");
const select3 = document.getElementById("select3");
select3.style.display = "none";
loaderElement.style.display = "none";
var selectValue = "0";

renderer = new THREE.WebGLRenderer({
    preserveDrawingBuffer: true,
    antialias: true,
    alpha: true
});
renderer.setClearColor(new THREE.Color('lightgrey'), 0);
renderer.domElement.style.position = 'absolute';
renderer.domElement.style.top = '0px';
renderer.domElement.style.left = '0px';
renderer.shadowMap.enabled = true;
renderer.shadowMapSoft = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap; 
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.domElement.addEventListener('click', onDocumentMouseClick, false);
document.body.appendChild(renderer.domElement);

let AR = {
    source: null,
    context: null,
}
let camera, scene;

function onResize() {
    AR.source.onResizeElement();
    AR.source.copyElementSizeTo(renderer.domElement);
    if (AR.context.arController !== null) {
        AR.source.copyElementSizeTo(AR.context.arController.canvas);
    }
}
window.addEventListener('resize', function(){
    onResize();
});

function createVirtualObj(){
    var cube = new THREE.BoxBufferGeometry(vObjHeight, vObjHeight * vObjRatio, vObjHeight);
    var wood = new THREE.MeshLambertMaterial({map: loader.load("my-textures/face/wood.png")});
    var transparentMaterial = new THREE.MeshBasicMaterial({
        map: loader.load("my-textures/face/wood.png"),
        transparent: true,
        opacity: 1,
    });
    return vObj = new THREE.Mesh(cube, transparentMaterial);
}

function create_scene() {
    let scene = new THREE.Scene();
    origLight = new THREE.DirectionalLight(0xffffff);
    origLight.castShadow = true;
    var d = vObjRatio * vObjHeight * 10;
    origLight.shadow.camera.left = -d;
    origLight.shadow.camera.right = d;
    origLight.shadow.camera.top = d;
    origLight.shadow.camera.bottom = -d;

    origLight.shadow.mapSize.width = 4096;
    origLight.shadow.mapSize.height = 4096;

    light = origLight.clone();
    scene.add(light);

    camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 1000);

    scene.add(camera);
    vObj = createVirtualObj();
    scene.add(vObj);
    vObj.position.set(adjustX, vObjRatio * vObjHeight / 2, adjustZ);
    vObj.castShadow = false;

    camera = camera;
    scene = scene;

    return [scene, camera];
}

var aux = create_scene();
scene = aux[0];
camera = aux[1];
var helper = new THREE.CameraHelper(camera);

export function updateAR() {
    if (AR.source) {
        if (AR.source.ready === false) return;
        AR.context.update(AR.source.domElement);
        scene.visible = camera.visible;
        return camera;
    }
    return null;
}

function setSource(type, url) {
    AR.source = new THREEx.ArToolkitSource({
        sourceType: type,
        sourceUrl: url,
    });
}

export function setARStuff(source) {
    switch (source) {
        case 'image':
            setSource('image', "my-images/new_imagem_4.jpg");
            break;
        case 'video':
            setSource('video', "my-videos/video4.MOV");
            break;
        case 'webcam':
            setSource('webcam', null);
            break;
    }

    AR.source.init(function onReady() {
        onResize();
    });

    AR.context = new THREEx.ArToolkitContext({
        cameraParametersUrl: 'data/camera_para.dat',
        detectionMode: 'mono',
    });

    AR.context.init(function onCompleted() {
        camera.projectionMatrix.copy(AR.context.getProjectionMatrix());
    });

    new THREEx.ArMarkerControls(AR.context, camera, {
        type: "pattern",
        patternUrl: "data/kanji.patt",
        changeMatrixMode: 'cameraTransformMatrix'
    });

    scene.visible = true;
}

setARStuff('image');

var shadowMat = new THREE.ShadowMaterial({
    opacity: 0.75,
    side: THREE.DoubleSide,
});
var splane = new THREE.PlaneGeometry(sPlaneSize, sPlaneSize, sPlaneSegments, sPlaneSegments);
shadowPlane = new THREE.Mesh(splane, shadowMat);
shadowPlane.receiveShadow = true;
shadowPlane.rotation.x = -Math.PI / 2;
shadowPlane.position.y = -0.01;
scene.add(shadowPlane);

document.getElementById("select2").addEventListener("change", async () => {
    const select = document.getElementById("select2");
    let value = select.value;
    if (selectValue != value) {
        selectValue = value;
        switch (value) {
            case '0':
                setSource('webcam', null);
                break;
            case '1':
                setSource('image', 'my-images/img_extobj_1.jpeg');
                break;
            case '2':
                setSource('image', 'my-images/img_extobj_2.jpeg');
                break;
            case '3':
                setSource('image', 'my-images/img_extobj_3.jpeg');
                break;
        }
    }
});

returnBtn.addEventListener('click', async () => {
    let value = select2.value;
    switch (value) {
        case '0':
            setSource('webcam', null);
            break;
        case '1':
            setSource('image', 'my-images/img_extobj_1.jpeg');
            break;
        case '2':
            setSource('image', 'my-images/img_extobj_2.jpeg');
            break;
        case '3':
            setSource('image', 'my-images/img_extobj_3.jpeg');
            break;
    }
    //select.style.display = "block";
    select2.style.display = "block";
    submitBtn.style.display = "block";
    select3.style.display = "none";
    returnBtn.style.display = "none";
    light.position.set(0, 10, 0);
});

async function loadPyodideAndPackages() {
    let pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.1/full/"
    });
    await pyodide.loadPackage("micropip");
    await pyodide.runPythonAsync(`
        import micropip
        await micropip.install('numpy')
        await micropip.install('opencv-python')
    `);
    return pyodide;
}

const pythonCode = `
import cv2 as cv
import numpy as np
import math

def read_image(image_data, channels):
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR if channels == 3 else cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Erro ao carregar a imagem")
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
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    shadow_mask = np.zeros_like(labels, dtype=np.uint8)
    for i in range(nb_classes):
        segment_mask = (labels == i)
        if np.any(segment_mask):
            segment_lab = lab_image[segment_mask]
            l_channel = segment_lab[:, 0]
            mean_l = np.mean(l_channel)
            if mean_l < 50:
                shadow_mask[segment_mask] = 255
    return shadow_mask

def apply_grabcut(image):
    if image is None:
        return None

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 15, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image * mask2[:, :, np.newaxis]
    return foreground, mask2 * 255

def combine_masks(segmentation_mask, contour_mask, color=128):
    combined_mask = contour_mask.copy()
    combined_mask[segmentation_mask > 200] = color
    return combined_mask

def extract_object_masks(image):
    _, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    if binary_image is not None:
        kernel = np.ones((5, 5), np.uint8)
        binary_image = cv.erode(binary_image, kernel, iterations=1)
        binary_image = cv.dilate(binary_image, kernel, iterations=1)
        binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
    num_labels, labels = cv.connectedComponents(binary_image)
    masks = []
    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        masks.append(mask)
    return masks

def find_largest_object(masks):
    max_area = 0
    largest_mask = None
    for mask in masks:
        current_area = np.count_nonzero(mask == 255)
        if current_area > max_area:
            max_area = current_area
            largest_mask = mask
    return largest_mask

def calculate_center_of_mass(image):
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    moments = cv.moments(image)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

def create_image_with_center_marks(image, object_center, shadow_center):
    output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    cv.circle(output_image, object_center, 5, (0, 255, 0), -1)
    cv.circle(output_image, shadow_center, 5, (255, 0, 0), -1)
    cv.line(output_image, object_center, shadow_center, (0, 0, 255), 2)
    return output_image

def calculate_shadow_angle(object_center, shadow_center):
    ox, oy = object_center
    sx, sy = shadow_center
    vector = (sx - ox, sy - oy)
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    return angle

def find_closest_shadow(object_mask, shadow_mask):
    object_center = calculate_center_of_mass(object_mask)
    object_base = find_base_from_center_of_mass(object_mask, object_center)
    if object_base is None:
        return None
    num_labels, shadow_labels = cv.connectedComponents((shadow_mask == 255).astype(np.uint8))
    min_score = float('inf')
    closest_shadow = None
    for label in range(1, num_labels):
        shadow_component_mask = (shadow_labels == label).astype(np.uint8) * 255
        shadow_center = calculate_center_of_mass(shadow_component_mask)
        dist = np.sqrt((object_base[0] - shadow_center[0]) ** 2 + (object_base[1] - shadow_center[1]) ** 2)
        area = cv.countNonZero(shadow_component_mask)
        adjusted_area = area / (np.exp(dist / 100) + 1)
        score = dist / (adjusted_area + 1)
        if score < min_score:
            min_score = score
            closest_shadow = shadow_component_mask
    if closest_shadow is None:
        closest_shadow = np.zeros_like(shadow_mask)
    return closest_shadow

def combine_object_and_shadow_mask(object_mask, shadow_mask):
    closest_shadow = find_closest_shadow(object_mask, shadow_mask)
    combined_mask = object_mask.copy()
    combined_mask[closest_shadow == 255] = 128
    return combined_mask

def find_base_from_center_of_mass(image, center):
    h, w = image.shape
    cx, cy = center
    for y in range(cy, h):
        if image[y, cx] == 0:
            return (cx, y - 1)
    return None

def calculate_proportion(largest_object_mask):
    x, y, w, h = cv.boundingRect(largest_object_mask)
    proportion = w / h
    return proportion

def process_image(image_data, mask_data):
    image = read_image(image_data, 3)
    mask = read_image(mask_data, 1)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    masked_img = cv.bitwise_and(img_gray, cv.bitwise_not(mask))
    img_blur = cv.GaussianBlur(masked_img, (3, 3), 0)
    blured_image = cv.GaussianBlur(image, (5, 5), 0)
    grabCut_image, grabCut_mask = apply_grabcut(blured_image)
    nb_classes = 10
    segmented = kmeans_segmentation(blured_image, nb_classes, use_color=True)
    normalized_segmented = normalize_segments(segmented)
    segmentation_mask = analyze_segments_for_shadows(blured_image, segmented, nb_classes)
    combined_mask = combine_masks(segmentation_mask, grabCut_mask, 128)
    objectsWithShadowsImage = combine_masks(mask, combined_mask, 0)
    objectsWithoutShadow = combine_masks(segmentation_mask, objectsWithShadowsImage, 0)
    objectsWithoutShadow = combine_masks(mask, objectsWithoutShadow, 0)
    grabCut_masks = extract_object_masks(objectsWithoutShadow)
    largest_object_mask = find_largest_object(grabCut_masks)
    object_center = calculate_center_of_mass(largest_object_mask)
    combined_mask2 = combine_object_and_shadow_mask(largest_object_mask, segmentation_mask)
    shadow_center = calculate_center_of_mass((combined_mask2 == 128).astype(np.uint8))
    shadow_angle = calculate_shadow_angle(object_center, shadow_center)
    proportion = calculate_proportion(largest_object_mask)
    result = {
        "objectCenter": list(object_center),
        "shadowCenter": list(shadow_center),
        "shadowAngle": shadow_angle,
        "proportion": proportion
    }
    return result
`;

async function processImageWithPyodide(imageSrc, maskSrc) {
    const pyodide = await loadPyodideAndPackages();
    await pyodide.runPythonAsync(pythonCode);

    const imageResponse = await fetch(imageSrc);
    const imageData = await imageResponse.arrayBuffer();

    const maskResponse = await fetch(maskSrc);
    const maskData = await maskResponse.arrayBuffer();

    pyodide.globals.set("image_data", pyodide.toPy(new Uint8Array(imageData)));
    pyodide.globals.set("mask_data", pyodide.toPy(new Uint8Array(maskData)));

    const result = pyodide.runPython(`
        result = process_image(image_data, mask_data)
        result
    `);

    return result.toJs();
}

document.getElementById("submitButtonInput").addEventListener("click", async () => {
    try {
        //let value = select.value;
        loaderElement.style.display = "block";
        select2.style.display = "none";
        submitBtn.style.display = "none";

        light.position.set(0, 10, 0);

        var vw, vh;
        if (AR.source.parameters.sourceType == "webcam" || AR.source.parameters.sourceType == "video") {
            vw = AR.source.domElement.videoWidth;
            vh = AR.source.domElement.videoHeight;
        } else {
            vw = AR.source.domElement.naturalWidth;
            vh = AR.source.domElement.naturalHeight;
        }
        var w = renderer.domElement.width;
        var h = renderer.domElement.height;
        var cw = renderer.domElement.clientWidth;
        var ch = renderer.domElement.clientHeight;
        var pw = (cw > ch) ? Math.floor((cw - ch) / 2.0) : 0;
        var ph = (ch > cw) ? Math.floor((ch - cw) / 2.0) : 0;
        var pvw = (vw > vh) ? Math.floor((vw - vh) / 2.0) : 0;
        var pvh = (vh > vw) ? Math.floor((vh - vw) / 2.0) : 0;
        // Processar a imagem diretamente no navegador
        var canvas = document.createElement("canvas");
        var client = document.createElement("canvas");
        canvas.width = 256;
        canvas.height = 256;
        client.width = window.innerWidth;
        client.height = window.innerHeight;
        var ctx = canvas.getContext("2d");
        var aux = client.getContext("2d");
        ctx.drawImage(AR.source.domElement, 0, 0, 256, 256);
        aux.drawImage(renderer.domElement, 0, 0, window.innerWidth, window.innerHeight, 0, 0, 256, 256);
        ctx.drawImage(client, 0, 0, 256, 256);
        var img = canvas.toDataURL("image/jpeg");
        ctx.clearRect(0, 0, 256, 256);
        ctx.drawImage(client, 0, 0, 256, 256);
        var data = ctx.getImageData(0, 0, 256, 256);
        for (var i = 0; i < 256 * 256 * 4; i += 4) {
            if (data.data[i] > 0 || data.data[i + 1] > 0 || data.data[i + 2] > 0) {
                data.data[i] = 255;
                data.data[i + 1] = 255;
                data.data[i + 2] = 255;
            }
            data.data[i + 3] = 255;
        }
        ctx.putImageData(data, 0, 0);
        var mask = canvas.toDataURL("image/jpeg");

        const result = await processImageWithPyodide(img, mask);
        if (!result) {
            throw new Error("Erro ao processar a imagem com OpenCV.");
        }

        // Processar os resultados conforme necessário
        var object_center = new THREE.Vector2(result.objectCenter[0], result.objectCenter[1]);
        var shadow_center = new THREE.Vector2(result.shadowCenter[0], result.shadowCenter[1]);
        var proportion = result.proportion;
        console.log("Centro de Massa do Objeto: (" + result.objectCenter[0] + ", " + result.objectCenter[1] + ")");
        console.log("Centro de Massa da Sombra: (" + result.shadowCenter[0] + ", " + result.shadowCenter[1] + ")");
        console.log("Proporção: " + proportion );

        // Reescala os pontos para a altura da cena
        var scaleFactor = window.innerHeight / 256;
        object_center.x *= scaleFactor;
        object_center.y *= scaleFactor;
        shadow_center.x *= scaleFactor;
        shadow_center.y *= scaleFactor;

        console.log("Scale Factor: " + scaleFactor);
        console.log(object_center);
        console.log(shadow_center);

        // Converte os centros de massa para coordenadas normalizadas
        object_center.x = (object_center.x / window.innerWidth) * 2 - 1;
        object_center.y = -(object_center.y / window.innerHeight) * 2 + 1;
        shadow_center.x = (shadow_center.x / window.innerWidth) * 2 - 1;
        shadow_center.y = -(shadow_center.y / window.innerHeight) * 2 + 1;

        console.log("Object_center: " + object_center);
        console.log("Shadow_center: " + shadow_center);

        // Encontra a posição na tela do centro geométrico do cubo
        var object_geometry_center = vObj.position.clone();

        console.log("Object_geometry_center: " + object_geometry_center.x + " " + object_geometry_center.y + " " + object_geometry_center.z);
        var screenPos = object_geometry_center.clone();
        screenPos.project(camera);
        screenPos.x = (screenPos.x * window.innerWidth / 2) + window.innerWidth / 2;
        screenPos.y = -(screenPos.y * window.innerHeight / 2) + window.innerHeight / 2;

        // Converte a posição em tela para coordenadas normalizadas
        var screenPos2D = new THREE.Vector2(
            (screenPos.x / window.innerWidth) * 2 - 1,
            -(screenPos.y / window.innerHeight) * 2 + 1
        );

        console.log("ScreenPos2D: " + screenPos2D.x + " " + screenPos2D.y);

        // Calcula a diferença entre a posição em tela do centro geométrico do cubo e o centro de massa do objeto
        var diff = new THREE.Vector2();
        diff.subVectors(screenPos2D, object_center);

        console.log("Diff: " + diff.x + " " + diff.y);

        // Aplica essa diferença ao centro de massa da sombra para obter sua posição em tela
        var shadow_screen_center = new THREE.Vector2();
        shadow_screen_center.addVectors(shadow_center, diff);

        // Ajuste a posição do ponto de raycasting considerando a proporção
        let scalee = 1;
        shadow_screen_center.x *= proportion * scalee;
        shadow_screen_center.y *= proportion * scalee;

        let shadow_center_position;
        // Faz raycasting para obter a posição 3D correspondente à posição da tela do centro de massa da sombra
        var raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(shadow_screen_center, camera);
        var intersects = raycaster.intersectObject(shadowPlane);
        if (intersects.length > 0) {
            var intersect = intersects[0];
            console.log("Interseção do centro de massa da sombra 3D:", intersect.point);

            // Adiciona uma esfera para visualizar o centro de massa da sombra
            addSphereAtPoint(intersect.point, 0x0000ff);  // Azul para sombra
            shadow_center_position = intersect.point;
            // Adicionar linha visualizando o vetor entre os dois pontos
            addLineBetweenPoints(object_geometry_center, intersect.point);

            // Calcular a direção da luz considerando a proporção
            var lightDirection = new THREE.Vector3();
            lightDirection.subVectors(intersect.point, object_geometry_center).normalize();
            var lightTarget = object_geometry_center.clone().add(lightDirection.multiplyScalar(2)); // multiplicar para colocar o target à frente da luz

            // Definir a posição da luz e seu target, ajustando a posição da luz para trás do objeto
            var lightPosition = object_geometry_center.clone().sub(lightDirection.multiplyScalar(1)); // ajustar a posição da luz para trás do objeto

            light.position.copy(lightPosition);
            light.target.position.copy(lightTarget);
            light.target.updateMatrixWorld();
            // Visualizar a luz
            var lightHelper = new THREE.DirectionalLightHelper(light, 6); // 5 pode ser ajustado conforme necessário
            scene.add(lightHelper);

            // Visualizar a direção da luz com uma flecha
            var arrowHelper = new THREE.ArrowHelper(lightDirection, lightPosition, 10, 0xff0000); // 10 pode ser ajustado conforme necessário, 0xff0000 é a cor vermelha
            scene.add(arrowHelper);

            vObj.castShadow = true;
        } else {
            console.log("Nenhuma interseção encontrada para a sombra.");
        }

        // Adiciona uma esfera para visualizar o centro de massa do objeto
        addSphereAtPoint(object_geometry_center, 0x00ff00);  // Verde para o objeto

        // Adiciona bounding box
        addBoundingBox(vObj);

        // Debug prints
        console.log("Centro do Cubo 3D:", object_geometry_center);
        console.log("Posição de Tela do Centro do Cubo 3D:", screenPos);
        console.log("Posição de Tela do Centro de Massa da Sombra:", shadow_screen_center);
        console.log("Posição de Tela do Centro de Massa do Objeto:", object_center);
        console.log("Direção da Luz:", light.position, light.target.position);

        // FIM TESTES
        switch (selectValue) {
            case '0':
                //setSource('webcam',null)
                break;
            case '1':
                //setSource('image','my-images/imagem_1.jpg')
                break;
            case '2':
                //setSource('image', 'my-images/frame2.jpg')
                break;
            case '3':
                setSource("video", "my-videos/video4.MOV")
            break;
        }

        submitBtn.style.display = "none";
        //select.style.display = "none";
        select2.style.display = "none";
        //select3.style.display = "none";
        select3.style.display = "block";
        returnBtn.style.display = "block";
        loaderElement.style.display = "none";
    } catch (error) {
        console.error(error);
        loaderElement.style.display = "none";
    }
});

function radianosParaGraus(radianos) {
    return radianos * (180 / Math.PI);
}

function onDocumentMouseClick(event) {
    event.preventDefault();

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    ray.setFromCamera(mouse, camera);

    var intersects = ray.intersectObjects([shadowPlane, vObj]);

    if (intersects.length > 0) {
        var intersect = intersects[0];
        console.log("Interseção no ponto de clique:", intersect.point);
        addSphereAtPoint(intersect.point, 0xff0000);  // Vermelho para o ponto de clique
    } else {
        console.log("Nenhuma interseção encontrada no ponto de clique.");
    }
}

function addSphereAtPoint(point, color) {
    var geometry = new THREE.SphereGeometry(0.1, 16, 16);
    var material = new THREE.MeshBasicMaterial({ color: color });
    var sphere = new THREE.Mesh(geometry, material);
    sphere.position.copy(point);
    scene.add(sphere);
}

function addLineBetweenPoints(point1, point2) {
    var material = new THREE.LineBasicMaterial({ color: 0xff0000 });
    var points = [];
    points.push(point1);
    points.push(point2);
    var geometry = new THREE.BufferGeometry().setFromPoints(points);
    var line = new THREE.Line(geometry, material);
    scene.add(line);
}

function addBoundingBox(object) {
    var box = new THREE.BoxHelper(object, 0xffff00);  // Amarelo para a bounding box
    scene.add(box);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Funções para carregar objetos GLTF
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
var loadedObjects = {};
var currentObject = vObj; // Inicializa com o cubo como objeto padrão
var currentFile = null; // Arquivo GLTF atualmente selecionado

function onError() { };

function onProgress(xhr, model) {
    if (xhr.lengthComputable) {
        var percentComplete = xhr.loaded / xhr.total * 100;
    }
}

export function getMaxSize(obj) {
    var maxSize;
    var box = new THREE.Box3().setFromObject(obj);
    var min = box.min;
    var max = box.max;

    var size = new THREE.Box3();
    size.x = max.x - min.x;
    size.y = max.y - min.y;
    size.z = max.z - min.z;

    if (size.x >= size.y && size.x >= size.z)
        maxSize = size.x;
    else {
        if (size.y >= size.z)
            maxSize = size.y;
        else {
            maxSize = size.z;
        }
    }
    return maxSize;
}

// Normalize scale and multiple by the newScale
function normalizeAndRescale(obj, newScale) {
    var scale = getMaxSize(obj);
    obj.scale.set(newScale * (1.0 / scale),
        newScale * (1.0 / scale),
        newScale * (1.0 / scale));
    return obj;
}

function fixPosition(obj) {
    // Fix position of the object over the ground plane
    var box = new THREE.Box3().setFromObject(obj);
    if (box.min.y > 0)
        obj.translateY(-box.min.y);
    else
        obj.translateY(-1 * box.min.y);
    return obj;
}

function loadGLTFFile(file, desiredScale, angle) {
    var loader = new THREE.GLTFLoader();
    loader.load(file, function (gltf) {
        var obj = gltf.scene;
        obj.castShadow = true;
        obj.traverse(function (child) {
            if (child) {
                child.castShadow = true;
            }
        });
        obj.traverse(function (node) {
            if (node.material) node.material.side = THREE.DoubleSide;
        });

        obj = normalizeAndRescale(obj, desiredScale);
        obj = fixPosition(obj);
        obj.rotateY(THREE.MathUtils.degToRad(angle));

        loadedObjects[file] = obj; // Armazena o objeto carregado
        if (file === currentFile) { // Verifica se o objeto é o atualmente selecionado
            currentObject = obj;
            scene.add(currentObject);
        }
    }, onProgress, onError);
}

document.getElementById('select3').addEventListener('change', function () {
    var selectedValue = this.value;
    var file;
    var desiredScale = 1;
    var angle = 0;

    // Remove o objeto atualmente exibido
    if (currentObject) {
        scene.remove(currentObject);
    }

    // Define o arquivo e parâmetros com base no valor selecionado
    switch (selectedValue) {
        case '0':
            currentObject = vObj;
            desiredScale = 1;
            file = 'cube'; // Valor especial para o cubo
            break;
        case '1':
            file = 'assets/objs/basket.glb';
            desiredScale = 1.5;
            break;
        case '2':
            file = 'assets/objs/cubwooden.glb';
            desiredScale = 2;
            break;
        case '3':
            file = 'assets/objs/dog.glb';
            desiredScale = 3;
            break;
        case '4':
            file = 'assets/objs/house.glb';
            desiredScale = 2;
            break;
        case '5':
            file = 'assets/objs/statueLaRenommee.glb';
            desiredScale = 2.5;
            break;
        case '6':
            file = 'assets/objs/woodenGoose.glb';
            desiredScale = 2.5;
            break;
        default:
            currentObject = vObj; // Valor padrão para o cubo
            desiredScale = 1;
            file = 'cube';
    }

    // Atualiza o arquivo atualmente selecionado
    currentFile = file;

    // Carrega e exibe o objeto selecionado
    if (file !== 'cube') {
        if (loadedObjects[file]) {
            currentObject = loadedObjects[file];
            scene.add(currentObject);
        } else {
            loadGLTFFile(file, desiredScale, angle);
        }
    } else {
        scene.add(currentObject);
    }
});

function render() {
    updateAR();
    requestAnimationFrame(render);
    renderer.render(scene, camera);

    if (camera.visible) {
        camera.getWorldPosition(position);
        camera.getWorldQuaternion(quaternion);
    }
}

render();
