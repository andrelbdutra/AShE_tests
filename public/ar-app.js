var  renderer, rend;
var emptyObj, vObj, vObjMask, light, origLight, shadowPlane, wPlane, dPlane;
var sphere
var adjustX, adjustZ;
let position = new THREE.Vector3(); // Para armazenar posição da câmera virtual
let quaternion = new THREE.Quaternion(); // Para armazenar a rotação da câmera virtual
var mouse = new THREE.Vector2();
var emptyPlane
var ray    = new THREE.Raycaster();
var point  = new THREE.Vector2();
var loader = new THREE.TextureLoader();

var planeSize      = 150.00;
var sPlaneSize     =  150.00;
var sPlaneSegments = 300.00;
var vObjHeight     =   1.20;
var vObjRatio      =   1.00;
var adjustX        =   0.00;
var adjustZ        =   0.00;
var done           =  false;

const loaderElement = document.createElement("div");
loaderElement.setAttribute("class", "loader");	
loaderElement.setAttribute("id", "loader");
document.body.appendChild(loaderElement);
const select = document.getElementById("select");
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
//renderer.setSize(1280, 720);
renderer.setSize(window.innerWidth, window.innerHeight); // Change here to render in low resolution (for example 640 x 480)
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
	var cube   = new THREE.BoxBufferGeometry(vObjHeight, vObjHeight * vObjRatio, vObjHeight);
	var wood = new THREE.MeshLambertMaterial({map: loader.load("my-textures/face/wood.png")});
    var transparentMaterial = new THREE.MeshBasicMaterial({
        map: loader.load("my-textures/face/wood.png"),
        //color: 0x00ff00,
        transparent: true,
        //opacity: 0.5,
        opacity: 1,
    });
	return vObj        = new THREE.Mesh(cube,   transparentMaterial);

}

function create_scene()
{
	let scene = new THREE.Scene();
	//var ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
	origLight = new THREE.DirectionalLight(0xffffff);
	origLight.castShadow = true;
	var d = vObjRatio * vObjHeight * 10;
	origLight.shadow.camera.left   = -d;
	origLight.shadow.camera.right  =  d;
	origLight.shadow.camera.top    =  d;
	origLight.shadow.camera.bottom = -d;

	origLight.shadow.mapSize.width  = 4096;
	origLight.shadow.mapSize.height = 4096;

	light = origLight.clone();
	scene.add(light);
	//light.position.set(0, 10, 0);
	//light.intensity = 3.0;

	//let camera = new THREE.Camera();
	camera = new THREE.PerspectiveCamera(30,  window.innerWidth / window.innerHeight, 0.1, 1000);

	scene.add( camera );
	vObj = createVirtualObj();
	scene.add( vObj );
	vObj.position.set     (adjustX, vObjRatio * vObjHeight / 2, adjustZ);
	vObj.castShadow = false;

	camera = camera;
	scene = scene;  

	return [scene, camera];
}

var aux = create_scene();
scene 	= aux[0];
camera = aux[1];
var helper = new THREE.CameraHelper(camera);
//scene.add(helper);

//-- ALL AR STUFF HERE ---------------------------------------------------------
export function updateAR()
{
   if(AR.source)
   {
      if( AR.source.ready === false )	return
      AR.context.update( AR.source.domElement )
      scene.visible = camera.visible  
      return camera;
   }
   return null;
}

function setSource(type, url)
{
   AR.source = new THREEx.ArToolkitSource({	
      sourceType : type,
      sourceUrl : url,
   })
}

export function setARStuff(source)
{
   switch (source)
   {
      case 'image':
         setSource('image', "my-images/new_imagem_4.jpg");
         //setSource('image', "my-images/img_extobj_5.jpeg");
         break;
      case 'video':
         setSource('video', "my-videos/video4.MOV");
         break;
      case 'webcam':
         setSource('webcam', null);
         break;
   }   
   
   AR.source.init(function onReady(){
		onResize()
	});
   //----------------------------------------------------------------------------
   // initialize arToolkitContext
   AR.context = new THREEx.ArToolkitContext({
      cameraParametersUrl:'data/camera_para.dat',
      detectionMode: 'mono',
   })
   
   // initialize it
   AR.context.init(function onCompleted(){
      camera.projectionMatrix.copy( AR.context.getProjectionMatrix() );
   })
   
   //----------------------------------------------------------------------------
   // Create a ArMarkerControls
   new THREEx.ArMarkerControls(AR.context, camera, {	
		type: "pattern", 
		patternUrl: "data/kanji.patt",
      	changeMatrixMode: 'cameraTransformMatrix' 
   })
   // as we do changeMatrixMode: 'cameraTransformMatrix', start with invisible scene
   scene.visible = true   
}

// Possible sources: 'image', 'video', 'webcam' 
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

document.getElementById("select2").addEventListener("click", async () => {
	const select = document.getElementById("select2");
	let value = select.value;
	if(selectValue != value) {
		selectValue = value;
		switch (value)
        {
        	case '0':
              	setSource('webcam',null)
              	break;
        	case '1':
              	setSource('image','my-images/img_extobj_1.jpeg')         
              	break;
        	case '2':
              	setSource('image', 'my-images/img_extobj_2.jpeg')                     
              	break;
			case '3':
				setSource('image','my-images/img_extobj_3.jpeg')         
				break;
        }
	}
})

returnBtn.addEventListener('click', async () => {
	let value = select2.value;
		switch (value)
        {
        	case '0':
              	setSource('webcam',null)
              	break;
			case '1':
				setSource('image','my-images/img_extobj_1.jpeg')         
				break;
			case '2':
				setSource('image', 'my-images/img_extobj_2.jpeg')                     
				break;
			case '3':
				setSource('image','my-images/img_extobj_3.jpeg')         
				break;
		  }
	select.style.display = "block";
	select2.style.display = "block";
	submitBtn.style.display = "block";
	select3.style.display = "none";
	returnBtn.style.display = "none";
	light.position.set(0, 10, 0);
});

// Funções utilitárias do OpenCV.js
function applyGrabCut(img) {
    let mask = new cv.Mat();
    let bgdModel = new cv.Mat();
    let fgdModel = new cv.Mat();
    let rect = new cv.Rect(10, 10, img.cols - 20, img.rows - 20);
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT);
    let mask2 = new cv.Mat();
    cv.compare(mask, new cv.Mat(mask.rows, mask.cols, mask.type(), new cv.Scalar(cv.GC_FGD)), mask2, cv.CMP_EQ);
    let fg = new cv.Mat(img.size(), img.type());
    img.copyTo(fg, mask2);
    return [fg, mask2];
}

function kmeansSegmentation(image, nb_classes, use_color) {
    let samples = image.reshape(1, image.cols * image.rows);
    samples.convertTo(samples, cv.CV_32F);
    let kmeans = new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 10, 1.0);
    let labels = new cv.Mat();
    let centers = new cv.Mat();
    cv.kmeans(samples, nb_classes, labels, kmeans, 10, cv.KMEANS_RANDOM_CENTERS, centers);
    let segmented = labels.reshape(1, image.rows);
    segmented.convertTo(segmented, cv.CV_8U);
    return segmented;
}

function normalizeSegments(segmented) {
    let uniqueValues = [...new Set(segmented.data)];
    let step = Math.floor(255 / uniqueValues.length);
    let normalizedImage = new cv.Mat(segmented.rows, segmented.cols, segmented.type(), new cv.Scalar(0));
    for (let i = 0; i < uniqueValues.length; i++) {
        let mask = new cv.Mat();
        cv.compare(segmented, uniqueValues[i], mask, cv.CMP_EQ);
        normalizedImage.setTo(i * step, mask);
        mask.delete();
    }
    return normalizedImage;
}

function analyzeSegmentsForShadows(image, labels, nb_classes) {
    let labImage = new cv.Mat();
    cv.cvtColor(image, labImage, cv.COLOR_BGR2Lab);
    let shadowMask = new cv.Mat(labels.rows, labels.cols, cv.CV_8U, new cv.Scalar(0));
    for (let i = 0; i < nb_classes; i++) {
        let mask = new cv.Mat();
        cv.compare(labels, i, mask, cv.CMP_EQ);
        let mean = cv.mean(labImage, mask);
        if (mean[0] < 50) {
            shadowMask.setTo(255, mask);
        }
        mask.delete();
    }
    labImage.delete();
    return shadowMask;
}

function combineMasks(segmentationMask, contourMask, color) {
    let combinedMask = contourMask.clone();
    let mask = new cv.Mat();
    cv.compare(segmentationMask, 255, mask, cv.CMP_GE);
    combinedMask.setTo(color, mask);
    mask.delete();
    return combinedMask;
}

function extractObjectMasks(image) {
    let binaryImage = new cv.Mat();
    cv.threshold(image, binaryImage, 127, 255, cv.THRESH_BINARY);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(binaryImage, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    let masks = [];
    for (let i = 0; i < contours.size(); ++i) {
        let mask = cv.Mat.zeros(binaryImage.rows, binaryImage.cols, cv.CV_8U);
        cv.drawContours(mask, contours, i, new cv.Scalar(255), cv.FILLED, 8, hierarchy, 0);
        masks.push(mask);
    }
    binaryImage.delete();
    contours.delete();
    hierarchy.delete();
    return masks;
}

function findLargestObject(masks) {
    let maxArea = 0;
    let largestMask = null;
    for (let mask of masks) {
        let currentArea = cv.countNonZero(mask);
        if (currentArea > maxArea) {
            maxArea = currentArea;
            largestMask = mask;
        }
    }
    return largestMask;
}

function calculateCenterOfMass(image) {
    let moments = cv.moments(image, true);
    let cx = moments.m10 / moments.m00;
    let cy = moments.m01 / moments.m00;
    return [cx, cy];
}

function combineObjectAndShadowMask(objectMask, shadowMask) {
    let combinedMask = objectMask.clone();
    let shadowMask8U = new cv.Mat();
    shadowMask.convertTo(shadowMask8U, cv.CV_8U);
    combinedMask.setTo(128, shadowMask8U);
    shadowMask8U.delete();
    return combinedMask;
}

function calculateShadowAngle(objectCenter, shadowCenter) {
    let dx = shadowCenter[0] - objectCenter[0];
    let dy = shadowCenter[1] - objectCenter[1];
    return Math.atan2(dy, dx) * 180 / Math.PI;
}

function calculateProportion(largestObjectMask) {
    let rect = cv.boundingRect(largestObjectMask);
    return rect.width / rect.height;
}

async function processImageWithOpenCV(imageSrc, maskSrc) {
    // Carregar a imagem original
    let imgElement = document.createElement('img');
    imgElement.src = imageSrc;
    await imgElement.decode();

    // Carregar a máscara
    let maskElement = document.createElement('img');
    maskElement.src = maskSrc;
    await maskElement.decode();

    // Criar um canvas para desenhar as imagens
    let canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');

    // Definir o tamanho do canvas
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;

    // Desenhar a imagem no canvas
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);
    let img = cv.imread(canvas);

    // Desenhar a máscara no canvas
    ctx.drawImage(maskElement, 0, 0, maskElement.width, maskElement.height);
    let mask = cv.imread(canvas);

    // Continuar com o processamento
    let [grabCutImage, grabCutMask] = applyGrabCut(img);
    let nb_classes = 10;
    let segmented = kmeansSegmentation(img, nb_classes, true);
    let normalizedSegmented = normalizeSegments(segmented);
    let segmentationMask = analyzeSegmentsForShadows(img, segmented, nb_classes);
    let combinedMask = combineMasks(segmentationMask, grabCutMask, 128);
    let objectsWithShadowsImage = combineMasks(mask, combinedMask, 0);
    let objectsWithoutShadow = combineMasks(segmentationMask, objectsWithShadowsImage, 0);
    objectsWithoutShadow = combineMasks(mask, objectsWithoutShadow, 0);
    let grabCutMasks = extractObjectMasks(objectsWithoutShadow);
    let largestObjectMask = findLargestObject(grabCutMasks);
    let objectCenter = calculateCenterOfMass(largestObjectMask);
    let combinedMask2 = combineObjectAndShadowMask(largestObjectMask, segmentationMask);
    let shadowCenter = calculateCenterOfMass(combinedMask2);
    let shadowAngle = calculateShadowAngle(objectCenter, shadowCenter);
    let proportion = calculateProportion(largestObjectMask);

    // Deletar as imagens para liberar memória
    img.delete();
    mask.delete();
    grabCutImage.delete();
    grabCutMask.delete();
    segmented.delete();
    normalizedSegmented.delete();
    segmentationMask.delete();
    combinedMask.delete();
    objectsWithShadowsImage.delete();
    objectsWithoutShadow.delete();
    largestObjectMask.delete();
    combinedMask2.delete();

    // Retornar os valores
    return {
        objectCenter,
        shadowCenter,
        proportion
    };
}

document.getElementById("submitButtonInput").addEventListener("click", async () => {
    let value = select.value;
    loaderElement.style.display = "block";

    light.position.set(0, 10, 0);

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
    aux.drawImage(renderer.domElement, 0, 0, client.width, client.height, 0, 0, 256, 256);
    ctx.drawImage(client, 0, 0, client.width, client.height, 0, 0, 256, 256);
    var img = canvas.toDataURL("image/jpeg");
    ctx.clearRect(0, 0, 256, 256);
    ctx.drawImage(client, 0, 0, client.width, client.height, 0, 0, 256, 256);
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

    const result = await processImageWithOpenCV(img, mask);

    // Processar os resultados conforme necessário
    var object_center = new THREE.Vector2(result.objectCenter[0], result.objectCenter[1]);
    var shadow_center = new THREE.Vector2(result.shadowCenter[0], result.shadowCenter[1]);
    var proportion = result.proportion;

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
    select.style.display = "none";
    select2.style.display = "none";
    select3.style.display = "none";

    returnBtn.style.display = "block";
    loaderElement.style.display = "none";
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

function onProgress ( xhr, model ) {
    if ( xhr.lengthComputable ) {
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
    loader.load(file, function(gltf) {
        var obj = gltf.scene;
        obj.castShadow = true;
        obj.traverse(function(child) {
            if (child) {
                child.castShadow = true;
            }
        });
        obj.traverse(function(node) {
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


document.getElementById('select3').addEventListener('change', function() {
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


function render(){
	updateAR(); 
	requestAnimationFrame(render);
	//renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
	renderer.render(scene, camera);
	
	//virtualCamera.visible = false;
	if(camera.visible){
		// Copia a posição da câmera real
		camera.getWorldPosition(position);
		camera.getWorldQuaternion(quaternion);
		//console.log("Camera position: " + position.x + " " + position.y + " " + position.z);
	}
}
		
render();