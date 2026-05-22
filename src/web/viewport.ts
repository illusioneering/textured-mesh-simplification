import {
  AmbientLight,
  Box3,
  Color,
  DirectionalLight,
  GridHelper,
  Material,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  cloneMaterialForPbrPreview,
  defaultPbrMaterialPropertyState,
  type PbrMaterialPropertyState,
} from './pbrControls';

export type RenderingMode = 'wireframe' | 'geometry' | 'pbr';

export interface ViewportGridFrame {
  centerX: number;
  centerZ: number;
  y: number;
  size: number;
  divisions: number;
  spacing: number;
}

const DEFAULT_GRID_FRAME: ViewportGridFrame = {
  centerX: 0,
  centerZ: 0,
  y: -0.001,
  size: 10,
  divisions: 10,
  spacing: 1,
};

function gridSpacingForSize(size: number): number {
  if (!Number.isFinite(size) || size <= 0) return DEFAULT_GRID_FRAME.spacing;
  return 10 ** Math.round(Math.log10(size / 20));
}

function gridDivisionsForSize(size: number, spacing: number): number {
  if (!Number.isFinite(size) || size <= 0 || !Number.isFinite(spacing) || spacing <= 0) {
    return DEFAULT_GRID_FRAME.divisions;
  }
  return Math.max(2, Math.ceil(size / spacing));
}

export function gridFrameForObjectBounds(object: Object3D): ViewportGridFrame {
  const box = new Box3().setFromObject(object);
  if (box.isEmpty()) return { ...DEFAULT_GRID_FRAME };

  const center = box.getCenter(new Vector3());
  const size = box.getSize(new Vector3());
  const xzSize = Math.max(size.x, size.z);
  const rawGridSize = xzSize > 0 ? xzSize * 2 : DEFAULT_GRID_FRAME.size;
  const spacing = gridSpacingForSize(rawGridSize);
  const divisions = gridDivisionsForSize(rawGridSize, spacing);
  const gridSize = divisions * spacing;

  return {
    centerX: center.x,
    centerZ: center.z,
    y: box.min.y,
    size: gridSize,
    divisions,
    spacing,
  };
}

export interface ViewportCameraPose {
  position: { x: number; y: number; z: number };
  target: { x: number; y: number; z: number };
  near: number;
  far: number;
  zoom: number;
}

interface ViewportVector3Values {
  x: number;
  y: number;
  z: number;
}

interface MutableViewportVector3Values extends ViewportVector3Values {
  set(x: number, y: number, z: number): unknown;
}

interface MutableViewportCameraValues {
  position: MutableViewportVector3Values;
  near: number;
  far: number;
  zoom: number;
  updateProjectionMatrix(): unknown;
}

interface MutableViewportControlsValues {
  target: MutableViewportVector3Values;
  update(): unknown;
}

export function copyViewportCameraPose(pose: ViewportCameraPose): ViewportCameraPose {
  return {
    position: { ...pose.position },
    target: { ...pose.target },
    near: pose.near,
    far: pose.far,
    zoom: pose.zoom,
  };
}

export function viewportCameraPoseFromValues(
  position: ViewportVector3Values,
  target: ViewportVector3Values,
  near: number,
  far: number,
  zoom: number,
): ViewportCameraPose {
  return copyViewportCameraPose({ position, target, near, far, zoom });
}

export function applyViewportCameraPoseValues(
  pose: ViewportCameraPose,
  camera: MutableViewportCameraValues,
  controls: MutableViewportControlsValues,
): void {
  const copied = copyViewportCameraPose(pose);
  camera.position.set(copied.position.x, copied.position.y, copied.position.z);
  controls.target.set(copied.target.x, copied.target.y, copied.target.z);
  camera.near = copied.near;
  camera.far = copied.far;
  camera.zoom = copied.zoom;
  camera.updateProjectionMatrix();
  controls.update();
}

export function gridFrameForModel(model: Object3D | null, updateGrid: boolean): ViewportGridFrame | null {
  if (!updateGrid) return null;
  if (!model) return { ...DEFAULT_GRID_FRAME };
  return gridFrameForObjectBounds(model);
}

export interface SetModelOptions {
  fitCamera?: boolean;
  updateGrid?: boolean;
}

export interface ModelViewport {
  setModel(model: Object3D | null, options?: SetModelOptions): void;
  setGridFromObject(object: Object3D): void;
  getCameraPose(): ViewportCameraPose;
  setCameraPose(pose: ViewportCameraPose): void;
  setRenderingMode(mode: RenderingMode): void;
  setPbrPropertyState(state: PbrMaterialPropertyState): void;
  resize(): void;
  dispose(): void;
}

function fitCameraToObject(camera: PerspectiveCamera, controls: OrbitControls, object: Object3D): void {
  const box = new Box3().setFromObject(object);
  if (box.isEmpty()) {
    camera.position.set(2, 2, 2);
    controls.target.set(0, 0, 0);
    controls.update();
    return;
  }

  const center = box.getCenter(new Vector3());
  const size = box.getSize(new Vector3());

  const vFov = camera.fov * Math.PI / 180;
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);

  // Fit width and height, not max xyz
  const fitHeightDistance = size.y / (2 * Math.tan(vFov / 2));
  const fitWidthDistance = size.x / (2 * Math.tan(hFov / 2));

  const distance = Math.max(fitHeightDistance, fitWidthDistance, 1e-3);

  const direction = new Vector3(0.85, 0.65, 1).normalize();

  // Try 1.1–1.25 instead of 1.8
  camera.position.copy(center).addScaledVector(direction, distance * 2);

  camera.near = Math.max(distance / 100, 0.001);
  camera.far = Math.max(distance * 100, 1000);
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}

function isMesh(object: Object3D): object is Mesh {
  return (object as Mesh).isMesh === true;
}

export function createModelViewport(container: HTMLElement): ModelViewport {
  const scene = new Scene();
  scene.background = new Color(0x020617);

  const camera = new PerspectiveCamera(45, 1, 0.01, 1000);
  camera.position.set(2, 2, 2);

  const renderer = new WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = SRGBColorSpace;
  container.replaceChildren(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  scene.add(new AmbientLight(0xffffff, 0.55));
  const keyLight = new DirectionalLight(0xffffff, 1.75);
  keyLight.position.set(4, 5, 6);
  scene.add(keyLight);
  const fillLight = new DirectionalLight(0x93c5fd, 0.65);
  fillLight.position.set(-3, 2, -4);
  scene.add(fillLight);
  let grid = new GridHelper(DEFAULT_GRID_FRAME.size, DEFAULT_GRID_FRAME.divisions, 0x334155, 0x1e293b);
  grid.position.set(DEFAULT_GRID_FRAME.centerX, DEFAULT_GRID_FRAME.y, DEFAULT_GRID_FRAME.centerZ);
  scene.add(grid);

  const originalMaterials = new WeakMap<Mesh, Material | Material[]>();
  const pbrMaterialCache = new Map<Material, Material>();
  const geometryMaterial = new MeshStandardMaterial({ color: 0xcbd5e1, roughness: 0.78, metalness: 0 });
  const wireframeMaterial = new MeshBasicMaterial({ color: 0x93c5fd, wireframe: true });

  let currentModel: Object3D | null = null;
  let renderingMode: RenderingMode = 'pbr';
  let pbrPropertyState = defaultPbrMaterialPropertyState(true);
  let animationFrame = 0;

  const applyGridFrame = (frame: ViewportGridFrame): void => {
    scene.remove(grid);
    grid.dispose();
    grid = new GridHelper(frame.size, frame.divisions, 0x334155, 0x1e293b);
    grid.position.set(frame.centerX, frame.y, frame.centerZ);
    scene.add(grid);
  };

  const cameraPose = (): ViewportCameraPose =>
    viewportCameraPoseFromValues(camera.position, controls.target, camera.near, camera.far, camera.zoom);

  const applyCameraPose = (pose: ViewportCameraPose): void => {
    applyViewportCameraPoseValues(pose, camera, controls);
  };

  const resize = (): void => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  const disposeCachedPbrMaterials = (): void => {
    for (const material of pbrMaterialCache.values()) material.dispose();
    pbrMaterialCache.clear();
  };

  const pbrMaterialFor = (material: Material): Material => {
    const cached = pbrMaterialCache.get(material);
    if (cached) return cached;
    const preview = cloneMaterialForPbrPreview(material, pbrPropertyState);
    pbrMaterialCache.set(material, preview);
    return preview;
  };

  const pbrMaterialArrayFor = (materials: Material[]): Material[] => {
    return materials.map((material) => pbrMaterialFor(material));
  };

  const applyRenderingMode = (): void => {
    if (!currentModel) return;
    currentModel.traverse((object) => {
      if (!isMesh(object)) return;
      if (!originalMaterials.has(object)) originalMaterials.set(object, object.material);
      if (renderingMode === 'pbr') {
        const original = originalMaterials.get(object);
        if (Array.isArray(original)) {
          object.material = pbrMaterialArrayFor(original);
        } else if (original) {
          object.material = pbrMaterialFor(original);
        }
      } else if (renderingMode === 'geometry') {
        object.material = geometryMaterial;
      } else {
        object.material = wireframeMaterial;
      }
    });
  };

  const render = (): void => {
    controls.update();
    renderer.render(scene, camera);
    animationFrame = requestAnimationFrame(render);
  };

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();
  render();

  return {
    setModel(model: Object3D | null, options: SetModelOptions = {}): void {
      const { fitCamera = true, updateGrid = true } = options;
      if (currentModel) {
        scene.remove(currentModel);
        currentModel = null;
      }
      disposeCachedPbrMaterials();
      if (!model) {
        const gridFrame = gridFrameForModel(null, updateGrid);
        if (gridFrame) applyGridFrame(gridFrame);
        return;
      }
      currentModel = model;
      currentModel.traverse((object) => {
        if (isMesh(object)) originalMaterials.set(object, object.material);
      });
      scene.add(currentModel);
      applyRenderingMode();
      const gridFrame = gridFrameForModel(currentModel, updateGrid);
      if (gridFrame) applyGridFrame(gridFrame);
      if (fitCamera) fitCameraToObject(camera, controls, currentModel);
    },
    setGridFromObject(object: Object3D): void {
      applyGridFrame(gridFrameForObjectBounds(object));
    },
    getCameraPose(): ViewportCameraPose {
      return cameraPose();
    },
    setCameraPose(pose: ViewportCameraPose): void {
      applyCameraPose(pose);
    },
    setRenderingMode(mode: RenderingMode): void {
      renderingMode = mode;
      applyRenderingMode();
    },
    setPbrPropertyState(state: PbrMaterialPropertyState): void {
      pbrPropertyState = { ...state };
      disposeCachedPbrMaterials();
      applyRenderingMode();
    },
    resize,
    dispose(): void {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      if (currentModel) scene.remove(currentModel);
      disposeCachedPbrMaterials();
      geometryMaterial.dispose();
      wireframeMaterial.dispose();
      grid.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
