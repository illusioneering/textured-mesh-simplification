import { Matrix4, Mesh, Object3D, Scene } from 'three';

export interface BrowserPreservedNodeTransform {
  name: string;
  matrix: number[];
}

export interface BrowserSceneTransformContext {
  sceneName: string;
  nodes: BrowserPreservedNodeTransform[];
  attachmentWorldMatrix: number[];
  supported: boolean;
  warning?: string;
}

function matrixArray(matrix: Matrix4): number[] {
  return matrix.toArray().map((value) => value);
}

export function identitySceneTransformContext(sceneName = 'Scene', warning?: string): BrowserSceneTransformContext {
  return {
    sceneName,
    nodes: [],
    attachmentWorldMatrix: new Matrix4().toArray(),
    supported: false,
    ...(warning ? { warning } : {}),
  };
}

function isMeshObject(object: Object3D): object is Mesh {
  return (object as Mesh).isMesh === true && Boolean((object as Mesh).geometry);
}

function collectMeshNodePaths(root: Object3D): Object3D[][] {
  const paths: Object3D[][] = [];
  const visit = (object: Object3D, path: Object3D[]): void => {
    const nextPath = [...path, object];
    if (isMeshObject(object)) paths.push(nextPath);
    for (const child of object.children) visit(child, nextPath);
  };
  for (const child of root.children) visit(child, []);
  return paths;
}

export function sceneTransformContextFromObject(root: Object3D): BrowserSceneTransformContext {
  root.updateMatrixWorld(true);
  const sceneName = root.name || 'Scene';
  const meshPaths = collectMeshNodePaths(root);
  if (meshPaths.length === 0) return identitySceneTransformContext(sceneName, 'No mesh nodes found; output will use identity transform.');
  if (meshPaths.length > 1) {
    return identitySceneTransformContext(sceneName);
  }

  const path = meshPaths[0]!;
  const attachment = path[path.length - 1]!;
  return {
    sceneName,
    nodes: path.map((node, index) => ({
      name: node.name || (index === path.length - 1 ? 'simplified-geometry' : `node-${index}`),
      matrix: matrixArray(node.matrix),
    })),
    attachmentWorldMatrix: matrixArray(attachment.matrixWorld),
    supported: true,
  };
}

export function attachmentLocalMatrix(context?: BrowserSceneTransformContext): Matrix4 {
  if (!context?.supported) return new Matrix4();
  return new Matrix4().fromArray(context.attachmentWorldMatrix).invert();
}

export function createSceneWithPreservedTransform(
  mesh: Mesh,
  context: BrowserSceneTransformContext | undefined,
  meshNodeName: string,
): Scene {
  const scene = new Scene();
  scene.name = context?.sceneName || 'Scene';
  if (!context?.supported || context.nodes.length === 0) {
    mesh.name = meshNodeName;
    scene.add(mesh);
    return scene;
  }

  let parent: Object3D | null = null;
  context.nodes.forEach((preserved, index) => {
    const isLast = index === context.nodes.length - 1;
    const node = isLast ? mesh : new Object3D();
    node.name = preserved.name || (isLast ? meshNodeName : `node-${index}`);
    node.matrixAutoUpdate = false;
    node.matrix.fromArray(preserved.matrix);
    if (parent) parent.add(node);
    else scene.add(node);
    parent = node;
  });
  scene.updateMatrixWorld(true);
  return scene;
}
