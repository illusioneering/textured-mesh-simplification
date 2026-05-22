import { Matrix4 } from 'three';
import type { Document, Mesh, Node as GltfNode, Scene } from '@gltf-transform/core';

export interface PreservedNodeTransform {
  name: string;
  matrix: Mat4Array;
}

type Mat4Array = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

export interface SceneTransformContext {
  sceneName: string;
  nodes: PreservedNodeTransform[];
  attachmentWorldMatrix: Matrix4;
}

function cloneMatrixArray(matrix: readonly number[]): Mat4Array {
  if (matrix.length !== 16) throw new Error(`Expected 16 matrix elements, got ${matrix.length}.`);
  return Array.from(matrix, (value) => value) as Mat4Array;
}

export function identitySceneTransformContext(sceneName = 'Scene'): SceneTransformContext {
  return {
    sceneName,
    nodes: [],
    attachmentWorldMatrix: new Matrix4(),
  };
}

function collectMeshNodePaths(scene: Scene): GltfNode[][] {
  const paths: GltfNode[][] = [];
  const visit = (node: GltfNode, path: GltfNode[]): void => {
    const nextPath = [...path, node];
    if (node.getMesh()) paths.push(nextPath);
    for (const child of node.listChildren()) visit(child, nextPath);
  };
  for (const child of scene.listChildren()) visit(child, []);
  return paths;
}

export function sceneTransformContextFromScene(scene: Scene): SceneTransformContext {
  const meshPaths = collectMeshNodePaths(scene);
  if (meshPaths.length !== 1) return identitySceneTransformContext(scene.getName() || 'Scene');

  const path = meshPaths[0]!;
  const attachmentNode = path[path.length - 1]!;
  return {
    sceneName: scene.getName() || 'Scene',
    nodes: path.map((node, index) => ({
      name: node.getName() || (index === path.length - 1 ? 'simplified-geometry' : `node-${index}`),
      matrix: cloneMatrixArray(node.getMatrix()),
    })),
    attachmentWorldMatrix: new Matrix4().fromArray(attachmentNode.getWorldMatrix()),
  };
}

export function sceneTransformContextFromScenes(scenes: Scene[]): SceneTransformContext {
  if (scenes.length !== 1) return identitySceneTransformContext();
  return sceneTransformContextFromScene(scenes[0]!);
}

export function attachmentLocalMatrix(context?: SceneTransformContext): Matrix4 {
  return (context?.attachmentWorldMatrix ?? new Matrix4()).clone().invert();
}

export function createOutputSceneWithTransform(options: {
  document: Document;
  mesh: Mesh;
  context?: SceneTransformContext | undefined;
  sceneName?: string | undefined;
  meshNodeName: string;
}): void {
  const context = options.context ?? identitySceneTransformContext(options.sceneName ?? 'Scene');
  const scene = options.document.createScene(context.sceneName || options.sceneName || 'Scene');
  let parent: GltfNode | null = null;

  if (context.nodes.length === 0) {
    const node = options.document.createNode(options.meshNodeName).setMesh(options.mesh);
    scene.addChild(node);
    options.document.getRoot().setDefaultScene(scene);
    return;
  }

  context.nodes.forEach((preserved, index) => {
    const node = options.document.createNode(index === context.nodes.length - 1 ? preserved.name || options.meshNodeName : preserved.name)
      .setMatrix(cloneMatrixArray(preserved.matrix));
    if (index === context.nodes.length - 1) node.setMesh(options.mesh);
    if (parent) parent.addChild(node);
    else scene.addChild(node);
    parent = node;
  });

  options.document.getRoot().setDefaultScene(scene);
}
