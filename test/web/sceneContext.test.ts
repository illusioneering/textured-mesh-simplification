import { describe, expect, it } from 'vitest';
import { BoxGeometry, Group, Matrix4, Mesh, MeshBasicMaterial, Object3D } from 'three';
import {
  attachmentLocalMatrix,
  createSceneWithPreservedTransform,
  identitySceneTransformContext,
  sceneTransformContextFromObject,
} from '../../src/web/sceneContext';

function expectMatrixClose(actual: Matrix4, expected: Matrix4): void {
  const a = actual.toArray();
  const e = expected.toArray();
  for (let i = 0; i < e.length; i += 1) expect(a[i]).toBeCloseTo(e[i]!, 5);
}

describe('browser scene transform context', () => {
  it('captures a single mesh-node path and attachment world matrix', () => {
    const root = new Group();
    root.name = 'Scene';
    const parent = new Object3D();
    parent.name = 'parent';
    parent.matrixAutoUpdate = false;
    parent.matrix.makeTranslation(1, 2, 3);
    const mesh = new Mesh(new BoxGeometry(), new MeshBasicMaterial());
    mesh.name = 'mesh-node';
    mesh.matrixAutoUpdate = false;
    mesh.matrix.makeRotationZ(Math.PI / 2);
    parent.add(mesh);
    root.add(parent);
    root.updateMatrixWorld(true);

    const context = sceneTransformContextFromObject(root);
    expect(context.supported).toBe(true);
    expect(context.nodes.map((node) => node.name)).toEqual(['parent', 'mesh-node']);
    expectMatrixClose(new Matrix4().fromArray(context.attachmentWorldMatrix), mesh.matrixWorld);
    expectMatrixClose(attachmentLocalMatrix(context), mesh.matrixWorld.clone().invert());
  });

  it('uses identity attachment context without warning for multiple mesh nodes', () => {
    const root = new Group();
    root.add(new Mesh(new BoxGeometry(), new MeshBasicMaterial()));
    root.add(new Mesh(new BoxGeometry(), new MeshBasicMaterial()));
    const context = sceneTransformContextFromObject(root);
    expect(context.supported).toBe(false);
    expect(context.nodes).toEqual([]);
    expect(context.warning).toBeUndefined();
    expectMatrixClose(attachmentLocalMatrix(context), new Matrix4());
  });

  it('recreates the preserved node path for output scenes', () => {
    const context = identitySceneTransformContext('Scene');
    context.supported = true;
    context.nodes = [
      { name: 'root-node', matrix: new Matrix4().makeTranslation(1, 0, 0).toArray() },
      { name: 'mesh-node', matrix: new Matrix4().makeScale(2, 2, 2).toArray() },
    ];
    const mesh = new Mesh(new BoxGeometry(), new MeshBasicMaterial());
    const scene = createSceneWithPreservedTransform(mesh, context, 'fallback-name');

    expect(scene.name).toBe('Scene');
    expect(scene.children[0]!.name).toBe('root-node');
    expect(scene.children[0]!.children[0]!.name).toBe('mesh-node');
    expect((scene.children[0]!.children[0] as Mesh).isMesh).toBe(true);
  });
});
