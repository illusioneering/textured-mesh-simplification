import { describe, expect, it } from 'vitest';
import { BoxGeometry, Mesh, MeshBasicMaterial, Object3D } from 'three';
import { cloneSceneForViewport } from '../../src/web/previewScene';

function firstMesh(object: Object3D): Mesh {
  let mesh: Mesh | null = null;
  object.traverse((child) => {
    if ((child as Mesh).isMesh) mesh = child as Mesh;
  });
  if (!mesh) throw new Error('Expected mesh in scene.');
  return mesh;
}

describe('viewport preview scene cloning', () => {
  it('allows preview material overrides without mutating the export scene', () => {
    const sourceMaterial = new MeshBasicMaterial({ color: 0xff0000 });
    const overrideMaterial = new MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
    const source = new Object3D();
    source.add(new Mesh(new BoxGeometry(), sourceMaterial));

    const preview = cloneSceneForViewport(source);
    firstMesh(preview).material = overrideMaterial;

    expect(firstMesh(source).material).toBe(sourceMaterial);
    expect(firstMesh(preview).material).toBe(overrideMaterial);
  });
});
