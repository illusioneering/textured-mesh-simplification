import type { Object3D } from 'three';

export function cloneSceneForViewport(scene: Object3D): Object3D {
  return scene.clone(true);
}
