import { Vector3 } from 'three';
import type { RawMesh } from './types';

const EPSILON = 1e-20;

/**
 * Computes area-weighted smooth vertex normals for a raw indexed triangle mesh.
 *
 * The returned array is keyed by the original RawMesh vertex id, not by any
 * serialized GLB vertex id. Preserve-mode writers can copy these normals to UV
 * seam duplicates to avoid introducing lighting seams; baked-atlas writers
 * should recompute normals after atlas vertex duplication so final shading
 * follows the serialized simplified geometry.
 */
export function computeAreaWeightedVertexNormals(mesh: RawMesh): Vector3[] {
  const normals = mesh.positions.map(() => new Vector3());

  for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex += 1) {
    const face = mesh.faces[faceIndex]!;
    const a = mesh.positions[face[0]];
    const b = mesh.positions[face[1]];
    const c = mesh.positions[face[2]];
    if (!a || !b || !c) throw new Error(`Face ${faceIndex} references a missing vertex.`);

    const faceNormal = new Vector3()
      .subVectors(b, a)
      .cross(new Vector3().subVectors(c, a));
    if (faceNormal.lengthSq() <= EPSILON) continue;

    normals[face[0]]!.add(faceNormal);
    normals[face[1]]!.add(faceNormal);
    normals[face[2]]!.add(faceNormal);
  }

  for (const normal of normals) {
    if (normal.lengthSq() <= EPSILON) normal.set(0, 0, 1);
    else normal.normalize();
  }

  return normals;
}
