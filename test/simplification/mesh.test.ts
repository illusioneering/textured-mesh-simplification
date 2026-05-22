import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { SimplificationMesh } from '../../src/simplification/mesh';
import { MIN_GENERATED_FACE_QUALITY } from '../../src/simplification/faceQuality';
import type { RawMesh } from '../../src/simplification/types';

function duplicateFaceMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(0, 1, 0),
      new Vector3(0, 0, 1),
    ],
    faces: [
      [0, 1, 2],
      [2, 1, 0],
      [0, 1, 3],
    ],
  };
}

describe('SimplificationMesh', () => {
  it('builds typed-array state and removes duplicate faces deterministically', () => {
    const mesh = SimplificationMesh.fromRawMesh(duplicateFaceMesh());

    expect(mesh.activeVertexCount()).toBe(4);
    expect(mesh.activeFaceCount()).toBe(2);
    expect(mesh.activeEdgeCount(false)).toBe(5);
    expect(mesh.outputFaceIds()).toEqual([0, 2]);
  });

  it('uses a numeric edge-key encoding within safe integer bounds', () => {
    const mesh = SimplificationMesh.fromRawMesh(duplicateFaceMesh());

    expect(mesh.edgeKey(0, 3)).toBe(3);
    expect(mesh.edgeKey(3, 0)).toBe(3);
  });

  it('reports scale-invariant triangle quality for simplification faces', () => {
    const mesh = SimplificationMesh.fromRawMesh({
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0.5, Math.sqrt(3) / 2, 0),
        new Vector3(2, 1e-5, 0),
        new Vector3(0, 0, 0),
        new Vector3(8, 0, 0),
        new Vector3(4, 4 * Math.sqrt(3), 0),
        new Vector3(0, 0, 0),
        new Vector3(1e-11, 0, 0),
        new Vector3(0.5e-11, (Math.sqrt(3) / 2) * 1e-11, 0),
      ],
      faces: [[0, 1, 2], [0, 1, 3], [4, 5, 6], [7, 8, 9]],
    });

    const unitQuality = mesh.triangleQualityByVertices(0, 1, 2);
    const scaledQuality = mesh.triangleQualityByVertices(4, 5, 6);
    const tinyQuality = mesh.triangleQualityByVertices(7, 8, 9);

    expect(unitQuality).toBeCloseTo(1, 12);
    expect(scaledQuality).toBeCloseTo(1, 12);
    expect(scaledQuality).toBeCloseTo(unitQuality, 12);
    expect(tinyQuality).toBeCloseTo(1, 12);
    expect(tinyQuality).toBeCloseTo(unitQuality, 12);
    expect(mesh.triangleQualityByVertices(0, 1, 3)).toBeLessThan(MIN_GENERATED_FACE_QUALITY);
  });

  it('evaluates triangle quality with a candidate replacement position without mutating the mesh', () => {
    const mesh = SimplificationMesh.fromRawMesh({
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(2, 1e-5, 0),
      ],
      faces: [[0, 1, 2]],
    });

    const before = mesh.triangleQualityByFace(0);
    const after = mesh.triangleQualityWithVertexPosition(0, 2, [2, 1e-5, 0]);

    expect(before).toBeGreaterThan(MIN_GENERATED_FACE_QUALITY);
    expect(after).toBeLessThan(MIN_GENERATED_FACE_QUALITY);
    expect(mesh.position(2)).toEqual([0, 1, 0]);
  });
});
