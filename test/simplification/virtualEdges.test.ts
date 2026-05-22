import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { SimplificationMesh } from '../../src/simplification/mesh';
import {
  addVirtualEdges,
  computeAutoGlobalRadius,
  computeConnectedComponents,
  computeLocalTriangleRadii,
} from '../../src/simplification/virtualEdges';
import type { RawMesh } from '../../src/simplification/types';

function twoTriangleMesh(offset = 0.1): SimplificationMesh {
  return SimplificationMesh.fromRawMesh({
    positions: [
      new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0),
      new Vector3(0, 0, offset), new Vector3(1, 0, offset), new Vector3(0, 1, offset),
    ],
    faces: [[0, 1, 2], [3, 4, 5]],
  });
}

describe('virtual edges', () => {
  it('computes connected components using physical edges', () => {
    const components = computeConnectedComponents(twoTriangleMesh());
    expect(components.count).toBe(2);
    expect(components.faceComponents[0]).not.toBe(components.faceComponents[1]);
  });

  it('adds manual-global-radius virtual edges when triangle distance is below 2r', () => {
    const mesh = twoTriangleMesh(0.1);
    const stats = addVirtualEdges(mesh, { mode: 'manual-global-radius', radius: 0.1 });
    expect(stats.added).toBeGreaterThan(0);
    expect(mesh.activeEdgeIds(true).some((edgeId) => mesh.edgeVirtual[edgeId])).toBe(true);
  });

  it('does not add manual-global-radius virtual edges for far triangles', () => {
    const mesh = twoTriangleMesh(10);
    const stats = addVirtualEdges(mesh, { mode: 'manual-global-radius', radius: 0.1 });
    expect(stats.added).toBe(0);
  });

  it('computes clamped auto-local triangle radii from one-ring edge lengths', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0),
        new Vector3(10, 0, 0), new Vector3(110, 0, 0), new Vector3(10, 100, 0),
        new Vector3(0, 0, 0.1), new Vector3(1, 0, 0.1), new Vector3(0, 1, 0.1),
      ],
      faces: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    };
    const mesh = SimplificationMesh.fromRawMesh(raw);

    const radii = computeLocalTriangleRadii(mesh);

    expect(radii.mode).toBe('auto-local-radius');
    expect(radii.radiusScale).toBe(0.5);
    expect(radii.clampMin).toBeGreaterThan(0);
    expect(radii.clampMax).toBeGreaterThan(radii.clampMin);
    expect(radii.faceRadii[1]).toBeLessThanOrEqual(radii.clampMax * 0.5);
  });

  it('uses auto-local-radius radii with the r_i plus r_j threshold', () => {
    const mesh = SimplificationMesh.fromRawMesh({
      positions: [
        new Vector3(0, 0, 0), new Vector3(100, 0, 0), new Vector3(0, 100, 0),
        new Vector3(0, 0, 0.9), new Vector3(100, 0, 0.9), new Vector3(0, 100, 0.9),
      ],
      faces: [[0, 1, 2], [3, 4, 5]],
    });

    const stats = addVirtualEdges(mesh, { mode: 'auto-local-radius' });

    expect(stats.added).toBeGreaterThan(0);
    expect(stats.mode).toBe('auto-local-radius');
  });

  it('computes uncapped auto-global-radius from the median physical edge length', () => {
    const mesh = SimplificationMesh.fromRawMesh({
      positions: [
        new Vector3(0, 0, 0), new Vector3(2, 0, 0), new Vector3(0, 2, 0),
        new Vector3(0, 0, 1.5), new Vector3(2, 0, 1.5), new Vector3(0, 2, 1.5),
        new Vector3(1000, 0, 0), new Vector3(1002, 0, 0), new Vector3(1000, 2, 0),
      ],
      faces: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    });

    const radius = computeAutoGlobalRadius(mesh);

    expect(radius).toMatchObject({
      mode: 'auto-global-radius',
      medianEdgeLength: 2,
      radiusScale: 0.5,
      radius: 1,
    });
    expect(radius.bboxDistanceLimit).toBeGreaterThan(2);
  });

  it('caps auto-global-radius by half the bbox distance limit', () => {
    const mesh = twoTriangleMesh(0.1);

    const radius = computeAutoGlobalRadius(mesh);

    expect(radius.medianEdgeLength).toBe(1);
    expect(radius.radius).toBeCloseTo(radius.bboxDistanceLimit * 0.5);
    expect(radius.radius).toBeLessThan(radius.medianEdgeLength * 0.5);
  });

  it('adds auto-global-radius virtual edges using the computed global radius', () => {
    const mesh = SimplificationMesh.fromRawMesh({
      positions: [
        new Vector3(0, 0, 0), new Vector3(2, 0, 0), new Vector3(0, 2, 0),
        new Vector3(0, 0, 1.5), new Vector3(2, 0, 1.5), new Vector3(0, 2, 1.5),
        new Vector3(1000, 0, 0), new Vector3(1002, 0, 0), new Vector3(1000, 2, 0),
      ],
      faces: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    });

    const stats = addVirtualEdges(mesh, { mode: 'auto-global-radius' });

    expect(stats.added).toBeGreaterThan(0);
    expect(stats).toMatchObject({
      mode: 'auto-global-radius',
      radius: 1,
      radiusScale: 0.5,
    });
    expect(stats.bboxDistanceLimit).toBeGreaterThan(2);
  });

  it('discards duplicate virtual vertex pairs before applying the component-pair cap', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(1, 1, 0),
        new Vector3(0, 0, 0.05), new Vector3(1, 0, 0.05), new Vector3(0, 1, 0.05), new Vector3(1, 1, 0.05),
      ],
      faces: [[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]],
    };
    const mesh = SimplificationMesh.fromRawMesh(raw);

    const stats = addVirtualEdges(mesh, { mode: 'manual-global-radius', radius: 1 });

    expect(stats.duplicateVertexPairCandidates).toBeGreaterThan(0);
    expect(stats.added).toBe(stats.generatedVirtualEdges);
  });

  it('does not cap manual-global-radius virtual-edge candidates', () => {
    const positions: Vector3[] = [];
    const faces: RawMesh['faces'] = [];
    for (let i = 0; i < 6; i += 1) {
      positions.push(new Vector3(i, 0, 0), new Vector3(i, 1, 0));
      positions.push(new Vector3(i, 0, 0.05), new Vector3(i, 1, 0.05));
      if (i > 0) {
        const a = (i - 1) * 4;
        const b = i * 4;
        faces.push([a, b, a + 1], [b, b + 1, a + 1]);
        faces.push([a + 2, b + 2, a + 3], [b + 2, b + 3, a + 3]);
      }
    }
    const mesh = SimplificationMesh.fromRawMesh({ positions, faces });

    const stats = addVirtualEdges(mesh, { mode: 'manual-global-radius', radius: 100 });

    expect(stats.added).toBeGreaterThan(4);
    expect(stats.maxPairsPerComponentPair).toBeUndefined();
    expect(stats.cappedVirtualEdgeCandidates).toBe(0);
  });

  it('does not cap auto-local-radius virtual-edge candidates by default', () => {
    const positions: Vector3[] = [];
    const faces: RawMesh['faces'] = [];
    for (let i = 0; i < 6; i += 1) {
      positions.push(new Vector3(i, 0, 0), new Vector3(i, 1, 0));
      positions.push(new Vector3(i, 0, 0.05), new Vector3(i, 1, 0.05));
      if (i > 0) {
        const a = (i - 1) * 4;
        const b = i * 4;
        faces.push([a, b, a + 1], [b, b + 1, a + 1]);
        faces.push([a + 2, b + 2, a + 3], [b + 2, b + 3, a + 3]);
      }
    }
    const mesh = SimplificationMesh.fromRawMesh({ positions, faces });

    const stats = addVirtualEdges(mesh, { mode: 'auto-local-radius' });

    expect(stats.added).toBeGreaterThan(4);
    expect(stats.maxPairsPerComponentPair).toBeNull();
    expect(stats.cappedVirtualEdgeCandidates).toBe(0);
  });

  it('keeps the nearest four unique auto-local-radius virtual-edge candidates per component pair when explicitly capped', () => {
    const positions: Vector3[] = [];
    const faces: RawMesh['faces'] = [];
    for (let i = 0; i < 6; i += 1) {
      positions.push(new Vector3(i, 0, 0), new Vector3(i, 1, 0));
      positions.push(new Vector3(i, 0, 0.05), new Vector3(i, 1, 0.05));
      if (i > 0) {
        const a = (i - 1) * 4;
        const b = i * 4;
        faces.push([a, b, a + 1], [b, b + 1, a + 1]);
        faces.push([a + 2, b + 2, a + 3], [b + 2, b + 3, a + 3]);
      }
    }
    const mesh = SimplificationMesh.fromRawMesh({ positions, faces });

    const stats = addVirtualEdges(mesh, { mode: 'auto-local-radius', maxPairsPerComponentPair: 4 });

    expect(stats.added).toBe(4);
    expect(stats.maxPairsPerComponentPair).toBe(4);
    expect(stats.cappedVirtualEdgeCandidates).toBeGreaterThan(0);
  });
});
