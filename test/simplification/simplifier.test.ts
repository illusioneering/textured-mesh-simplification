import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  createSimplificationState,
  finalizeSimplification,
  runSimplificationStage,
  runVirtualEdgeStage,
  simplifyRawMeshCore,
} from '../../src/simplification/simplifier';
import { MIN_GENERATED_FACE_QUALITY } from '../../src/simplification/faceQuality';
import { simplifyRawMesh } from '../../src/simplification/simplify';
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

function tetraRawMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(0, 1, 0),
      new Vector3(0, 0, 1),
    ],
    faces: [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
  };
}

function triangleQuality(mesh: RawMesh, face: readonly [number, number, number]): number {
  const a = mesh.positions[face[0]]!;
  const b = mesh.positions[face[1]]!;
  const c = mesh.positions[face[2]]!;
  const ab = b.clone().sub(a);
  const bc = c.clone().sub(b);
  const ca = a.clone().sub(c);
  const area = ab.clone().cross(c.clone().sub(a)).length() * 0.5;
  return (4 * Math.sqrt(3) * area) / (ab.lengthSq() + bc.lengthSq() + ca.lengthSq());
}

describe('simplification path', () => {
  it('matches monolithic simplification when run through explicit simplification stages', () => {
    const raw = tetraRawMesh();
    const options = {
      targetFaceCount: 1,
      virtualEdges: { mode: 'manual-global-radius' as const, radius: 0 },
    };

    const monolithic = simplifyRawMeshCore(raw, options);
    const state = createSimplificationState(raw, options);
    const virtualEdges = runVirtualEdgeStage(state);
    const simplification = runSimplificationStage(state);
    const staged = finalizeSimplification(state);

    expect(virtualEdges.generatedVirtualEdges).toBe(monolithic.stats.virtualEdges);
    expect(simplification.collapses).toBe(monolithic.stats.collapses);
    expect(staged.stats).toEqual(monolithic.stats);
    expect(staged.outputFaceIds).toEqual(monolithic.outputFaceIds);
    expect(staged.rawMesh.faces).toEqual(monolithic.rawMesh.faces);
    expect(staged.history).toHaveLength(monolithic.history.length);
    expect(staged.history).toHaveLength(staged.stats.collapses);
  });

  it('rejects running simplification before virtual-edge search', () => {
    const state = createSimplificationState(tetraRawMesh(), {
      targetFaceCount: 1,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });

    expect(() => runSimplificationStage(state)).toThrow(/Run virtual-edge search before simplification/i);
  });

  it('simplifies through the internal simplification entry point', () => {
    const result = simplifyRawMeshCore(tetraRawMesh(), {
      targetFaceCount: 1,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });

    expect(result.stats.inputFaces).toBe(4);
    expect(result.stats.collapses).toBeGreaterThan(0);
    expect(result.history).toHaveLength(result.stats.collapses);
    expect(result.rawMesh.faces.length).toBeLessThanOrEqual(1);
  });

  it('removes duplicate faces deterministically before simplification', () => {
    const result = simplifyRawMesh(duplicateFaceMesh(), {
      targetFaceCount: 10,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });

    expect(result.stats.inputFaces).toBe(3);
    expect(result.stats.outputFaces).toBe(2);
    expect(result.outputFaceIds).toEqual([0, 2]);
  });

  it('captures collapse history for every successful collapse', () => {
    const raw = tetraRawMesh();

    const result = simplifyRawMesh(raw, {
      targetFaceCount: 1,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });

    expect(result.stats.collapses).toBeGreaterThan(0);
    expect(result.history).toHaveLength(result.stats.collapses);
    for (const record of result.history) {
      expect(record.beforeFaces.length).toBeGreaterThan(0);
      expect(record.afterFaceIds.every((faceId) => Number.isInteger(faceId))).toBe(true);
    }
  });

  it('does not accept collapses that turn healthy surviving faces into slivers', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(-1.9148719333702346, 2.989552972254061, -2.11191533760765),
        new Vector3(-1.8768999576568604, 3.0063693523406982, -2.0282397270202637),
        new Vector3(-1.9176561832427979, 3.0265119075775146, -2.078902244567871),
        new Vector3(-1.9225964546203613, 2.9140491485595703, -2.0753931999206543),
        new Vector3(-1.9374239444732666, 2.9396920204162598, -2.1197617053985596),
        new Vector3(-1.838927984237671, 3.0231857299804688, -1.9445643424987793),
        new Vector3(-1.9204411506652832, 3.0634708404541016, -2.0458896160125732),
        new Vector3(-1.8838839530944824, 2.9678096771240234, -2.0414698123931885),
        new Vector3(-1.852895975112915, 2.946066379547119, -1.9710242748260498),
        new Vector3(-1.9188148975372314, 3.02372145652771, -2.1506714820861816),
        new Vector3(-1.9400184154510498, 2.969877243041992, -2.173226833343506),
        new Vector3(-1.9262773990631104, 3.0532267093658447, -2.1085009574890137),
      ],
      faces: [
        [0, 1, 2],
        [3, 0, 4],
        [1, 5, 6],
        [7, 8, 1],
        [6, 2, 1],
        [0, 9, 10],
        [0, 2, 11],
        [4, 0, 10],
        [3, 7, 0],
        [1, 0, 7],
        [9, 0, 11],
        [5, 1, 8],
      ],
    };

    expect(raw.faces.every((face) => triangleQuality(raw, face) >= MIN_GENERATED_FACE_QUALITY)).toBe(true);

    const result = simplifyRawMesh(raw, {
      targetFaceCount: 11,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });

    for (const face of result.rawMesh.faces) {
      expect(triangleQuality(result.rawMesh, face)).toBeGreaterThanOrEqual(MIN_GENERATED_FACE_QUALITY);
    }
  });

  it('reports exact grid diagnostics for auto-local-radius virtual edges', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0), new Vector3(100, 0, 0), new Vector3(0, 100, 0),
        new Vector3(0, 0, 0.5), new Vector3(100, 0, 0.5), new Vector3(0, 100, 0.5),
      ],
      faces: [[0, 1, 2], [3, 4, 5]],
    };

    const progress: string[] = [];
    const first = simplifyRawMesh(raw, {
      targetRatio: 1,
      virtualEdges: { mode: 'auto-local-radius' },
      onVirtualEdgeProgress: (event) => progress.push(event.phase),
    });
    const second = simplifyRawMesh(raw, { targetRatio: 1, virtualEdges: { mode: 'auto-local-radius' } });

    expect(first.stats.virtualEdges).toBeGreaterThan(0);
    expect(first.stats.virtualEdges).toBe(second.stats.virtualEdges);
    expect(first.stats.virtualEdgeDiagnostics).toMatchObject({
      mode: 'auto-local-radius',
      searchStrategy: 'grid-exact',
      candidateFacePairs: 1,
      exactDistanceTests: 1,
      generatedVirtualEdges: first.stats.virtualEdges,
      maxPairsPerComponentPair: null,
    });
    expect(first.stats.virtualEdgeDiagnostics?.aabbRejectedPairs).toBe(0);
    expect(first.stats.virtualEdgeDiagnostics?.distanceRejectedPairs).toBe(0);
    expect(progress).toContain('building-buckets');
    expect(progress).toContain('searching-pairs');
  });

  it('reports exact grid diagnostics for manual-global-radius virtual edges', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0),
        new Vector3(0, 0, 0.05), new Vector3(1, 0, 0.05), new Vector3(0, 1, 0.05),
      ],
      faces: [[0, 1, 2], [3, 4, 5]],
    };

    const result = simplifyRawMesh(raw, { targetRatio: 1, virtualEdges: { mode: 'manual-global-radius', radius: 0.1 } });

    expect(result.stats.virtualEdgeDiagnostics).toMatchObject({
      mode: 'manual-global-radius',
      radius: 0.1,
      searchStrategy: 'grid-exact',
      candidateFacePairs: 1,
      exactDistanceTests: 1,
      generatedVirtualEdges: result.stats.virtualEdges,
    });
  });

  it('reports grid diagnostics for auto-global-radius virtual edges', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0), new Vector3(2, 0, 0), new Vector3(0, 2, 0),
        new Vector3(0, 0, 1.5), new Vector3(2, 0, 1.5), new Vector3(0, 2, 1.5),
        new Vector3(1000, 0, 0), new Vector3(1002, 0, 0), new Vector3(1000, 2, 0),
      ],
      faces: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    };

    const result = simplifyRawMesh(raw, { targetRatio: 1, virtualEdges: { mode: 'auto-global-radius' } });

    expect(result.stats.virtualEdges).toBeGreaterThan(0);
    expect(result.stats.virtualEdgeDiagnostics).toMatchObject({
      mode: 'auto-global-radius',
      radius: 1,
      radiusScale: 0.5,
      searchStrategy: 'grid-exact',
      generatedVirtualEdges: result.stats.virtualEdges,
    });
    expect(result.stats.virtualEdgeDiagnostics?.bboxDistanceLimit).toBeGreaterThan(2);
  });

  it('does not create virtual edges between nearby faces in the same physical component', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(1, 1, 0),
      ],
      faces: [[0, 1, 2], [1, 3, 2]],
    };

    const result = simplifyRawMesh(raw, { targetRatio: 1, virtualEdges: { mode: 'manual-global-radius', radius: 10 } });

    expect(result.stats.virtualEdges).toBe(0);
    expect(result.stats.virtualEdgeDiagnostics).toMatchObject({
      mode: 'manual-global-radius',
      searchStrategy: 'grid-exact',
      candidateFacePairs: 0,
      exactDistanceTests: 0,
      generatedVirtualEdges: 0,
    });
  });
});
