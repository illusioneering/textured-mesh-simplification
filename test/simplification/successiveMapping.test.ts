import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { simplifyRawMesh } from '../../src/simplification/simplify';
import type { Barycentric, CollapseHistoryRecord, RawMesh } from '../../src/simplification/types';
import {
  createHistoryTraceIndex,
  mapOutputSampleToInput,
  type MappedSample,
} from '../../src/simplification/successiveMapping';

function triangleRawMesh(): RawMesh {
  return {
    positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
    faces: [[0, 1, 2]],
  };
}

function expectMappedSamplesToMatch(actual: MappedSample, expected: MappedSample): void {
  expect(actual.faceId).toBe(expected.faceId);
  expect(actual.distanceSquared).toBeCloseTo(expected.distanceSquared);
  expect(actual.projections).toBe(expected.projections);
  expect(actual.barycentric[0]).toBeCloseTo(expected.barycentric[0]);
  expect(actual.barycentric[1]).toBeCloseTo(expected.barycentric[1]);
  expect(actual.barycentric[2]).toBeCloseTo(expected.barycentric[2]);
}

describe('successive mapping', () => {
  it('maps directly when no collapse history exists', () => {
    const mapped = mapOutputSampleToInput({
      outputRawMesh: triangleRawMesh(),
      outputFaceIds: [42],
      outputFaceIndex: 0,
      outputBarycentric: [0.2, 0.3, 0.5],
      history: [],
    });

    expect(mapped.faceId).toBe(42);
    expect(mapped.barycentric).toEqual([0.2, 0.3, 0.5]);
    expect(mapped.projections).toBe(0);
  });

  it('maps directly with an empty history trace index', () => {
    const history: CollapseHistoryRecord[] = [];
    const indexed = mapOutputSampleToInput({
      outputRawMesh: triangleRawMesh(),
      outputFaceIds: [42],
      outputFaceIndex: 0,
      outputBarycentric: [0.2, 0.3, 0.5],
      history,
      historyIndex: createHistoryTraceIndex(history),
    });

    expect(indexed.faceId).toBe(42);
    expect(indexed.barycentric).toEqual([0.2, 0.3, 0.5]);
    expect(indexed.projections).toBe(0);
  });

  it('projects through matching after-face one-rings only', () => {
    const history: CollapseHistoryRecord[] = [{
      keepVertexId: 0,
      removedVertexId: 1,
      afterFaceIds: [7],
      beforeFaces: [{
        faceId: 3,
        vertices: [0, 1, 2],
        positions: [new Vector3(0, 0, 0), new Vector3(2, 0, 0), new Vector3(0, 2, 0)],
      }],
    }];

    const mapped = mapOutputSampleToInput({
      outputRawMesh: triangleRawMesh(),
      outputFaceIds: [7],
      outputFaceIndex: 0,
      outputBarycentric: [0.5, 0.25, 0.25],
      history,
    });

    expect(mapped.faceId).toBe(3);
    expect(mapped.projections).toBe(1);
    expect(mapped.barycentric[0] + mapped.barycentric[1] + mapped.barycentric[2]).toBeCloseTo(1);
  });

  it('chains face correspondence backward across multiple collapse records', () => {
    const history: CollapseHistoryRecord[] = [
      {
        keepVertexId: 10,
        removedVertexId: 11,
        afterFaceIds: [1],
        beforeFaces: [
          {
            faceId: 10,
            vertices: [0, 1, 2],
            positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
          },
          {
            faceId: 11,
            vertices: [3, 4, 5],
            positions: [new Vector3(100, 0, 0), new Vector3(101, 0, 0), new Vector3(100, 1, 0)],
          },
        ],
      },
      {
        keepVertexId: 20,
        removedVertexId: 21,
        afterFaceIds: [2],
        beforeFaces: [{
          faceId: 1,
          vertices: [6, 7, 8],
          positions: [new Vector3(100, 0, 0), new Vector3(101, 0, 0), new Vector3(100, 1, 0)],
        }],
      },
    ];

    const mapped = mapOutputSampleToInput({
      outputRawMesh: triangleRawMesh(),
      outputFaceIds: [2],
      outputFaceIndex: 0,
      outputBarycentric: [1, 0, 0],
      history,
    });

    expect(mapped.faceId).toBe(10);
    expect(mapped.projections).toBe(2);
  });

  it('matches unindexed mapping when using a history trace index', () => {
    const history: CollapseHistoryRecord[] = [
      {
        keepVertexId: 10,
        removedVertexId: 11,
        afterFaceIds: [1],
        beforeFaces: [
          {
            faceId: 10,
            vertices: [0, 1, 2],
            positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
          },
          {
            faceId: 11,
            vertices: [3, 4, 5],
            positions: [new Vector3(100, 0, 0), new Vector3(101, 0, 0), new Vector3(100, 1, 0)],
          },
        ],
      },
      {
        keepVertexId: 20,
        removedVertexId: 21,
        afterFaceIds: [2],
        beforeFaces: [{
          faceId: 1,
          vertices: [6, 7, 8],
          positions: [new Vector3(100, 0, 0), new Vector3(101, 0, 0), new Vector3(100, 1, 0)],
        }],
      },
    ];
    const shared = {
      outputRawMesh: triangleRawMesh(),
      outputFaceIds: [2],
      outputFaceIndex: 0,
      outputBarycentric: [1, 0, 0] as Barycentric,
      history,
    };

    const unindexed = mapOutputSampleToInput(shared);
    const indexed = mapOutputSampleToInput({
      ...shared,
      historyIndex: createHistoryTraceIndex(history),
    });

    expectMappedSamplesToMatch(indexed, unindexed);
  });

  it('keeps the final simplified point fixed while walking collapse history', () => {
    const outputRawMesh: RawMesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    };
    const history: CollapseHistoryRecord[] = [
      {
        keepVertexId: 0,
        removedVertexId: 1,
        afterFaceIds: [10],
        beforeFaces: [
          {
            faceId: 1,
            vertices: [0, 1, 2],
            positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
          },
          {
            faceId: 2,
            vertices: [3, 4, 5],
            positions: [new Vector3(5, 0, 0), new Vector3(6, 0, 0), new Vector3(5, 1, 0)],
          },
        ],
      },
      {
        keepVertexId: 10,
        removedVertexId: 11,
        afterFaceIds: [20],
        beforeFaces: [
          {
            faceId: 10,
            vertices: [6, 7, 8],
            positions: [new Vector3(20, 0, 0), new Vector3(21, 0, 0), new Vector3(20, 1, 0)],
          },
        ],
      },
    ];

    const mapped = mapOutputSampleToInput({
      outputRawMesh,
      outputFaceIds: [20],
      outputFaceIndex: 0,
      outputBarycentric: [1, 0, 0],
      history,
    });

    expect(mapped.faceId).toBe(1);
    expect(mapped.projections).toBe(2);
  });

  it('maps a captured simplification result to a valid original face id', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(0, 0, 1),
      ],
      faces: [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    };
    const simplified = simplifyRawMesh(raw, {
      targetFaceCount: 1,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });
    const mapped = mapOutputSampleToInput({
      outputRawMesh: simplified.rawMesh,
      outputFaceIds: simplified.outputFaceIds,
      outputFaceIndex: 0,
      outputBarycentric: [1 / 3, 1 / 3, 1 / 3],
      history: simplified.history,
    });

    expect(mapped.faceId).toBeGreaterThanOrEqual(0);
    expect(mapped.faceId).toBeLessThan(raw.faces.length);
    expect(mapped.barycentric[0] + mapped.barycentric[1] + mapped.barycentric[2]).toBeCloseTo(1);
  });

  it('matches unindexed mapping across multiple samples from captured simplification history', () => {
    const raw: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(0, 0, 1),
        new Vector3(1, 1, 0),
      ],
      faces: [[0, 1, 2], [1, 4, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 4, 3]],
    };
    const simplified = simplifyRawMesh(raw, {
      targetFaceCount: 2,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });
    const index = createHistoryTraceIndex(simplified.history);
    const barycentrics: Barycentric[] = [
      [1 / 3, 1 / 3, 1 / 3],
      [0.8, 0.1, 0.1],
      [0.1, 0.8, 0.1],
      [0.1, 0.1, 0.8],
    ];

    for (let outputFaceIndex = 0; outputFaceIndex < simplified.rawMesh.faces.length; outputFaceIndex += 1) {
      for (const outputBarycentric of barycentrics) {
        const shared = {
          outputRawMesh: simplified.rawMesh,
          outputFaceIds: simplified.outputFaceIds,
          outputFaceIndex,
          outputBarycentric,
          history: simplified.history,
        };
        const unindexed = mapOutputSampleToInput(shared);
        const indexed = mapOutputSampleToInput({ ...shared, historyIndex: index });
        expectMappedSamplesToMatch(indexed, unindexed);
      }
    }
  });
});
