import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import type { RawMesh } from '../../src/simplification/types';
import { createHistoryTraceIndex, mapOutputSampleToInput } from '../../src/texture/successiveMapping';

describe('texture successive mapping compatibility exports', () => {
  it('re-exports output sample mapping from simplification', () => {
    const outputRawMesh: RawMesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    };

    const mapped = mapOutputSampleToInput({
      outputRawMesh,
      outputFaceIds: [4],
      outputFaceIndex: 0,
      outputBarycentric: [0.2, 0.3, 0.5],
      history: [],
      historyIndex: createHistoryTraceIndex([]),
    });

    expect(mapped.faceId).toBe(4);
    expect(mapped.barycentric).toEqual([0.2, 0.3, 0.5]);
  });
});
