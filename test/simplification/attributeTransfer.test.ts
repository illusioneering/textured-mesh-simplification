import { describe, expect, it } from 'vitest';
import { Vector2, Vector3, Vector4 } from 'three';
import type { RawMesh, RawSimplificationResult } from '../../src/simplification/types';
import type { SourceFaceAttributes } from '../../src/simplification/attributes';
import { faceUvSet, transferredVertexUv } from '../../src/simplification/attributes';
import { transferVertexAttributesToSimplifiedMesh } from '../../src/simplification/attributeTransfer';

function rawResult(rawMesh: RawMesh, outputFaceIds = rawMesh.faces.map((_, index) => index)): RawSimplificationResult {
  return {
    rawMesh,
    outputFaceIds,
    history: [],
    stats: {
      inputVertices: rawMesh.positions.length,
      inputFaces: rawMesh.faces.length,
      outputVertices: rawMesh.positions.length,
      outputFaces: rawMesh.faces.length,
      physicalEdges: 0,
      virtualEdges: 0,
      collapses: 0,
      stoppedReason: 'target-reached',
    },
  };
}

function singleTriangleResult(): RawSimplificationResult {
  return rawResult({
    positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
    faces: [[0, 1, 2]],
  });
}

function triangleAttributes(): SourceFaceAttributes[] {
  return [{
    materialId: 0,
    uvSets: [
      { texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] },
      { texCoord: 1, uvs: [new Vector2(0.1, 0.2), new Vector2(0.3, 0.4), new Vector2(0.5, 0.6)] },
    ],
    normalCorners: [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)],
    tangentCorners: [new Vector4(1, 0, 0, 1), new Vector4(0, 1, 0, 1), new Vector4(0, 0, 1, -1)],
    colorCorners: [new Vector4(1, 0, 0, 1), new Vector4(0, 1, 0, 1), new Vector4(0, 0, 1, 1)],
    colorItemSize: 3,
    normalMapYScale: -1,
  }];
}

describe('transferVertexAttributesToSimplifiedMesh', () => {
  it('transfers UVs, normals, and tangents onto welded output vertices', () => {
    const transferred = transferVertexAttributesToSimplifiedMesh({
      sourceFaceAttributes: triangleAttributes(),
      raw: singleTriangleResult(),
    });

    expect(transferred.vertices).toHaveLength(3);
    expect(faceUvSet(transferred.vertices[0]!, 0)).toEqual(new Vector2(0, 0));
    expect(faceUvSet(transferred.vertices[1]!, 1)).toEqual(new Vector2(0.3, 0.4));
    expect(transferred.vertices[0]!.normal).toEqual(new Vector3(1, 0, 0));
    expect(transferred.vertices[1]!.normal).toEqual(new Vector3(0, 1, 0));
    expect(transferred.vertices[2]!.normal).toEqual(new Vector3(0, 0, 1));
    expect(transferred.vertices[0]!.tangent).toEqual(new Vector4(1, 0, 0, 1));
    expect(transferred.vertices[1]!.tangent).toEqual(new Vector4(0, 1, 0, 1));
    expect(transferred.vertices[2]!.tangent).toEqual(new Vector4(0, 0, 1, -1));
    expect(transferred.colorItemSize).toBe(3);
    expect(transferred.vertices[0]!.color).toEqual(new Vector4(1, 0, 0, 1));
    expect(transferred.vertices[1]!.color).toEqual(new Vector4(0, 1, 0, 1));
    expect(transferred.vertices[2]!.color).toEqual(new Vector4(0, 0, 1, 1));
    expect(transferred.normalMapYScale).toBe(-1);
    expect(transferred.hasSourceTangents).toBe(true);
  });

  it('marks source tangent provenance when only some source faces have tangents', () => {
    const raw = rawResult({
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(1, 1, 0),
      ],
      faces: [[0, 1, 2], [1, 3, 2]],
    });
    const sourceFaceAttributes: SourceFaceAttributes[] = [
      {
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
        tangentCorners: [new Vector4(1, 0, 0, 1), new Vector4(1, 0, 0, 1), new Vector4(1, 0, 0, 1)],
      },
      {
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(1, 0), new Vector2(1, 1), new Vector2(0, 1)] }],
      },
    ];

    const transferred = transferVertexAttributesToSimplifiedMesh({ sourceFaceAttributes, raw });

    expect(transferred.hasSourceTangents).toBe(true);
  });

  it('walks collapse history before sampling source attributes', () => {
    const outputRawMesh: RawMesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    };
    const raw: RawSimplificationResult = {
      ...rawResult(outputRawMesh, [10]),
      history: [{
        keepVertexId: 0,
        removedVertexId: 1,
        afterFaceIds: [10],
        beforeFaces: [{
          faceId: 2,
          vertices: [0, 1, 2],
          positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
        }],
      }],
    };
    const sourceFaceAttributes: SourceFaceAttributes[] = [
      { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(9, 9), new Vector2(9, 9), new Vector2(9, 9)] }] },
      { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(8, 8), new Vector2(8, 8), new Vector2(8, 8)] }] },
      { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }] },
    ];

    const transferred = transferVertexAttributesToSimplifiedMesh({ sourceFaceAttributes, raw });

    expect(transferredVertexUv(transferred.vertices[0]!, 0)).toEqual(new Vector2(0, 0));
    expect(transferredVertexUv(transferred.vertices[1]!, 0)).toEqual(new Vector2(1, 0));
    expect(transferredVertexUv(transferred.vertices[2]!, 0)).toEqual(new Vector2(0, 1));
  });

  it('area-weights conflicting welded vertex samples without splitting topology', () => {
    const raw = rawResult({
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(2, 0, 0),
        new Vector3(0, 2, 0),
        new Vector3(0, 0.5, 0),
      ],
      faces: [[0, 1, 2], [0, 3, 1]],
    });
    const sourceFaceAttributes: SourceFaceAttributes[] = [
      {
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(0, 0), new Vector2(0, 0)] }],
        normalCorners: [new Vector3(1, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 0, 0)],
      },
      {
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(10, 0), new Vector2(10, 0), new Vector2(10, 0)] }],
        normalCorners: [new Vector3(0, 1, 0), new Vector3(0, 1, 0), new Vector3(0, 1, 0)],
      },
    ];

    const transferred = transferVertexAttributesToSimplifiedMesh({ sourceFaceAttributes, raw });

    expect(transferred.vertices).toHaveLength(raw.rawMesh.positions.length);
    expect(faceUvSet(transferred.vertices[0]!, 0)?.x).toBeCloseTo(2);
    expect(transferred.vertices[0]!.normal?.x).toBeCloseTo(4 / Math.sqrt(17));
    expect(transferred.vertices[0]!.normal?.y).toBeCloseTo(1 / Math.sqrt(17));
  });

  it('area-weights conflicting welded vertex colors without splitting topology', () => {
    const raw = rawResult({
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(2, 0, 0),
        new Vector3(0, 2, 0),
        new Vector3(0, 0.5, 0),
      ],
      faces: [[0, 1, 2], [0, 3, 1]],
    });
    const sourceFaceAttributes: SourceFaceAttributes[] = [
      {
        materialId: 0,
        uvSets: [],
        colorCorners: [new Vector4(1, 0, 0, 1), new Vector4(1, 0, 0, 1), new Vector4(1, 0, 0, 1)],
        colorItemSize: 4,
      },
      {
        materialId: 0,
        uvSets: [],
        colorCorners: [new Vector4(0, 1, 0, 1), new Vector4(0, 1, 0, 1), new Vector4(0, 1, 0, 1)],
        colorItemSize: 4,
      },
    ];

    const transferred = transferVertexAttributesToSimplifiedMesh({ sourceFaceAttributes, raw });

    expect(transferred.vertices).toHaveLength(raw.rawMesh.positions.length);
    expect(transferred.vertices[0]!.color?.x).toBeCloseTo(0.8);
    expect(transferred.vertices[0]!.color?.y).toBeCloseTo(0.2);
    expect(transferred.vertices[0]!.color?.z).toBeCloseTo(0);
    expect(transferred.vertices[0]!.color?.w).toBeCloseTo(1);
    expect(transferred.colorItemSize).toBe(4);
  });

  it('omits optional UV sets missing from any sample for a welded vertex', () => {
    const raw = rawResult({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, -1, 0)],
      faces: [[0, 1, 2], [0, 3, 1]],
    });
    const sourceFaceAttributes: SourceFaceAttributes[] = [
      {
        materialId: 0,
        uvSets: [
          { texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] },
          { texCoord: 1, uvs: [new Vector2(2, 0), new Vector2(2, 1), new Vector2(2, 2)] },
        ],
      },
      {
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(10, 0), new Vector2(11, 0), new Vector2(12, 0)] }],
      },
    ];

    const transferred = transferVertexAttributesToSimplifiedMesh({ sourceFaceAttributes, raw });

    expect(transferredVertexUv(transferred.vertices[0]!, 0)).toBeDefined();
    expect(faceUvSet(transferred.vertices[0]!, 1)).toBeUndefined();
    expect(faceUvSet(transferred.vertices[2]!, 1)).toEqual(new Vector2(2, 2));
  });

  it('reports final progress for simplified output faces', () => {
    const raw = singleTriangleResult();
    const updates: Array<{ processedFaces: number; totalFaces: number }> = [];

    transferVertexAttributesToSimplifiedMesh({
      sourceFaceAttributes: triangleAttributes(),
      raw,
      onProgress: (progress) => updates.push(progress),
    });

    expect(updates).toEqual([{ processedFaces: 1, totalFaces: 1 }]);
  });
});
