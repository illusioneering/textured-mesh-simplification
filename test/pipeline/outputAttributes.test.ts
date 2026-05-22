import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { SourceFaceAttributes } from '../../src/simplification/attributes';
import type { RawMesh, RawSimplificationResult } from '../../src/simplification/types';
import { prepareOutputTransferredAttributes } from '../../src/pipeline/outputAttributes';

function rawResult(rawMesh: RawMesh): RawSimplificationResult {
  return {
    rawMesh,
    outputFaceIds: rawMesh.faces.map((_, index) => index),
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

function flatXyTriangle(): RawSimplificationResult {
  return rawResult({
    positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
    faces: [[0, 1, 2]],
  });
}

function triangleAttributes(normals: [Vector3, Vector3, Vector3]): SourceFaceAttributes[] {
  return [{
    materialId: 0,
    uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
    normalCorners: normals,
  }];
}

describe('prepareOutputTransferredAttributes', () => {
  it('ignores authored source normals when recomputeNormals is true', () => {
    const transferred = prepareOutputTransferredAttributes({
      raw: flatXyTriangle(),
      sourceFaceAttributes: triangleAttributes([
        new Vector3(1, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 0, 0),
      ]),
      recomputeNormals: true,
    });

    expect(transferred.vertices.map((vertex) => vertex.normal)).toEqual([
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
    ]);
  });

  it('uses authored source normals when recomputeNormals is false', () => {
    const transferred = prepareOutputTransferredAttributes({
      raw: flatXyTriangle(),
      sourceFaceAttributes: triangleAttributes([
        new Vector3(1, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 0, 0),
      ]),
      recomputeNormals: false,
    });

    expect(transferred.vertices.map((vertex) => vertex.normal)).toEqual([
      new Vector3(1, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(1, 0, 0),
    ]);
  });

  it('falls back per vertex to computed normals when source normals are unavailable or invalid', () => {
    const transferred = prepareOutputTransferredAttributes({
      raw: flatXyTriangle(),
      sourceFaceAttributes: triangleAttributes([
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 0, 0),
      ]),
      recomputeNormals: false,
    });

    expect(transferred.vertices.map((vertex) => vertex.normal)).toEqual([
      new Vector3(0, 0, 1),
      new Vector3(1, 0, 0),
      new Vector3(1, 0, 0),
    ]);
  });

  it('falls back to computed normals when source face attributes do not cover output face ids', () => {
    const raw = rawResult({
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(2, 0, 0),
      ],
      faces: [[0, 1, 2], [1, 3, 2]],
    });
    raw.outputFaceIds = [0, 4];

    const transferred = prepareOutputTransferredAttributes({
      raw,
      sourceFaceAttributes: triangleAttributes([
        new Vector3(1, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 0, 0),
      ]),
      recomputeNormals: false,
    });

    expect(transferred.vertices.map((vertex) => vertex.normal)).toEqual([
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
    ]);
  });

  it('falls back to computed normals when mapped source normal corners are malformed', () => {
    const sourceFaceAttributes = [{
      materialId: 0,
      uvSets: [],
      normalCorners: [new Vector3(1, 0, 0), new Vector3(1, 0, 0)],
    }] as unknown as SourceFaceAttributes[];

    const transferred = prepareOutputTransferredAttributes({
      raw: flatXyTriangle(),
      sourceFaceAttributes,
      recomputeNormals: false,
    });

    expect(transferred).toEqual({
      vertices: [
        { uvSets: [], normal: new Vector3(0, 0, 1) },
        { uvSets: [], normal: new Vector3(0, 0, 1) },
        { uvSets: [], normal: new Vector3(0, 0, 1) },
      ],
    });
  });

  it('falls back to computed normals when mapped source UV tuples are malformed', () => {
    const sourceFaceAttributes = [{
      materialId: 0,
      uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0)] }],
      normalCorners: [new Vector3(1, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 0, 0)],
    }] as unknown as SourceFaceAttributes[];

    const transferred = prepareOutputTransferredAttributes({
      raw: flatXyTriangle(),
      sourceFaceAttributes,
      recomputeNormals: false,
    });

    expect(transferred).toEqual({
      vertices: [
        { uvSets: [], normal: new Vector3(0, 0, 1) },
        { uvSets: [], normal: new Vector3(0, 0, 1) },
        { uvSets: [], normal: new Vector3(0, 0, 1) },
      ],
    });
  });
});
