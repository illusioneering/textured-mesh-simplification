import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { TransferredMeshAttributes } from '../../src/simplification/attributes';
import type { CollapseHistoryRecord, RawMesh } from '../../src/simplification/types';
import { deriveWatlasInputFaceUvs } from '../../src/texture/watlasInputUvs';
import type { SourceMaterial, TexturedRawMesh } from '../../src/texture/types';

const sampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'linear' as const };

function material(): SourceMaterial {
  return {
    name: 'material',
    baseColorFactor: [1, 1, 1, 1],
    textureSlots: [{ slot: 'baseColor', texCoord: 0, sampler, hasImage: true }],
    alphaMode: 'OPAQUE',
    alphaCutoff: 0.5,
    doubleSided: false,
    emissiveFactor: [0, 0, 0],
    metallicFactor: 1,
    roughnessFactor: 1,
    normalScale: 1,
    occlusionStrength: 1,
  };
}

function rawMesh(): RawMesh {
  return {
    positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
    faces: [[0, 1, 2]],
  };
}

function sourceWithUvSets(uvSets: TexturedRawMesh['faceAttributes'][number]['uvSets']): TexturedRawMesh {
  return {
    rawMesh: rawMesh(),
    faceAttributes: [{ materialId: 0, uvSets }],
    materials: [material()],
  };
}

describe('deriveWatlasInputFaceUvs', () => {
  it('uses transferred welded vertex TEXCOORD_0 as atlas input UVs when complete', () => {
    const source = sourceWithUvSets([]);
    const transferredAttributes: TransferredMeshAttributes = {
      vertices: [
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.2, 0.3) }] },
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.8, 0.3) }] },
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.2, 0.9) }] },
      ],
    };

    const result = deriveWatlasInputFaceUvs({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
      transferredAttributes,
    });

    expect(result).toHaveLength(1);
    expect(result?.[0]).toEqual([
      new Vector2(0.2, 0.3),
      new Vector2(0.8, 0.3),
      new Vector2(0.2, 0.9),
    ]);
    expect(result?.[0]?.[0]).not.toBe(transferredAttributes.vertices[0]?.uvSets[0]?.uv);
  });

  it('falls back to source UV derivation when transferred TEXCOORD_0 is incomplete', () => {
    const source = sourceWithUvSets([
      { texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] },
    ]);
    const transferredAttributes: TransferredMeshAttributes = {
      vertices: [
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.2, 0.3) }] },
        { uvSets: [] },
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.2, 0.9) }] },
      ],
    };

    const result = deriveWatlasInputFaceUvs({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
      transferredAttributes,
    });

    expect(result).toHaveLength(1);
    expect(result?.[0]).toEqual([
      new Vector2(0, 0),
      new Vector2(1, 0),
      new Vector2(0, 1),
    ]);
  });

  it('prefers transferred TEXCOORD_0 when it is available', () => {
    const source = sourceWithUvSets([
      { texCoord: 1, uvs: [new Vector2(0.1, 0.1), new Vector2(0.9, 0.1), new Vector2(0.1, 0.9)] },
      { texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] },
    ]);

    const result = deriveWatlasInputFaceUvs({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
    });

    expect(result).toHaveLength(1);
    expect(result?.[0]?.[1].x).toBe(1);
    expect(result?.[0]?.[2].y).toBe(1);
  });

  it('falls back to the lowest common TEXCOORD when TEXCOORD_0 is absent', () => {
    const source = sourceWithUvSets([
      { texCoord: 3, uvs: [new Vector2(0.3, 0.3), new Vector2(0.8, 0.3), new Vector2(0.3, 0.8)] },
      { texCoord: 1, uvs: [new Vector2(0.1, 0.1), new Vector2(0.7, 0.1), new Vector2(0.1, 0.7)] },
    ]);

    const result = deriveWatlasInputFaceUvs({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
    });

    expect(result?.[0]?.[1].x).toBeCloseTo(0.7);
    expect(result?.[0]?.[2].y).toBeCloseTo(0.7);
  });

  it('returns undefined when no TEXCOORD is present on every mapped output corner', () => {
    const source = sourceWithUvSets([]);

    const result = deriveWatlasInputFaceUvs({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
    });

    expect(result).toBeUndefined();
  });

  it('uses collapse history barycentrics for output corners', () => {
    const source = sourceWithUvSets([
      { texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] },
    ]);
    const outputRawMesh: RawMesh = {
      positions: [new Vector3(0.5, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    };
    const history: CollapseHistoryRecord[] = [{
      keepVertexId: 1,
      removedVertexId: 0,
      beforeFaces: [{
        faceId: 0,
        vertices: [0, 1, 2],
        positions: [
          new Vector3(0, 0, 0),
          new Vector3(1, 0, 0),
          new Vector3(0, 1, 0),
        ],
      }],
      afterFaceIds: [0],
    }];

    const result = deriveWatlasInputFaceUvs({
      source,
      outputRawMesh,
      outputFaceIds: [0],
      history,
    });

    expect(result?.[0]?.[0].x).toBeCloseTo(0.5);
    expect(result?.[0]?.[0].y).toBeCloseTo(0);
  });
});
