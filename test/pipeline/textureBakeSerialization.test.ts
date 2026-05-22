import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import {
  deserializeTextureBakeBatchRunInput,
  serializeTextureBakeBatchRunInput,
} from '../../src/pipeline/textureBakeSerialization';
import type { TextureBakeBatchRunInput } from '../../src/texture/bakeBatch';
import type { SourceMaterial } from '../../src/texture/types';

const sampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'nearest' as const };

function material(overrides: Partial<SourceMaterial> = {}): SourceMaterial {
  return {
    name: 'mat',
    baseColorFactor: [1, 1, 1, 1],
    textureSlots: [],
    alphaMode: 'OPAQUE',
    alphaCutoff: 0.5,
    doubleSided: false,
    emissiveFactor: [0, 0, 0],
    metallicFactor: 1,
    roughnessFactor: 1,
    normalScale: 1,
    occlusionStrength: 1,
    ...overrides,
  };
}

describe('texture bake serialization', () => {
  it('round-trips the preserved output normal scale and prepared output normals in worker payloads', () => {
    const input: TextureBakeBatchRunInput = {
      source: {
        rawMesh: {
          positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
          faces: [[0, 1, 2]],
        },
        faceAttributes: [{
          materialId: 0,
          uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
        }],
        materials: [material({
          normalScale: 1,
          textureSlots: [{
            slot: 'normal',
            texCoord: 0,
            sampler,
            hasImage: true,
            image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) },
          }],
        })],
      },
      outputRawMesh: {
        positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
        faces: [[0, 1, 2]],
      },
      outputFaceIds: [0],
      history: [],
      atlas: {
        textureSize: 4,
        padding: 0,
        faceUvs: [[new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)]],
        facePixelTriangles: [[[0, 0], [4, 0], [0, 4]]],
      },
      activeSlots: ['normal'],
      outputNormalScale: 0.375,
      outputVertexNormals: [
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(0, 0, 1),
      ],
      batches: [{ id: 0, startFaceIndex: 0, endFaceIndex: 1, sampleCount: 8 }],
      totalFaces: 1,
      totalSamples: 8,
    };

    const serialized = serializeTextureBakeBatchRunInput(input);
    const deserialized = deserializeTextureBakeBatchRunInput(serialized);

    expect(serialized.outputNormalScale).toBe(0.375);
    expect(serialized.outputVertexNormals).toBeInstanceOf(Float32Array);
    expect(Array.from(serialized.outputVertexNormals ?? [])).toEqual([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    expect(deserialized.outputNormalScale).toBe(0.375);
    expect(deserialized.outputVertexNormals?.map((normal) => normal.toArray())).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
  });
});
