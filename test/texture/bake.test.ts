import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { TransferredMeshAttributes } from '../../src/simplification/attributes';
import { bakeStandardMaterialTextures, estimateBakeOutputTextureBytes } from '../../src/texture/bake';
import { simplifyRawMesh } from '../../src/simplification/simplify';
import type { RawMesh } from '../../src/simplification/types';
import type { TextureBakeBatchResult, TextureBakeBatchRunInput } from '../../src/texture/bakeBatch';
import type { SourceMaterial, TexturedRawMesh } from '../../src/texture/types';

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

function squareSource(): TexturedRawMesh {
  const rawMesh: RawMesh = {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(1, 1, 0),
      new Vector3(0, 1, 0),
    ],
    faces: [[0, 1, 2], [0, 2, 3]],
  };
  return {
    rawMesh,
    faceAttributes: [{ materialId: 0, uvSets: [] }, { materialId: 0, uvSets: [] }],
    materials: [material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })],
  };
}

function colorForBatchId(batchId: number): [number, number, number, number] {
  return [batchId * 40 + 20, 255 - batchId * 40, 0, 255];
}

function resultForBatchId(batchId: number): TextureBakeBatchResult {
  return {
    batchId,
    processedFaces: 1,
    sampleCount: 1,
    mappedPixels: 1,
    unmappedPixels: 0,
    pixelIndices: new Uint32Array([0]),
    baseColor: new Uint8Array(colorForBatchId(batchId)),
    additionalTextures: [],
  };
}

describe('bakeStandardMaterialTextures', () => {
  it('estimates output atlas image and filled-mask bytes', () => {
    expect(estimateBakeOutputTextureBytes(4096, ['normal', 'metallicRoughness', 'occlusion', 'emissive'])).toBe(
      (4096 * 4096 * 4 * 5) + (4096 * 4096),
    );
  });

  it('bakes constant material colors into covered texels', async () => {
    const source: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{ materialId: 0, uvSets: [] }],
      materials: [material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })],
    };
    const result = await bakeStandardMaterialTextures({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
      textureSize: 32,
      padding: 2,
      gutterPasses: 1,
    });
    expect(result.stats.filledPixels).toBeGreaterThan(0);
    expect(result.stats.mappedPixels).toBe(result.stats.filledPixels);
    let sawRed = false;
    for (let i = 0; i < result.image.data.length; i += 4) {
      if (result.image.data[i + 3]! > 0) {
        expect(result.image.data[i]).toBe(255);
        expect(result.image.data[i + 1]).toBe(0);
        expect(result.image.data[i + 2]).toBe(0);
        sawRed = true;
        break;
      }
    }
    expect(sawRed).toBe(true);
  });

  it('bakes adjacent output faces with generated atlas UVs', async () => {
    const rawMesh: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2], [0, 2, 3]],
    };
    const source: TexturedRawMesh = {
      rawMesh,
      faceAttributes: [{ materialId: 0, uvSets: [] }, { materialId: 0, uvSets: [] }],
      materials: [material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })],
    };

    const baked = await bakeStandardMaterialTextures({
      source,
      outputRawMesh: rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      textureSize: 64,
      padding: 2,
      gutterPasses: 0,
    });

    expect(baked.atlas.faceUvs).toHaveLength(2);
    expect(baked.atlas.facePixelTriangles).toHaveLength(2);
    expect(baked.atlas.islandCount).toBeGreaterThan(0);
    expect(baked.stats.unmappedPixels).toBe(0);
  });

  it('bakes standard material textures into the same generated atlas', async () => {
    const rawMesh: RawMesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    };
    const source: TexturedRawMesh = {
      rawMesh,
      faceAttributes: [{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0.25, 0.25), new Vector2(0.25, 0.25), new Vector2(0.25, 0.25)] }],
      }],
      materials: [material({
        name: 'full',
        baseColorTexture: {
          image: { width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) },
          sampler,
          texCoord: 0,
        },
        textureSlots: [
          { slot: 'baseColor', texCoord: 0, sampler, hasImage: true },
          { slot: 'normal', texCoord: 0, sampler, hasImage: true, image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) } },
          { slot: 'metallicRoughness', texCoord: 0, sampler, hasImage: true, image: { width: 1, height: 1, data: new Uint8Array([0, 127, 255, 255]) } },
          { slot: 'occlusion', texCoord: 0, sampler, hasImage: true, image: { width: 1, height: 1, data: new Uint8Array([200, 200, 200, 255]) } },
          { slot: 'emissive', texCoord: 0, sampler, hasImage: true, image: { width: 1, height: 1, data: new Uint8Array([0, 0, 255, 255]) } },
        ],
      })],
    };

    const baked = await bakeStandardMaterialTextures({
      source,
      outputRawMesh: rawMesh,
      outputFaceIds: [0],
      history: [],
      textureSize: 16,
      padding: 1,
      gutterPasses: 0,
    });

    expect(baked.additionalTextures.map((texture) => texture.slot).sort()).toEqual([
      'emissive',
      'metallicRoughness',
      'normal',
      'occlusion',
    ]);
    for (const texture of baked.additionalTextures) {
      expect(texture.image.width).toBe(baked.image.width);
      expect(texture.image.height).toBe(baked.image.height);
      expect(texture.image.data).toHaveLength(baked.image.data.length);
    }

    const firstFilled = Array.from({ length: baked.image.data.length / 4 }, (_value, index) => index)
      .find((pixel) => baked.image.data[pixel * 4 + 3]! > 0);
    expect(firstFilled).not.toBeUndefined();
    const sampledBySlot = new Map(baked.additionalTextures.map((texture) => [
      texture.slot,
      Array.from(texture.image.data.slice(firstFilled! * 4, firstFilled! * 4 + 4)),
    ]));
    expect(sampledBySlot.get('normal')).toEqual([128, 128, 255, 255]);
    expect(sampledBySlot.get('metallicRoughness')).toEqual([0, 127, 255, 255]);
    expect(sampledBySlot.get('occlusion')).toEqual([200, 200, 200, 255]);
    expect(sampledBySlot.get('emissive')).toEqual([0, 0, 255, 255]);
  });

  it('throws instead of writing transparent pixels when a mapped source face is missing', async () => {
    const source: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [],
      materials: [material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })],
    };
    await expect(bakeStandardMaterialTextures({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
      textureSize: 32,
      padding: 2,
    })).rejects.toThrow(/face attributes/i);
  });

  it('rejects atlas output that exceeds the configured allocation cap before baking pixels', async () => {
    const source: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{ materialId: 0, uvSets: [] }],
      materials: [material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })],
    };
    let batchRunnerCalled = false;

    await expect(bakeStandardMaterialTextures({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
      textureSize: 16,
      padding: 1,
      maxOutputTextureBytes: 1,
      batchRunner: async () => {
        batchRunnerCalled = true;
      },
    })).rejects.toThrow(/Texture atlas output would allocate/);
    expect(batchRunnerCalled).toBe(false);
  });

  it('reports atlas, resampling, gutter, and completion progress', async () => {
    const rawMesh: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2], [0, 2, 3]],
    };
    const source: TexturedRawMesh = {
      rawMesh,
      faceAttributes: [{ materialId: 0, uvSets: [] }, { materialId: 0, uvSets: [] }],
      materials: [material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })],
    };
    const stages: string[] = [];

    await bakeStandardMaterialTextures({
      source,
      outputRawMesh: rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      textureSize: 32,
      padding: 2,
      gutterPasses: 1,
      onProgress: (progress) => stages.push(progress.stage),
    });

    expect(stages[0]).toBe('atlas-created');
    expect(stages).toContain('resampling');
    expect(stages).toContain('dilating');
    expect(stages.at(-1)).toBe('complete');
  });

  it('merges streamed batch results in ascending batch order', async () => {
    const source = squareSource();
    let plannedBatchIds: number[] = [];

    const baked = await bakeStandardMaterialTextures({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      textureSize: 512,
      padding: 0,
      gutterPasses: 0,
      batchRunner: async (input, onBatchResult) => {
        plannedBatchIds = input.batches.map((batch) => batch.id);
        expect(plannedBatchIds.length).toBeGreaterThan(1);
        for (const batch of [...input.batches].reverse()) {
          onBatchResult(resultForBatchId(batch.id));
        }
      },
    });

    expect(Array.from(baked.image.data.slice(0, 4))).toEqual(colorForBatchId(plannedBatchIds.at(-1)!));
    expect(baked.stats).toEqual({ filledPixels: 1, mappedPixels: plannedBatchIds.length, unmappedPixels: 0 });
  });

  it('omits prepared output normals from batch input when no normal texture is active', async () => {
    const source = squareSource();
    const transferredAttributes: TransferredMeshAttributes = {
      vertices: source.rawMesh.positions.map(() => ({
        uvSets: [],
        normal: new Vector3(1, 0, 0),
      })),
    };
    let capturedInput: TextureBakeBatchRunInput | undefined;

    await bakeStandardMaterialTextures({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      transferredAttributes,
      textureSize: 64,
      padding: 0,
      gutterPasses: 0,
      batchRunner: async (input, onBatchResult) => {
        capturedInput = input;
        for (const batch of input.batches) onBatchResult(resultForBatchId(batch.id));
      },
    });

    expect(capturedInput?.activeSlots).toEqual([]);
    expect(capturedInput?.outputVertexNormals).toBeUndefined();
  });

  it('throws when a streaming batch runner omits trailing planned batches', async () => {
    const source = squareSource();

    await expect(bakeStandardMaterialTextures({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      textureSize: 512,
      padding: 0,
      gutterPasses: 0,
      batchRunner: async (input, onBatchResult) => {
        expect(input.batches.length).toBeGreaterThan(1);
        for (const batch of input.batches.slice(0, -1)) {
          onBatchResult(resultForBatchId(batch.id));
        }
      },
    })).rejects.toThrow(/missing.*batch/i);
  });

  it('throws when a streaming batch runner emits duplicate or out-of-range batch ids', async () => {
    const source = squareSource();
    const options = {
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      textureSize: 512,
      padding: 0,
      gutterPasses: 0,
    };

    await expect(bakeStandardMaterialTextures({
      ...options,
      batchRunner: async (input, onBatchResult) => {
        expect(input.batches.length).toBeGreaterThan(1);
        onBatchResult(resultForBatchId(input.batches[1]!.id));
        onBatchResult(resultForBatchId(input.batches[1]!.id));
      },
    })).rejects.toThrow(/duplicate.*batch/i);

    await expect(bakeStandardMaterialTextures({
      ...options,
      batchRunner: async (input, onBatchResult) => {
        onBatchResult(resultForBatchId(input.batches.length));
      },
    })).rejects.toThrow(/invalid.*batch/i);
  });

  it('runs an end-to-end simplify-with-history and bake workflow on a synthetic square', async () => {
    const rawMesh: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2], [0, 2, 3]],
    };
    const source: TexturedRawMesh = {
      rawMesh,
      faceAttributes: [
        { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1)] }] },
        { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 1)] }] },
      ],
      materials: [material({
        name: 'checker',
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: {
          image: {
          width: 2,
          height: 2,
          data: new Uint8Array([
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 255, 255,
          ]),
          },
          sampler,
          texCoord: 0,
        },
        textureSlots: [{ slot: 'baseColor', texCoord: 0, sampler, hasImage: true }],
      })],
    };
    const simplified = simplifyRawMesh(rawMesh, {
      targetFaceCount: 1,
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    });
    expect(simplified.rawMesh.faces.length).toBeGreaterThan(0);
    const baked = await bakeStandardMaterialTextures({
      source,
      outputRawMesh: simplified.rawMesh,
      outputFaceIds: simplified.outputFaceIds,
      history: simplified.history,
      textureSize: 64,
      padding: 2,
    });
    expect(baked.stats.filledPixels).toBeGreaterThan(0);
    expect(baked.atlas.islandCount).toBe(baked.atlas.faceUvs.length);
    expect(baked.stats.unmappedPixels).toBe(0);
    let coloredPixels = 0;
    for (let i = 0; i < baked.image.data.length; i += 4) {
      if (baked.image.data[i + 3]! > 0) coloredPixels += 1;
    }
    expect(coloredPixels).toBeGreaterThan(baked.stats.filledPixels);
  });
});
