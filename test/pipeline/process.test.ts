import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { TransferredMeshAttributes } from '../../src/simplification/attributes';
import type { RawMesh, RawSimplificationResult } from '../../src/simplification/types';
import type { AtlasOptions } from '../../src/texture/atlas';
import type { SourceMaterial, TexturedRawMesh } from '../../src/texture/types';
import {
  bakeTextureForSimplifiedGeometry,
  combinedMaterialSettings,
  hasBakeableTextureData,
  hasImageBackedTextureBakeData,
  hasImageBackedTextureTransferData,
  hasPreservableMaterialData,
  hasUsableTextureData,
  processGeometryOnly,
  processTextured,
} from '../../src/pipeline/process';

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

function squareRawMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(1, 1, 0),
      new Vector3(0, 1, 0),
    ],
    faces: [[0, 1, 2], [0, 2, 3]],
  };
}

function texturedSquare(): TexturedRawMesh {
  const rawMesh = squareRawMesh();
  return {
    rawMesh,
    faceAttributes: [
      { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1)] }] },
      { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 1)] }] },
    ],
    materials: [material({
      name: 'mat',
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
      alphaMode: 'MASK',
      alphaCutoff: 0.4,
      doubleSided: true,
      metallicFactor: 0,
      roughnessFactor: 0.42,
    })],
  };
}

describe('processing pipeline', () => {
  afterEach(() => {
    vi.doUnmock('../../src/texture/bake');
    vi.doUnmock('../../src/texture/atlas');
  });

  it('detects whether materials have image-backed standard texture data for baking', () => {
    expect(hasUsableTextureData([material({ name: 'default', baseColorFactor: [1, 1, 1, 1] })])).toBe(false);
    expect(hasUsableTextureData([material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })])).toBe(false);
    expect(hasUsableTextureData(texturedSquare().materials)).toBe(true);
    expect(hasUsableTextureData([material({
      textureSlots: [{
        slot: 'normal',
        texCoord: 0,
        sampler,
        hasImage: true,
        image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) },
      }],
    })])).toBe(true);
  });

  it('separates preservable factor data from image-backed transfer and bake data', () => {
    const factorOnly = material({
      name: 'red',
      baseColorFactor: [1, 0, 0, 1],
      metallicFactor: 0,
      roughnessFactor: 0.42,
    });

    expect(hasPreservableMaterialData(factorOnly)).toBe(true);
    expect(hasImageBackedTextureTransferData(factorOnly)).toBe(false);
    expect(hasImageBackedTextureBakeData(factorOnly)).toBe(false);
    expect(hasUsableTextureData([factorOnly])).toBe(false);

    const textured = texturedSquare().materials[0]!;
    expect(hasPreservableMaterialData(textured)).toBe(true);
    expect(hasImageBackedTextureTransferData(textured)).toBe(true);
    expect(hasImageBackedTextureBakeData(textured)).toBe(true);
  });

  it('checks bakeability using standard material texture UV requirements', () => {
    const source = texturedSquare();
    delete source.materials[0]!.baseColorTexture;
    source.materials[0]!.textureSlots = [{
      slot: 'normal',
      texCoord: 0,
      sampler,
      hasImage: true,
      image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) },
    }];
    expect(hasBakeableTextureData(source)).toBe(true);

    source.materials[0]!.textureSlots[0]!.texCoord = 1;
    expect(hasBakeableTextureData(source)).toBe(false);
  });

  it('does not classify factor-only sources as texture bakeable', () => {
    const source = texturedSquare();
    delete source.materials[0]!.baseColorTexture;
    source.materials[0]!.textureSlots = [];
    source.materials[0]!.baseColorFactor = [1, 0, 0, 1];

    expect(hasBakeableTextureData(source)).toBe(false);
  });

  it('combines scalar material settings for a baked output material', () => {
    expect(combinedMaterialSettings(texturedSquare().materials)).toEqual({
      alphaMode: 'MASK',
      alphaCutoff: 0.4,
      doubleSided: true,
      metallicFactor: 0,
      roughnessFactor: 0.42,
      emissiveFactor: [0, 0, 0],
      normalScale: 1,
      occlusionStrength: 1,
    });
  });

  it('runs geometry-only simplification and forwards progress', () => {
    const progressIterations: number[] = [];
    const result = processGeometryOnly(squareRawMesh(), {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    }, (progress) => progressIterations.push(progress.iteration));

    expect(result.raw.rawMesh.faces.length).toBeLessThanOrEqual(1);
    expect(result.raw.stats.inputFaces).toBe(2);
    expect(result.elapsedSeconds).toBeGreaterThanOrEqual(0);
    expect(progressIterations.length).toBeGreaterThan(0);
  });

  it('runs textured simplification without mutating source material filters', async () => {
    const source = texturedSquare();
    const result = await processTextured(source, {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });

    expect(source.materials[0]!.baseColorTexture!.sampler.filter).toBe('nearest');
    expect(source.materials[0]!.textureSlots[0]!.sampler.filter).toBe('nearest');
    expect(result.raw.rawMesh.faces.length).toBeGreaterThan(0);
    expect(result.baked.stats.filledPixels).toBeGreaterThan(0);
    expect(result.baked.atlas.islandCount).toBeGreaterThan(0);
    expect(result.materialSettings.metallicFactor).toBe(0);
  });

  it('passes texture slot images and requested sampler filters to the texture bake path', async () => {
    vi.resetModules();
    const capturedSources: TexturedRawMesh[] = [];
    vi.doMock('../../src/texture/bake', () => ({
      bakeStandardMaterialTextures: vi.fn(async (options: {
        source: TexturedRawMesh;
        outputRawMesh: RawMesh;
      }) => {
        capturedSources.push(options.source);
        return {
          image: { width: 1, height: 1, data: new Uint8Array([1, 2, 3, 4]) },
          additionalTextures: [],
          atlas: {
            textureSize: 1,
            padding: 0,
            faceUvs: options.outputRawMesh.faces.map(() => [
              new Vector2(0, 0),
              new Vector2(1, 0),
              new Vector2(0, 1),
            ]),
            facePixelTriangles: options.outputRawMesh.faces.map(() => [[0, 0], [0, 0], [0, 0]]),
          },
          stats: { filledPixels: 1, mappedPixels: 1, unmappedPixels: 0 },
        };
      }),
    }));
    const { processTextured: mockedProcessTextured } = await import('../../src/pipeline/process');
    const source = texturedSquare();
    const slotImage = { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) };
    source.materials[0]!.textureSlots = [{
      slot: 'baseColor',
      texCoord: 0,
      sampler,
      hasImage: true,
      image: slotImage,
    }];

    await mockedProcessTextured(source, {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });

    expect(capturedSources[0]!.materials[0]!.textureSlots[0]!.image).toBe(slotImage);
    expect(capturedSources[0]!.materials[0]!.textureSlots[0]!.sampler.filter).toBe('linear');
    expect(source.materials[0]!.textureSlots[0]!.sampler.filter).toBe('nearest');
  });

  it('forwards the max output texture allocation cap to the texture bake path', async () => {
    vi.resetModules();
    const capturedOptions: Array<{ maxOutputTextureBytes?: number }> = [];
    vi.doMock('../../src/texture/bake', () => ({
      bakeStandardMaterialTextures: vi.fn(async (options: {
        outputRawMesh: RawMesh;
        maxOutputTextureBytes?: number;
      }) => {
        capturedOptions.push(options);
        return {
          image: { width: 1, height: 1, data: new Uint8Array([1, 2, 3, 4]) },
          additionalTextures: [],
          atlas: {
            textureSize: 1,
            padding: 0,
            faceUvs: options.outputRawMesh.faces.map(() => [
              new Vector2(0, 0),
              new Vector2(1, 0),
              new Vector2(0, 1),
            ]),
            facePixelTriangles: options.outputRawMesh.faces.map(() => [[0, 0], [0, 0], [0, 0]]),
          },
          stats: { filledPixels: 1, mappedPixels: 1, unmappedPixels: 0 },
        };
      }),
    }));
    const {
      bakeTextureForSimplifiedGeometry: mockedBakeTextureForSimplifiedGeometry,
      processGeometryOnly: mockedProcessGeometryOnly,
    } = await import('../../src/pipeline/process');
    const source = texturedSquare();
    const simplified = mockedProcessGeometryOnly(source.rawMesh, {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });

    await mockedBakeTextureForSimplifiedGeometry(source, simplified.raw, {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    }, undefined, { maxOutputTextureBytes: 1234 });

    expect(capturedOptions[0]!.maxOutputTextureBytes).toBe(1234);
  });

  it('processes normal-only source materials with standard texture data', async () => {
    const source = texturedSquare();
    delete source.materials[0]!.baseColorTexture;
    source.materials[0]!.baseColorFactor = [1, 1, 1, 1];
    source.materials[0]!.textureSlots = [{
      slot: 'normal',
      texCoord: 0,
      sampler,
      hasImage: true,
      image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) },
    }];

    const result = await processTextured(source, {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });

    expect(result.baked.additionalTextures.map((texture) => texture.slot)).toContain('normal');
  });

  it('does not pass transferred vertex UVs into atlas generation during deferred baking', async () => {
    vi.resetModules();
    const capturedAtlasOptions: AtlasOptions[] = [];
    vi.doMock('../../src/texture/atlas', async () => {
      const actual = await vi.importActual<typeof import('../../src/texture/atlas')>('../../src/texture/atlas');
      return {
        ...actual,
        createInjectiveAtlas: vi.fn(async (_mesh: RawMesh, options: AtlasOptions) => {
          capturedAtlasOptions.push(options);
          return {
            textureSize: options.textureSize,
            padding: options.padding,
            faceUvs: [],
            facePixelTriangles: [],
            islandCount: 0,
          };
        }),
      };
    });
    const { bakeTextureForSimplifiedGeometry: mockedBakeTextureForSimplifiedGeometry } = await import('../../src/pipeline/process');
    const rawMesh: RawMesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    };
    const source: TexturedRawMesh = {
      rawMesh,
      faceAttributes: [{ materialId: 0, uvSets: [] }],
      materials: [material({
        baseColorTexture: {
          image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) },
          sampler,
          texCoord: 0,
        },
      })],
    };
    const raw: RawSimplificationResult = {
      rawMesh,
      outputFaceIds: [0],
      history: [],
      stats: {
        inputVertices: 3,
        inputFaces: 1,
        outputVertices: 3,
        outputFaces: 1,
        physicalEdges: 3,
        virtualEdges: 0,
        collapses: 0,
        stoppedReason: 'target-reached',
      },
    };
    const transferredAttributes: TransferredMeshAttributes = {
      vertices: [
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.25, 0.25) }] },
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.75, 0.25) }] },
        { uvSets: [{ texCoord: 0, uv: new Vector2(0.25, 0.75) }] },
      ],
    };

    await mockedBakeTextureForSimplifiedGeometry(source, raw, {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 16,
      texturePadding: 1,
      textureFilter: 'nearest',
    }, undefined, transferredAttributes, {
      batchRunner: async (input) => {
        expect(input.batches).toEqual([]);
      },
    });

    expect(capturedAtlasOptions).toEqual([{ textureSize: 16, padding: 1 }]);
  });

  it('bakes repeatedly from a captured geometry simplification without re-running simplification', async () => {
    const source = texturedSquare();
    const progressIterations: number[] = [];
    const simplified = processGeometryOnly(source.rawMesh, {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    }, (progress) => progressIterations.push(progress.iteration));

    const bake32 = await bakeTextureForSimplifiedGeometry(source, simplified.raw, {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    const bake64 = await bakeTextureForSimplifiedGeometry(source, simplified.raw, {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 64,
      texturePadding: 2,
      textureFilter: 'linear',
    });

    expect(progressIterations.length).toBeGreaterThan(0);
    expect(simplified.raw.history.length).toBeGreaterThan(0);
    expect(bake32.raw.stats.outputFaces).toBe(simplified.raw.stats.outputFaces);
    expect(bake64.raw.stats.outputFaces).toBe(simplified.raw.stats.outputFaces);
    expect(bake32.baked.image.width).toBe(32);
    expect(bake64.baked.image.width).toBe(64);
    expect(source.materials[0]!.baseColorTexture!.sampler.filter).toBe('nearest');
  });

  it('reports geometry completion before texture baking progress', async () => {
    const events: string[] = [];

    await processTextured(texturedSquare(), {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    }, undefined, (raw) => {
      events.push(`geometry:${raw.stats.outputFaces}`);
    }, (progress) => {
      events.push(progress.stage);
    });

    expect(events[0]).toMatch(/^geometry:/);
    expect(events).toContain('atlas-created');
    expect(events).toContain('resampling');
    expect(events).toContain('complete');
  });

  it('passes a custom texture bake batch runner to the bake path', async () => {
    const source = texturedSquare();
    const simplified = processGeometryOnly(source.rawMesh, {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    let runnerCalls = 0;

    await bakeTextureForSimplifiedGeometry(source, simplified.raw, {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    }, undefined, {
      batchRunner: async (input, onBatchResult, onProgress) => {
        runnerCalls += 1;
        const { runTextureBakeBatchesSerial } = await import('../../src/texture/bakeBatch');
        await runTextureBakeBatchesSerial(input, onBatchResult, onProgress);
      },
    });

    expect(runnerCalls).toBe(1);
  });

  it('rejects texture transfer when no standard texture data is present', async () => {
    const source = texturedSquare();
    delete source.materials[0]!.baseColorTexture;
    source.materials[0]!.textureSlots = [];
    source.materials[0]!.baseColorFactor = [1, 1, 1, 1];
    await expect(processTextured(source, {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    })).rejects.toThrow(/no image-backed standard material texture data/i);
  });
});
