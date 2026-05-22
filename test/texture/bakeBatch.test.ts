import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { Vector2, Vector3, Vector4 } from 'three';
import type { RawMesh } from '../../src/simplification/types';
import {
  createEmptyBakeImages,
  mergeTextureBakeBatchResult,
  mergeTextureBakeBatchResults,
  planTextureBakeBatches,
  readPixel,
  runTextureBakeBatchGroup,
  runTextureBakeBatchesSerial,
  type TextureBakeBatchResult,
} from '../../src/texture/bakeBatch';
import { rasterizeAtlasTriangle } from '../../src/texture/rasterize';
import {
  computeAuthoredTangentFrame,
  computeFaceTangentFrame,
  normalRgbToVector,
  tangentNormalToWorld,
  worldNormalToTangent,
} from '../../src/texture/tangentSpace';
import type { AtlasLayout, Rgba, SourceMaterial, StandardBakedTextureSlot, TexturedRawMesh } from '../../src/texture/types';

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

function squareMesh(): RawMesh {
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

function squareAtlas(): AtlasLayout {
  return {
    textureSize: 8,
    padding: 0,
    faceUvs: [
      [new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1)],
      [new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 1)],
    ],
    facePixelTriangles: [
      [[0, 0], [8, 0], [8, 8]],
      [[0, 0], [8, 8], [0, 8]],
    ],
    islandCount: 1,
  };
}

function texturedSquare(): TexturedRawMesh {
  const rawMesh = squareMesh();
  return {
    rawMesh,
    faceAttributes: [
      { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1)] }] },
      { materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 1)] }] },
    ],
    materials: [material({
      name: 'textured',
      baseColorTexture: {
        image: { width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) },
        sampler,
        texCoord: 0,
      },
      textureSlots: [{
        slot: 'normal',
        texCoord: 0,
        sampler,
        hasImage: true,
        image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) },
      }],
    })],
  };
}

function singleTriangleAtlas(faceUvs: [Vector2, Vector2, Vector2]): AtlasLayout {
  return {
    textureSize: 2,
    padding: 0,
    faceUvs: [faceUvs],
    facePixelTriangles: [[[0, 0], [2, 0], [0, 2]]],
    islandCount: 1,
  };
}

function texturedTriangle(options: {
  slot: StandardBakedTextureSlot;
  color: Rgba;
  normalScale?: number;
  normalMapYScale?: 1 | -1;
  uvs?: [Vector2, Vector2, Vector2];
  normalCorners?: [Vector3, Vector3, Vector3];
  tangentCorners?: [Vector4, Vector4, Vector4];
}): TexturedRawMesh {
  const source: TexturedRawMesh = {
    rawMesh: {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    },
    faceAttributes: [{
      materialId: 0,
      uvSets: [{
        texCoord: 0,
        uvs: options.uvs ?? [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)],
      }],
      ...(options.normalCorners ? { normalCorners: options.normalCorners } : {}),
      ...(options.tangentCorners ? { tangentCorners: options.tangentCorners } : {}),
    }],
    materials: [material({
      normalScale: options.normalScale ?? 1,
      textureSlots: [{
        slot: options.slot,
        texCoord: 0,
        sampler,
        hasImage: true,
        image: { width: 1, height: 1, data: new Uint8Array(options.color) },
      }],
    })],
  };
  if (options.normalMapYScale !== undefined) {
    (source.faceAttributes[0] as typeof source.faceAttributes[0] & { normalMapYScale: 1 | -1 }).normalMapYScale = options.normalMapYScale;
  }
  return source;
}

async function bakeSingleTriangleAdditionalTexture(
  source: TexturedRawMesh,
  slot: StandardBakedTextureSlot,
  atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)]),
  outputNormalScale = source.materials[0]?.normalScale ?? 1,
): Promise<Uint8Array> {
  const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 16 });
  const results: TextureBakeBatchResult[] = [];
  await runTextureBakeBatchesSerial({
    source,
    outputRawMesh: source.rawMesh,
    outputFaceIds: [0],
    history: [],
    atlas,
    activeSlots: [slot],
    batches: plan.batches,
    totalFaces: plan.totalFaces,
    totalSamples: plan.totalSamples,
    outputNormalScale,
  }, (result) => results.push(result));
  const data = results[0]?.additionalTextures[0]?.data;
  if (!data) throw new Error('Expected a baked texture result.');
  return data;
}

function applyNormalScale(color: Rgba, normalScale: number): Vector3 {
  const normal = normalRgbToVector(color);
  return new Vector3(normal.x * normalScale, normal.y * normalScale, normal.z).normalize();
}

function applyNormalScale2D(color: Rgba, xScale: number, yScale: number): Vector3 {
  const normal = normalRgbToVector(color);
  return new Vector3(normal.x * xScale, normal.y * yScale, normal.z).normalize();
}

describe('texture bake batches', () => {
  it('plans contiguous face batches from atlas sample counts', () => {
    const plan = planTextureBakeBatches(squareAtlas(), { targetSamplesPerBatch: 10 });

    expect(plan.totalFaces).toBe(2);
    expect(plan.totalSamples).toBeGreaterThan(0);
    expect(plan.batches.length).toBeGreaterThan(0);
    expect(plan.batches[0]?.startFaceIndex).toBe(0);
    expect(plan.batches.at(-1)?.endFaceIndex).toBe(2);
    for (let index = 0; index < plan.batches.length; index += 1) {
      const batch = plan.batches[index]!;
      expect(batch.id).toBe(index);
      if (index > 0) {
        expect(batch.startFaceIndex).toBe(plan.batches[index - 1]!.endFaceIndex);
      }
      expect(batch.endFaceIndex).toBeGreaterThan(batch.startFaceIndex);
      expect(batch.sampleCount).toBeGreaterThan(0);
    }
  });

  it('runs planned batches serially and aggregates progress', async () => {
    const source = texturedSquare();
    const atlas = squareAtlas();
    const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 10 });
    const progress: Array<{ processedSamples: number }> = [];
    const results: TextureBakeBatchResult[] = [];

    await runTextureBakeBatchesSerial({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      atlas,
      activeSlots: ['normal'],
      batches: plan.batches,
      totalFaces: plan.totalFaces,
      totalSamples: plan.totalSamples,
    }, (result) => results.push(result), (update) => progress.push({ processedSamples: update.processedSamples }));

    expect(results).toHaveLength(plan.batches.length);
    expect(results.reduce((total, result) => total + result.mappedPixels, 0)).toBe(plan.totalSamples);
    expect(progress.at(-1)?.processedSamples).toBe(plan.totalSamples);
  });

  it('runs planned batches serially through callbacks without exposing collected results', async () => {
    const source = texturedSquare();
    const atlas = squareAtlas();
    const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 10 });
    const callbackResults: TextureBakeBatchResult[] = [];

    const returnedResults = await runTextureBakeBatchesSerial({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      atlas,
      activeSlots: ['normal'],
      batches: plan.batches,
      totalFaces: plan.totalFaces,
      totalSamples: plan.totalSamples,
    }, (result) => callbackResults.push(result));

    expect(returnedResults).toBeUndefined();
    expect(callbackResults.map((result) => result.batchId)).toEqual(plan.batches.map((batch) => batch.id));
    expect(callbackResults.reduce((total, result) => total + result.mappedPixels, 0)).toBe(plan.totalSamples);
  });

  it('opts serial batch execution out of returned result collection', () => {
    const sourceText = readFileSync(new URL('../../src/texture/bakeBatch.ts', import.meta.url), 'utf8');

    expect(sourceText).toContain(
      'runTextureBakeBatchGroup(input, onBatchResult, onProgress, { collectResults: false })',
    );
  });

  it('returns batch results while also delivering callback results from a batch group', async () => {
    const source = texturedSquare();
    const atlas = squareAtlas();
    const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 10 });
    const callbackResults: TextureBakeBatchResult[] = [];
    const progress: Array<{ processedSamples: number }> = [];

    const returnedResults = await runTextureBakeBatchGroup({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      atlas,
      activeSlots: ['normal'],
      batches: plan.batches,
      totalFaces: plan.totalFaces,
      totalSamples: plan.totalSamples,
    }, (result) => callbackResults.push(result), (update) => progress.push({ processedSamples: update.processedSamples }));

    expect(callbackResults.map((result) => result.batchId)).toEqual(plan.batches.map((batch) => batch.id));
    expect(returnedResults.map((result) => result.batchId)).toEqual(callbackResults.map((result) => result.batchId));
    expect(returnedResults.reduce((total, result) => total + result.mappedPixels, 0)).toBe(plan.totalSamples);
    expect(progress.at(-1)?.processedSamples).toBe(plan.totalSamples);
  });

  it('can deliver callback results without retaining returned batch results', async () => {
    const source = texturedSquare();
    const atlas = squareAtlas();
    const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 10 });
    const callbackResults: TextureBakeBatchResult[] = [];

    const returnedResults = await runTextureBakeBatchGroup({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0, 1],
      history: [],
      atlas,
      activeSlots: ['normal'],
      batches: plan.batches,
      totalFaces: plan.totalFaces,
      totalSamples: plan.totalSamples,
    }, (result) => callbackResults.push(result), undefined, { collectResults: false });

    expect(callbackResults.map((result) => result.batchId)).toEqual(plan.batches.map((batch) => batch.id));
    expect(callbackResults.reduce((total, result) => total + result.mappedPixels, 0)).toBe(plan.totalSamples);
    expect(returnedResults).toEqual([]);
  });

  it('merges a single batch result into existing bake images', () => {
    const images = createEmptyBakeImages(1, ['normal']);
    const result: TextureBakeBatchResult = {
      batchId: 0,
      processedFaces: 1,
      sampleCount: 1,
      mappedPixels: 1,
      unmappedPixels: 0,
      pixelIndices: new Uint32Array([0]),
      baseColor: new Uint8Array([12, 34, 56, 255]),
      additionalTextures: [{ slot: 'normal', data: new Uint8Array([128, 128, 255, 255]) }],
    };

    const stats = mergeTextureBakeBatchResult(images, result);

    expect(readPixel(images.image, 0, 0)).toEqual([12, 34, 56, 255]);
    expect(readPixel(images.additionalTextures[0]!.image, 0, 0)).toEqual([128, 128, 255, 255]);
    expect(images.filled[0]).toBe(1);
    expect(stats).toEqual({ filledPixels: 1, mappedPixels: 1, unmappedPixels: 0 });
  });

  it('merges sparse batch results in deterministic batch order', () => {
    const images = createEmptyBakeImages(1, ['normal']);
    const results: TextureBakeBatchResult[] = [
      {
        batchId: 1,
        processedFaces: 1,
        sampleCount: 1,
        mappedPixels: 1,
        unmappedPixels: 0,
        pixelIndices: new Uint32Array([0]),
        baseColor: new Uint8Array([255, 0, 0, 255]),
        additionalTextures: [{ slot: 'normal', data: new Uint8Array([128, 128, 255, 255]) }],
      },
      {
        batchId: 0,
        processedFaces: 1,
        sampleCount: 1,
        mappedPixels: 1,
        unmappedPixels: 0,
        pixelIndices: new Uint32Array([0]),
        baseColor: new Uint8Array([0, 255, 0, 255]),
        additionalTextures: [{ slot: 'normal', data: new Uint8Array([1, 2, 3, 4]) }],
      },
    ];

    const stats = mergeTextureBakeBatchResults(images, results);

    expect(readPixel(images.image, 0, 0)).toEqual([255, 0, 0, 255]);
    expect(readPixel(images.additionalTextures[0]!.image, 0, 0)).toEqual([128, 128, 255, 255]);
    expect(stats).toEqual({ filledPixels: 1, mappedPixels: 2, unmappedPixels: 0 });
  });

  it('keeps neutral normal-map samples neutral when source and output tangent frames match', async () => {
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: [128, 128, 255, 255],
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
    }), 'normal');

    expect(Array.from(baked.slice(0, 4))).toEqual([128, 128, 255, 255]);
  });

  it('uses provided output vertex normals when rebaking normal-map samples', async () => {
    const source = texturedTriangle({
      slot: 'normal',
      color: [128, 128, 255, 255],
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
    });
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)]);
    const outputVertexNormals = [
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
    ];
    const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 16 });
    const results: TextureBakeBatchResult[] = [];

    await runTextureBakeBatchesSerial({
      source,
      outputRawMesh: source.rawMesh,
      outputFaceIds: [0],
      history: [],
      atlas,
      activeSlots: ['normal'],
      outputVertexNormals,
      batches: plan.batches,
      totalFaces: plan.totalFaces,
      totalSamples: plan.totalSamples,
    }, (result) => results.push(result));

    const baked = results[0]?.additionalTextures[0]?.data;
    if (!baked) throw new Error('Expected a baked normal texture result.');
    const firstSample = rasterizeAtlasTriangle(0, atlas.facePixelTriangles[0]!, atlas.textureSize)[0]!;
    const outputFrame = computeFaceTangentFrame({
      positions: source.rawMesh.positions as [Vector3, Vector3, Vector3],
      uvs: atlas.faceUvs[0]!,
      normal: outputVertexNormals[0]!,
    })!;
    const expected = worldNormalToTangent(new Vector3(0, 0, 1), outputFrame);
    const viewerApplied = normalRgbToVector([baked[0]!, baked[1]!, baked[2]!, baked[3]!]);

    expect(firstSample.barycentric[0]).toBeGreaterThan(0);
    expect(viewerApplied.x).toBeCloseTo(expected.x, 1);
    expect(viewerApplied.y).toBeCloseTo(expected.y, 1);
    expect(viewerApplied.z).toBeCloseTo(expected.z, 1);
    expect(baked[1]).toBeLessThan(80);
  });

  it('rebakes normal-map samples into the output atlas tangent frame', async () => {
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(0, 1), new Vector2(-1, 0)]);
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: [255, 128, 128, 255],
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
    }), 'normal', atlas);

    expect(baked[0]).toBeGreaterThanOrEqual(127);
    expect(baked[0]).toBeLessThanOrEqual(128);
    expect(baked[1]).toBeGreaterThan(250);
    expect(baked[2]).toBeGreaterThanOrEqual(127);
    expect(baked[2]).toBeLessThanOrEqual(128);
    expect(baked[3]).toBe(255);
  });

  it('compensates rebaked normal-map RGB for preserved material normalScale', async () => {
    const normalScale = 0.5;
    const sourceColor: Rgba = [255, 128, 255, 255];
    const sourceNormals = [
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
    ] as [Vector3, Vector3, Vector3];
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(0, 1), new Vector2(-1, 0)]);
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: sourceColor,
      normalScale,
      normalCorners: sourceNormals,
    }), 'normal', atlas);

    const sourceFrame = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)],
      normal: sourceNormals[0],
    })!;
    const outputFrame = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: atlas.faceUvs[0]!,
      normal: new Vector3(0, 0, 1),
    })!;
    const expected = worldNormalToTangent(tangentNormalToWorld(applyNormalScale(sourceColor, normalScale), sourceFrame), outputFrame);
    const viewerApplied = applyNormalScale([baked[0]!, baked[1]!, baked[2]!, baked[3]!], normalScale);
    expect(viewerApplied.x).toBeCloseTo(expected.x, 1);
    expect(viewerApplied.y).toBeCloseTo(expected.y, 1);
    expect(viewerApplied.z).toBeCloseTo(expected.z, 1);
  });

  it('uses separate source and output normal scales for mixed-material normal-map bakes', async () => {
    const outputNormalScale = 0.5;
    const sourceNormalScale = 1;
    const sourceColor: Rgba = [255, 128, 255, 255];
    const sourceNormals = [
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
    ] as [Vector3, Vector3, Vector3];
    const source = texturedTriangle({
      slot: 'normal',
      color: sourceColor,
      normalScale: outputNormalScale,
      normalCorners: sourceNormals,
    });
    source.materials.push(material({
      name: 'strong-normal',
      normalScale: sourceNormalScale,
      textureSlots: [{
        slot: 'normal',
        texCoord: 0,
        sampler,
        hasImage: true,
        image: { width: 1, height: 1, data: new Uint8Array(sourceColor) },
      }],
    }));
    source.rawMesh.faces.push([0, 1, 2]);
    source.faceAttributes.push({
      materialId: 1,
      uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
      normalCorners: sourceNormals,
    });
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(0, 1), new Vector2(-1, 0)]);
    const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 16 });
    const results: TextureBakeBatchResult[] = [];
    await runTextureBakeBatchesSerial({
      source,
      outputRawMesh: {
        positions: source.rawMesh.positions,
        faces: [[0, 1, 2]],
      },
      outputFaceIds: [1],
      history: [],
      atlas,
      activeSlots: ['normal'],
      batches: plan.batches,
      totalFaces: plan.totalFaces,
      totalSamples: plan.totalSamples,
      outputNormalScale,
    }, (result) => results.push(result));
    const baked = results[0]?.additionalTextures[0]?.data;
    if (!baked) throw new Error('Expected a baked normal texture result.');

    const sourceFrame = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)],
      normal: sourceNormals[0],
    })!;
    const outputFrame = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: atlas.faceUvs[0]!,
      normal: new Vector3(0, 0, 1),
    })!;
    const expected = worldNormalToTangent(tangentNormalToWorld(applyNormalScale(sourceColor, sourceNormalScale), sourceFrame), outputFrame);
    const viewerApplied = applyNormalScale([baked[0]!, baked[1]!, baked[2]!, baked[3]!], outputNormalScale);
    expect(viewerApplied.x).toBeCloseTo(expected.x, 1);
    expect(viewerApplied.y).toBeCloseTo(expected.y, 1);
    expect(viewerApplied.z).toBeCloseTo(expected.z, 1);
  });

  it('handles zero preserved material normalScale deterministically', async () => {
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(0, 1), new Vector2(-1, 0)]);
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: [255, 128, 255, 255],
      normalScale: 0,
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
    }), 'normal', atlas, 0);

    expect(Array.from(baked.slice(0, 4))).toEqual([128, 128, 255, 255]);
  });

  it('compensates normalScale when falling back from a degenerate tangent frame', async () => {
    const sourceNormalScale = 1;
    const outputNormalScale = 0.5;
    const sourceColor: Rgba = [200, 128, 240, 255];
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(0, 0), new Vector2(0, 0)]);
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: sourceColor,
      normalScale: sourceNormalScale,
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
    }), 'normal', atlas, outputNormalScale);

    const expected = applyNormalScale(sourceColor, sourceNormalScale);
    const viewerApplied = applyNormalScale([baked[0]!, baked[1]!, baked[2]!, baked[3]!], outputNormalScale);
    expect(viewerApplied.x).toBeCloseTo(expected.x, 1);
    expect(viewerApplied.y).toBeCloseTo(expected.y, 1);
    expect(viewerApplied.z).toBeCloseTo(expected.z, 1);
  });

  it('preserves source normal-map Y scale when rebaking into tangent-bearing output', async () => {
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: [128, 255, 255, 255],
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
      normalMapYScale: -1,
    }), 'normal');

    expect(baked[0]).toBeGreaterThanOrEqual(127);
    expect(baked[0]).toBeLessThanOrEqual(128);
    expect(baked[1]).toBeLessThan(64);
    expect(baked[2]).toBeGreaterThan(200);
    expect(baked[2]).toBeLessThan(230);
    expect(baked[3]).toBe(255);
  });

  it('uses authored source tangent handedness when rebaking normal-map samples', async () => {
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: [128, 255, 255, 255],
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
      tangentCorners: [new Vector4(1, 0, 0, -1), new Vector4(1, 0, 0, -1), new Vector4(1, 0, 0, -1)],
    }), 'normal');

    expect(baked[0]).toBeGreaterThanOrEqual(127);
    expect(baked[0]).toBeLessThanOrEqual(128);
    expect(baked[1]).toBeLessThan(64);
    expect(baked[2]).toBeGreaterThan(200);
    expect(baked[2]).toBeLessThan(230);
    expect(baked[3]).toBe(255);
  });

  it('interpolates authored bitangents instead of tangent handedness when rebaking normal maps', async () => {
    const sourceColor: Rgba = [128, 255, 255, 255];
    const tangents = [
      new Vector4(1, 0, 0, -1),
      new Vector4(1, 0, 0, 1),
      new Vector4(0, 1, 0, 1),
    ] as [Vector4, Vector4, Vector4];
    const normals = [
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
    ] as [Vector3, Vector3, Vector3];

    const source = texturedTriangle({
      slot: 'normal',
      color: sourceColor,
      normalCorners: normals,
      tangentCorners: tangents,
    });
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)]);
    const baked = await bakeSingleTriangleAdditionalTexture(source, 'normal', atlas);
    const firstSample = rasterizeAtlasTriangle(0, atlas.facePixelTriangles[0]!, atlas.textureSize)[0]!;

    const sourceFrame = computeAuthoredTangentFrame({
      normals,
      tangents,
      barycentric: firstSample.barycentric,
    })!;
    const outputFrame = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)],
      normal: new Vector3(0, 0, 1),
    })!;
    const expected = worldNormalToTangent(tangentNormalToWorld(applyNormalScale2D(sourceColor, 1, 1), sourceFrame), outputFrame);
    const viewerApplied = normalRgbToVector([baked[0]!, baked[1]!, baked[2]!, baked[3]!]);

    expect(viewerApplied.x).toBeCloseTo(expected.x, 1);
    expect(viewerApplied.y).toBeCloseTo(expected.y, 1);
    expect(viewerApplied.z).toBeCloseTo(expected.z, 1);
  });

  it('uses authored source normals when rebaking normal-map samples', async () => {
    const tiltedNormals = [
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
      new Vector3(0, 1, 1).normalize(),
    ] as [Vector3, Vector3, Vector3];
    const withAuthoredNormals = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: [128, 128, 255, 255],
      normalCorners: tiltedNormals,
    }), 'normal');
    const withFallbackNormals = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'normal',
      color: [128, 128, 255, 255],
    }), 'normal');

    expect(withFallbackNormals[1]).toBe(128);
    expect(withFallbackNormals[2]).toBe(255);
    expect(withAuthoredNormals[1]).toBeGreaterThan(withFallbackNormals[1]!);
    expect(withAuthoredNormals[2]).toBeLessThan(withFallbackNormals[2]!);
  });

  it('continues to resample non-normal texture slots as colors', async () => {
    const atlas = singleTriangleAtlas([new Vector2(0, 0), new Vector2(0, 1), new Vector2(-1, 0)]);
    const baked = await bakeSingleTriangleAdditionalTexture(texturedTriangle({
      slot: 'emissive',
      color: [10, 20, 30, 255],
      normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)],
    }), 'emissive', atlas);

    expect(Array.from(baked.slice(0, 4))).toEqual([10, 20, 30, 255]);
  });
});
