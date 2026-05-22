import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { RawMesh } from '../../src/simplification/types';
import type { ProcessingOptions } from '../../src/pipeline/options';
import type { SourceMaterial } from '../../src/texture/types';
import {
  aggregatePrimitiveStats,
  allocatePrimitiveProcessingOptions,
  bakePrimitiveTextures,
  processPrimitiveGeometries,
  type ScenePrimitiveProcessingEntry,
} from '../../src/pipeline/sceneProcessing';
import { processPrimitiveGeometriesStaged } from '../../src/pipeline/primitiveStagedProcessing';

const sampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'linear' as const };

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

function strip(faceCount: number, scale = 1): RawMesh {
  const positions: Vector3[] = [];
  for (let i = 0; i < faceCount + 2; i += 1) positions.push(new Vector3(i * scale, (i % 2) * scale, 0));
  const faces: RawMesh['faces'] = [];
  for (let i = 0; i < faceCount; i += 1) faces.push([i, i + 1, i + 2]);
  return { positions, faces };
}

function entry(id: string, faceCount: number, scale = 1): ScenePrimitiveProcessingEntry {
  const rawMesh = strip(faceCount, scale);
  return { id, label: id, rawMesh, bakeable: false, hasTexturedMaterial: false };
}

function texturedEntry(id: string): ScenePrimitiveProcessingEntry {
  const rawMesh = strip(1);
  return {
    id,
    label: id,
    rawMesh,
    bakeable: true,
    hasTexturedMaterial: true,
    texturedRawMesh: {
      rawMesh,
      faceAttributes: [{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
      }],
      materials: [material({
        name: 'mat',
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: {
          image: { width: 2, height: 2, data: new Uint8Array(16).fill(255) },
          sampler,
          texCoord: 0,
        },
        textureSlots: [{ slot: 'baseColor', texCoord: 0, sampler, hasImage: true }],
      })],
    },
  };
}

function authoredNormalEntry(id: string): ScenePrimitiveProcessingEntry {
  const rawMesh: RawMesh = {
    positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
    faces: [[0, 1, 2]],
  };
  return {
    id,
    label: id,
    rawMesh,
    bakeable: false,
    hasTexturedMaterial: false,
    texturedRawMesh: {
      rawMesh,
      faceAttributes: [{
        materialId: 0,
        uvSets: [],
        normalCorners: [new Vector3(1, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 0, 0)],
      }],
      materials: [],
    },
  };
}

function texturedSquareEntry(id: string): ScenePrimitiveProcessingEntry {
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
    id,
    label: id,
    rawMesh,
    bakeable: true,
    hasTexturedMaterial: true,
    texturedRawMesh: {
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
      })],
    },
  };
}

function options(target: ProcessingOptions['target']): ProcessingOptions {
  return {
    target,
    primitiveGrouping: 'material-parent',
    virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    weldVertices: true,
    recomputeNormals: true,
    transferTextures: false,
    textureSize: 32,
    texturePadding: 1,
    textureFilter: 'linear',
  };
}

describe('scene primitive processing', () => {
  it('allocates absolute target faces proportionally and deterministically', () => {
    const allocated = allocatePrimitiveProcessingOptions([
      entry('small', 2),
      entry('large', 6),
      entry('empty', 0),
    ], options({ kind: 'faces', targetFaceCount: 4 }));

    expect(allocated.map((item) => [item.entry.id, item.options.target])).toEqual([
      ['small', { kind: 'faces', targetFaceCount: 1 }],
      ['large', { kind: 'faces', targetFaceCount: 3 }],
      ['empty', { kind: 'faces', targetFaceCount: 0 }],
    ]);
  });

  it('keeps ratio targets as per-primitive ratios', () => {
    const allocated = allocatePrimitiveProcessingOptions([
      entry('a', 2),
      entry('b', 6),
    ], options({ kind: 'ratio', ratio: 0.5 }));

    expect(allocated.map((item) => item.options.target)).toEqual([
      { kind: 'ratio', ratio: 0.5 },
      { kind: 'ratio', ratio: 0.5 },
    ]);
  });

  it('aggregates per-primitive simplification stats', () => {
    const processed = processPrimitiveGeometries([
      entry('a', 2),
      entry('b', 2),
    ], options({ kind: 'faces', targetFaceCount: 2 }));
    const stats = aggregatePrimitiveStats(processed.entries.map((item) => item.geometry.raw.stats));

    expect(stats.inputFaces).toBe(4);
    expect(stats.outputFaces).toBeLessThanOrEqual(2);
    expect(stats.inputVertices).toBe(8);
    expect(stats.outputVertices).toBeGreaterThan(0);
    expect(stats.stoppedReason).toBe('target-reached');
  });

  it('passes virtual-edge mode to each primitive simplification', () => {
    const processed = processPrimitiveGeometries([
      entry('small', 2, 1),
      entry('large', 2, 10),
    ], {
      ...options({ kind: 'ratio', ratio: 1 }),
      virtualEdges: { mode: 'auto-local-radius' },
    });

    const modes = processed.entries.map((item) => item.geometry.raw.stats.virtualEdgeDiagnostics?.mode);

    expect(modes).toEqual(['auto-local-radius', 'auto-local-radius']);
  });

  it('emits primitive lifecycle callbacks around geometry simplification and texture baking', async () => {
    const entries = [texturedEntry('first'), texturedEntry('second')];
    const events: string[] = [];
    const processingOptions = {
      ...options({ kind: 'ratio', ratio: 1 }),
      transferTextures: true,
    };
    const geometry = processPrimitiveGeometries(entries, processingOptions, {
      onGeometryStart: (id) => events.push(`geometry-start:${id}`),
      onGeometryComplete: (id, result) => events.push(`geometry-complete:${id}:${result.rawMesh.faces.length}`),
    });

    await bakePrimitiveTextures(geometry, processingOptions, {
      onBakeStart: (id) => events.push(`bake-start:${id}`),
      onBakeComplete: (id, result) => events.push(`bake-complete:${id}:${result.stats.unmappedPixels}`),
    });

    expect(events).toEqual([
      'geometry-start:first',
      'geometry-complete:first:1',
      'geometry-start:second',
      'geometry-complete:second:1',
      'bake-start:first',
      'bake-complete:first:0',
      'bake-start:second',
      'bake-complete:second:0',
    ]);
  });

  it('captures collapse history for deferred primitive texture baking', async () => {
    const entry = texturedSquareEntry('square');
    const processingOptions = options({ kind: 'faces', targetFaceCount: 1 });
    const geometry = processPrimitiveGeometries([entry], processingOptions);

    const raw = geometry.entries[0]!.geometry.raw;
    const baked = await bakePrimitiveTextures(geometry, { ...processingOptions, transferTextures: true });

    expect(raw.stats.collapses).toBeGreaterThan(0);
    expect(raw.history).toHaveLength(raw.stats.collapses);
    expect(baked.entries[0]!.baked?.baked.stats.unmappedPixels).toBe(0);
  });

  it('passes a custom texture bake batch runner to primitive texture baking', async () => {
    const entries = [texturedEntry('first'), texturedEntry('second')];
    const processingOptions = {
      ...options({ kind: 'ratio', ratio: 1 }),
      transferTextures: true,
    };
    const geometry = processPrimitiveGeometries(entries, processingOptions);
    let runnerCalls = 0;

    await bakePrimitiveTextures(geometry, processingOptions, {}, {
      batchRunner: async (input, onBatchResult, onProgress) => {
        runnerCalls += 1;
        const { runTextureBakeBatchesSerial } = await import('../../src/texture/bakeBatch');
        await runTextureBakeBatchesSerial(input, onBatchResult, onProgress);
      },
    });

    expect(runnerCalls).toBe(2);
  });

  it('captures collapse history for primitive attribute transfer without baking', () => {
    const entry = {
      ...texturedSquareEntry('square'),
      bakeable: false,
      requiresAttributeTransfer: true,
    };
    const geometry = processPrimitiveGeometries([entry], options({ kind: 'faces', targetFaceCount: 1 }));
    const raw = geometry.entries[0]!.geometry.raw;

    expect(raw.stats.collapses).toBeGreaterThan(0);
    expect(raw.history).toHaveLength(raw.stats.collapses);
  });

  it('preserves authored primitive normals when recomputeNormals is false', () => {
    const result = processPrimitiveGeometries([authoredNormalEntry('normal-only')], {
      ...options({ kind: 'ratio', ratio: 1 }),
      recomputeNormals: false,
    });

    expect(result.entries[0]!.transferredAttributes?.vertices.map((vertex) => vertex.normal)).toEqual([
      new Vector3(1, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(1, 0, 0),
    ]);
  });

  it('captures staged history for every simplified primitive while transferring attributes conditionally', () => {
    const result = processPrimitiveGeometriesStaged([
      {
        entry: entry('plain', 2),
        options: options({ kind: 'faces', targetFaceCount: 1 }),
      },
      {
        entry: {
          ...texturedSquareEntry('transfer'),
          bakeable: false,
          requiresAttributeTransfer: true,
        },
        options: options({ kind: 'faces', targetFaceCount: 1 }),
      },
    ]);
    const plain = result.entries.find((item) => item.id === 'plain')!;
    const transfer = result.entries.find((item) => item.id === 'transfer')!;

    expect(plain.raw.stats.collapses).toBeGreaterThan(0);
    expect(plain.raw.history).toHaveLength(plain.raw.stats.collapses);
    expect(plain.transferredAttributes).toBeUndefined();
    expect(transfer.raw.stats.collapses).toBeGreaterThan(0);
    expect(transfer.raw.history).toHaveLength(transfer.raw.stats.collapses);
    expect(transfer.transferredAttributes?.vertices).toHaveLength(transfer.raw.rawMesh.positions.length);
  });

  it('processes all primitives in strict global stage order', () => {
    const events: string[] = [];
    const result = processPrimitiveGeometriesStaged([
      {
        entry: entry('first', 2),
        options: options({ kind: 'faces', targetFaceCount: 1 }),
      },
      {
        entry: {
          ...texturedSquareEntry('second'),
          bakeable: false,
          requiresAttributeTransfer: true,
        },
        options: options({ kind: 'faces', targetFaceCount: 1 }),
      },
    ], {
      onVirtualEdgeStageStart: () => events.push('stage:virtual-start'),
      onVirtualEdgePrimitiveComplete: (id) => events.push(`virtual-complete:${id}`),
      onSimplificationStageStart: () => events.push('stage:simplify-start'),
      onSimplificationPrimitiveComplete: (id) => events.push(`simplify-complete:${id}`),
      onAttributeTransferStageStart: () => events.push('stage:attribute-start'),
      onAttributeTransferPrimitiveComplete: (id) => events.push(`attribute-complete:${id}`),
    });

    expect(events.indexOf('stage:simplify-start')).toBeGreaterThan(events.indexOf('virtual-complete:second'));
    expect(events.indexOf('stage:attribute-start')).toBeGreaterThan(events.indexOf('simplify-complete:second'));
    expect(result.entries).toHaveLength(2);
    const processed = result.entries[1]!;
    expect(processed.transferredAttributes?.vertices).toHaveLength(processed.raw.rawMesh.positions.length);
  });
});
