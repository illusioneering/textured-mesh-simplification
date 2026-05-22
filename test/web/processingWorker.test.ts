import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { RawMesh, RawSimplificationResult } from '../../src/simplification/types';
import type { ProcessingOptions } from '../../src/pipeline/options';
import type { GeometryProcessingResult, TexturedProcessingResult } from '../../src/pipeline/process';
import type {
  PrimitiveGeometryProcessingResult,
  PrimitiveSceneProcessingResult,
  ProcessablePrimitiveEntry,
} from '../../src/pipeline/sceneProcessing';
import type { SourceMaterial, TexturedRawMesh } from '../../src/texture/types';
import {
  serializePrimitiveEntries,
  serializeRawMesh,
  serializeTexturedRawMesh,
} from '../../src/web/serialization';
import type { WorkerRequestMessage, WorkerResponseMessage } from '../../src/web/workerProtocol';

const BROWSER_MAX_BAKE_OUTPUT_TEXTURE_BYTES = 256 * 1024 * 1024;

type WorkerScopeStub = {
  postMessage: ReturnType<typeof vi.fn>;
  onmessage: ((event: MessageEvent<WorkerRequestMessage>) => void) | null;
  posted: WorkerResponseMessage[];
};

type SerializablePrimitiveEntry = ProcessablePrimitiveEntry & {
  label: string;
  meshOrdinal: number;
  texturedRawMesh: TexturedRawMesh;
  hasTexturedMaterial: boolean;
};

const originalSelfDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'self');

const sampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'nearest' as const };

function rawMesh(): RawMesh {
  return {
    positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
    faces: [[0, 1, 2]],
  };
}

function rawResult(mesh: RawMesh = rawMesh()): RawSimplificationResult {
  return {
    rawMesh: mesh,
    outputFaceIds: [0],
    history: [],
    stats: {
      inputVertices: mesh.positions.length,
      inputFaces: mesh.faces.length,
      outputVertices: mesh.positions.length,
      outputFaces: mesh.faces.length,
      physicalEdges: 3,
      virtualEdges: 0,
      collapses: 0,
      stoppedReason: 'target-reached',
    },
  };
}

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

function texturedRawMesh(): TexturedRawMesh {
  const mesh = rawMesh();
  return {
    rawMesh: mesh,
    faceAttributes: [{
      materialId: 0,
      uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
    }],
    materials: [material({
      baseColorTexture: {
        image: { width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) },
        sampler,
        texCoord: 0,
      },
      textureSlots: [{ slot: 'baseColor', texCoord: 0, sampler, hasImage: true }],
    })],
  };
}

function primitiveEntry(): SerializablePrimitiveEntry {
  const source = texturedRawMesh();
  return {
    id: 'primitive-0',
    label: 'Primitive 0',
    meshOrdinal: 0,
    rawMesh: source.rawMesh,
    texturedRawMesh: source,
    bakeable: true,
    hasTexturedMaterial: true,
  };
}

function processingOptions(): ProcessingOptions {
  return {
    target: { kind: 'ratio', ratio: 1 },
    primitiveGrouping: 'material-parent',
    virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    weldVertices: true,
    recomputeNormals: true,
    transferTextures: true,
    textureSize: 32,
    texturePadding: 1,
    textureFilter: 'linear',
  };
}

function bakedResult(raw: RawSimplificationResult = rawResult()): TexturedProcessingResult {
  return {
    raw,
    elapsedSeconds: 0,
    materialSettings: {
      alphaMode: 'OPAQUE',
      alphaCutoff: 0.5,
      doubleSided: false,
      metallicFactor: 1,
      roughnessFactor: 1,
      emissiveFactor: [0, 0, 0],
      normalScale: 1,
      occlusionStrength: 1,
    },
    baked: {
      image: { width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) },
      additionalTextures: [],
      atlas: {
        textureSize: 1,
        padding: 0,
        faceUvs: [[new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)]],
        facePixelTriangles: [[[0, 0], [0, 0], [0, 0]]],
      },
      stats: { filledPixels: 1, mappedPixels: 1, unmappedPixels: 0 },
    },
  };
}

function installWorkerScope(): WorkerScopeStub {
  const scope: WorkerScopeStub = {
    onmessage: null,
    posted: [],
    postMessage: vi.fn((message: WorkerResponseMessage) => {
      scope.posted.push(message);
    }),
  };
  Object.defineProperty(globalThis, 'self', {
    configurable: true,
    value: scope,
  });
  return scope;
}

function restoreSelf(): void {
  if (originalSelfDescriptor) {
    Object.defineProperty(globalThis, 'self', originalSelfDescriptor);
  } else {
    delete (globalThis as Record<string, unknown>).self;
  }
}

async function waitForDone(scope: WorkerScopeStub, id: string): Promise<void> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (scope.posted.some((message) => message.type === 'done' && message.id === id)) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error(`Timed out waiting for done message ${id}.`);
}

async function setupWorker(): Promise<{
  scope: WorkerScopeStub;
  processGeometryOnly: ReturnType<typeof vi.fn>;
  bakeTextureForSimplifiedGeometry: ReturnType<typeof vi.fn>;
  processPrimitiveGeometriesInBrowserWorkers: ReturnType<typeof vi.fn>;
  bakePrimitiveTextures: ReturnType<typeof vi.fn>;
  batchRunner: unknown;
}> {
  vi.resetModules();
  const scope = installWorkerScope();
  const batchRunner = vi.fn();
  const processGeometryOnly = vi.fn((mesh: RawMesh): GeometryProcessingResult => ({
    raw: rawResult(mesh),
    elapsedSeconds: 0,
  }));
  const bakeTextureForSimplifiedGeometry = vi.fn(async (
    _source: TexturedRawMesh,
    raw: RawSimplificationResult,
  ): Promise<TexturedProcessingResult> => bakedResult(raw));
  const processPrimitiveGeometriesInBrowserWorkers = vi.fn(async (
    entries: ProcessablePrimitiveEntry[],
  ): Promise<PrimitiveGeometryProcessingResult> => {
    const processed = entries.map((entry) => {
      const raw = rawResult(entry.rawMesh);
      return {
        id: entry.id,
        source: entry,
        geometry: { raw, elapsedSeconds: 0 },
        raw,
      };
    });
    return {
      entries: processed,
      stats: rawResult(entries[0]?.rawMesh ?? rawMesh()).stats,
      elapsedSeconds: 0,
    };
  });
  const bakePrimitiveTextures = vi.fn(async (
    geometryResult: PrimitiveGeometryProcessingResult,
  ): Promise<PrimitiveSceneProcessingResult> => ({
    entries: geometryResult.entries,
    stats: geometryResult.stats,
    elapsedSeconds: 0,
  }));

  vi.doMock('../../src/pipeline/process', () => ({
    processGeometryOnly,
    bakeTextureForSimplifiedGeometry,
  }));
  vi.doMock('../../src/web/primitiveWorkerPool', () => ({
    processPrimitiveGeometriesInBrowserWorkers,
  }));
  vi.doMock('../../src/pipeline/sceneProcessing', () => ({
    bakePrimitiveTextures,
  }));
  vi.doMock('../../src/web/textureBakeWorkerPool', () => ({
    createBrowserTextureBakeBatchRunner: vi.fn(() => batchRunner),
  }));

  await import('../../src/web/processingWorker');

  return {
    scope,
    processGeometryOnly,
    bakeTextureForSimplifiedGeometry,
    processPrimitiveGeometriesInBrowserWorkers,
    bakePrimitiveTextures,
    batchRunner,
  };
}

describe('processingWorker', () => {
  afterEach(() => {
    vi.doUnmock('../../src/pipeline/process');
    vi.doUnmock('../../src/web/primitiveWorkerPool');
    vi.doUnmock('../../src/pipeline/sceneProcessing');
    vi.doUnmock('../../src/web/textureBakeWorkerPool');
    vi.resetModules();
    vi.restoreAllMocks();
    restoreSelf();
  });

  it('passes the browser output texture allocation cap to single-mesh baking', async () => {
    const { scope, processGeometryOnly, bakeTextureForSimplifiedGeometry, batchRunner } = await setupWorker();
    const source = texturedRawMesh();
    const options = processingOptions();

    scope.onmessage?.({
      data: {
        type: 'simplify',
        id: 'simplify-single',
        input: { kind: 'geometry', rawMesh: serializeRawMesh(source.rawMesh) },
        options,
      },
    } as MessageEvent<WorkerRequestMessage>);
    await waitForDone(scope, 'simplify-single');

    expect(processGeometryOnly).toHaveBeenCalledWith(
      expect.anything(),
      options,
      expect.any(Function),
      expect.objectContaining({
        onVirtualEdgeProgress: expect.any(Function),
      }),
    );

    scope.onmessage?.({
      data: {
        type: 'bake',
        id: 'bake-single',
        source: serializeTexturedRawMesh(source),
        options: processingOptions(),
      },
    } as MessageEvent<WorkerRequestMessage>);
    await waitForDone(scope, 'bake-single');

    expect(bakeTextureForSimplifiedGeometry).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.anything(),
      expect.any(Function),
      expect.objectContaining({
        batchRunner,
        maxOutputTextureBytes: BROWSER_MAX_BAKE_OUTPUT_TEXTURE_BYTES,
      }),
    );
  });

  it('passes the browser output texture allocation cap to primitive baking', async () => {
    const { scope, processPrimitiveGeometriesInBrowserWorkers, bakePrimitiveTextures, batchRunner } = await setupWorker();
    const entry = primitiveEntry();
    const options = processingOptions();

    scope.onmessage?.({
      data: {
        type: 'simplify',
        id: 'simplify-primitives',
        input: {
          kind: 'primitives',
          entries: serializePrimitiveEntries([entry], { includeImages: false }),
        },
        options,
      },
    } as MessageEvent<WorkerRequestMessage>);
    await waitForDone(scope, 'simplify-primitives');

    expect(processPrimitiveGeometriesInBrowserWorkers).toHaveBeenCalledWith(
      expect.any(Array),
      options,
      expect.objectContaining({
        onAggregateProgress: expect.any(Function),
      }),
    );

    scope.onmessage?.({
      data: {
        type: 'bake',
        id: 'bake-primitives',
        source: {
          kind: 'primitives',
          entries: serializePrimitiveEntries([entry]),
        },
        options: processingOptions(),
      },
    } as MessageEvent<WorkerRequestMessage>);
    await waitForDone(scope, 'bake-primitives');

    expect(bakePrimitiveTextures).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.anything(),
      expect.objectContaining({
        batchRunner,
        maxOutputTextureBytes: BROWSER_MAX_BAKE_OUTPUT_TEXTURE_BYTES,
      }),
    );
  });
});
