import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { Vector2, Vector3, Vector4 } from 'three';
import type { CollapseHistoryRecord, RawMesh } from '../../src/simplification/types';
import {
  planTextureBakeBatches,
  type TextureBakeBatch,
  type TextureBakeBatchProgress,
  type TextureBakeBatchResult,
  type TextureBakeBatchRunInput,
} from '../../src/texture/bakeBatch';
import type { AtlasLayout, SourceFaceAttributes, SourceMaterial, TexturedRawMesh } from '../../src/texture/types';
import type { TextureBakeWorkerRequest, TextureBakeWorkerResponse } from '../../src/pipeline/textureBakeWorkerProtocol';
import {
  browserTextureBakeWorkerCount,
  createBrowserTextureBakeBatchRunner,
  estimateTextureBakeReplicatedPayloadBytes,
} from '../../src/web/textureBakeWorkerPool';

type WorkerBehavior = (worker: StubTextureBakeWorker, request: TextureBakeWorkerRequest) => void;

const originalWorkerDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
let workerBehavior: WorkerBehavior = () => {};

class StubTextureBakeWorker {
  static instances: StubTextureBakeWorker[] = [];

  readonly index: number;
  readonly requests: TextureBakeWorkerRequest[] = [];
  readonly transferLists: Array<Transferable[] | undefined> = [];
  terminated = false;
  onmessage: ((event: MessageEvent<TextureBakeWorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  constructor() {
    this.index = StubTextureBakeWorker.instances.length;
    StubTextureBakeWorker.instances.push(this);
  }

  postMessage(request: TextureBakeWorkerRequest, transfer?: Transferable[]): void {
    this.requests.push(request);
    this.transferLists.push(transfer);
    workerBehavior(this, request);
  }

  terminate(): void {
    this.terminated = true;
  }

  respond(response: TextureBakeWorkerResponse): void {
    this.onmessage?.({ data: response } as MessageEvent<TextureBakeWorkerResponse>);
  }
}

function installWorkerStub(): void {
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    writable: true,
    value: StubTextureBakeWorker as unknown as typeof Worker,
  });
}

function setHardwareConcurrency(value: number): void {
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { hardwareConcurrency: value },
  });
}

function restoreGlobalProperty(name: 'Worker' | 'navigator', descriptor: PropertyDescriptor | undefined): void {
  if (descriptor) {
    Object.defineProperty(globalThis, name, descriptor);
  } else {
    delete (globalThis as Record<string, unknown>)[name];
  }
}

const sampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'nearest' as const };
const mebibyte = 1024 * 1024;

function material(overrides: Partial<SourceMaterial> = {}): SourceMaterial {
  return {
    name: 'mat',
    baseColorFactor: [0.25, 0.5, 0.75, 1],
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

function squareAtlas(textureSize = 8): AtlasLayout {
  return {
    textureSize,
    padding: 0,
    faceUvs: [
      [new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1)],
      [new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 1)],
    ],
    facePixelTriangles: [
      [[0, 0], [textureSize, 0], [textureSize, textureSize]],
      [[0, 0], [textureSize, textureSize], [0, textureSize]],
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
      baseColorTexture: {
        image: { width: 1, height: 1, data: new Uint8Array([255, 64, 32, 255]) },
        sampler,
        texCoord: 0,
      },
    })],
  };
}

function serialBakeInput(): TextureBakeBatchRunInput {
  const source = texturedSquare();
  const atlas = squareAtlas();
  const plan = planTextureBakeBatches(atlas, { targetSamplesPerBatch: 10 });
  return {
    source,
    outputRawMesh: source.rawMesh,
    outputFaceIds: [0, 1],
    history: [],
    atlas,
    activeSlots: [],
    batches: plan.batches,
    totalFaces: plan.totalFaces,
    totalSamples: plan.totalSamples,
  };
}

function workerBakeInput(): TextureBakeBatchRunInput {
  const source = texturedSquare();
  return {
    source,
    outputRawMesh: source.rawMesh,
    outputFaceIds: [0, 1],
    history: [],
    atlas: squareAtlas(),
    activeSlots: [],
    batches: [
      { id: 0, startFaceIndex: 0, endFaceIndex: 1, sampleCount: 10 },
      { id: 1, startFaceIndex: 1, endFaceIndex: 2, sampleCount: 20 },
      { id: 2, startFaceIndex: 2, endFaceIndex: 3, sampleCount: 5 },
    ],
    totalFaces: 3,
    totalSamples: 35,
  };
}

function sparseVector3Array(length: number, values: Vector3[]): Vector3[] {
  const array = [...values];
  array.length = Math.max(length, values.length);
  return array;
}

function sparseVector2Triples(length: number): [Vector2, Vector2, Vector2][] {
  const array: [Vector2, Vector2, Vector2][] = [
    [new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1)],
    [new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 1)],
  ];
  array.length = Math.max(length, array.length);
  return array;
}

function sparsePixelTriangles(length: number): [[number, number], [number, number], [number, number]][] {
  const array: [[number, number], [number, number], [number, number]][] = [
    [[0, 0], [1, 0], [1, 1]],
    [[0, 0], [1, 1], [0, 1]],
  ];
  array.length = Math.max(length, array.length);
  return array;
}

function sparseFaces(length: number): RawMesh['faces'] {
  const array: RawMesh['faces'] = [[0, 1, 2], [0, 2, 3]];
  array.length = Math.max(length, array.length);
  return array;
}

function fakeImage(byteLength: number) {
  return {
    width: 1,
    height: 1,
    data: {
      0: 255,
      1: 128,
      2: 64,
      3: 255,
      length: byteLength,
      byteLength,
    } as unknown as Uint8Array,
  };
}

function inputWithEstimatedBytes(targetBytes: number): TextureBakeBatchRunInput {
  const input = scaledBakeInput({
    imageBytes: 0,
    batchCount: 16,
    totalSamples: 1_000,
  });
  const overheadBytes = estimateTextureBakeReplicatedPayloadBytes(input);
  input.source.materials[0]!.baseColorTexture!.image = fakeImage(targetBytes - overheadBytes);
  return input;
}

function scaledBakeInput(options: {
  sourceVertices?: number;
  sourceFaces?: number;
  outputVertices?: number;
  outputFaces?: number;
  historyRecords?: number;
  imageBytes?: number;
  activeSlots?: TextureBakeBatchRunInput['activeSlots'];
  batchCount?: number;
  totalSamples: number;
  outputVertexNormals?: boolean;
}): TextureBakeBatchRunInput {
  const sourceVertices = options.sourceVertices ?? 4;
  const sourceFaces = options.sourceFaces ?? 2;
  const outputVertices = options.outputVertices ?? sourceVertices;
  const outputFaces = options.outputFaces ?? sourceFaces;
  const activeSlots = options.activeSlots ?? [];
  const batchCount = options.batchCount ?? 8;
  const sourcePositions = sparseVector3Array(sourceVertices, [
    new Vector3(0, 0, 0),
    new Vector3(1, 0, 0),
    new Vector3(1, 1, 0),
    new Vector3(0, 1, 0),
  ]);
  const outputPositions = sparseVector3Array(outputVertices, [
    new Vector3(0, 0, 0),
    new Vector3(1, 0, 0),
    new Vector3(1, 1, 0),
    new Vector3(0, 1, 0),
  ]);
  const faceAttributes: SourceFaceAttributes[] = texturedSquare().faceAttributes.map((attributes) => ({
    ...attributes,
    normalCorners: [new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1)] as [Vector3, Vector3, Vector3],
    tangentCorners: [new Vector4(1, 0, 0, 1), new Vector4(1, 0, 0, 1), new Vector4(1, 0, 0, 1)] as [Vector4, Vector4, Vector4],
    normalMapYScale: 1,
  }));
  faceAttributes.length = Math.max(sourceFaces, faceAttributes.length);
  const image = fakeImage(options.imageBytes ?? 0);
  const history: CollapseHistoryRecord[] = Array.from({ length: options.historyRecords ?? 0 }, (_, index) => ({
    keepVertexId: index,
    removedVertexId: index + 1,
    beforeFaces: [{
      faceId: index,
      vertices: [0, 1, 2],
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 1, 0)],
    }],
    afterFaceIds: [index, index + 1],
  }));
  const batchSamples = Math.max(1, Math.ceil(options.totalSamples / batchCount));

  const input: TextureBakeBatchRunInput = {
    source: {
      rawMesh: { positions: sourcePositions, faces: sparseFaces(sourceFaces) },
      faceAttributes,
      materials: [material({
        baseColorTexture: { image, sampler, texCoord: 0 },
        textureSlots: activeSlots.map((slot) => ({
          slot,
          texCoord: 0,
          sampler,
          hasImage: true,
          image: fakeImage(1024),
        })),
      })],
    },
    outputRawMesh: { positions: outputPositions, faces: sparseFaces(outputFaces) },
    outputFaceIds: Array.from({ length: outputFaces }, (_, index) => index),
    history,
    atlas: {
      textureSize: 8,
      padding: 0,
      faceUvs: sparseVector2Triples(outputFaces),
      facePixelTriangles: sparsePixelTriangles(outputFaces),
      islandCount: 1,
    },
    activeSlots,
    batches: Array.from({ length: batchCount }, (_, index) => ({
      id: index,
      startFaceIndex: index < 2 ? index : 0,
      endFaceIndex: index < 1 ? index + 1 : 2,
      sampleCount: batchSamples,
    })),
    totalFaces: outputFaces,
    totalSamples: options.totalSamples,
  };
  if (options.outputVertexNormals) {
    input.outputVertexNormals = outputPositions.map(() => new Vector3(0, 0, 1));
  }
  return input;
}

function batchResult(batch: TextureBakeBatch): TextureBakeBatchResult {
  return {
    batchId: batch.id,
    processedFaces: batch.endFaceIndex - batch.startFaceIndex,
    sampleCount: batch.sampleCount,
    mappedPixels: batch.sampleCount,
    unmappedPixels: 0,
    pixelIndices: new Uint32Array([batch.id]),
    baseColor: new Uint8Array([batch.id, 0, 0, 255]),
    additionalTextures: [],
  };
}

function progressFor(
  batch: TextureBakeBatch,
  completedBatches: number,
  partitionTotal: number,
  processedSamples: number,
): TextureBakeBatchProgress {
  return {
    batchId: batch.id,
    completedBatches,
    totalBatches: partitionTotal,
    processedFaces: completedBatches,
    totalFaces: 3,
    processedSamples,
    totalSamples: 35,
    mappedPixels: processedSamples,
    unmappedPixels: 0,
  };
}

beforeEach(() => {
  StubTextureBakeWorker.instances = [];
  workerBehavior = () => {};
  installWorkerStub();
  setHardwareConcurrency(3);
});

afterEach(() => {
  restoreGlobalProperty('Worker', originalWorkerDescriptor);
  restoreGlobalProperty('navigator', originalNavigatorDescriptor);
});

describe('browser texture bake worker pool', () => {
  it('estimates replicated payload bytes above raw image bytes for scaled inputs', () => {
    const imageBytes = 8 * 1024 * 1024;
    const input = scaledBakeInput({
      sourceVertices: 20_000,
      sourceFaces: 40_000,
      outputVertices: 10_000,
      outputFaces: 20_000,
      historyRecords: 500,
      imageBytes,
      activeSlots: ['normal', 'metallicRoughness'],
      totalSamples: 100_000,
    });

    expect(estimateTextureBakeReplicatedPayloadBytes(input)).toBeGreaterThan(imageBytes);
  });

  it('uses one browser worker count for very large replicated payloads', () => {
    const input = scaledBakeInput({
      imageBytes: 260 * 1024 * 1024,
      batchCount: 16,
      totalSamples: 1_000,
    });

    expect(browserTextureBakeWorkerCount(input, 16)).toBe(1);
  });

  it('caps medium replicated payloads at two browser workers', () => {
    const input = scaledBakeInput({
      imageBytes: 129 * 1024 * 1024,
      batchCount: 16,
      totalSamples: 1_000,
    });

    expect(browserTextureBakeWorkerCount(input, 16)).toBe(2);
  });

  it('caps small replicated payloads at four browser workers', () => {
    const input = scaledBakeInput({
      imageBytes: 1024,
      batchCount: 16,
      totalSamples: 1_000,
    });

    expect(browserTextureBakeWorkerCount(input, 16)).toBe(4);
  });

  it('treats memory thresholds as exclusive upper bounds', () => {
    expect(browserTextureBakeWorkerCount(inputWithEstimatedBytes(128 * mebibyte), 16)).toBe(4);
    expect(browserTextureBakeWorkerCount(inputWithEstimatedBytes(128 * mebibyte + 1), 16)).toBe(2);
    expect(browserTextureBakeWorkerCount(inputWithEstimatedBytes(256 * mebibyte), 16)).toBe(2);
    expect(browserTextureBakeWorkerCount(inputWithEstimatedBytes(256 * mebibyte + 1), 16)).toBe(1);
  });

  it('includes normal-map scratch normals in replicated payload estimates', () => {
    const base = scaledBakeInput({
      sourceVertices: 60_000,
      outputVertices: 60_000,
      imageBytes: 115 * mebibyte,
      batchCount: 16,
      totalSamples: 1_000,
    });
    const withNormal = scaledBakeInput({
      sourceVertices: 60_000,
      outputVertices: 60_000,
      imageBytes: 115 * mebibyte,
      activeSlots: ['normal'],
      batchCount: 16,
      totalSamples: 1_000,
    });

    expect(estimateTextureBakeReplicatedPayloadBytes(base)).toBeLessThanOrEqual(128 * mebibyte);
    expect(browserTextureBakeWorkerCount(base, 16)).toBe(4);
    expect(estimateTextureBakeReplicatedPayloadBytes(withNormal)).toBeGreaterThan(128 * mebibyte);
    expect(browserTextureBakeWorkerCount(withNormal, 16)).toBe(2);
  });

  it('counts provided output normals as serialized payload and live worker memory', () => {
    const computed = scaledBakeInput({
      sourceVertices: 60_000,
      outputVertices: 60_000,
      imageBytes: 115 * mebibyte,
      activeSlots: ['normal'],
      batchCount: 16,
      totalSamples: 1_000,
    });
    const provided = scaledBakeInput({
      sourceVertices: 60_000,
      outputVertices: 60_000,
      imageBytes: 115 * mebibyte,
      activeSlots: ['normal'],
      batchCount: 16,
      totalSamples: 1_000,
      outputVertexNormals: true,
    });
    const base = scaledBakeInput({
      sourceVertices: 60_000,
      outputVertices: 60_000,
      imageBytes: 115 * mebibyte,
      batchCount: 16,
      totalSamples: 1_000,
    });

    expect(estimateTextureBakeReplicatedPayloadBytes(provided)).toBeGreaterThan(
      estimateTextureBakeReplicatedPayloadBytes(base),
    );
    expect(estimateTextureBakeReplicatedPayloadBytes(provided)).toBeGreaterThan(
      estimateTextureBakeReplicatedPayloadBytes(computed),
    );
  });

  it('runs very large replicated payloads through the serial path without creating subworkers', async () => {
    const input = scaledBakeInput({
      imageBytes: 260 * 1024 * 1024,
      batchCount: 16,
      totalSamples: 32,
    });
    const results: TextureBakeBatchResult[] = [];

    await createBrowserTextureBakeBatchRunner()(input, (result) => results.push(result));

    expect(StubTextureBakeWorker.instances).toHaveLength(0);
    expect(results).toHaveLength(input.batches.length);
    expect(results.reduce((total, result) => total + result.mappedPixels, 0)).toBeGreaterThan(0);
  });

  it('terminates created workers and falls back when serialization fails before initialization', async () => {
    const input = scaledBakeInput({
      imageBytes: 1024,
      batchCount: 16,
      totalSamples: 32,
    });
    const results: TextureBakeBatchResult[] = [];

    await createBrowserTextureBakeBatchRunner()(input, (result) => results.push(result));

    expect(StubTextureBakeWorker.instances).toHaveLength(2);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.requests.length === 0)).toBe(true);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.onmessage === null && worker.onerror === null)).toBe(true);
    expect(results).toHaveLength(input.batches.length);
  });

  it('runs texture bake partitions in multiple workers and aggregates progress by worker', async () => {
    workerBehavior = (worker, request) => {
      if (request.type === 'init') {
        worker.respond({ type: 'ready', id: request.id });
        return;
      }
      if (request.type !== 'run-batches') return;

      if (worker.index === 0) {
        const batch = request.batches[0]!;
        worker.respond({ type: 'batch-progress', id: request.id, progress: progressFor(batch, 1, 1, 20) });
        worker.respond({ type: 'batch-result', id: request.id, result: batchResult(batch) });
        worker.respond({ type: 'batches-complete', id: request.id });
        return;
      }

      const first = request.batches[0]!;
      const second = request.batches[1]!;
      worker.respond({ type: 'batch-progress', id: request.id, progress: progressFor(first, 1, 2, 10) });
      worker.respond({ type: 'batch-result', id: request.id, result: batchResult(first) });
      worker.respond({ type: 'batch-progress', id: request.id, progress: progressFor(second, 2, 2, 15) });
      worker.respond({ type: 'batch-result', id: request.id, result: batchResult(second) });
      worker.respond({ type: 'batches-complete', id: request.id });
    };
    const input = workerBakeInput();
    const progress: TextureBakeBatchProgress[] = [];
    const results: TextureBakeBatchResult[] = [];

    await createBrowserTextureBakeBatchRunner()(input, (result) => results.push(result), (update) => progress.push(update));

    expect(StubTextureBakeWorker.instances).toHaveLength(2);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
    expect(StubTextureBakeWorker.instances.map((worker) => (
      worker.requests.filter((request) => request.type === 'init').length
    ))).toEqual([1, 1]);
    expect(StubTextureBakeWorker.instances.map((worker) => (
      worker.requests.filter((request) => request.type === 'run-batches').map((request) => request.batches.map((batch) => batch.id))
    ))).toEqual([[[1]], [[0, 2]]]);
    expect(StubTextureBakeWorker.instances.flatMap((worker) => worker.transferLists)).toEqual([
      undefined,
      undefined,
      undefined,
      undefined,
    ]);
    expect(results.map((result) => result.batchId)).toEqual([1, 0, 2]);
    expect(progress.map((update) => ({
      batchId: update.batchId,
      completedBatches: update.completedBatches,
      totalBatches: update.totalBatches,
      processedFaces: update.processedFaces,
      processedSamples: update.processedSamples,
      mappedPixels: update.mappedPixels,
    }))).toEqual([
      { batchId: -1, completedBatches: 1, totalBatches: 3, processedFaces: 1, processedSamples: 20, mappedPixels: 20 },
      { batchId: -1, completedBatches: 2, totalBatches: 3, processedFaces: 2, processedSamples: 30, mappedPixels: 30 },
      { batchId: -1, completedBatches: 3, totalBatches: 3, processedFaces: 3, processedSamples: 35, mappedPixels: 35 },
    ]);
  });

  it('falls back to serial texture baking when worker initialization fails', async () => {
    workerBehavior = (worker, request) => {
      if (request.type === 'init') {
        worker.respond({ type: 'error', id: request.id, message: 'init failed' });
      }
    };
    const input = serialBakeInput();
    const progress: TextureBakeBatchProgress[] = [];
    const results: TextureBakeBatchResult[] = [];

    await createBrowserTextureBakeBatchRunner()(input, (result) => results.push(result), (update) => progress.push(update));

    expect(results).toHaveLength(input.batches.length);
    expect(results.reduce((total, result) => total + result.mappedPixels, 0)).toBe(input.totalSamples);
    expect(progress.at(-1)?.processedSamples).toBe(input.totalSamples);
    expect(StubTextureBakeWorker.instances.length).toBeGreaterThan(0);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
  });

  it('rejects runtime worker errors without falling back and terminates workers', async () => {
    workerBehavior = (worker, request) => {
      if (request.type === 'init') {
        worker.respond({ type: 'ready', id: request.id });
        return;
      }
      if (request.type !== 'run-batches') return;
      if (worker.index === 0) {
        worker.respond({ type: 'error', id: request.id, message: 'batch failed' });
        return;
      }
      for (const batch of request.batches) {
        worker.respond({ type: 'batch-result', id: request.id, result: batchResult(batch) });
      }
      worker.respond({ type: 'batches-complete', id: request.id });
    };

    await expect(createBrowserTextureBakeBatchRunner()(workerBakeInput(), () => undefined)).rejects.toThrow('batch failed');
    expect(StubTextureBakeWorker.instances).toHaveLength(2);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
  });

  it('rejects callback errors from streamed batch results and terminates workers', async () => {
    workerBehavior = (worker, request) => {
      if (request.type === 'init') {
        worker.respond({ type: 'ready', id: request.id });
        return;
      }
      if (request.type !== 'run-batches') return;
      if (worker.index !== 0) return;
      const batch = request.batches[0]!;
      worker.respond({ type: 'batch-result', id: request.id, result: batchResult(batch) });
    };

    await expect(createBrowserTextureBakeBatchRunner()(workerBakeInput(), () => {
      throw new Error('merge failed');
    })).rejects.toThrow('merge failed');

    expect(StubTextureBakeWorker.instances).toHaveLength(2);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.onmessage === null && worker.onerror === null)).toBe(true);
  });
});
