import { EventEmitter } from 'node:events';
import { execFileSync } from 'node:child_process';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { RawMesh } from '../../src/simplification/types';
import {
  planTextureBakeBatches,
  type TextureBakeBatch,
  type TextureBakeBatchProgress,
  type TextureBakeBatchResult,
  type TextureBakeBatchRunInput,
} from '../../src/texture/bakeBatch';
import type { AtlasLayout, SourceMaterial, TexturedRawMesh } from '../../src/texture/types';
import type { TextureBakeWorkerRequest, TextureBakeWorkerResponse } from '../../src/pipeline/textureBakeWorkerProtocol';

type WorkerBehavior = (worker: StubTextureBakeWorker, request: TextureBakeWorkerRequest) => void;

let workerBehavior: WorkerBehavior = () => {};

class StubTextureBakeWorker extends EventEmitter {
  static instances: StubTextureBakeWorker[] = [];

  readonly index: number;
  readonly requests: TextureBakeWorkerRequest[] = [];
  terminated = false;
  terminateCalls = 0;

  constructor() {
    super();
    this.index = StubTextureBakeWorker.instances.length;
    StubTextureBakeWorker.instances.push(this);
  }

  postMessage(request: TextureBakeWorkerRequest): void {
    this.requests.push(request);
    workerBehavior(this, request);
  }

  async terminate(): Promise<number> {
    this.terminated = true;
    this.terminateCalls += 1;
    this.emit('exit', 1);
    return 1;
  }

  respond(response: TextureBakeWorkerResponse): void {
    this.emit('message', response);
  }
}

const sampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'nearest' as const };

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

function fanMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(0, 1, 0),
      new Vector3(-1, 0, 0),
      new Vector3(0, -1, 0),
    ],
    faces: [
      [0, 1, 2],
      [0, 2, 3],
      [0, 3, 4],
      [0, 4, 1],
    ],
  };
}

function atlas(textureSize = 8): AtlasLayout {
  return {
    textureSize,
    padding: 0,
    faceUvs: [
      [new Vector2(0, 0), new Vector2(0.5, 0), new Vector2(0, 0.5)],
      [new Vector2(0.5, 0), new Vector2(1, 0), new Vector2(1, 0.5)],
      [new Vector2(0, 0.5), new Vector2(0.5, 0.5), new Vector2(0, 1)],
      [new Vector2(0.5, 0.5), new Vector2(1, 0.5), new Vector2(1, 1)],
    ],
    facePixelTriangles: [
      [[0, 0], [4, 0], [0, 4]],
      [[4, 0], [8, 0], [8, 4]],
      [[0, 4], [4, 4], [0, 8]],
      [[4, 4], [8, 4], [8, 8]],
    ],
    islandCount: 4,
  };
}

function texturedFan(): TexturedRawMesh {
  const rawMesh = fanMesh();
  return {
    rawMesh,
    faceAttributes: rawMesh.faces.map(() => ({
      materialId: 0,
      uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
    })),
    materials: [material({
      baseColorTexture: {
        image: { width: 1, height: 1, data: new Uint8Array([255, 64, 32, 255]) },
        sampler,
        texCoord: 0,
      },
    })],
  };
}

function workerBakeInput(): TextureBakeBatchRunInput {
  const source = texturedFan();
  const layout = atlas();
  const plan = planTextureBakeBatches(layout, { targetSamplesPerBatch: 1 });
  return {
    source,
    outputRawMesh: source.rawMesh,
    outputFaceIds: [0, 1, 2, 3],
    history: [],
    atlas: layout,
    activeSlots: [],
    batches: plan.batches,
    totalFaces: plan.totalFaces,
    totalSamples: plan.totalSamples,
  };
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

function progressFor(batch: TextureBakeBatch, completedBatches: number, processedSamples: number): TextureBakeBatchProgress {
  return {
    batchId: batch.id,
    completedBatches,
    totalBatches: completedBatches,
    processedFaces: completedBatches,
    totalFaces: 4,
    processedSamples,
    totalSamples: 40,
    mappedPixels: processedSamples,
    unmappedPixels: 0,
  };
}

async function loadPoolWithStubWorkers() {
  vi.resetModules();
  StubTextureBakeWorker.instances = [];
  vi.doMock('node:os', () => ({ availableParallelism: () => 3 }));
  vi.doMock('node:worker_threads', () => ({ Worker: StubTextureBakeWorker }));
  return import('../../src/local/textureBakeWorkerPool');
}

afterEach(() => {
  vi.resetModules();
  vi.doUnmock('node:os');
  vi.doUnmock('node:worker_threads');
  StubTextureBakeWorker.instances = [];
  workerBehavior = () => {};
});

describe('Node texture bake worker pool', () => {
  it('runs texture bake batches in real Node workers and emits aggregate progress', async () => {
    const output = execFileSync(process.execPath, ['--import', 'tsx', '--input-type=module', '--eval', `
      import { createRequire, syncBuiltinESMExports } from 'node:module';
      const require = createRequire(import.meta.url);
      const os = require('node:os');
      os.availableParallelism = () => 4;
      syncBuiltinESMExports();

      const { Vector2, Vector3 } = await import('three');
      const { planTextureBakeBatches } = await import('./src/texture/bakeBatch.ts');
      const { createNodeTextureBakeBatchRunner } = await import('./src/local/textureBakeWorkerPool.ts');

      const sampler = { wrapS: 'clamp', wrapT: 'clamp', filter: 'nearest' };
      const rawMesh = {
        positions: [
          new Vector3(0, 0, 0),
          new Vector3(1, 0, 0),
          new Vector3(0, 1, 0),
          new Vector3(-1, 0, 0),
          new Vector3(0, -1, 0),
        ],
        faces: [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
      };
      const layout = {
        textureSize: 8,
        padding: 0,
        faceUvs: [
          [new Vector2(0, 0), new Vector2(0.5, 0), new Vector2(0, 0.5)],
          [new Vector2(0.5, 0), new Vector2(1, 0), new Vector2(1, 0.5)],
          [new Vector2(0, 0.5), new Vector2(0.5, 0.5), new Vector2(0, 1)],
          [new Vector2(0.5, 0.5), new Vector2(1, 0.5), new Vector2(1, 1)],
        ],
        facePixelTriangles: [
          [[0, 0], [4, 0], [0, 4]],
          [[4, 0], [8, 0], [8, 4]],
          [[0, 4], [4, 4], [0, 8]],
          [[4, 4], [8, 4], [8, 8]],
        ],
        islandCount: 4,
      };
      const plan = planTextureBakeBatches(layout, { targetSamplesPerBatch: 1 });
      const input = {
        source: {
          rawMesh,
          faceAttributes: rawMesh.faces.map(() => ({
            materialId: 0,
            uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
          })),
          materials: [{
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
            baseColorTexture: {
              image: { width: 1, height: 1, data: new Uint8Array([255, 64, 32, 255]) },
              sampler,
              texCoord: 0,
            },
          }],
        },
        outputRawMesh: rawMesh,
        outputFaceIds: [0, 1, 2, 3],
        history: [],
        atlas: layout,
        activeSlots: [],
        batches: plan.batches,
        totalFaces: plan.totalFaces,
        totalSamples: plan.totalSamples,
      };
      const progress = [];
      const results = [];
      await createNodeTextureBakeBatchRunner()(input, (result) => results.push(result), (update) => progress.push(update));
      console.log(JSON.stringify({
        batchIds: input.batches.map((batch) => batch.id),
        totalSamples: input.totalSamples,
        resultBatchIds: results.map((result) => result.batchId),
        mappedPixels: results.reduce((total, result) => total + result.mappedPixels, 0),
        progress,
      }));
    `], {
      cwd: process.cwd(),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const summary = JSON.parse(output) as {
      batchIds: number[];
      totalSamples: number;
      resultBatchIds: number[];
      mappedPixels: number;
      progress: TextureBakeBatchProgress[];
    };

    expect(summary.batchIds.length).toBeGreaterThan(1);
    expect(summary.resultBatchIds.sort((left, right) => left - right)).toEqual(summary.batchIds);
    expect(summary.mappedPixels).toBe(summary.totalSamples);
    expect(summary.progress.length).toBeGreaterThan(0);
    expect(summary.progress.every((update) => update.batchId === -1)).toBe(true);
    expect(summary.progress.at(-1)).toMatchObject({
      completedBatches: summary.batchIds.length,
      totalBatches: summary.batchIds.length,
      processedSamples: summary.totalSamples,
      mappedPixels: summary.totalSamples,
    });
  });

  it('terminates workers after successful batch execution', async () => {
    workerBehavior = (worker, request) => {
      if (request.type === 'init') {
        worker.respond({ type: 'ready', id: request.id });
        return;
      }
      if (request.type !== 'run-batches') return;
      let processedSamples = 0;
      request.batches.forEach((batch, index) => {
        processedSamples += batch.sampleCount;
        worker.respond({
          type: 'batch-progress',
          id: request.id,
          progress: progressFor(batch, index + 1, processedSamples),
        });
        worker.respond({ type: 'batch-result', id: request.id, result: batchResult(batch) });
      });
      worker.respond({ type: 'batches-complete', id: request.id });
    };
    const { createNodeTextureBakeBatchRunner } = await loadPoolWithStubWorkers();
    const results: TextureBakeBatchResult[] = [];

    await createNodeTextureBakeBatchRunner()(workerBakeInput(), (result) => results.push(result));

    expect(StubTextureBakeWorker.instances).toHaveLength(2);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminateCalls === 1)).toBe(true);
    expect(results.map((result) => result.batchId)).toEqual([0, 2, 1, 3]);
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
      }
    };
    const { createNodeTextureBakeBatchRunner } = await loadPoolWithStubWorkers();

    await expect(createNodeTextureBakeBatchRunner()(workerBakeInput(), () => undefined)).rejects.toThrow('batch failed');
    expect(StubTextureBakeWorker.instances).toHaveLength(2);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminateCalls === 1)).toBe(true);
  });

  it('rejects callback errors from streamed batch results and terminates workers', async () => {
    workerBehavior = (worker, request) => {
      if (request.type === 'init') {
        worker.respond({ type: 'ready', id: request.id });
        return;
      }
      if (request.type !== 'run-batches') return;
      const batch = request.batches[0]!;
      worker.respond({ type: 'batch-result', id: request.id, result: batchResult(batch) });
      worker.respond({ type: 'batches-complete', id: request.id });
    };
    const { createNodeTextureBakeBatchRunner } = await loadPoolWithStubWorkers();

    await expect(createNodeTextureBakeBatchRunner()(workerBakeInput(), () => {
      throw new Error('merge failed');
    })).rejects.toThrow('merge failed');

    expect(StubTextureBakeWorker.instances).toHaveLength(2);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminated)).toBe(true);
    expect(StubTextureBakeWorker.instances.every((worker) => worker.terminateCalls === 1)).toBe(true);
    expect(StubTextureBakeWorker.instances.every((worker) => (
      worker.listenerCount('message') === 0
      && worker.listenerCount('error') === 0
      && worker.listenerCount('exit') === 0
    ))).toBe(true);
  });
});
