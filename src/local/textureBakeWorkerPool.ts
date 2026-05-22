import { availableParallelism } from 'node:os';
import { Worker, type WorkerOptions } from 'node:worker_threads';
import {
  runTextureBakeBatchesSerial,
  type TextureBakeBatch,
  type TextureBakeBatchProgress,
  type TextureBakeBatchResult,
  type TextureBakeBatchRunner,
  type TextureBakeBatchRunInput,
} from '../texture/bakeBatch';
import {
  aggregateTextureBakeProgress,
  automaticTextureBakeWorkerCount,
  partitionTextureBakeBatches,
} from '../pipeline/textureBakeWorkerCoordinator';
import { serializeTextureBakeBatchRunInput } from '../pipeline/textureBakeSerialization';
import type { TextureBakeWorkerRequest, TextureBakeWorkerResponse } from '../pipeline/textureBakeWorkerProtocol';

interface TextureBakeWorkerHandle {
  worker: Worker;
  batches: TextureBakeBatch[];
  index: number;
}

let requestCounter = 0;

function nextRequestId(prefix: string): string {
  requestCounter += 1;
  return `${prefix}-${requestCounter.toString(36)}`;
}

export function nodeTextureBakeWorkerCount(
  batchCount: number,
  hardware = availableParallelism(),
): number {
  return automaticTextureBakeWorkerCount(batchCount, hardware);
}

function createWorker(): Worker {
  const workerUrl = new URL('./textureBakeWorker.ts', import.meta.url);
  return new Worker(`import ${JSON.stringify(workerUrl.href)};`, {
    eval: true,
    type: 'module',
  } as WorkerOptions & { type: 'module' });
}

async function terminateWorkers(handles: readonly TextureBakeWorkerHandle[]): Promise<void> {
  await Promise.all(handles.map((handle) => handle.worker.terminate().catch(() => undefined)));
}

function createWorkerHandles(partitions: readonly TextureBakeBatch[][]): TextureBakeWorkerHandle[] | null {
  const handles: TextureBakeWorkerHandle[] = [];
  try {
    for (const [index, batches] of partitions.entries()) {
      handles.push({
        worker: createWorker(),
        batches,
        index,
      });
    }
    return handles;
  } catch {
    void terminateWorkers(handles);
    return null;
  }
}

function errorFromResponse(response: Extract<TextureBakeWorkerResponse, { type: 'error' }>): Error {
  const error = new Error(response.message);
  if (response.stack) error.stack = response.stack;
  return error;
}

function postWorkerRequest(worker: Worker, request: TextureBakeWorkerRequest): void {
  worker.postMessage(request);
}

function initializeWorker(
  handle: TextureBakeWorkerHandle,
  input: Extract<TextureBakeWorkerRequest, { type: 'init' }>['input'],
): Promise<void> {
  const id = nextRequestId(`texture-init-${handle.index}`);
  const request: TextureBakeWorkerRequest = { type: 'init', id, input };
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      handle.worker.off('message', onMessage);
      handle.worker.off('error', onError);
      handle.worker.off('exit', onExit);
    };
    const onError = (error: Error): void => {
      cleanup();
      reject(error);
    };
    const onExit = (code: number): void => {
      cleanup();
      reject(new Error(`Texture bake worker ${handle.index} exited with code ${code} during initialization.`));
    };
    const onMessage = (response: TextureBakeWorkerResponse): void => {
      if (response.id !== id) {
        cleanup();
        reject(new Error(`Texture bake worker ${handle.index} returned mismatched response ${response.id} for ${id}.`));
        return;
      }
      if (response.type === 'error') {
        cleanup();
        reject(errorFromResponse(response));
        return;
      }
      if (response.type !== 'ready') {
        cleanup();
        reject(new Error(`Texture bake worker ${handle.index} returned ${response.type} during initialization.`));
        return;
      }
      cleanup();
      resolve();
    };
    handle.worker.on('message', onMessage);
    handle.worker.on('error', onError);
    handle.worker.on('exit', onExit);
    postWorkerRequest(handle.worker, request);
  });
}

function completedBatchCount(latestByWorker: ReadonlyMap<number, TextureBakeBatchProgress>): number {
  let completed = 0;
  for (const progress of latestByWorker.values()) completed += progress.completedBatches;
  return completed;
}

function aggregateProgress(
  input: TextureBakeBatchRunInput,
  latestByWorker: ReadonlyMap<number, TextureBakeBatchProgress>,
): TextureBakeBatchProgress {
  return {
    batchId: -1,
    ...aggregateTextureBakeProgress({
      completedBatches: completedBatchCount(latestByWorker),
      totalBatches: input.batches.length,
      latestByWorker,
      totalFaces: input.totalFaces,
      totalSamples: input.totalSamples,
    }),
  };
}

function runPartition(
  handle: TextureBakeWorkerHandle,
  input: TextureBakeBatchRunInput,
  latestByWorker: Map<number, TextureBakeBatchProgress>,
  onBatchResult: (result: TextureBakeBatchResult) => void,
  onProgress?: (progress: TextureBakeBatchProgress) => void,
): Promise<void> {
  const id = nextRequestId(`texture-batches-${handle.index}`);
  const request: TextureBakeWorkerRequest = { type: 'run-batches', id, batches: handle.batches };
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      handle.worker.off('message', onMessage);
      handle.worker.off('error', onError);
      handle.worker.off('exit', onExit);
    };
    const onError = (error: Error): void => {
      cleanup();
      reject(error);
    };
    const onExit = (code: number): void => {
      cleanup();
      reject(new Error(`Texture bake worker ${handle.index} exited with code ${code} during texture baking.`));
    };
    const onMessage = (response: TextureBakeWorkerResponse): void => {
      if (response.id !== id) {
        cleanup();
        reject(new Error(`Texture bake worker ${handle.index} returned mismatched response ${response.id} for ${id}.`));
        return;
      }
      if (response.type === 'error') {
        cleanup();
        reject(errorFromResponse(response));
        return;
      }
      if (response.type === 'batch-progress') {
        latestByWorker.set(handle.index, response.progress);
        onProgress?.(aggregateProgress(input, latestByWorker));
        return;
      }
      if (response.type === 'batch-result') {
        try {
          onBatchResult(response.result);
        } catch (error: unknown) {
          cleanup();
          reject(error);
        }
        return;
      }
      if (response.type === 'batches-complete') {
        cleanup();
        resolve();
        return;
      }
      cleanup();
      reject(new Error(`Texture bake worker ${handle.index} returned ${response.type} during texture baking.`));
    };
    handle.worker.on('message', onMessage);
    handle.worker.on('error', onError);
    handle.worker.on('exit', onExit);
    postWorkerRequest(handle.worker, request);
  });
}

async function runInNodeTextureBakeWorkers(
  input: TextureBakeBatchRunInput,
  workerCount: number,
  onBatchResult: (result: TextureBakeBatchResult) => void,
  onProgress?: (progress: TextureBakeBatchProgress) => void,
): Promise<boolean> {
  const partitions = partitionTextureBakeBatches(input.batches, workerCount).filter((partition) => partition.length > 0);
  const handles = createWorkerHandles(partitions);
  if (!handles) return false;

  let serializedInput: ReturnType<typeof serializeTextureBakeBatchRunInput> | null = null;
  try {
    serializedInput = serializeTextureBakeBatchRunInput(input);
    await Promise.all(handles.map((handle) => initializeWorker(handle, serializedInput!)));
    serializedInput = null;
  } catch {
    serializedInput = null;
    await terminateWorkers(handles);
    return false;
  }

  try {
    const latestByWorker = new Map<number, TextureBakeBatchProgress>();
    await Promise.all(handles.map((handle) => (
      runPartition(handle, input, latestByWorker, onBatchResult, onProgress)
    )));
    return true;
  } finally {
    await terminateWorkers(handles);
  }
}

export function createNodeTextureBakeBatchRunner(): TextureBakeBatchRunner {
  return async (input, onBatchResult, onProgress) => {
    const workerCount = nodeTextureBakeWorkerCount(input.batches.length);
    if (input.batches.length <= 1 || workerCount <= 1) {
      await runTextureBakeBatchesSerial(input, onBatchResult, onProgress);
      return;
    }
    const completed = await runInNodeTextureBakeWorkers(input, workerCount, onBatchResult, onProgress);
    if (!completed) {
      await runTextureBakeBatchesSerial(input, onBatchResult, onProgress);
    }
  };
}
