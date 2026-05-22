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

const MEBIBYTE = 1024 * 1024;
const MEDIUM_REPLICATED_PAYLOAD_BYTES = 128 * MEBIBYTE;
const LARGE_REPLICATED_PAYLOAD_BYTES = 256 * MEBIBYTE;

const ARRAY_HEADER_BYTES = 32;
const ARRAY_SLOT_BYTES = 8;
const OBJECT_HEADER_BYTES = 48;
const NUMBER_BYTES = 8;
const VECTOR2_BYTES = 56;
const VECTOR3_BYTES = 72;
const VECTOR4_BYTES = 88;
const FACE_INDICES_BYTES = OBJECT_HEADER_BYTES + 3 * NUMBER_BYTES;
const COORDINATE_PAIR_BYTES = ARRAY_HEADER_BYTES + 2 * (ARRAY_SLOT_BYTES + NUMBER_BYTES);
const PIXEL_TRIANGLE_BYTES = arrayShellBytes(3) + 3 * COORDINATE_PAIR_BYTES;
const BATCH_RESULT_SAMPLE_BYTES = 4 + 4;

let requestCounter = 0;

function nextRequestId(prefix: string): string {
  requestCounter += 1;
  return `${prefix}-${requestCounter.toString(36)}`;
}

function browserHardwareConcurrency(): number {
  return typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
    ? navigator.hardwareConcurrency
    : 4;
}

function arrayShellBytes(length: number): number {
  return ARRAY_HEADER_BYTES + length * ARRAY_SLOT_BYTES;
}

function imageBytes(image: { data: { byteLength?: number; length: number } } | undefined): number {
  if (!image) return 0;
  return image.data.byteLength ?? image.data.length;
}

function estimateRawMeshBytes(input: TextureBakeBatchRunInput['outputRawMesh']): number {
  return arrayShellBytes(input.positions.length)
    + input.positions.length * VECTOR3_BYTES
    + arrayShellBytes(input.faces.length)
    + input.faces.length * FACE_INDICES_BYTES;
}

function estimateSourceFaceAttributesBytes(input: TextureBakeBatchRunInput): number {
  let bytes = arrayShellBytes(input.source.faceAttributes.length);
  for (const attributes of input.source.faceAttributes) {
    bytes += OBJECT_HEADER_BYTES + NUMBER_BYTES + ARRAY_HEADER_BYTES;
    if (!attributes) continue;
    bytes += attributes.uvSets.length * (ARRAY_SLOT_BYTES + OBJECT_HEADER_BYTES + NUMBER_BYTES + arrayShellBytes(3) + 3 * VECTOR2_BYTES);
    if (attributes.normalCorners) bytes += arrayShellBytes(3) + 3 * VECTOR3_BYTES;
    if (attributes.tangentCorners) bytes += arrayShellBytes(3) + 3 * VECTOR4_BYTES;
    if (attributes.normalMapYScale !== undefined) bytes += NUMBER_BYTES;
  }
  return bytes;
}

function estimateSourceMaterialsBytes(input: TextureBakeBatchRunInput): number {
  let bytes = arrayShellBytes(input.source.materials.length);
  for (const material of input.source.materials) {
    bytes += OBJECT_HEADER_BYTES
      + 4 * NUMBER_BYTES
      + 3 * NUMBER_BYTES
      + 5 * NUMBER_BYTES
      + ARRAY_HEADER_BYTES;
    bytes += imageBytes(material.baseColorTexture?.image);
    for (const slot of material.textureSlots) {
      bytes += ARRAY_SLOT_BYTES + OBJECT_HEADER_BYTES + 2 * NUMBER_BYTES;
      bytes += imageBytes(slot.image);
    }
  }
  return bytes;
}

function estimateHistoryBytes(input: TextureBakeBatchRunInput): number {
  let bytes = arrayShellBytes(input.history.length);
  for (const record of input.history) {
    bytes += OBJECT_HEADER_BYTES
      + 2 * NUMBER_BYTES
      + arrayShellBytes(record.beforeFaces.length)
      + arrayShellBytes(record.afterFaceIds.length)
      + record.afterFaceIds.length * NUMBER_BYTES;
    for (let index = 0; index < record.beforeFaces.length; index += 1) {
      bytes += ARRAY_SLOT_BYTES
        + OBJECT_HEADER_BYTES
        + NUMBER_BYTES
        + FACE_INDICES_BYTES
        + arrayShellBytes(3)
        + 3 * VECTOR3_BYTES;
    }
  }
  return bytes;
}

function estimateAtlasBytes(input: TextureBakeBatchRunInput): number {
  return OBJECT_HEADER_BYTES
    + 3 * NUMBER_BYTES
    + arrayShellBytes(input.atlas.faceUvs.length)
    + input.atlas.faceUvs.length * (arrayShellBytes(3) + 3 * VECTOR2_BYTES)
    + arrayShellBytes(input.atlas.facePixelTriangles.length)
    + input.atlas.facePixelTriangles.length * PIXEL_TRIANGLE_BYTES;
}

function estimateBatchesBytes(input: TextureBakeBatchRunInput): number {
  return arrayShellBytes(input.batches.length)
    + input.batches.length * (OBJECT_HEADER_BYTES + 4 * NUMBER_BYTES);
}

function estimateNormalScratchBytes(input: TextureBakeBatchRunInput): number {
  if (!input.activeSlots.includes('normal')) return 0;
  const sourceNormals = arrayShellBytes(input.source.rawMesh.positions.length)
    + input.source.rawMesh.positions.length * VECTOR3_BYTES;
  const outputNormals = arrayShellBytes(input.outputRawMesh.positions.length)
    + input.outputRawMesh.positions.length * VECTOR3_BYTES;
  return sourceNormals + outputNormals;
}

function estimateOutputVertexNormalsPayloadBytes(input: TextureBakeBatchRunInput): number {
  if (!input.outputVertexNormals) return 0;
  return ARRAY_HEADER_BYTES + input.outputVertexNormals.length * 3 * Float32Array.BYTES_PER_ELEMENT;
}

export function estimateTextureBakeReplicatedPayloadBytes(input: TextureBakeBatchRunInput): number {
  const activeSlotCount = input.activeSlots.length;
  const plannedBatchResultBytes = input.totalSamples * (BATCH_RESULT_SAMPLE_BYTES + activeSlotCount * 4);
  return estimateRawMeshBytes(input.source.rawMesh)
    + estimateSourceFaceAttributesBytes(input)
    + estimateSourceMaterialsBytes(input)
    + estimateRawMeshBytes(input.outputRawMesh)
    + arrayShellBytes(input.outputFaceIds.length)
    + input.outputFaceIds.length * NUMBER_BYTES
    + estimateHistoryBytes(input)
    + estimateAtlasBytes(input)
    + arrayShellBytes(input.activeSlots.length)
    + input.activeSlots.length * ARRAY_SLOT_BYTES
    + estimateBatchesBytes(input)
    + estimateOutputVertexNormalsPayloadBytes(input)
    + estimateNormalScratchBytes(input)
    + plannedBatchResultBytes;
}

export function browserTextureBakeWorkerCount(
  input: TextureBakeBatchRunInput,
  hardware = browserHardwareConcurrency(),
): number {
  const cpuWorkerCount = automaticTextureBakeWorkerCount(input.batches.length, hardware);
  const estimatedBytes = estimateTextureBakeReplicatedPayloadBytes(input);
  if (estimatedBytes > LARGE_REPLICATED_PAYLOAD_BYTES) return 1;
  if (estimatedBytes > MEDIUM_REPLICATED_PAYLOAD_BYTES) return Math.min(cpuWorkerCount, 2);
  return Math.min(cpuWorkerCount, 4);
}

function createTextureBakeWorker(): Worker {
  return new Worker(new URL('./textureBakeWorker.ts', import.meta.url), { type: 'module' });
}

function terminateWorkers(handles: readonly TextureBakeWorkerHandle[]): void {
  for (const handle of handles) {
    handle.worker.onerror = null;
    handle.worker.onmessage = null;
    handle.worker.terminate();
  }
}

function createWorkerHandles(partitions: readonly TextureBakeBatch[][]): TextureBakeWorkerHandle[] | null {
  const handles: TextureBakeWorkerHandle[] = [];
  try {
    for (const [index, batches] of partitions.entries()) {
      handles.push({
        worker: createTextureBakeWorker(),
        batches,
        index,
      });
    }
    return handles;
  } catch {
    terminateWorkers(handles);
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
    handle.worker.onerror = (event) => {
      reject(new Error(event.message));
    };
    handle.worker.onmessage = (event: MessageEvent<TextureBakeWorkerResponse>) => {
      const response = event.data;
      if (response.id !== id) {
        reject(new Error(`Texture bake worker ${handle.index} returned mismatched response ${response.id} for ${id}.`));
        return;
      }
      if (response.type === 'error') {
        reject(errorFromResponse(response));
        return;
      }
      if (response.type !== 'ready') {
        reject(new Error(`Texture bake worker ${handle.index} returned ${response.type} during initialization.`));
        return;
      }
      resolve();
    };
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
      handle.worker.onerror = null;
      handle.worker.onmessage = null;
    };
    handle.worker.onerror = (event) => {
      cleanup();
      reject(new Error(event.message));
    };
    handle.worker.onmessage = (event: MessageEvent<TextureBakeWorkerResponse>) => {
      const response = event.data;
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
    postWorkerRequest(handle.worker, request);
  });
}

async function runInBrowserTextureBakeWorkers(
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
    terminateWorkers(handles);
    return false;
  }

  try {
    const latestByWorker = new Map<number, TextureBakeBatchProgress>();
    await Promise.all(handles.map((handle) => (
      runPartition(handle, input, latestByWorker, onBatchResult, onProgress)
    )));
    return true;
  } finally {
    terminateWorkers(handles);
  }
}

export function createBrowserTextureBakeBatchRunner(): TextureBakeBatchRunner {
  return async (input, onBatchResult, onProgress) => {
    const workerCount = browserTextureBakeWorkerCount(input);
    if (input.batches.length <= 1 || workerCount <= 1) {
      await runTextureBakeBatchesSerial(input, onBatchResult, onProgress);
      return;
    }
    const completed = await runInBrowserTextureBakeWorkers(input, workerCount, onBatchResult, onProgress);
    if (!completed) {
      await runTextureBakeBatchesSerial(input, onBatchResult, onProgress);
    }
  };
}
