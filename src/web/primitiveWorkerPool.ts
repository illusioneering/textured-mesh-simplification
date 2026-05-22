import type { SimplificationProgress, VirtualEdgeProgress } from '../simplification/types';
import type { AttributeTransferProgress } from '../simplification/attributeTransfer';
import { allocatePrimitiveProcessingOptions, type ProcessablePrimitiveEntry } from '../pipeline/sceneProcessing';
import { processPrimitiveGeometriesStaged } from '../pipeline/primitiveStagedProcessing';
import type { ProcessingOptions } from '../pipeline/options';
import {
  aggregateAttributeTransferProgress,
  aggregateSimplificationProgress,
  aggregateVirtualProgress,
  automaticPrimitiveWorkerCount,
  deserializePrimitiveStageResults,
  partitionPrimitiveWorkerInputs,
} from '../pipeline/primitiveWorkerCoordinator';
import type {
  AggregatePrimitiveStageProgress,
  PrimitiveParallelStage,
  PrimitiveStageWorkerRequest,
  PrimitiveStageWorkerResponse,
  PrimitiveWorkerInput,
  SerializedPrimitiveStageResult,
} from '../pipeline/primitiveWorkerProtocol';
import { deserializePrimitiveEntries, serializePrimitiveEntries } from './serialization';

type ProcessablePrimitiveEntryWithOrdinal = ProcessablePrimitiveEntry & { meshOrdinal: number };

interface PrimitiveWorkerPoolCallbacks {
  onAggregateProgress?: (progress: AggregatePrimitiveStageProgress) => void;
}

interface PrimitiveWorkerHandle {
  worker: Worker;
  inputs: PrimitiveWorkerInput[];
  index: number;
}

type StageProgressMap =
  | Map<string, VirtualEdgeProgress>
  | Map<string, SimplificationProgress>
  | Map<string, AttributeTransferProgress>;

let requestCounter = 0;

function nextRequestId(prefix: string): string {
  requestCounter += 1;
  return `${prefix}-${requestCounter.toString(36)}`;
}

export function browserPrimitiveWorkerCount(primitiveCount: number): number {
  const hardware = typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
    ? navigator.hardwareConcurrency
    : 4;
  return automaticPrimitiveWorkerCount(primitiveCount, hardware);
}

function createPrimitiveStageWorker(): Worker {
  return new Worker(new URL('./primitiveStageWorker.ts', import.meta.url), { type: 'module' });
}

function terminateWorkers(handles: readonly PrimitiveWorkerHandle[]): void {
  for (const handle of handles) handle.worker.terminate();
}

function createWorkerHandles(partitions: readonly PrimitiveWorkerInput[][]): PrimitiveWorkerHandle[] | null {
  const handles: PrimitiveWorkerHandle[] = [];
  try {
    for (const [index, partition] of partitions.entries()) {
      handles.push({
        worker: createPrimitiveStageWorker(),
        inputs: partition,
        index,
      });
    }
    return handles;
  } catch {
    terminateWorkers(handles);
    return null;
  }
}

function errorFromResponse(response: Extract<PrimitiveStageWorkerResponse, { type: 'error' }>): Error {
  const error = new Error(response.message);
  if (response.stack) error.stack = response.stack;
  return error;
}

function postWorkerRequest(worker: Worker, request: PrimitiveStageWorkerRequest): void {
  worker.postMessage(request);
}

function initializeWorker(handle: PrimitiveWorkerHandle): Promise<void> {
  const id = nextRequestId(`init-${handle.index}`);
  const request: PrimitiveStageWorkerRequest = { type: 'init', id, inputs: handle.inputs };
  return new Promise((resolve, reject) => {
    handle.worker.onerror = (event) => {
      reject(new Error(event.message));
    };
    handle.worker.onmessage = (event: MessageEvent<PrimitiveStageWorkerResponse>) => {
      const response = event.data;
      if (response.id !== id) {
        reject(new Error(`Primitive worker ${handle.index} returned mismatched response ${response.id} for ${id}.`));
        return;
      }
      if (response.type === 'error') {
        reject(errorFromResponse(response));
        return;
      }
      if (response.type !== 'ready') {
        reject(new Error(`Primitive worker ${handle.index} returned ${response.type} during initialization.`));
        return;
      }
      resolve();
    };
    postWorkerRequest(handle.worker, request);
  });
}

function aggregateStageProgress(
  stage: PrimitiveParallelStage,
  completedPrimitives: number,
  totalPrimitives: number,
  latestByPrimitive: StageProgressMap,
): AggregatePrimitiveStageProgress {
  if (stage === 'virtual-edges') {
    return aggregateVirtualProgress(
      completedPrimitives,
      totalPrimitives,
      latestByPrimitive as Map<string, VirtualEdgeProgress>,
    );
  }
  if (stage === 'simplification') {
    return aggregateSimplificationProgress(
      completedPrimitives,
      totalPrimitives,
      latestByPrimitive as Map<string, SimplificationProgress>,
    );
  }
  return aggregateAttributeTransferProgress(
    completedPrimitives,
    totalPrimitives,
    latestByPrimitive as Map<string, AttributeTransferProgress>,
  );
}

function runStage(
  handles: readonly PrimitiveWorkerHandle[],
  stage: PrimitiveParallelStage,
  totalPrimitives: number,
  callbacks: PrimitiveWorkerPoolCallbacks,
): Promise<SerializedPrimitiveStageResult[]> {
  let completedPrimitives = 0;
  const latestByPrimitive: StageProgressMap = new Map();

  const runWorkerStage = (handle: PrimitiveWorkerHandle): Promise<SerializedPrimitiveStageResult[]> => {
    const id = nextRequestId(`${stage}-${handle.index}`);
    const request: PrimitiveStageWorkerRequest = { type: 'run-stage', id, stage };
    return new Promise((resolve, reject) => {
      handle.worker.onerror = (event) => {
        reject(new Error(event.message));
      };
      handle.worker.onmessage = (event: MessageEvent<PrimitiveStageWorkerResponse>) => {
        const response = event.data;
        if (response.id !== id) {
          reject(new Error(`Primitive worker ${handle.index} returned mismatched response ${response.id} for ${id}.`));
          return;
        }
        if (response.type === 'error') {
          reject(errorFromResponse(response));
          return;
        }
        if (response.type === 'stage-progress') {
          if (response.stage !== stage) {
            reject(new Error(`Primitive worker ${handle.index} returned ${response.stage} progress during ${stage}.`));
            return;
          }
          latestByPrimitive.set(response.primitiveId, response.progress as never);
          callbacks.onAggregateProgress?.(
            aggregateStageProgress(stage, completedPrimitives, totalPrimitives, latestByPrimitive),
          );
          return;
        }
        if (response.type === 'stage-complete') {
          if (response.stage !== stage) {
            reject(new Error(`Primitive worker ${handle.index} completed ${response.stage} during ${stage}.`));
            return;
          }
          completedPrimitives += handle.inputs.length;
          callbacks.onAggregateProgress?.(
            aggregateStageProgress(stage, completedPrimitives, totalPrimitives, latestByPrimitive),
          );
          resolve(response.results ?? []);
          return;
        }
        reject(new Error(`Primitive worker ${handle.index} returned ${response.type} during ${stage}.`));
      };
      postWorkerRequest(handle.worker, request);
    });
  };

  return Promise.all(handles.map(runWorkerStage)).then((workerResults) => workerResults.flat());
}

function stagedFallback(
  entries: readonly ProcessablePrimitiveEntryWithOrdinal[],
  options: ProcessingOptions,
  callbacks: PrimitiveWorkerPoolCallbacks = {},
) {
  const totalPrimitives = entries.length;
  let completedVirtualPrimitives = 0;
  let completedSimplificationPrimitives = 0;
  let completedAttributeTransferPrimitives = 0;
  const latestVirtualProgress = new Map<string, VirtualEdgeProgress>();
  const latestSimplificationProgress = new Map<string, SimplificationProgress>();
  const latestAttributeTransferProgress = new Map<string, AttributeTransferProgress>();
  const allocated = allocatePrimitiveProcessingOptions(entries, options);
  return processPrimitiveGeometriesStaged(entries.map((entry, index) => ({
    entry,
    options: allocated[index]?.options ?? options,
  })), {
    onVirtualEdgeProgress: (id, progress) => {
      latestVirtualProgress.set(id, progress);
      callbacks.onAggregateProgress?.(
        aggregateVirtualProgress(completedVirtualPrimitives, totalPrimitives, latestVirtualProgress),
      );
    },
    onVirtualEdgePrimitiveComplete: () => {
      completedVirtualPrimitives += 1;
      callbacks.onAggregateProgress?.(
        aggregateVirtualProgress(completedVirtualPrimitives, totalPrimitives, latestVirtualProgress),
      );
    },
    onSimplificationProgress: (id, progress) => {
      latestSimplificationProgress.set(id, progress);
      callbacks.onAggregateProgress?.(
        aggregateSimplificationProgress(
          completedSimplificationPrimitives,
          totalPrimitives,
          latestSimplificationProgress,
        ),
      );
    },
    onSimplificationPrimitiveComplete: () => {
      completedSimplificationPrimitives += 1;
      callbacks.onAggregateProgress?.(
        aggregateSimplificationProgress(
          completedSimplificationPrimitives,
          totalPrimitives,
          latestSimplificationProgress,
        ),
      );
    },
    onAttributeTransferProgress: (id, progress) => {
      latestAttributeTransferProgress.set(id, progress);
      callbacks.onAggregateProgress?.(
        aggregateAttributeTransferProgress(
          completedAttributeTransferPrimitives,
          totalPrimitives,
          latestAttributeTransferProgress,
        ),
      );
    },
    onAttributeTransferPrimitiveComplete: () => {
      completedAttributeTransferPrimitives += 1;
      callbacks.onAggregateProgress?.(
        aggregateAttributeTransferProgress(
          completedAttributeTransferPrimitives,
          totalPrimitives,
          latestAttributeTransferProgress,
        ),
      );
    },
  });
}

function serializablePrimitiveEntries(entries: readonly ProcessablePrimitiveEntryWithOrdinal[]) {
  return entries.map((entry) => {
    const textured = entry.texturedRawMesh;
    if (!textured) throw new Error(`Primitive ${entry.id} is missing textured metadata for worker processing.`);
    return {
      id: entry.id,
      label: entry.label ?? entry.id,
      meshOrdinal: entry.meshOrdinal,
      rawMesh: entry.rawMesh,
      texturedRawMesh: textured,
      bakeable: entry.bakeable,
      hasTexturedMaterial: entry.hasTexturedMaterial ?? false,
      ...(entry.requiresAttributeTransfer === true ? { requiresAttributeTransfer: true } : {}),
    };
  });
}

export async function processPrimitiveGeometriesInBrowserWorkers(
  entries: readonly ProcessablePrimitiveEntryWithOrdinal[],
  options: ProcessingOptions,
  callbacks: PrimitiveWorkerPoolCallbacks = {},
) {
  const workerCount = browserPrimitiveWorkerCount(entries.length);
  if (entries.length <= 1 || workerCount <= 1) {
    return stagedFallback(entries, options, callbacks);
  }

  const allocated = allocatePrimitiveProcessingOptions(entries, options);
  const serializedEntries = serializePrimitiveEntries(serializablePrimitiveEntries(entries), { includeImages: false });
  const inputs: PrimitiveWorkerInput[] = serializedEntries.map((entry, index) => ({
    entry,
    options: allocated[index]?.options ?? options,
  }));
  const partitions = partitionPrimitiveWorkerInputs(inputs, workerCount).filter((partition) => partition.length > 0);
  const handles = createWorkerHandles(partitions);
  if (!handles) return stagedFallback(entries, options, callbacks);

  try {
    await Promise.all(handles.map(initializeWorker));
  } catch {
    terminateWorkers(handles);
    return stagedFallback(entries, options, callbacks);
  }

  try {
    await runStage(handles, 'virtual-edges', entries.length, callbacks);
    await runStage(handles, 'simplification', entries.length, callbacks);
    const results = await runStage(handles, 'attribute-transfer', entries.length, callbacks);
    return deserializePrimitiveStageResults(results, deserializePrimitiveEntries(serializedEntries));
  } finally {
    terminateWorkers(handles);
  }
}
