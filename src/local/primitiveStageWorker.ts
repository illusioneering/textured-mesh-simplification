import { parentPort } from 'node:worker_threads';
import { PrimitiveStageBatch, type StagedPrimitiveInput } from '../pipeline/primitiveStagedProcessing';
import {
  deserializePrimitiveEntries,
  serializeFullRawSimplificationResult,
  serializeTransferredMeshAttributes,
} from '../pipeline/primitiveSerialization';
import type {
  PrimitiveStageWorkerRequest,
  PrimitiveStageWorkerResponse,
  SerializedPrimitiveStageResult,
} from '../pipeline/primitiveWorkerProtocol';

let batch: PrimitiveStageBatch | null = null;
let meshOrdinalById = new Map<string, number>();

function post(message: PrimitiveStageWorkerResponse): void {
  parentPort?.postMessage(message);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function errorStack(error: unknown): string | undefined {
  return error instanceof Error ? error.stack : undefined;
}

function requireBatch(): PrimitiveStageBatch {
  if (!batch) throw new Error('Primitive stage worker has not been initialized.');
  return batch;
}

function handleInit(request: Extract<PrimitiveStageWorkerRequest, { type: 'init' }>): void {
  const entries = deserializePrimitiveEntries(request.inputs.map((input) => input.entry));
  meshOrdinalById = new Map(entries.map((entry) => [entry.id, entry.meshOrdinal]));
  const stagedInputs: StagedPrimitiveInput[] = entries.map((entry, index) => {
    const input = request.inputs[index];
    if (!input) throw new Error(`Missing worker input for primitive ${entry.id}.`);
    return {
      entry,
      options: input.options,
    };
  });
  batch = new PrimitiveStageBatch(stagedInputs);
  post({ type: 'ready', id: request.id });
}

function serializeStageResults(): SerializedPrimitiveStageResult[] {
  return requireBatch().result().entries.map((entry) => ({
    id: entry.id,
    meshOrdinal: meshOrdinalById.get(entry.id) ?? 0,
    raw: serializeFullRawSimplificationResult(entry.geometry.raw),
    ...(entry.transferredAttributes
      ? { transferredAttributes: serializeTransferredMeshAttributes(entry.transferredAttributes) }
      : {}),
    elapsedSeconds: entry.geometry.elapsedSeconds,
  }));
}

function handleRunStage(request: Extract<PrimitiveStageWorkerRequest, { type: 'run-stage' }>): void {
  const activeBatch = requireBatch();
  if (request.stage === 'virtual-edges') {
    activeBatch.runVirtualEdgeStage({
      onVirtualEdgeProgress: (primitiveId, progress) => {
        post({ type: 'stage-progress', id: request.id, stage: request.stage, primitiveId, progress });
      },
    });
    post({ type: 'stage-complete', id: request.id, stage: request.stage });
    return;
  }
  if (request.stage === 'simplification') {
    activeBatch.runSimplificationStage({
      onSimplificationProgress: (primitiveId, progress) => {
        post({ type: 'stage-progress', id: request.id, stage: request.stage, primitiveId, progress });
      },
    });
    post({ type: 'stage-complete', id: request.id, stage: request.stage });
    return;
  }
  activeBatch.runAttributeTransferStage({
    onAttributeTransferProgress: (primitiveId, progress) => {
      post({ type: 'stage-progress', id: request.id, stage: request.stage, primitiveId, progress });
    },
  });
  post({ type: 'stage-complete', id: request.id, stage: request.stage, results: serializeStageResults() });
}

function handleDispose(request: Extract<PrimitiveStageWorkerRequest, { type: 'dispose' }>): void {
  batch = null;
  meshOrdinalById = new Map();
  post({ type: 'disposed', id: request.id });
}

if (!parentPort) throw new Error('Primitive stage worker requires a worker_threads parentPort.');

parentPort.on('message', (request: PrimitiveStageWorkerRequest) => {
  try {
    if (request.type === 'init') handleInit(request);
    else if (request.type === 'run-stage') handleRunStage(request);
    else handleDispose(request);
  } catch (error: unknown) {
    const stack = errorStack(error);
    post({
      type: 'error',
      id: request.id,
      message: errorMessage(error),
      ...(stack ? { stack } : {}),
    });
  }
});
