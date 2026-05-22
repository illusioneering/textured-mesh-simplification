import { bakeTextureForSimplifiedGeometry, processGeometryOnly } from '../pipeline/process';
import { bakePrimitiveTextures, type PrimitiveGeometryProcessingResult } from '../pipeline/sceneProcessing';
import type { RawSimplificationResult, SimplificationProgress, VirtualEdgeProgress } from '../simplification/types';
import type { AggregatePrimitiveStageProgress } from '../pipeline/primitiveWorkerProtocol';
import type { BakeTextureProgress } from '../texture/types';
import {
  aggregateSimplificationProgressMessage,
  aggregateAttributeTransferProgressMessage,
  aggregateVirtualEdgeProgressMessage,
  bakeProgressMessage,
  primitiveBakeCompleteMessage,
  primitiveBakeStartMessage,
  shouldPostAggregatePrimitiveStageProgress,
  shouldPostBakeProgress,
  shouldPostSimplificationProgress,
  simplificationProgressMessage,
  virtualEdgeProgressMessage,
  type SimplificationProgressPostState,
} from './progressMessages';
import { processPrimitiveGeometriesInBrowserWorkers } from './primitiveWorkerPool';
import { createBrowserTextureBakeBatchRunner } from './textureBakeWorkerPool';
import {
  collectTransferables,
  deserializePrimitiveEntries,
  deserializeRawMesh,
  deserializeTexturedRawMesh,
  serializeGeometryProcessingResult,
  serializePrimitiveGeometryProcessingResult,
  serializePrimitiveSceneProcessingResult,
  serializeTexturedProcessingResult,
} from './serialization';
import type { WorkerRequestMessage, WorkerResponseMessage } from './workerProtocol';

type WorkerLikeScope = {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<WorkerRequestMessage>) => void) | null;
};

const workerSelf = self as unknown as WorkerLikeScope;
const BROWSER_MAX_BAKE_OUTPUT_TEXTURE_BYTES = 256 * 1024 * 1024;
let cachedSimplification: RawSimplificationResult | null = null;
let cachedPrimitiveSimplification: PrimitiveGeometryProcessingResult | null = null;
let cachedPrimitiveEntries: Array<{ id: string; meshOrdinal: number }> = [];

interface PrimitiveMessageMeta {
  index: number;
  total: number;
  label: string;
}

function post(message: WorkerResponseMessage): void {
  if (message.type === 'done') workerSelf.postMessage(message, collectTransferables(message));
  else workerSelf.postMessage(message);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function errorStack(error: unknown): string | undefined {
  return error instanceof Error ? error.stack : undefined;
}

function createSimplificationProgressPoster(id: string): (progress: SimplificationProgress) => void {
  const progressState: SimplificationProgressPostState = { lastPostTimeMs: performance.now() };
  return (progress: SimplificationProgress): void => {
    const now = performance.now();
    if (!shouldPostSimplificationProgress(progress, progressState, now)) return;
    progressState.lastPostTimeMs = now;
    post({
      type: 'progress',
      id,
      phase: 'simplifying',
      message: simplificationProgressMessage(progress),
      progress,
    });
  };
}

function createVirtualEdgeProgressPoster(id: string): (progress: VirtualEdgeProgress) => void {
  const progressState: SimplificationProgressPostState = { lastPostTimeMs: 0 };
  return (progress: VirtualEdgeProgress): void => {
    const now = performance.now();
    if (progress.phase === 'searching-pairs' && !shouldPostSimplificationProgress({
      iteration: progress.processedFaces,
      activeFaces: progress.totalFaces,
      activeVertices: 0,
      activeEdges: progress.generatedVirtualEdges,
      lastCost: progress.exactDistanceTests,
    }, progressState, now)) {
      return;
    }
    progressState.lastPostTimeMs = now;
    post({
      type: 'progress',
      id,
      phase: 'simplifying',
      message: virtualEdgeProgressMessage(progress),
      virtualEdgeProgress: progress,
    });
  };
}

function primitiveMessageMetadata(entries: ReadonlyArray<{ id: string; label?: string }>): Map<string, PrimitiveMessageMeta> {
  return new Map(entries.map((entry, index) => [
    entry.id,
    {
      index,
      total: entries.length,
      label: entry.label && entry.label.length > 0 ? entry.label : `Primitive ${index + 1}`,
    },
  ]));
}

function primitiveMetaFor(id: string, metadata: ReadonlyMap<string, PrimitiveMessageMeta>): PrimitiveMessageMeta {
  return metadata.get(id) ?? { index: 0, total: 1, label: id };
}

function aggregatePrimitiveStageMessage(progress: AggregatePrimitiveStageProgress): string {
  if (progress.stage === 'virtual-edges') return aggregateVirtualEdgeProgressMessage(progress);
  if (progress.stage === 'simplification') return aggregateSimplificationProgressMessage(progress);
  return aggregateAttributeTransferProgressMessage(progress);
}

function createAggregatePrimitiveStageProgressPoster(id: string): (progress: AggregatePrimitiveStageProgress) => void {
  const progressState: SimplificationProgressPostState = { lastPostTimeMs: 0 };
  return (primitiveStageProgress: AggregatePrimitiveStageProgress): void => {
    const now = performance.now();
    if (!shouldPostAggregatePrimitiveStageProgress(progressState, now)) return;
    progressState.lastPostTimeMs = now;
    post({
      type: 'progress',
      id,
      phase: 'simplifying',
      message: aggregatePrimitiveStageMessage(primitiveStageProgress),
      primitiveStageProgress,
    });
  };
}

function createBakeProgressPoster(id: string): (progress: BakeTextureProgress) => void {
  const progressState: SimplificationProgressPostState = { lastPostTimeMs: 0 };
  return (progress: BakeTextureProgress): void => {
    const now = performance.now();
    if (progress.stage === 'resampling') {
      if (!shouldPostBakeProgress(progressState, now)) return;
      progressState.lastPostTimeMs = now;
    }
    post({
      type: 'progress',
      id,
      phase: 'baking',
      message: bakeProgressMessage(progress),
      bakeProgress: progress,
    });
  };
}

async function handleSimplify(request: Extract<WorkerRequestMessage, { type: 'simplify' }>): Promise<void> {
  cachedSimplification = null;
  cachedPrimitiveSimplification = null;
  cachedPrimitiveEntries = [];
  post({ type: 'progress', id: request.id, phase: 'simplifying', message: 'Starting geometry simplification…' });
  if (request.input.kind === 'primitives') {
    const entries = deserializePrimitiveEntries(request.input.entries);
    const postAggregatePrimitiveStageProgress = createAggregatePrimitiveStageProgressPoster(request.id);
    const result = await processPrimitiveGeometriesInBrowserWorkers(
      entries,
      request.options,
      {
        onAggregateProgress: postAggregatePrimitiveStageProgress,
      },
    );
    cachedPrimitiveSimplification = result;
    cachedPrimitiveEntries = entries.map((entry) => ({ id: entry.id, meshOrdinal: entry.meshOrdinal }));
    post({ type: 'done', id: request.id, result: serializePrimitiveGeometryProcessingResult(result, cachedPrimitiveEntries) });
    return;
  }
  const result = processGeometryOnly(
    deserializeRawMesh(request.input.rawMesh),
    request.options,
    createSimplificationProgressPoster(request.id),
    {
      onVirtualEdgeProgress: createVirtualEdgeProgressPoster(request.id),
    },
  );
  cachedSimplification = result.raw;
  post({ type: 'done', id: request.id, result: serializeGeometryProcessingResult(result) });
}

async function handleBake(request: Extract<WorkerRequestMessage, { type: 'bake' }>): Promise<void> {
  if (!('rawMesh' in request.source)) {
    if (!cachedPrimitiveSimplification) throw new Error('Simplify geometry before baking a texture atlas.');
    const entries = deserializePrimitiveEntries(request.source.entries);
    const sourceById = new Map(entries.map((entry) => [entry.id, entry]));
    cachedPrimitiveSimplification.entries.forEach((entry) => {
      const source = sourceById.get(entry.id);
      if (source) entry.source = source;
    });
    post({ type: 'progress', id: request.id, phase: 'baking', message: 'Baking standard material texture atlases…' });
    const primitiveMetadata = primitiveMessageMetadata(entries);
    const batchRunner = createBrowserTextureBakeBatchRunner();
    const postBakeProgress = createBakeProgressPoster(request.id);
    const result = await bakePrimitiveTextures(cachedPrimitiveSimplification, request.options, {
      onBakeStart: (entryId) => {
        post({
          type: 'progress',
          id: request.id,
          phase: 'baking',
          message: primitiveBakeStartMessage(primitiveMetaFor(entryId, primitiveMetadata)),
        });
      },
      onBakeComplete: (entryId, baked) => {
        post({
          type: 'progress',
          id: request.id,
          phase: 'baking',
          message: primitiveBakeCompleteMessage({
            ...primitiveMetaFor(entryId, primitiveMetadata),
            filledPixels: baked.stats.filledPixels,
            unmappedPixels: baked.stats.unmappedPixels,
          }),
        });
      },
      onBakeProgress: (_entryId, progress) => {
        postBakeProgress(progress);
      },
    }, {
      batchRunner,
      maxOutputTextureBytes: BROWSER_MAX_BAKE_OUTPUT_TEXTURE_BYTES,
    });
    post({ type: 'done', id: request.id, result: serializePrimitiveSceneProcessingResult(result, cachedPrimitiveEntries) });
    return;
  }
  if (!cachedSimplification) throw new Error('Simplify geometry before baking a texture atlas.');
  post({ type: 'progress', id: request.id, phase: 'baking', message: 'Baking standard material texture atlas…' });
  const batchRunner = createBrowserTextureBakeBatchRunner();
  const postBakeProgress = createBakeProgressPoster(request.id);
  const result = await bakeTextureForSimplifiedGeometry(
    deserializeTexturedRawMesh(request.source),
    cachedSimplification,
    request.options,
    postBakeProgress,
    {
      batchRunner,
      maxOutputTextureBytes: BROWSER_MAX_BAKE_OUTPUT_TEXTURE_BYTES,
    },
  );
  post({ type: 'done', id: request.id, result: serializeTexturedProcessingResult(result) });
}

async function handleRequest(request: WorkerRequestMessage): Promise<void> {
  if (request.type === 'simplify') {
    await handleSimplify(request);
    return;
  }
  if (request.type === 'bake') await handleBake(request);
}

workerSelf.onmessage = (event: MessageEvent<WorkerRequestMessage>) => {
  void handleRequest(event.data).catch((error: unknown) => {
    const stack = errorStack(error);
    post({
      type: 'error',
      id: event.data.id,
      message: errorMessage(error),
      ...(stack ? { stack } : {}),
    });
  });
};

export {};
