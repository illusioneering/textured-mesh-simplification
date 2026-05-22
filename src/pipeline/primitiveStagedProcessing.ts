import {
  createSimplificationState,
  finalizeSimplification,
  runSimplificationStage,
  runVirtualEdgeStage,
  type SimplificationState,
} from '../simplification/simplifier';
import type {
  RawSimplificationResult,
  SimplificationProgress,
  SimplificationResult,
  VirtualEdgeDiagnostics,
  VirtualEdgeProgress,
} from '../simplification/types';
import type { TransferredMeshAttributes } from '../simplification/attributes';
import type { AttributeTransferProgress } from '../simplification/attributeTransfer';
import type { GeometryProcessingResult } from './process';
import { type ProcessingOptions, toSimplifyOptions } from './options';
import { prepareOutputTransferredAttributes, shouldAttachPreparedOutputAttributes } from './outputAttributes';
import type {
  ProcessablePrimitiveEntry,
  PrimitiveGeometryProcessingResult,
  ProcessedPrimitiveGeometryEntry,
} from './sceneProcessing';

export interface StagedPrimitiveInput {
  entry: ProcessablePrimitiveEntry;
  options: ProcessingOptions;
}

export interface PrimitiveStageCallbacks {
  onVirtualEdgeStageStart?: (totalPrimitives: number) => void;
  onVirtualEdgeProgress?: (id: string, progress: VirtualEdgeProgress) => void;
  onVirtualEdgePrimitiveComplete?: (id: string, diagnostics: VirtualEdgeDiagnostics) => void;
  onSimplificationStageStart?: (totalPrimitives: number) => void;
  onSimplificationProgress?: (id: string, progress: SimplificationProgress) => void;
  onSimplificationPrimitiveComplete?: (id: string, result: RawSimplificationResult) => void;
  onAttributeTransferStageStart?: (totalPrimitives: number) => void;
  onAttributeTransferProgress?: (id: string, progress: AttributeTransferProgress) => void;
  onAttributeTransferPrimitiveComplete?: (id: string, attributes: TransferredMeshAttributes | undefined) => void;
}

interface PrimitiveStageItem {
  input: StagedPrimitiveInput;
  state: SimplificationState;
  elapsedSeconds: number;
  raw: RawSimplificationResult | undefined;
  transferredAttributes: TransferredMeshAttributes | undefined;
}

interface WeightedStats {
  inputVertices: number;
  inputFaces: number;
  outputVertices: number;
  outputFaces: number;
  physicalEdges: number;
  virtualEdges: number;
  collapses: number;
  stoppedReason: SimplificationResult['stats']['stoppedReason'];
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function combineStoppedReason(
  current: SimplificationResult['stats']['stoppedReason'],
  next: SimplificationResult['stats']['stoppedReason'],
): SimplificationResult['stats']['stoppedReason'] {
  if (current === 'max-iterations' || next === 'max-iterations') return 'max-iterations';
  if (current === 'queue-empty' || next === 'queue-empty') return 'queue-empty';
  return 'target-reached';
}

export function aggregatePrimitiveStats(
  statsEntries: readonly SimplificationResult['stats'][],
): SimplificationResult['stats'] {
  let stoppedReason: SimplificationResult['stats']['stoppedReason'] = 'target-reached';

  return statsEntries.reduce<WeightedStats>((sum, statsEntry) => {
    stoppedReason = combineStoppedReason(stoppedReason, statsEntry.stoppedReason);
    return {
      inputVertices: sum.inputVertices + statsEntry.inputVertices,
      inputFaces: sum.inputFaces + statsEntry.inputFaces,
      outputVertices: sum.outputVertices + statsEntry.outputVertices,
      outputFaces: sum.outputFaces + statsEntry.outputFaces,
      physicalEdges: sum.physicalEdges + statsEntry.physicalEdges,
      virtualEdges: sum.virtualEdges + statsEntry.virtualEdges,
      collapses: sum.collapses + statsEntry.collapses,
      stoppedReason,
    };
  }, {
    inputVertices: 0,
    inputFaces: 0,
    outputVertices: 0,
    outputFaces: 0,
    physicalEdges: 0,
    virtualEdges: 0,
    collapses: 0,
    stoppedReason: 'target-reached',
  });
}

export class PrimitiveStageBatch {
  private readonly items: PrimitiveStageItem[];

  constructor(inputs: readonly StagedPrimitiveInput[]) {
    this.items = inputs.map((input) => ({
      input,
      state: createSimplificationState(input.entry.rawMesh, toSimplifyOptions(input.options)),
      elapsedSeconds: 0,
      raw: undefined,
      transferredAttributes: undefined,
    }));
  }

  runVirtualEdgeStage(callbacks: PrimitiveStageCallbacks = {}): void {
    callbacks.onVirtualEdgeStageStart?.(this.items.length);
    for (const item of this.items) {
      const started = nowMs();
      const id = item.input.entry.id;
      const diagnostics = runVirtualEdgeStage(
        item.state,
        callbacks.onVirtualEdgeProgress
          ? (progress) => callbacks.onVirtualEdgeProgress?.(id, progress)
          : undefined,
      );
      item.elapsedSeconds += (nowMs() - started) / 1000;
      callbacks.onVirtualEdgePrimitiveComplete?.(id, diagnostics);
    }
  }

  runSimplificationStage(callbacks: PrimitiveStageCallbacks = {}): void {
    callbacks.onSimplificationStageStart?.(this.items.length);
    for (const item of this.items) {
      const started = nowMs();
      const id = item.input.entry.id;
      runSimplificationStage(
        item.state,
        callbacks.onSimplificationProgress
          ? (progress) => callbacks.onSimplificationProgress?.(id, progress)
          : undefined,
      );
      item.raw = finalizeSimplification(item.state);
      item.elapsedSeconds += (nowMs() - started) / 1000;
      callbacks.onSimplificationPrimitiveComplete?.(id, item.raw);
    }
  }

  runAttributeTransferStage(callbacks: PrimitiveStageCallbacks = {}): void {
    callbacks.onAttributeTransferStageStart?.(this.items.length);
    for (const item of this.items) {
      const started = nowMs();
      const id = item.input.entry.id;
      const source = item.input.entry.texturedRawMesh ?? null;
      const raw = this.requireRaw(item);
      const sourceFaceAttributes = source?.faceAttributes;
      const preparedAttributes = prepareOutputTransferredAttributes({
        raw,
        recomputeNormals: item.input.options.recomputeNormals,
        ...(sourceFaceAttributes ? { sourceFaceAttributes } : {}),
        ...(callbacks.onAttributeTransferProgress
          ? {
              onProgress: (progress: AttributeTransferProgress) => {
                callbacks.onAttributeTransferProgress?.(id, progress);
              },
            }
          : {}),
      });
      const attributes = shouldAttachPreparedOutputAttributes({
        raw,
        recomputeNormals: item.input.options.recomputeNormals,
        requiresAttributeTransfer: item.input.entry.requiresAttributeTransfer === true,
        ...(sourceFaceAttributes ? { sourceFaceAttributes } : {}),
      })
        ? preparedAttributes
        : undefined;
      item.transferredAttributes = attributes;
      item.elapsedSeconds += (nowMs() - started) / 1000;
      callbacks.onAttributeTransferPrimitiveComplete?.(id, attributes);
    }
  }

  result(): PrimitiveGeometryProcessingResult {
    const entries = this.items.map<ProcessedPrimitiveGeometryEntry>((item) => {
      const raw = this.requireRaw(item);
      const geometry: GeometryProcessingResult = {
        raw,
        elapsedSeconds: item.elapsedSeconds,
      };
      return {
        id: item.input.entry.id,
        source: item.input.entry,
        geometry,
        raw,
        ...(item.transferredAttributes ? { transferredAttributes: item.transferredAttributes } : {}),
      };
    });

    return {
      entries,
      stats: aggregatePrimitiveStats(entries.map((entry) => entry.geometry.raw.stats)),
      elapsedSeconds: this.items.reduce((sum, item) => sum + item.elapsedSeconds, 0),
    };
  }

  private requireRaw(item: PrimitiveStageItem): RawSimplificationResult {
    if (!item.raw) {
      throw new Error(`Primitive ${item.input.entry.id} has not been simplified.`);
    }
    return item.raw;
  }
}

export function processPrimitiveGeometriesStaged(
  inputs: readonly StagedPrimitiveInput[],
  callbacks: PrimitiveStageCallbacks = {},
): PrimitiveGeometryProcessingResult {
  const batch = new PrimitiveStageBatch(inputs);
  batch.runVirtualEdgeStage(callbacks);
  batch.runSimplificationStage(callbacks);
  batch.runAttributeTransferStage(callbacks);
  return batch.result();
}
