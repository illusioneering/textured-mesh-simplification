import type { ProcessingOptions } from './options';
import type {
  SerializedFullRawSimplificationResult,
  SerializedPrimitiveProcessingEntry,
  SerializedTransferredMeshAttributes,
} from './primitiveSerialization';

export type PrimitiveParallelStage = 'virtual-edges' | 'simplification' | 'attribute-transfer';

export interface PrimitiveWorkerInput {
  entry: SerializedPrimitiveProcessingEntry;
  options: ProcessingOptions;
}

export interface AggregateVirtualEdgeStageProgress {
  stage: 'virtual-edges';
  completedPrimitives: number;
  totalPrimitives: number;
  processedFaces: number;
  totalFaces: number;
  candidateFacePairs: number;
  exactDistanceTests: number;
  generatedVirtualEdges: number;
}

export interface AggregateSimplificationStageProgress {
  stage: 'simplification';
  completedPrimitives: number;
  totalPrimitives: number;
  collapses: number;
  activeFaces: number;
  activeVertices: number;
  activeEdges: number;
}

export interface AggregateAttributeTransferStageProgress {
  stage: 'attribute-transfer';
  completedPrimitives: number;
  totalPrimitives: number;
  processedFaces: number;
  totalFaces: number;
}

export type AggregatePrimitiveStageProgress =
  | AggregateVirtualEdgeStageProgress
  | AggregateSimplificationStageProgress
  | AggregateAttributeTransferStageProgress;

export type PrimitiveStageWorkerRequest =
  | { type: 'init'; id: string; inputs: PrimitiveWorkerInput[] }
  | { type: 'run-stage'; id: string; stage: PrimitiveParallelStage }
  | { type: 'dispose'; id: string };

export interface SerializedPrimitiveStageResult {
  id: string;
  meshOrdinal: number;
  raw: SerializedFullRawSimplificationResult;
  transferredAttributes?: SerializedTransferredMeshAttributes;
  elapsedSeconds: number;
}

export type PrimitiveStageWorkerResponse =
  | { type: 'ready'; id: string }
  | { type: 'stage-progress'; id: string; stage: PrimitiveParallelStage; primitiveId: string; progress: unknown }
  | { type: 'stage-complete'; id: string; stage: PrimitiveParallelStage; results?: SerializedPrimitiveStageResult[] }
  | { type: 'disposed'; id: string }
  | { type: 'error'; id: string; message: string; stack?: string };
