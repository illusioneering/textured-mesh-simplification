import type { SimplificationProgress, VirtualEdgeProgress } from '../simplification/types';
import type { BakeTextureProgress } from '../texture/types';
import type { ProcessingOptions } from '../pipeline/options';
import type { AggregatePrimitiveStageProgress } from '../pipeline/primitiveWorkerProtocol';
import type {
  SerializedPrimitiveProcessingEntry,
  SerializedProcessingResult,
  SerializedRawMesh,
  SerializedTexturedRawMesh,
} from './serialization';

export type ProcessingPhase = 'idle' | 'reading' | 'simplifying' | 'baking' | 'exporting' | 'done' | 'error';

export interface WorkerSimplifyRequest {
  type: 'simplify';
  id: string;
  input:
    | { kind: 'geometry'; rawMesh: SerializedRawMesh }
    | { kind: 'primitives'; entries: SerializedPrimitiveProcessingEntry[] };
  options: ProcessingOptions;
}

export interface WorkerBakeRequest {
  type: 'bake';
  id: string;
  source:
    | SerializedTexturedRawMesh
    | { kind: 'primitives'; entries: SerializedPrimitiveProcessingEntry[] };
  options: ProcessingOptions;
}

export interface WorkerProgressMessage {
  type: 'progress';
  id: string;
  phase: ProcessingPhase;
  message: string;
  progress?: SimplificationProgress;
  virtualEdgeProgress?: VirtualEdgeProgress;
  primitiveStageProgress?: AggregatePrimitiveStageProgress;
  bakeProgress?: BakeTextureProgress;
}

export interface WorkerDoneMessage {
  type: 'done';
  id: string;
  result: SerializedProcessingResult;
}

export interface WorkerErrorMessage {
  type: 'error';
  id: string;
  message: string;
  stack?: string;
}

export type WorkerRequestMessage = WorkerSimplifyRequest | WorkerBakeRequest;
export type WorkerResponseMessage = WorkerProgressMessage | WorkerDoneMessage | WorkerErrorMessage;
