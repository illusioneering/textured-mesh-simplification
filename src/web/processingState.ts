import type { Object3D } from 'three';
import type { PrimitiveGroupingMode, ProcessingOptions } from '../pipeline/options';
import type { PrimitiveExtractionOptions, PrimitiveExtractionResult } from '../pipeline/primitiveExtraction';
import type { SerializedPrimitiveProcessingResult } from './serialization';
import type { BrowserPrimitiveApplyMetadata } from './browserGltfIo';

export interface SimplificationOptionsKey {
  primitiveGrouping: PrimitiveGroupingMode;
  weldVertices: boolean;
  recomputeNormals: boolean;
}

export interface ActiveProcessedState {
  assetRevision: number;
  simplifyOptions: ProcessingOptions;
  simplificationOptionsKey: SimplificationOptionsKey;
  bakeableEntryCount: number;
  texturedBakeableEntryCount: number;
  applyMetadataByEntryId: Map<string, BrowserPrimitiveApplyMetadata>;
  simplifyResult: SerializedPrimitiveProcessingResult;
  outputScene?: Object3D;
}

export interface PendingSimplifyState {
  simplifyOptions: ProcessingOptions;
  simplificationOptionsKey: SimplificationOptionsKey;
  extraction: PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>;
}

export function createSimplificationOptionsKey(options: SimplificationOptionsKey): SimplificationOptionsKey {
  return {
    primitiveGrouping: options.primitiveGrouping,
    weldVertices: options.weldVertices,
    recomputeNormals: options.recomputeNormals,
  };
}

function sameSimplificationOptionsKey(left: SimplificationOptionsKey, right: SimplificationOptionsKey): boolean {
  return left.primitiveGrouping === right.primitiveGrouping
    && left.weldVertices === right.weldVertices
    && left.recomputeNormals === right.recomputeNormals;
}

export function isActiveProcessedStateCurrent(
  state: ActiveProcessedState | null,
  assetRevision: number,
  simplificationOptionsKey: SimplificationOptionsKey,
): state is ActiveProcessedState {
  return state !== null
    && state.assetRevision === assetRevision
    && sameSimplificationOptionsKey(state.simplificationOptionsKey, simplificationOptionsKey);
}

export function isActiveProcessedOutputCurrent(
  state: ActiveProcessedState | null,
  outputScene: Object3D | null,
  assetRevision: number,
  simplificationOptionsKey: SimplificationOptionsKey,
): state is ActiveProcessedState {
  return outputScene !== null && isActiveProcessedStateCurrent(state, assetRevision, simplificationOptionsKey);
}

export function isActiveProcessedStateForAsset(
  state: ActiveProcessedState | null,
  assetRevision: number,
): state is ActiveProcessedState {
  return state !== null && state.assetRevision === assetRevision;
}

export function isActiveProcessedOutputForAsset(
  state: ActiveProcessedState | null,
  outputScene: Object3D | null,
  assetRevision: number,
): state is ActiveProcessedState {
  return outputScene !== null && isActiveProcessedStateForAsset(state, assetRevision);
}

export function isActiveProcessedBakeAvailable(
  state: ActiveProcessedState | null,
  assetRevision: number,
  simplificationOptionsKey: SimplificationOptionsKey,
): state is ActiveProcessedState {
  return isActiveProcessedStateCurrent(state, assetRevision, simplificationOptionsKey)
    && state.texturedBakeableEntryCount > 0;
}

export function isActiveProcessedBakeAvailableForAsset(
  state: ActiveProcessedState | null,
  assetRevision: number,
): state is ActiveProcessedState {
  return isActiveProcessedStateForAsset(state, assetRevision)
    && state.texturedBakeableEntryCount > 0;
}

export function createActiveProcessedState(options: {
  assetRevision: number;
  pending: PendingSimplifyState;
  simplifyResult: SerializedPrimitiveProcessingResult;
  outputScene?: Object3D;
}): ActiveProcessedState {
  return {
    assetRevision: options.assetRevision,
    simplifyOptions: options.pending.simplifyOptions,
    simplificationOptionsKey: options.pending.simplificationOptionsKey,
    bakeableEntryCount: options.pending.extraction.summary.bakeableEntryCount,
    texturedBakeableEntryCount: options.pending.extraction.summary.bakeableEntryCount,
    applyMetadataByEntryId: new Map(options.pending.extraction.applyMetadataByEntryId),
    simplifyResult: options.simplifyResult,
    ...(options.outputScene ? { outputScene: options.outputScene } : {}),
  };
}
