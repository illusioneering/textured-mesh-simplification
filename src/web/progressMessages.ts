import type { SimplificationProgress, VirtualEdgeProgress } from '../simplification/types';
import type { BakeTextureProgress } from '../texture/types';
import type {
  AggregateAttributeTransferStageProgress,
  AggregateSimplificationStageProgress,
  AggregateVirtualEdgeStageProgress,
} from '../pipeline/primitiveWorkerProtocol';

export const SIMPLIFICATION_PROGRESS_INTERVAL_MS = 750;
export const AGGREGATE_PRIMITIVE_STAGE_PROGRESS_INTERVAL_MS = 500;
export const BAKE_PROGRESS_INTERVAL_MS = 500;

export interface SimplificationProgressPostState {
  lastPostTimeMs: number;
}

export function shouldPostSimplificationProgress(
  _progress: SimplificationProgress,
  state: SimplificationProgressPostState,
  nowMs: number,
): boolean {
  return nowMs - state.lastPostTimeMs >= SIMPLIFICATION_PROGRESS_INTERVAL_MS;
}

export function shouldPostAggregatePrimitiveStageProgress(
  state: SimplificationProgressPostState,
  nowMs: number,
): boolean {
  return nowMs - state.lastPostTimeMs >= AGGREGATE_PRIMITIVE_STAGE_PROGRESS_INTERVAL_MS;
}

export function shouldPostBakeProgress(
  state: SimplificationProgressPostState,
  nowMs: number,
): boolean {
  return nowMs - state.lastPostTimeMs >= BAKE_PROGRESS_INTERVAL_MS;
}

export function simplificationProgressMessage(progress: SimplificationProgress): string {
  return `Geometry iteration ${progress.iteration.toLocaleString()}: ${progress.activeFaces.toLocaleString()} active faces.`;
}

export function virtualEdgeProgressMessage(progress: VirtualEdgeProgress): string {
  if (progress.phase === 'building-buckets') {
    return `Virtual-edge search: indexing ${progress.totalFaces.toLocaleString()} faces.`;
  }
  const percent = progress.totalFaces > 0 ? Math.round((progress.processedFaces / progress.totalFaces) * 100) : 100;
  return `Virtual-edge search: ${percent.toLocaleString()}% of faces, `
    + `${progress.candidateFacePairs.toLocaleString()} candidate pairs, `
    + `${progress.exactDistanceTests.toLocaleString()} exact tests, `
    + `${progress.generatedVirtualEdges.toLocaleString()} virtual edges.`;
}

export function geometryCompleteMessage(outputFaces: number, outputVertices: number): string {
  return `Geometry simplification complete: ${outputFaces.toLocaleString()} faces, ${outputVertices.toLocaleString()} vertices.`;
}

export interface PrimitiveLifecycleMessageInput {
  index: number;
  total: number;
}

export interface PrimitiveStartMessageInput extends PrimitiveLifecycleMessageInput {
  label: string;
}

export interface PrimitiveGeometryCompleteMessageInput extends PrimitiveLifecycleMessageInput {
  outputFaces: number;
  outputVertices: number;
}

export interface PrimitiveBakeCompleteMessageInput extends PrimitiveLifecycleMessageInput {
  filledPixels: number;
  unmappedPixels: number;
}

function primitiveOrdinal(input: PrimitiveLifecycleMessageInput): string {
  return `${(input.index + 1).toLocaleString()}/${input.total.toLocaleString()}`;
}

function primitiveCompletion(progress: { completedPrimitives: number; totalPrimitives: number }): string {
  return `${progress.completedPrimitives.toLocaleString()}/${progress.totalPrimitives.toLocaleString()} primitives complete`;
}

function facePercent(progress: { processedFaces: number; totalFaces: number }): string {
  const percent = progress.totalFaces > 0 ? Math.round((progress.processedFaces / progress.totalFaces) * 100) : 100;
  return `${percent.toLocaleString()}%`;
}

export function primitiveGeometryStartMessage(input: PrimitiveStartMessageInput): string {
  return `Simplifying primitive ${primitiveOrdinal(input)}: ${input.label}…`;
}

export function primitiveGeometryCompleteMessage(input: PrimitiveGeometryCompleteMessageInput): string {
  return `Primitive ${primitiveOrdinal(input)} geometry complete: `
    + `${input.outputFaces.toLocaleString()} faces, ${input.outputVertices.toLocaleString()} vertices.`;
}

export function primitiveBakeStartMessage(input: PrimitiveStartMessageInput): string {
  return `Baking primitive ${primitiveOrdinal(input)}: ${input.label}…`;
}

export function primitiveBakeCompleteMessage(input: PrimitiveBakeCompleteMessageInput): string {
  return `Primitive ${primitiveOrdinal(input)} texture bake complete: `
    + `${input.filledPixels.toLocaleString()} filled pixels, ${input.unmappedPixels.toLocaleString()} unmapped samples.`;
}

export function aggregateVirtualEdgeProgressMessage(progress: AggregateVirtualEdgeStageProgress): string {
  return `Virtual-edge stage: ${facePercent(progress)} of faces, ${primitiveCompletion(progress)}, `
    + `${progress.candidateFacePairs.toLocaleString()} candidate pairs, `
    + `${progress.exactDistanceTests.toLocaleString()} exact tests, `
    + `${progress.generatedVirtualEdges.toLocaleString()} virtual edges.`;
}

export function aggregateSimplificationProgressMessage(progress: AggregateSimplificationStageProgress): string {
  return `Simplification stage: ${primitiveCompletion(progress)}, `
    + `${progress.collapses.toLocaleString()} collapses, ${progress.activeFaces.toLocaleString()} active faces.`;
}

export function aggregateAttributeTransferProgressMessage(progress: AggregateAttributeTransferStageProgress): string {
  return `Attribute transfer stage: ${facePercent(progress)} of faces, ${primitiveCompletion(progress)}.`;
}

export function bakeProgressMessage(progress: BakeTextureProgress): string {
  if (progress.stage === 'atlas-created') {
    const islandCount = progress.islandCount ?? progress.totalFaces;
    return `Texture atlas generated: ${islandCount.toLocaleString()} atlas chart islands for ${progress.totalFaces.toLocaleString()} faces. Baking standard material texture pixels…`;
  }
  if (progress.stage === 'resampling') {
    const percent = progress.totalSamples > 0 ? Math.round((progress.processedSamples / progress.totalSamples) * 100) : 100;
    return `Texture resampling: ${percent.toLocaleString()}% of atlas samples, `
      + `${progress.completedBatches.toLocaleString()}/${progress.totalBatches.toLocaleString()} batches complete, `
      + `${progress.mappedPixels.toLocaleString()} mapped samples, `
      + `${progress.unmappedPixels.toLocaleString()} unmapped samples.`;
  }
  if (progress.stage === 'dilating') {
    return `Expanding texture gutters (pass ${progress.gutterPass.toLocaleString()} of ${progress.gutterPasses.toLocaleString()})…`;
  }
  return `Texture baking complete: ${progress.filledPixels.toLocaleString()} filled pixels, ${progress.mappedPixels.toLocaleString()} mapped samples, ${progress.unmappedPixels.toLocaleString()} unmapped samples.`;
}
