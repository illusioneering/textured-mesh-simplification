import type { RawMesh, RawSimplificationResult, SimplificationProgress, SimplificationResult, VirtualEdgeProgress } from '../simplification/types';
import type { TransferredMeshAttributes } from '../simplification/attributes';
import type { BakeTextureProgress, TexturedRawMesh } from '../texture/types';
import type { ProcessingOptions } from './options';
import { prepareOutputTransferredAttributes, shouldAttachPreparedOutputAttributes } from './outputAttributes';
import {
  bakeTextureForSimplifiedGeometry,
  type GeometryProcessingResult,
  processGeometryOnly,
  type TextureBakeProcessingSettings,
  type TexturedProcessingResult,
} from './process';
import { aggregatePrimitiveStats } from './primitiveStagedProcessing';

export interface ProcessablePrimitiveEntry {
  id: string;
  label?: string;
  rawMesh: RawMesh;
  texturedRawMesh?: TexturedRawMesh;
  bakeable: boolean;
  hasPreservableMaterialData?: boolean;
  hasTexturedMaterial?: boolean;
  requiresAttributeTransfer?: boolean;
  transferredAttributes?: TransferredMeshAttributes;
}

export interface AllocatedPrimitiveProcessingOptions {
  id: string;
  entry: ProcessablePrimitiveEntry;
  options: ProcessingOptions;
}

export type ScenePrimitiveProcessingEntry = ProcessablePrimitiveEntry;

export interface ProcessedPrimitiveGeometryEntry {
  id: string;
  source: ProcessablePrimitiveEntry;
  geometry: GeometryProcessingResult;
  raw: GeometryProcessingResult['raw'];
  transferredAttributes?: TransferredMeshAttributes;
}

export interface ProcessedPrimitiveEntry extends ProcessedPrimitiveGeometryEntry {
  baked?: TexturedProcessingResult;
}

export interface PrimitiveGeometryProcessingResult {
  entries: ProcessedPrimitiveGeometryEntry[];
  stats: SimplificationResult['stats'];
  elapsedSeconds: number;
}

export interface PrimitiveSceneProcessingResult {
  entries: ProcessedPrimitiveEntry[];
  stats: SimplificationResult['stats'];
  elapsedSeconds: number;
}

export interface PrimitiveProcessingCallbacks {
  onGeometryStart?: (id: string, entry: ProcessablePrimitiveEntry) => void;
  onGeometryComplete?: (id: string, result: RawSimplificationResult) => void;
  onGeometryProgress?: (id: string, progress: SimplificationProgress) => void;
  onVirtualEdgeProgress?: (id: string, progress: VirtualEdgeProgress) => void;
  onBakeStart?: (id: string, entry: ProcessedPrimitiveGeometryEntry) => void;
  onBakeComplete?: (id: string, result: TexturedProcessingResult['baked']) => void;
  onBakeProgress?: (id: string, progress: BakeTextureProgress) => void;
}

interface WeightedAllocation {
  index: number;
  faceCount: number;
  targetFaceCount: number;
  remainder: number;
}

function cloneOptionsWithTargetFaceCount(options: ProcessingOptions, targetFaceCount: number): ProcessingOptions {
  return {
    ...options,
    target: { kind: 'faces', targetFaceCount },
  };
}

function compareRemainderDescending(a: WeightedAllocation, b: WeightedAllocation): number {
  if (b.remainder !== a.remainder) return b.remainder - a.remainder;
  if (b.faceCount !== a.faceCount) return b.faceCount - a.faceCount;
  return a.index - b.index;
}

export function allocatePrimitiveProcessingOptions(
  entries: readonly ProcessablePrimitiveEntry[],
  options: ProcessingOptions,
): AllocatedPrimitiveProcessingOptions[] {
  if (options.target.kind === 'ratio') {
    return entries.map((entry) => ({ id: entry.id, entry, options }));
  }

  const totalBudget = options.target.targetFaceCount;
  const nonEmptyEntries = entries
    .map((entry, index) => ({ entry, index, faceCount: entry.rawMesh.faces.length }))
    .filter((item) => item.faceCount > 0);
  const totalInputFaces = nonEmptyEntries.reduce((sum, item) => sum + item.faceCount, 0);
  const baseTarget = totalBudget >= nonEmptyEntries.length ? 1 : 0;
  const targets: number[] = entries.map((entry) => (entry.rawMesh.faces.length > 0 ? baseTarget : 0));
  let remainingBudget = totalBudget - targets.reduce((sum, target) => sum + target, 0);

  if (remainingBudget > 0 && totalInputFaces > 0) {
    let assignedByFloor = 0;
    const weighted = nonEmptyEntries.map<WeightedAllocation>(({ index, faceCount }) => {
      const share = (remainingBudget * faceCount) / totalInputFaces;
      const floor = Math.floor(share);
      targets[index] = (targets[index] ?? 0) + floor;
      assignedByFloor += floor;
      return {
        index,
        faceCount,
        targetFaceCount: targets[index] ?? 0,
        remainder: share - floor,
      };
    });
    remainingBudget -= assignedByFloor;

    weighted.sort(compareRemainderDescending);
    for (const item of weighted) {
      if (remainingBudget <= 0) break;
      targets[item.index] = (targets[item.index] ?? 0) + 1;
      remainingBudget -= 1;
    }
  }

  return entries.map((entry, index) => ({
    id: entry.id,
    entry,
    options: cloneOptionsWithTargetFaceCount(options, targets[index] ?? 0),
  }));
}

export function processPrimitiveGeometries(
  entries: readonly ProcessablePrimitiveEntry[],
  options: ProcessingOptions,
  callbacks: PrimitiveProcessingCallbacks = {},
): PrimitiveGeometryProcessingResult {
  const allocatedOptions = allocatePrimitiveProcessingOptions(entries, options);
  const processedEntries = entries.map<ProcessedPrimitiveGeometryEntry>((entry, index) => {
    const primitiveOptions = allocatedOptions[index]?.options ?? options;
    callbacks.onGeometryStart?.(entry.id, entry);
    const geometry = processGeometryOnly(
      entry.rawMesh,
      primitiveOptions,
      callbacks.onGeometryProgress ? (progress) => callbacks.onGeometryProgress?.(entry.id, progress) : undefined,
      {
        ...(callbacks.onVirtualEdgeProgress
          ? { onVirtualEdgeProgress: (progress) => callbacks.onVirtualEdgeProgress?.(entry.id, progress) }
          : {}),
      },
    );
    callbacks.onGeometryComplete?.(entry.id, geometry.raw);
    const sourceFaceAttributes = entry.texturedRawMesh?.faceAttributes;
    const transferredAttributes = prepareOutputTransferredAttributes({
      raw: geometry.raw,
      recomputeNormals: primitiveOptions.recomputeNormals,
      ...(sourceFaceAttributes ? { sourceFaceAttributes } : {}),
    });
    const attachTransferredAttributes = shouldAttachPreparedOutputAttributes({
      raw: geometry.raw,
      recomputeNormals: primitiveOptions.recomputeNormals,
      requiresAttributeTransfer: entry.requiresAttributeTransfer === true,
      ...(sourceFaceAttributes ? { sourceFaceAttributes } : {}),
    });
    return {
      id: entry.id,
      source: entry,
      geometry,
      raw: geometry.raw,
      ...(attachTransferredAttributes ? { transferredAttributes } : {}),
    };
  });

  return {
    entries: processedEntries,
    stats: aggregatePrimitiveStats(processedEntries.map((entry) => entry.geometry.raw.stats)),
    elapsedSeconds: processedEntries.reduce((sum, entry) => sum + entry.geometry.elapsedSeconds, 0),
  };
}

export async function bakePrimitiveTextures(
  geometryResult: PrimitiveGeometryProcessingResult,
  options: ProcessingOptions,
  callbacks: PrimitiveProcessingCallbacks = {},
  settings: TextureBakeProcessingSettings = {},
): Promise<PrimitiveSceneProcessingResult> {
  const entries: ProcessedPrimitiveEntry[] = [];
  for (const entry of geometryResult.entries) {
    const sourceTextured = entry.source.texturedRawMesh ?? null;
    if (!entry.source.bakeable || !sourceTextured) {
      entries.push(entry);
      continue;
    }
    callbacks.onBakeStart?.(entry.id, entry);
    const baked = await bakeTextureForSimplifiedGeometry(
      sourceTextured,
      entry.geometry.raw,
      options,
      callbacks.onBakeProgress ? (progress) => callbacks.onBakeProgress?.(entry.id, progress) : undefined,
      entry.transferredAttributes,
      settings,
    );
    callbacks.onBakeComplete?.(entry.id, baked.baked);
    entries.push({
      ...entry,
      baked,
    });
  }

  return {
    entries,
    stats: geometryResult.stats,
    elapsedSeconds: entries.reduce(
      (sum, entry) => sum + entry.geometry.elapsedSeconds + (entry.baked?.elapsedSeconds ?? 0),
      0,
    ),
  };
}

export async function processPrimitiveEntries(
  entries: readonly ProcessablePrimitiveEntry[],
  options: ProcessingOptions,
  callbacks: PrimitiveProcessingCallbacks = {},
): Promise<PrimitiveSceneProcessingResult> {
  const geometryResult = processPrimitiveGeometries(entries, options, callbacks);
  if (!options.transferTextures) {
    return {
      entries: geometryResult.entries,
      stats: geometryResult.stats,
      elapsedSeconds: geometryResult.elapsedSeconds,
    };
  }
  return await bakePrimitiveTextures(geometryResult, options, callbacks);
}

export function processPrimitiveGeometry(
  entries: readonly ProcessablePrimitiveEntry[],
  options: ProcessingOptions,
  callbacks: PrimitiveProcessingCallbacks = {},
): ProcessedPrimitiveGeometryEntry[] & PrimitiveGeometryProcessingResult {
  const result = processPrimitiveGeometries(entries, options, callbacks);
  return Object.assign(result.entries, {
    entries: result.entries,
    stats: result.stats,
    elapsedSeconds: result.elapsedSeconds,
  });
}

export {
  aggregatePrimitiveStats,
  PrimitiveStageBatch,
  processPrimitiveGeometriesStaged,
} from './primitiveStagedProcessing';
export type {
  PrimitiveStageCallbacks,
  StagedPrimitiveInput,
} from './primitiveStagedProcessing';
