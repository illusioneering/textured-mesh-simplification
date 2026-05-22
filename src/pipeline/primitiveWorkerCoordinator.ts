import type { SimplificationProgress, VirtualEdgeProgress } from '../simplification/types';
import type { AttributeTransferProgress } from '../simplification/attributeTransfer';
import {
  deserializeFullRawSimplificationResult,
  deserializeTransferredMeshAttributes,
} from './primitiveSerialization';
import type {
  AggregateAttributeTransferStageProgress,
  AggregateSimplificationStageProgress,
  AggregateVirtualEdgeStageProgress,
  PrimitiveWorkerInput,
  SerializedPrimitiveStageResult,
} from './primitiveWorkerProtocol';
import { aggregatePrimitiveStats } from './primitiveStagedProcessing';
import type {
  ProcessablePrimitiveEntry,
  PrimitiveGeometryProcessingResult,
  ProcessedPrimitiveGeometryEntry,
} from './sceneProcessing';
import type { GeometryProcessingResult } from './process';

type SourceEntryWithOrdinal = ProcessablePrimitiveEntry & { meshOrdinal: number };

interface WeightedWorkerPartition {
  inputs: PrimitiveWorkerInput[];
  weight: number;
  index: number;
}

export function automaticPrimitiveWorkerCount(primitiveCount: number, hardware: number): number {
  return Math.max(1, Math.min(Math.max(1, primitiveCount), Math.max(1, hardware - 1), 8));
}

export function partitionPrimitiveWorkerInputs(
  inputs: readonly PrimitiveWorkerInput[],
  workerCount: number,
): PrimitiveWorkerInput[][] {
  const partitionCount = Math.max(1, Math.min(workerCount, Math.max(1, inputs.length)));
  const partitions: WeightedWorkerPartition[] = Array.from({ length: partitionCount }, (_, index) => ({
    inputs: [],
    weight: 0,
    index,
  }));
  const sorted = inputs
    .map((input, index) => ({ input, index, weight: input.entry.rawMesh.indices.length }))
    .sort((a, b) => {
      if (b.weight !== a.weight) return b.weight - a.weight;
      return a.index - b.index;
    });

  for (const item of sorted) {
    partitions.sort((a, b) => {
      if (a.weight !== b.weight) return a.weight - b.weight;
      return a.index - b.index;
    });
    const partition = partitions[0];
    if (!partition) throw new Error('No primitive worker partition available.');
    partition.inputs.push(item.input);
    partition.weight += item.weight;
  }

  return partitions.sort((a, b) => a.index - b.index).map((partition) => partition.inputs);
}

export function deserializePrimitiveStageResults(
  results: readonly SerializedPrimitiveStageResult[],
  sourceEntries: readonly SourceEntryWithOrdinal[],
): PrimitiveGeometryProcessingResult {
  const orderById = new Map(sourceEntries.map((entry, index) => [entry.id, index]));
  const sourceById = new Map(sourceEntries.map((entry) => [entry.id, entry]));
  const entries = [...results]
    .sort((a, b) => {
      const orderA = orderById.get(a.id) ?? Number.MAX_SAFE_INTEGER;
      const orderB = orderById.get(b.id) ?? Number.MAX_SAFE_INTEGER;
      if (orderA !== orderB) return orderA - orderB;
      if (a.meshOrdinal !== b.meshOrdinal) return a.meshOrdinal - b.meshOrdinal;
      return a.id.localeCompare(b.id);
    })
    .map<ProcessedPrimitiveGeometryEntry>((result) => {
      const source = sourceById.get(result.id);
      if (!source) throw new Error(`Missing source primitive ${result.id} for worker result.`);
      const raw = deserializeFullRawSimplificationResult(result.raw);
      const geometry: GeometryProcessingResult = {
        raw,
        elapsedSeconds: result.elapsedSeconds,
      };
      return {
        id: result.id,
        source,
        geometry,
        raw,
        ...(result.transferredAttributes
          ? { transferredAttributes: deserializeTransferredMeshAttributes(result.transferredAttributes) }
          : {}),
      };
    });

  return {
    entries,
    stats: aggregatePrimitiveStats(entries.map((entry) => entry.geometry.raw.stats)),
    elapsedSeconds: entries.reduce((sum, entry) => sum + entry.geometry.elapsedSeconds, 0),
  };
}

export function aggregateVirtualProgress(
  completedPrimitives: number,
  totalPrimitives: number,
  latestByPrimitive: ReadonlyMap<string, VirtualEdgeProgress>,
): AggregateVirtualEdgeStageProgress {
  let processedFaces = 0;
  let totalFaces = 0;
  let candidateFacePairs = 0;
  let exactDistanceTests = 0;
  let generatedVirtualEdges = 0;

  for (const progress of latestByPrimitive.values()) {
    processedFaces += progress.processedFaces;
    totalFaces += progress.totalFaces;
    candidateFacePairs += progress.candidateFacePairs;
    exactDistanceTests += progress.exactDistanceTests;
    generatedVirtualEdges += progress.generatedVirtualEdges;
  }

  return {
    stage: 'virtual-edges',
    completedPrimitives,
    totalPrimitives,
    processedFaces,
    totalFaces,
    candidateFacePairs,
    exactDistanceTests,
    generatedVirtualEdges,
  };
}

export function aggregateSimplificationProgress(
  completedPrimitives: number,
  totalPrimitives: number,
  latestByPrimitive: ReadonlyMap<string, SimplificationProgress>,
): AggregateSimplificationStageProgress {
  let collapses = 0;
  let activeFaces = 0;
  let activeVertices = 0;
  let activeEdges = 0;

  for (const progress of latestByPrimitive.values()) {
    collapses += progress.iteration;
    activeFaces += progress.activeFaces;
    activeVertices += progress.activeVertices;
    activeEdges += progress.activeEdges;
  }

  return {
    stage: 'simplification',
    completedPrimitives,
    totalPrimitives,
    collapses,
    activeFaces,
    activeVertices,
    activeEdges,
  };
}

export function aggregateAttributeTransferProgress(
  completedPrimitives: number,
  totalPrimitives: number,
  latestByPrimitive: ReadonlyMap<string, AttributeTransferProgress>,
): AggregateAttributeTransferStageProgress {
  let processedFaces = 0;
  let totalFaces = 0;

  for (const progress of latestByPrimitive.values()) {
    processedFaces += progress.processedFaces;
    totalFaces += progress.totalFaces;
  }

  return {
    stage: 'attribute-transfer',
    completedPrimitives,
    totalPrimitives,
    processedFaces,
    totalFaces,
  };
}
