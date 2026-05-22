import type { TextureBakeBatch, TextureBakeBatchProgress } from '../texture/bakeBatch';

interface WeightedTextureBakePartition {
  batches: TextureBakeBatch[];
  weight: number;
  index: number;
}

export interface AggregateTextureBakeProgress {
  completedBatches: number;
  totalBatches: number;
  processedFaces: number;
  totalFaces: number;
  processedSamples: number;
  totalSamples: number;
  mappedPixels: number;
  unmappedPixels: number;
}

export interface AggregateTextureBakeProgressInput {
  completedBatches: number;
  totalBatches: number;
  latestByWorker: ReadonlyMap<number, TextureBakeBatchProgress>;
  totalFaces: number;
  totalSamples: number;
}

export function automaticTextureBakeWorkerCount(batchCount: number, hardware: number): number {
  return Math.max(1, Math.min(Math.max(1, batchCount), Math.max(1, hardware - 1), 8));
}

export function partitionTextureBakeBatches(
  batches: readonly TextureBakeBatch[],
  workerCount: number,
): TextureBakeBatch[][] {
  const partitionCount = Math.min(Math.max(1, Math.floor(workerCount)), batches.length);
  if (partitionCount === 0) return [];

  const partitions: WeightedTextureBakePartition[] = Array.from({ length: partitionCount }, (_, index) => ({
    batches: [],
    weight: 0,
    index,
  }));
  const sortedBatches = [...batches].sort((left, right) => (
    right.sampleCount - left.sampleCount || left.id - right.id
  ));

  for (const batch of sortedBatches) {
    let partition = partitions[0];
    for (const candidate of partitions) {
      if (!partition || candidate.weight < partition.weight || (
        candidate.weight === partition.weight && candidate.index < partition.index
      )) {
        partition = candidate;
      }
    }
    if (!partition) continue;
    partition.batches.push(batch);
    partition.weight += batch.sampleCount;
  }

  return partitions.sort((left, right) => left.index - right.index).map((partition) => partition.batches);
}

export function aggregateTextureBakeProgress(input: AggregateTextureBakeProgressInput): AggregateTextureBakeProgress {
  let processedFaces = 0;
  let processedSamples = 0;
  let mappedPixels = 0;
  let unmappedPixels = 0;

  for (const progress of input.latestByWorker.values()) {
    processedFaces += progress.processedFaces;
    processedSamples += progress.processedSamples;
    mappedPixels += progress.mappedPixels;
    unmappedPixels += progress.unmappedPixels;
  }

  return {
    completedBatches: input.completedBatches,
    totalBatches: input.totalBatches,
    processedFaces,
    totalFaces: input.totalFaces,
    processedSamples,
    totalSamples: input.totalSamples,
    mappedPixels,
    unmappedPixels,
  };
}
