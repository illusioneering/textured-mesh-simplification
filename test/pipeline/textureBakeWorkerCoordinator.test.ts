import { describe, expect, it } from 'vitest';
import type { TextureBakeBatch, TextureBakeBatchProgress } from '../../src/texture/bakeBatch';
import {
  aggregateTextureBakeProgress,
  automaticTextureBakeWorkerCount,
  partitionTextureBakeBatches,
} from '../../src/pipeline/textureBakeWorkerCoordinator';

function batch(id: number, sampleCount: number): TextureBakeBatch {
  return { id, startFaceIndex: id, endFaceIndex: id + 1, sampleCount };
}

describe('texture bake worker coordinator', () => {
  it('chooses deterministic worker counts', () => {
    expect(automaticTextureBakeWorkerCount(0, 8)).toBe(1);
    expect(automaticTextureBakeWorkerCount(1, 8)).toBe(1);
    expect(automaticTextureBakeWorkerCount(8, 16)).toBe(8);
    expect(automaticTextureBakeWorkerCount(20, 2)).toBe(1);
  });

  it('partitions batches by descending sample weight without dropping batch ids', () => {
    const partitions = partitionTextureBakeBatches([batch(0, 1), batch(1, 100), batch(2, 50)], 2);

    expect(partitions).toHaveLength(2);
    expect(partitions.flat().map((item) => item.id).sort((a, b) => a - b)).toEqual([0, 1, 2]);
    expect(partitions.every((partition) => partition.length > 0)).toBe(true);
  });

  it('aggregates latest per-worker cumulative progress into one resampling progress object', () => {
    const latestByWorker = new Map<number, TextureBakeBatchProgress>([
      [0, {
        batchId: 1,
        completedBatches: 2,
        totalBatches: 3,
        processedFaces: 3,
        totalFaces: 10,
        processedSamples: 30,
        totalSamples: 100,
        mappedPixels: 30,
        unmappedPixels: 0,
      }],
      [1, {
        batchId: 2,
        completedBatches: 1,
        totalBatches: 3,
        processedFaces: 2,
        totalFaces: 10,
        processedSamples: 20,
        totalSamples: 100,
        mappedPixels: 19,
        unmappedPixels: 1,
      }],
    ]);

    expect(aggregateTextureBakeProgress({
      completedBatches: 3,
      totalBatches: 3,
      latestByWorker,
      totalFaces: 10,
      totalSamples: 100,
    })).toEqual({
      completedBatches: 3,
      totalBatches: 3,
      processedFaces: 5,
      totalFaces: 10,
      processedSamples: 50,
      totalSamples: 100,
      mappedPixels: 49,
      unmappedPixels: 1,
    });
  });
});
