import { describe, expect, it } from 'vitest';
import type { SimplificationProgress } from '../../src/simplification/types';
import {
  aggregateAttributeTransferProgressMessage,
  aggregateSimplificationProgressMessage,
  aggregateVirtualEdgeProgressMessage,
  bakeProgressMessage,
  primitiveBakeCompleteMessage,
  primitiveBakeStartMessage,
  primitiveGeometryCompleteMessage,
  primitiveGeometryStartMessage,
  shouldPostAggregatePrimitiveStageProgress,
  shouldPostBakeProgress,
  shouldPostSimplificationProgress,
  simplificationProgressMessage,
  virtualEdgeProgressMessage,
  type SimplificationProgressPostState,
} from '../../src/web/progressMessages';

function progress(iteration: number, activeFaces = 1234): SimplificationProgress {
  return {
    iteration,
    activeFaces,
    activeVertices: 567,
    activeEdges: 890,
    lastCost: 0.00123,
  };
}

describe('web progress messages', () => {
  it('throttles geometry iteration status updates to reduce log bloat', () => {
    const state: SimplificationProgressPostState = { lastPostTimeMs: 1_000 };

    expect(shouldPostSimplificationProgress(progress(10), state, 1_250)).toBe(false);
    expect(shouldPostSimplificationProgress(progress(20), state, 1_749)).toBe(false);
    expect(shouldPostSimplificationProgress(progress(30), state, 1_750)).toBe(true);
  });

  it('throttles aggregate primitive-stage status updates to twice per second', () => {
    const state: SimplificationProgressPostState = { lastPostTimeMs: 1_000 };

    expect(shouldPostAggregatePrimitiveStageProgress(state, 1_499)).toBe(false);
    expect(shouldPostAggregatePrimitiveStageProgress(state, 1_500)).toBe(true);
  });

  it('throttles texture bake status updates to twice per second', () => {
    const state: SimplificationProgressPostState = { lastPostTimeMs: 1_000 };

    expect(shouldPostBakeProgress(state, 1_499)).toBe(false);
    expect(shouldPostBakeProgress(state, 1_500)).toBe(true);
  });

  it('formats concise geometry iteration messages', () => {
    expect(simplificationProgressMessage(progress(1200, 98_765))).toBe('Geometry iteration 1,200: 98,765 active faces.');
  });

  it('formats per-primitive lifecycle messages for worker status logs', () => {
    expect(primitiveGeometryStartMessage({ index: 0, total: 2, label: 'Chair seat' })).toBe(
      'Simplifying primitive 1/2: Chair seat…',
    );
    expect(primitiveGeometryCompleteMessage({ index: 0, total: 2, outputFaces: 1234, outputVertices: 567 })).toBe(
      'Primitive 1/2 geometry complete: 1,234 faces, 567 vertices.',
    );
    expect(primitiveBakeStartMessage({ index: 1, total: 2, label: 'Chair legs' })).toBe(
      'Baking primitive 2/2: Chair legs…',
    );
    expect(primitiveBakeCompleteMessage({ index: 1, total: 2, filledPixels: 2048, unmappedPixels: 3 })).toBe(
      'Primitive 2/2 texture bake complete: 2,048 filled pixels, 3 unmapped samples.',
    );
  });

  it('formats virtual-edge search progress messages for large meshes', () => {
    expect(virtualEdgeProgressMessage({
      phase: 'searching-pairs',
      processedFaces: 50_000,
      totalFaces: 100_000,
      candidateFacePairs: 12_345,
      exactDistanceTests: 456,
      generatedVirtualEdges: 78,
    })).toBe(
      'Virtual-edge search: 50% of faces, 12,345 candidate pairs, 456 exact tests, 78 virtual edges.',
    );
  });

  it('formats aggregate primitive stage progress messages', () => {
    expect(aggregateVirtualEdgeProgressMessage({
      stage: 'virtual-edges',
      completedPrimitives: 1,
      totalPrimitives: 2,
      processedFaces: 50_000,
      totalFaces: 100_000,
      candidateFacePairs: 12_345,
      exactDistanceTests: 456,
      generatedVirtualEdges: 78,
    })).toBe('Virtual-edge stage: 50% of faces, 1/2 primitives complete, 12,345 candidate pairs, 456 exact tests, 78 virtual edges.');

    expect(aggregateSimplificationProgressMessage({
      stage: 'simplification',
      completedPrimitives: 1,
      totalPrimitives: 2,
      collapses: 1200,
      activeFaces: 98_765,
      activeVertices: 54_321,
      activeEdges: 88_000,
    })).toBe('Simplification stage: 1/2 primitives complete, 1,200 collapses, 98,765 active faces.');

    expect(aggregateAttributeTransferProgressMessage({
      stage: 'attribute-transfer',
      completedPrimitives: 1,
      totalPrimitives: 2,
      processedFaces: 500,
      totalFaces: 1000,
    })).toBe('Attribute transfer stage: 50% of faces, 1/2 primitives complete.');
  });

  it('formats texture bake progress so the UI shows work after geometry simplification completes', () => {
    expect(bakeProgressMessage({ stage: 'atlas-created', totalFaces: 100, islandCount: 100 })).toBe(
      'Texture atlas generated: 100 atlas chart islands for 100 faces. Baking standard material texture pixels…',
    );
    expect(bakeProgressMessage({
      stage: 'resampling',
      completedBatches: 18,
      totalBatches: 40,
      processedFaces: 50,
      totalFaces: 100,
      processedSamples: 1234,
      totalSamples: 2468,
      mappedPixels: 1234,
      unmappedPixels: 2,
    })).toBe('Texture resampling: 50% of atlas samples, 18/40 batches complete, 1,234 mapped samples, 2 unmapped samples.');
    expect(bakeProgressMessage({ stage: 'dilating', gutterPass: 2, gutterPasses: 4 })).toBe(
      'Expanding texture gutters (pass 2 of 4)…',
    );
    expect(bakeProgressMessage({ stage: 'complete', filledPixels: 2048, mappedPixels: 4096, unmappedPixels: 0 })).toBe(
      'Texture baking complete: 2,048 filled pixels, 4,096 mapped samples, 0 unmapped samples.',
    );
  });
});
