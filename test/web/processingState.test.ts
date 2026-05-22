import { describe, expect, it } from 'vitest';
import { normalizeProcessingOptions, type ProcessingOptions, type ProcessingOptionsInput } from '../../src/pipeline/options';
import type { PrimitiveExtractionOptions, PrimitiveExtractionResult } from '../../src/pipeline/primitiveExtraction';
import type { BrowserPrimitiveApplyMetadata } from '../../src/web/browserGltfIo';
import {
  createActiveProcessedState,
  createSimplificationOptionsKey,
  isActiveProcessedBakeAvailable,
  isActiveProcessedBakeAvailableForAsset,
  isActiveProcessedOutputForAsset,
  isActiveProcessedOutputCurrent,
  isActiveProcessedStateForAsset,
  isActiveProcessedStateCurrent,
  type ActiveProcessedState,
  type PendingSimplifyState,
} from '../../src/web/processingState';

function processingOptions(input: ProcessingOptionsInput = {}): ProcessingOptions {
  return normalizeProcessingOptions({
    target: { kind: 'ratio', ratio: 0.5 },
    primitiveGrouping: 'material-parent',
    virtualEdges: { mode: 'auto-local-radius' },
    weldVertices: true,
    recomputeNormals: true,
    transferTextures: false,
    textureSize: 1024,
    texturePadding: 2,
    textureFilter: 'linear',
    ...input,
  });
}

function state(): ActiveProcessedState {
  const simplifyOptions = processingOptions();
  return {
    assetRevision: 1,
    simplifyOptions,
    simplificationOptionsKey: {
      primitiveGrouping: simplifyOptions.primitiveGrouping,
      weldVertices: simplifyOptions.weldVertices,
      recomputeNormals: simplifyOptions.recomputeNormals,
    },
    bakeableEntryCount: 1,
    texturedBakeableEntryCount: 1,
    applyMetadataByEntryId: new Map(),
    simplifyResult: {
      kind: 'primitives',
      entries: [],
      stats: {
        inputVertices: 0,
        inputFaces: 0,
        outputVertices: 0,
        outputFaces: 0,
        collapses: 0,
        activeEdges: 0,
        activeFaces: 0,
        activeVertices: 0,
        physicalEdges: 0,
        virtualEdges: 0,
        stoppedReason: 'target-reached',
      } as ActiveProcessedState['simplifyResult']['stats'],
      elapsedSeconds: 0,
    },
  };
}

describe('browser active processed state', () => {
  it('is current only for the same asset revision and simplification option key', () => {
    const key = createSimplificationOptionsKey({
      primitiveGrouping: 'material-parent',
      weldVertices: true,
      recomputeNormals: true,
    });

    expect(isActiveProcessedStateCurrent(state(), 1, key)).toBe(true);
    expect(isActiveProcessedStateCurrent(state(), 2, key)).toBe(false);
    expect(isActiveProcessedStateCurrent(state(), 1, { ...key, primitiveGrouping: 'none' })).toBe(false);
    expect(isActiveProcessedStateCurrent(state(), 1, { ...key, weldVertices: false })).toBe(false);
    expect(isActiveProcessedStateCurrent(state(), 1, { ...key, recomputeNormals: false })).toBe(false);
    expect(isActiveProcessedStateCurrent(null, 1, key)).toBe(false);
  });

  it('allows processed output only for current active state with an output scene', () => {
    const outputScene = {} as NonNullable<ActiveProcessedState['outputScene']>;
    const key = createSimplificationOptionsKey({
      primitiveGrouping: 'material-parent',
      weldVertices: true,
      recomputeNormals: true,
    });

    expect(isActiveProcessedOutputCurrent(state(), outputScene, 1, key)).toBe(true);
    expect(isActiveProcessedOutputCurrent(state(), outputScene, 1, { ...key, primitiveGrouping: 'none' })).toBe(false);
    expect(isActiveProcessedOutputCurrent(state(), outputScene, 1, { ...key, weldVertices: false })).toBe(false);
    expect(isActiveProcessedOutputCurrent(state(), outputScene, 1, { ...key, recomputeNormals: false })).toBe(false);
    expect(isActiveProcessedOutputCurrent(state(), null, 1, key)).toBe(false);
    expect(isActiveProcessedOutputCurrent(null, outputScene, 1, key)).toBe(false);
  });

  it('uses the pending simplify grouping when activating a worker result', () => {
    const applyMetadata: BrowserPrimitiveApplyMetadata = {
      meshOrdinal: 0,
      sourceMeshOrdinals: [0],
      sourceMaterial: null,
    };
    const pending: PendingSimplifyState = {
      simplifyOptions: processingOptions(),
      simplificationOptionsKey: {
        primitiveGrouping: 'material-parent',
        weldVertices: true,
        recomputeNormals: true,
      },
      extraction: {
        entries: [],
        applyMetadataByEntryId: new Map([['primitive-0', applyMetadata]]),
        extractionApplyState: { groupingMode: 'material-parent', mode: 'geometry-with-texture-metadata' },
        summary: {
          inputVertices: 0,
          inputFaces: 0,
          bakeableEntryCount: 0,
          hasTransferableTextureData: false,
          warnings: [],
        },
        releaseProcessingData: () => {},
        dispose: () => {},
      } satisfies PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>,
    };

    const active = createActiveProcessedState({
      assetRevision: 7,
      pending,
      simplifyResult: state().simplifyResult,
    });

    expect(active.simplificationOptionsKey).toEqual({
      primitiveGrouping: 'material-parent',
      weldVertices: true,
      recomputeNormals: true,
    });
    expect(active.simplifyOptions).toEqual(pending.simplifyOptions);
    expect(isActiveProcessedStateCurrent(active, 7, active.simplificationOptionsKey)).toBe(true);
    expect(isActiveProcessedStateCurrent(active, 7, { ...active.simplificationOptionsKey, primitiveGrouping: 'none' })).toBe(false);
    expect(active.applyMetadataByEntryId.get('primitive-0')).toBe(applyMetadata);
  });

  it('records bakeable entry count for the exact grouping used by simplify', () => {
    const pending: PendingSimplifyState = {
      simplifyOptions: processingOptions(),
      simplificationOptionsKey: {
        primitiveGrouping: 'material-parent',
        weldVertices: true,
        recomputeNormals: true,
      },
      extraction: {
        entries: [],
        applyMetadataByEntryId: new Map(),
        extractionApplyState: { groupingMode: 'material-parent', mode: 'geometry-with-texture-metadata' },
        summary: {
          inputVertices: 0,
          inputFaces: 0,
          bakeableEntryCount: 0,
          hasTransferableTextureData: true,
          warnings: [],
        },
        releaseProcessingData: () => {},
        dispose: () => {},
      } satisfies PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>,
    };

    const active = createActiveProcessedState({
      assetRevision: 7,
      pending,
      simplifyResult: state().simplifyResult,
    });

    expect(active.bakeableEntryCount).toBe(0);
  });

  it('keeps processed state available for the loaded asset even when current form options changed', () => {
    const active = state();
    const outputScene = {} as NonNullable<ActiveProcessedState['outputScene']>;

    expect(isActiveProcessedStateForAsset(active, 1)).toBe(true);
    expect(isActiveProcessedOutputForAsset(active, outputScene, 1)).toBe(true);
    expect(isActiveProcessedBakeAvailableForAsset(active, 1)).toBe(true);

    expect(isActiveProcessedStateForAsset(active, 2)).toBe(false);
    expect(isActiveProcessedOutputForAsset(active, null, 1)).toBe(false);
    expect(isActiveProcessedBakeAvailableForAsset({
      ...active,
      bakeableEntryCount: 0,
      texturedBakeableEntryCount: 0,
    }, 1)).toBe(false);
  });

  it('allows baking only when the current grouping produced bakeable entries', () => {
    const active = state();
    const key = active.simplificationOptionsKey;

    expect(isActiveProcessedBakeAvailable(active, 1, key)).toBe(true);
    expect(isActiveProcessedBakeAvailable({
      ...active,
      bakeableEntryCount: 0,
      texturedBakeableEntryCount: 0,
    }, 1, key)).toBe(false);
    expect(isActiveProcessedBakeAvailable(active, 1, { ...key, primitiveGrouping: 'none' })).toBe(false);
    expect(isActiveProcessedBakeAvailable(active, 1, { ...key, weldVertices: false })).toBe(false);
    expect(isActiveProcessedBakeAvailable(active, 1, { ...key, recomputeNormals: false })).toBe(false);
    expect(isActiveProcessedBakeAvailable(null, 1, key)).toBe(false);
  });

  it('does not allow UI baking for factor-only entries without texture maps', () => {
    const active = {
      ...state(),
      bakeableEntryCount: 1,
      texturedBakeableEntryCount: 0,
    } as ActiveProcessedState & { texturedBakeableEntryCount: number };

    expect(isActiveProcessedBakeAvailable(active, 1, active.simplificationOptionsKey)).toBe(false);
  });
});
