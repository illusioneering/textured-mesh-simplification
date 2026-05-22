import { describe, expect, it } from 'vitest';
import { createPrimitiveExtractionResult } from '../../src/pipeline/primitiveExtraction';
import type { ProcessablePrimitiveEntry } from '../../src/pipeline/sceneProcessing';

function entry(id: string): ProcessablePrimitiveEntry {
  return {
    id,
    label: id,
    rawMesh: {
      positions: [],
      faces: [],
    },
    bakeable: false,
    hasTexturedMaterial: false,
  };
}

describe('primitive extraction result lifecycle', () => {
  it('releases processing entries while preserving apply metadata and extraction apply state', () => {
    const result = createPrimitiveExtractionResult({
      entries: [entry('a')],
      applyMetadataByEntryId: new Map([['a', { meshOrdinal: 7 }]]),
      extractionApplyState: { documentId: 'doc' },
      summary: {
        inputVertices: 3,
        inputFaces: 1,
        bakeableEntryCount: 0,
        hasTransferableTextureData: false,
        warnings: [],
      },
    });

    result.releaseProcessingData();

    expect(result.entries).toEqual([]);
    expect(result.applyMetadataByEntryId.get('a')).toEqual({ meshOrdinal: 7 });
    expect(result.extractionApplyState).toEqual({ documentId: 'doc' });
  });

  it('disposes processing data and apply metadata', () => {
    const result = createPrimitiveExtractionResult({
      entries: [entry('a')],
      applyMetadataByEntryId: new Map([['a', { meshOrdinal: 7 }]]),
      extractionApplyState: undefined,
      summary: {
        inputVertices: 3,
        inputFaces: 1,
        bakeableEntryCount: 0,
        hasTransferableTextureData: false,
        warnings: [],
      },
    });

    result.dispose();

    expect(result.entries).toEqual([]);
    expect(result.applyMetadataByEntryId.size).toBe(0);
  });
});
