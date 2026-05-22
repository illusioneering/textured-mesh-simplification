import type { PrimitiveGroupingMode } from './options';
import type { ProcessablePrimitiveEntry, ProcessedPrimitiveEntry } from './sceneProcessing';

export type PrimitiveExtractionMode =
  | 'geometry'
  | 'geometry-with-texture-metadata'
  | 'bake';

export interface PrimitiveExtractionOptions {
  groupingMode: PrimitiveGroupingMode;
  mode: PrimitiveExtractionMode;
  weldVertices?: boolean;
}

export interface PrimitiveExtractionSummary {
  inputVertices: number;
  inputFaces: number;
  hasPreservableMaterialData?: boolean;
  hasImageBackedTextureTransferData?: boolean;
  hasImageBackedTextureBakeData?: boolean;
  hasTransferableVertexAttributes?: boolean;
  bakeableEntryCount: number;
  hasTransferableTextureData: boolean;
  warnings: string[];
}

export interface PrimitiveExtractionResult<TApplyMetadata, TExtractionApplyState = undefined> {
  entries: ProcessablePrimitiveEntry[];
  applyMetadataByEntryId: Map<string, TApplyMetadata>;
  extractionApplyState: TExtractionApplyState;
  summary: PrimitiveExtractionSummary;
  releaseProcessingData(): void;
  dispose(): void;
}

export interface PrimitiveSourceAdapter<TApplyMetadata, TOutput, TExtractionApplyState = undefined> {
  summarize(): Promise<PrimitiveExtractionSummary>;
  extractGroups(options: PrimitiveExtractionOptions): Promise<
    PrimitiveExtractionResult<TApplyMetadata, TExtractionApplyState>
  >;
  applyResults(
    extraction: PrimitiveExtractionResult<TApplyMetadata, TExtractionApplyState>,
    results: readonly ProcessedPrimitiveEntry[],
  ): Promise<TOutput>;
}

export interface CreatePrimitiveExtractionResultOptions<TApplyMetadata, TExtractionApplyState> {
  entries: ProcessablePrimitiveEntry[];
  applyMetadataByEntryId: Map<string, TApplyMetadata>;
  extractionApplyState: TExtractionApplyState;
  summary: PrimitiveExtractionSummary;
  onDispose?: () => void;
}

export function createPrimitiveExtractionResult<TApplyMetadata, TExtractionApplyState = undefined>(
  options: CreatePrimitiveExtractionResultOptions<TApplyMetadata, TExtractionApplyState>,
): PrimitiveExtractionResult<TApplyMetadata, TExtractionApplyState> {
  return {
    entries: options.entries,
    applyMetadataByEntryId: options.applyMetadataByEntryId,
    extractionApplyState: options.extractionApplyState,
    summary: options.summary,
    releaseProcessingData() {
      this.entries.length = 0;
    },
    dispose() {
      this.releaseProcessingData();
      this.applyMetadataByEntryId.clear();
      options.onDispose?.();
    },
  };
}
