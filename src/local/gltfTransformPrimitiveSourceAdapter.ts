import type {
  Document,
  Material,
  Mesh,
  Node as GltfNode,
  Primitive,
  Scene,
} from '@gltf-transform/core';
import {
  createPrimitiveExtractionResult,
  type PrimitiveExtractionOptions,
  type PrimitiveExtractionResult,
  type PrimitiveExtractionSummary,
  type PrimitiveSourceAdapter,
} from '../pipeline/primitiveExtraction';
import {
  hasEntryImageBackedTextureBakeData,
  hasEntryImageBackedTextureTransferData,
  hasEntryPreservableMaterialData,
  toProcessablePrimitiveEntry,
} from '../pipeline/primitiveEntryMetadata';
import type { ProcessablePrimitiveEntry, ProcessedPrimitiveEntry } from '../pipeline/sceneProcessing';
import {
  groupScenePrimitives,
  readGlbScenePrimitives,
  replaceScenePrimitiveGroupGeometry,
  replaceScenePrimitiveGroupTextured,
  type ScenePrimitiveEntry,
  type ScenePrimitiveGroup,
  type ScenePrimitiveReadResult,
} from './scenePrimitiveGltfIo';

export interface GltfTransformExtractionApplyState {
  document: Document;
  sourceByEntryId: Map<string, ScenePrimitiveGroup>;
}

export interface GltfTransformPrimitiveEntryMetadata {
  node: GltfNode;
  mesh: Mesh;
  primitive: Primitive;
  primitiveIndex: number;
  originalMaterial: Material | null;
}

export interface GltfTransformPrimitiveGroupMetadata {
  scene: Scene;
  parentNode: GltfNode | null;
  entries: GltfTransformPrimitiveEntryMetadata[];
  originalMaterial: Material | null;
}

function sourceId(group: ScenePrimitiveGroup): string {
  if (group.mode !== 'none') return group.id;
  const entry = group.entries[0];
  return entry ? String(entry.id) : group.id;
}

function metadataFor(group: ScenePrimitiveGroup): GltfTransformPrimitiveGroupMetadata {
  return {
    scene: group.scene,
    parentNode: group.parentNode,
    entries: group.entries.map((entry) => ({
      node: entry.node,
      mesh: entry.mesh,
      primitive: entry.primitive,
      primitiveIndex: entry.primitiveIndex,
      originalMaterial: entry.originalMaterial,
    })),
    originalMaterial: group.originalMaterial,
  };
}

function groupLabel(group: ScenePrimitiveGroup): string | undefined {
  const nodeNames = group.entries
    .map((entry) => entry.nodeName)
    .filter((name): name is string => name !== undefined && name.length > 0);
  if (nodeNames.length === 1) return nodeNames[0];
  const materialName = group.sourceMaterial.name;
  return materialName.length > 0 ? materialName : undefined;
}

function toProcessable(
  group: ScenePrimitiveGroup,
  options: PrimitiveExtractionOptions,
): ProcessablePrimitiveEntry {
  const label = groupLabel(group);
  return toProcessablePrimitiveEntry({
    id: sourceId(group),
    ...(label !== undefined ? { label } : {}),
    rawMesh: group.rawMesh,
    ...(options.mode === 'geometry'
      ? {}
      : {
          texturedRawMesh: {
            rawMesh: group.rawMesh,
            faceAttributes: group.faceAttributes,
            materials: [group.sourceMaterial],
          },
        }),
  });
}

function isSummaryBakeable(entry: ProcessablePrimitiveEntry): boolean {
  return entry.bakeable || (entry.texturedRawMesh ? hasEntryImageBackedTextureBakeData(entry.texturedRawMesh) : false);
}

function summaryFor(
  source: ScenePrimitiveReadResult,
  entries: readonly ProcessablePrimitiveEntry[],
): PrimitiveExtractionSummary {
  const hasImageBackedTextureTransferData = entries.some((entry) => (
    entry.texturedRawMesh ? hasEntryImageBackedTextureTransferData(entry.texturedRawMesh) : false
  ));
  return {
    inputVertices: source.entries.reduce((sum, entry) => sum + entry.rawMesh.positions.length, 0),
    inputFaces: source.entries.reduce((sum, entry) => sum + entry.rawMesh.faces.length, 0),
    bakeableEntryCount: entries.filter(isSummaryBakeable).length,
    hasTransferableTextureData: hasImageBackedTextureTransferData,
    hasPreservableMaterialData: entries.some((entry) => (
      entry.hasPreservableMaterialData === true
      || (entry.texturedRawMesh ? hasEntryPreservableMaterialData(entry.texturedRawMesh) : false)
    )),
    hasImageBackedTextureTransferData,
    hasImageBackedTextureBakeData: entries.some(isSummaryBakeable),
    hasTransferableVertexAttributes: entries.some((entry) => entry.requiresAttributeTransfer === true),
    warnings: source.warnings.map((warning) => {
      const node = warning.nodeName ? `${warning.nodeName}: ` : '';
      return `${node}skipped ${warning.meshName}[${warning.primitiveIndex}]: ${warning.reason}`;
    }),
  };
}

export class GltfTransformPrimitiveSourceAdapter
  implements PrimitiveSourceAdapter<
    GltfTransformPrimitiveGroupMetadata,
    Document,
    GltfTransformExtractionApplyState
  > {
  private constructor(private readonly inputPath: string) {}

  static async read(inputPath: string): Promise<GltfTransformPrimitiveSourceAdapter> {
    return new GltfTransformPrimitiveSourceAdapter(inputPath);
  }

  async summarize(): Promise<PrimitiveExtractionSummary> {
    const source = await readGlbScenePrimitives(this.inputPath, { mode: 'geometry-with-texture-metadata' });
    const groups = groupScenePrimitives(source.entries, 'material-parent', { weldVertices: true });
    return summaryFor(source, groups.map((group) => toProcessable(group, {
      groupingMode: 'material-parent',
      mode: 'geometry-with-texture-metadata',
      weldVertices: true,
    })));
  }

  async extractGroups(
    options: PrimitiveExtractionOptions,
  ): Promise<PrimitiveExtractionResult<GltfTransformPrimitiveGroupMetadata, GltfTransformExtractionApplyState>> {
    const source = options.mode === 'bake'
      ? await readGlbScenePrimitives(this.inputPath, { mode: 'bake' })
      : await readGlbScenePrimitives(this.inputPath, { mode: options.mode });
    const groups = groupScenePrimitives(source.entries, options.groupingMode, {
      weldVertices: options.weldVertices ?? true,
    });
    const applyMetadataByEntryId = new Map<string, GltfTransformPrimitiveGroupMetadata>();
    const sourceByEntryId = new Map<string, ScenePrimitiveGroup>();
    const entries = groups.map((group) => {
      const id = sourceId(group);
      const processable = toProcessable(group, options);
      applyMetadataByEntryId.set(id, metadataFor(group));
      sourceByEntryId.set(id, group);
      return processable;
    });

    return createPrimitiveExtractionResult({
      entries,
      applyMetadataByEntryId,
      extractionApplyState: {
        document: source.document,
        sourceByEntryId,
      },
      summary: summaryFor(source, entries),
      onDispose: () => sourceByEntryId.clear(),
    });
  }

  async applyResults(
    extraction: PrimitiveExtractionResult<GltfTransformPrimitiveGroupMetadata, GltfTransformExtractionApplyState>,
    results: readonly ProcessedPrimitiveEntry[],
  ): Promise<Document> {
    for (const processed of results) {
      const source = extraction.extractionApplyState.sourceByEntryId.get(processed.id);
      if (!source) throw new Error(`Missing source primitive group for processed entry ${processed.id}.`);
      if (processed.baked) {
        replaceScenePrimitiveGroupTextured(source, {
          outputRawMesh: processed.baked.raw.rawMesh,
          atlas: processed.baked.baked.atlas,
          image: processed.baked.baked.image,
          additionalTextures: processed.baked.baked.additionalTextures,
          ...(processed.transferredAttributes
            ? { transferredAttributes: processed.transferredAttributes }
            : {}),
        });
      } else {
        replaceScenePrimitiveGroupGeometry(
          source,
          processed.geometry.raw.rawMesh,
          processed.transferredAttributes,
        );
      }
    }
    return extraction.extractionApplyState.document;
  }
}

export type { ScenePrimitiveEntry };
