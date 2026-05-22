import type { RawMesh, SimplificationResult } from '../simplification/types';
import type { TexturedRawMesh } from '../texture/types';

export interface RawMeshStats {
  vertices: number;
  faces: number;
}

export interface InputModelSummaryStats {
  vertices: number;
  faces: number;
  materials: number;
  materialsWithTextures: number;
  materialsWithBaseColorImages: number;
  facesWithUvs: number;
  textureSlotKinds: string[];
  textureDimensions: string[];
}

export interface TexturedMeshStats extends InputModelSummaryStats {}

export interface ProcessingSummaryInput {
  stats: SimplificationResult['stats'];
  bake?: {
    filledPixels: number;
    mappedPixels: number;
    unmappedPixels: number;
    islandCount?: number;
    outputVertices?: number;
  };
  exportedBytes?: number;
  elapsedSeconds?: number;
}

export type ProcessingSummary = SimplificationResult['stats'] & {
  bakeFilledPixels?: number;
  bakeMappedPixels?: number;
  bakeUnmappedPixels?: number;
  bakeIslandCount?: number;
  bakeOutputVertices?: number;
  exportedBytes?: number;
  exportedBytesLabel?: string;
  elapsedSeconds?: number;
};

export type StatisticItem = [string, string | number | undefined];

export function formatScalar(value: number): string {
  if (!Number.isFinite(value)) return '0';
  if (value === 0) return '0';
  const absolute = Math.abs(value);
  if (absolute >= 1000) return Number(value.toFixed(3)).toString();
  if (absolute >= 1) return Number(value.toFixed(6)).toString();
  return Number(value.toPrecision(6)).toString();
}

export function summarizeRawMesh(rawMesh: RawMesh): RawMeshStats {
  return {
    vertices: rawMesh.positions.length,
    faces: rawMesh.faces.length,
  };
}

export function summarizeTexturedRawMesh(source: TexturedRawMesh): TexturedMeshStats {
  const textureDimensions = Array.from(new Set(source.materials
    .map((material) => {
      const image = material.baseColorTexture?.image;
      return image ? `${image.width}×${image.height}` : null;
    })
    .filter((value): value is string => value !== null)));
  return {
    ...summarizeRawMesh(source.rawMesh),
    materials: source.materials.length,
    materialsWithTextures: source.materials.filter((material) => material.textureSlots.some((slot) => slot.hasImage)).length,
    materialsWithBaseColorImages: source.materials.filter((material) => Boolean(material.baseColorTexture?.image)).length,
    facesWithUvs: source.faceAttributes.filter((attributes) => attributes.uvSets.length > 0).length,
    textureSlotKinds: Array.from(new Set(source.materials.flatMap((material) => (
      material.textureSlots.filter((slot) => slot.hasImage).map((slot) => slot.slot)
    )))).sort(),
    textureDimensions,
  };
}

export function inputModelStatItems(
  summary: InputModelSummaryStats,
  transformPreservation: string,
  gridSpacing: number | undefined,
): StatisticItem[] {
  return [
    ['Vertices', summary.vertices],
    ['Faces', summary.faces],
    ['Grid spacing', gridSpacing !== undefined ? formatScalar(gridSpacing) : undefined],
    ['Materials', summary.materials],
    ['Materials with textures', summary.materialsWithTextures],
    ['Materials with base-color textures', summary.materialsWithBaseColorImages],
    ['Material texture slots', summary.textureSlotKinds.length > 0 ? summary.textureSlotKinds.join(', ') : 'None'],
    ['Faces with UVs', summary.facesWithUvs],
    ['Base-color texture dimensions', summary.textureDimensions.length > 0 ? summary.textureDimensions.join(', ') : 'None'],
    ['Transform preservation', transformPreservation],
  ];
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

export function summarizeProcessingResult(input: ProcessingSummaryInput): ProcessingSummary {
  return {
    ...input.stats,
    ...(input.bake ? {
      bakeFilledPixels: input.bake.filledPixels,
      bakeMappedPixels: input.bake.mappedPixels,
      bakeUnmappedPixels: input.bake.unmappedPixels,
      ...(input.bake.islandCount !== undefined ? { bakeIslandCount: input.bake.islandCount } : {}),
      ...(input.bake.outputVertices !== undefined ? { bakeOutputVertices: input.bake.outputVertices } : {}),
    } : {}),
    ...(input.exportedBytes !== undefined ? {
      exportedBytes: input.exportedBytes,
      exportedBytesLabel: formatBytes(input.exportedBytes),
    } : {}),
    ...(input.elapsedSeconds !== undefined ? { elapsedSeconds: input.elapsedSeconds } : {}),
  };
}

export function formatProcessingCompleteStatus(
  operation: 'simplify' | 'bake',
  stats: Pick<SimplificationResult['stats'], 'outputFaces' | 'outputVertices'>,
  bake?: Pick<NonNullable<ProcessingSummaryInput['bake']>, 'outputVertices'>,
): string {
  const label = operation === 'bake' ? 'Texture atlas baking' : 'Geometry simplification';
  const outputVertices = operation === 'bake' && bake?.outputVertices !== undefined
    ? bake.outputVertices
    : stats.outputVertices;
  return `${label} complete: ${stats.outputFaces.toLocaleString()} faces, ${outputVertices.toLocaleString()} vertices.`;
}

export function processedOutputStatItems(summary: ProcessingSummary): StatisticItem[] {
  return [
    ['Output vertices', summary.outputVertices],
    ['Output faces', summary.outputFaces],
    ['Baked output vertices', summary.bakeOutputVertices],
    ['Collapses', summary.collapses],
    ['Physical edges', summary.physicalEdges],
    ['Virtual edges', summary.virtualEdges],
    ['Stopped reason', summary.stoppedReason],
    ['Elapsed seconds', summary.elapsedSeconds?.toFixed(2)],
    ['Atlas islands', summary.bakeIslandCount],
    ['Baked mapped pixels', summary.bakeMappedPixels],
    ['Baked unmapped pixels', summary.bakeUnmappedPixels],
    ['Exported GLB size', summary.exportedBytesLabel],
  ];
}
