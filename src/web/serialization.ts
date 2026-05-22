import type { RawMesh, RawSimplificationResult } from '../simplification/types';
import type {
  BakedMaterialTexture,
  BakedTextureResult,
} from '../texture/types';
import type { BakedMaterialSettings, GeometryProcessingResult, TexturedProcessingResult } from '../pipeline/process';
import type { PrimitiveGeometryProcessingResult, PrimitiveSceneProcessingResult } from '../pipeline/sceneProcessing';
import {
  deserializeRawMesh,
  deserializeRgbaImage,
  serializeTransferredMeshAttributes,
  serializeRawMesh,
  serializeRgbaImage,
  type SerializedRawMesh,
  type SerializedRgbaImage,
  type SerializedTransferredMeshAttributes,
} from '../pipeline/primitiveSerialization';
import {
  serializeAtlas,
  type SerializedAtlasLayout,
} from '../pipeline/textureBakeSerialization';
export {
  deserializePrimitiveEntries,
  deserializeRawMesh,
  deserializeRgbaImage,
  deserializeSourceFaceAttributes,
  deserializeSourceMaterial,
  deserializeSourceTexture,
  deserializeTexturedRawMesh,
  deserializeTransferredMeshAttributes,
  deserializeFullRawSimplificationResult,
  serializeFullRawSimplificationResult,
  serializePrimitiveEntries,
  serializeRawMesh,
  serializeRgbaImage,
  serializeSourceFaceAttributes,
  serializeSourceMaterial,
  serializeSourceTexture,
  serializeTexturedRawMesh,
  serializeTransferredMeshAttributes,
  type SerializationOptions,
  type SerializedFullCollapseHistory,
  type SerializedFullRawSimplificationResult,
  type SerializedPrimitiveProcessingEntry,
  type SerializedRawMesh,
  type SerializedRgbaImage,
  type SerializedSourceFaceAttributes,
  type SerializedSourceMaterial,
  type SerializedSourceMaterialTextureInfo,
  type SerializedSourceTexture,
  type SerializedTexturedRawMesh,
  type SerializedTransferredMeshAttributes,
} from '../pipeline/primitiveSerialization';
export {
  deserializeAtlas,
  serializeAtlas,
  type SerializedAtlasLayout,
} from '../pipeline/textureBakeSerialization';

export interface SerializedBakedMaterialTexture {
  slot: BakedMaterialTexture['slot'];
  image: SerializedRgbaImage;
}

export interface SerializedSimplifiedGeometryResult {
  rawMesh: SerializedRawMesh;
  stats: RawSimplificationResult['stats'];
}

export interface SerializedGeometryProcessingResult {
  kind: 'geometry';
  raw: SerializedSimplifiedGeometryResult;
  elapsedSeconds: number;
}

export interface SerializedTexturedProcessingResult {
  kind: 'textured';
  raw: SerializedSimplifiedGeometryResult;
  baked: {
    image: SerializedRgbaImage;
    additionalTextures: SerializedBakedMaterialTexture[];
    atlas: SerializedAtlasLayout;
    stats: BakedTextureResult['stats'];
  };
  materialSettings: BakedMaterialSettings;
  elapsedSeconds: number;
}

export interface SerializedPrimitiveProcessingResult {
  kind: 'primitives';
  entries: Array<{
    id: string;
    meshOrdinal: number;
    raw: SerializedSimplifiedGeometryResult;
    baked?: {
      image: SerializedRgbaImage;
      additionalTextures: SerializedBakedMaterialTexture[];
      atlas: SerializedAtlasLayout;
      stats: BakedTextureResult['stats'];
      materialSettings: BakedMaterialSettings;
    };
    transferredAttributes?: SerializedTransferredMeshAttributes;
  }>;
  stats: RawSimplificationResult['stats'];
  elapsedSeconds: number;
}

export type SerializedProcessingResult = SerializedGeometryProcessingResult | SerializedTexturedProcessingResult | SerializedPrimitiveProcessingResult;

export function serializeBakedMaterialTexture(texture: BakedMaterialTexture): SerializedBakedMaterialTexture {
  return {
    slot: texture.slot,
    image: serializeRgbaImage(texture.image),
  };
}

export function deserializeBakedMaterialTexture(texture: SerializedBakedMaterialTexture): BakedMaterialTexture {
  return {
    slot: texture.slot,
    image: deserializeRgbaImage(texture.image),
  };
}

export function serializeSimplifiedGeometryResult(result: RawSimplificationResult): SerializedSimplifiedGeometryResult {
  return {
    rawMesh: serializeRawMesh(result.rawMesh),
    stats: result.stats,
  };
}

export function deserializeSimplifiedRawMesh(result: SerializedSimplifiedGeometryResult): RawMesh {
  return deserializeRawMesh(result.rawMesh);
}

export function serializeGeometryProcessingResult(result: GeometryProcessingResult): SerializedGeometryProcessingResult {
  return { kind: 'geometry', raw: serializeSimplifiedGeometryResult(result.raw), elapsedSeconds: result.elapsedSeconds };
}

export function serializeTexturedProcessingResult(result: TexturedProcessingResult): SerializedTexturedProcessingResult {
  return {
    kind: 'textured',
    raw: serializeSimplifiedGeometryResult(result.raw),
    baked: {
      image: serializeRgbaImage(result.baked.image),
      additionalTextures: result.baked.additionalTextures.map(serializeBakedMaterialTexture),
      atlas: serializeAtlas(result.baked.atlas),
      stats: result.baked.stats,
    },
    materialSettings: result.materialSettings,
    elapsedSeconds: result.elapsedSeconds,
  };
}

export function serializePrimitiveGeometryProcessingResult(
  result: PrimitiveGeometryProcessingResult,
  sourceEntries: ReadonlyArray<{ id: string; meshOrdinal: number }>,
): SerializedPrimitiveProcessingResult {
  const meshOrdinals = new Map(sourceEntries.map((entry) => [entry.id, entry.meshOrdinal]));
  return {
    kind: 'primitives',
    entries: result.entries.map((entry) => ({
      id: entry.id,
      meshOrdinal: meshOrdinals.get(entry.id) ?? 0,
      raw: serializeSimplifiedGeometryResult(entry.geometry.raw),
      ...(entry.transferredAttributes
        ? { transferredAttributes: serializeTransferredMeshAttributes(entry.transferredAttributes) }
        : {}),
    })),
    stats: result.stats,
    elapsedSeconds: result.elapsedSeconds,
  };
}

export function serializePrimitiveSceneProcessingResult(
  result: PrimitiveSceneProcessingResult,
  sourceEntries: ReadonlyArray<{ id: string; meshOrdinal: number }>,
): SerializedPrimitiveProcessingResult {
  const meshOrdinals = new Map(sourceEntries.map((entry) => [entry.id, entry.meshOrdinal]));
  return {
    kind: 'primitives',
    entries: result.entries.map((entry) => ({
      id: entry.id,
      meshOrdinal: meshOrdinals.get(entry.id) ?? 0,
      raw: serializeSimplifiedGeometryResult(entry.geometry.raw),
      ...(entry.baked ? {
        baked: {
          image: serializeRgbaImage(entry.baked.baked.image),
          additionalTextures: entry.baked.baked.additionalTextures.map(serializeBakedMaterialTexture),
          atlas: serializeAtlas(entry.baked.baked.atlas),
          stats: entry.baked.baked.stats,
          materialSettings: entry.baked.materialSettings,
        },
      } : {}),
      ...(entry.transferredAttributes
        ? { transferredAttributes: serializeTransferredMeshAttributes(entry.transferredAttributes) }
        : {}),
    })),
    stats: result.stats,
    elapsedSeconds: result.elapsedSeconds,
  };
}

export function collectTransferables(value: unknown): Transferable[] {
  const transferables: Transferable[] = [];
  const seenObjects = new Set<object>();
  const seenBuffers = new Set<ArrayBuffer>();
  const visit = (item: unknown): void => {
    if (!item || typeof item !== 'object') return;
    if (ArrayBuffer.isView(item)) {
      const buffer = item.buffer;
      if (buffer instanceof ArrayBuffer && !seenBuffers.has(buffer)) {
        seenBuffers.add(buffer);
        transferables.push(buffer);
      }
      return;
    }
    if (item instanceof ArrayBuffer) {
      if (!seenBuffers.has(item)) {
        seenBuffers.add(item);
        transferables.push(item);
      }
      return;
    }
    if (seenObjects.has(item)) return;
    seenObjects.add(item);
    if (Array.isArray(item)) {
      item.forEach(visit);
      return;
    }
    for (const nested of Object.values(item as Record<string, unknown>)) visit(nested);
  };
  visit(value);
  return transferables;
}
