import type { RawMesh, RawSimplificationResult, SimplificationProgress, SimplifyOptions, VirtualEdgeProgress } from '../simplification/types';
import type { TransferredMeshAttributes } from '../simplification/attributes';
import { simplifyRawMesh } from '../simplification/simplify';
import { bakeStandardMaterialTextures } from '../texture/bake';
import type { TextureBakeBatchRunner } from '../texture/bakeBatch';
import type {
  BakedTextureResult,
  BakeTextureProgress,
  SourceMaterial,
  SourceMaterialTextureInfo,
  SourceTexture,
  StandardBakedTextureSlot,
  TextureSampler,
  TexturedRawMesh,
} from '../texture/types';
import { faceUvSet } from '../texture/types';
import { type ProcessingOptions, toSimplifyOptions } from './options';

export interface BakedMaterialSettings {
  alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff: number;
  doubleSided: boolean;
  metallicFactor: number;
  roughnessFactor: number;
  emissiveFactor: [number, number, number];
  normalScale: number;
  occlusionStrength: number;
}

export interface GeometryProcessingResult {
  raw: RawSimplificationResult;
  elapsedSeconds: number;
}

export interface TexturedProcessingResult extends GeometryProcessingResult {
  baked: BakedTextureResult;
  materialSettings: BakedMaterialSettings;
}

export interface GeometryProcessingSettings {
  onVirtualEdgeProgress?: (progress: VirtualEdgeProgress) => void;
}

export interface TextureBakeProcessingSettings {
  batchRunner?: TextureBakeBatchRunner;
  maxOutputTextureBytes?: number;
}

function isTransferredMeshAttributes(value: TransferredMeshAttributes | TextureBakeProcessingSettings | undefined): value is TransferredMeshAttributes {
  return value !== undefined && 'vertices' in value;
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
}

const STANDARD_BAKED_TEXTURE_SLOTS = new Set<StandardBakedTextureSlot>([
  'normal',
  'metallicRoughness',
  'occlusion',
  'emissive',
]);

const NO_USABLE_TEXTURE_DATA_MESSAGE = 'Input has no image-backed standard material texture data to bake.';

function hasNonDefaultBaseColorFactor(material: SourceMaterial): boolean {
  return material.baseColorFactor[0] !== 1
    || material.baseColorFactor[1] !== 1
    || material.baseColorFactor[2] !== 1
    || material.baseColorFactor[3] !== 1;
}

function hasNonDefaultEmissiveFactor(material: SourceMaterial): boolean {
  return material.emissiveFactor[0] !== 0
    || material.emissiveFactor[1] !== 0
    || material.emissiveFactor[2] !== 0;
}

function isStandardBakedTextureSlot(slot: SourceMaterialTextureInfo['slot']): slot is StandardBakedTextureSlot {
  return STANDARD_BAKED_TEXTURE_SLOTS.has(slot as StandardBakedTextureSlot);
}

function imageBackedStandardTextureSlots(material: SourceMaterial): SourceMaterialTextureInfo[] {
  return material.textureSlots.filter((slot) => isStandardBakedTextureSlot(slot.slot) && slot.hasImage && slot.image);
}

function imageBackedStandardTextureTransferSlots(material: SourceMaterial): SourceMaterialTextureInfo[] {
  return material.textureSlots.filter((slot) => isStandardBakedTextureSlot(slot.slot) && slot.hasImage);
}

export function hasPreservableMaterialData(material: SourceMaterial): boolean {
  return hasNonDefaultBaseColorFactor(material)
    || material.alphaMode !== 'OPAQUE'
    || material.alphaCutoff !== 0.5
    || material.doubleSided
    || material.metallicFactor !== 1
    || material.roughnessFactor !== 1
    || hasNonDefaultEmissiveFactor(material)
    || material.normalScale !== 1
    || material.occlusionStrength !== 1
    || Boolean(material.baseColorTexture)
    || material.textureSlots.some((slot) => slot.hasImage);
}

export function hasImageBackedTextureTransferData(material: SourceMaterial): boolean {
  return Boolean(material.baseColorTexture?.image)
    || material.textureSlots.some((slot) => slot.slot === 'baseColor' && slot.hasImage)
    || imageBackedStandardTextureTransferSlots(material).length > 0;
}

export function hasImageBackedTextureBakeData(material: SourceMaterial): boolean {
  return Boolean(material.baseColorTexture?.image)
    || imageBackedStandardTextureSlots(material).length > 0;
}

export function hasUsableTextureData(materials: SourceMaterial[]): boolean {
  return materials.some((material) => (
    hasImageBackedTextureBakeData(material)
  ));
}

export function hasBakeableTextureData(source: TexturedRawMesh): boolean {
  return source.materials.some(hasImageBackedTextureBakeData)
    && source.faceAttributes.length === source.rawMesh.faces.length
    && source.faceAttributes.every((attributes) => {
      const material = source.materials[attributes.materialId];
      if (!material) return false;
      if (!hasImageBackedTextureBakeData(material)) return true;
      if (material.baseColorTexture?.image && !faceUvSet(attributes, material.baseColorTexture.texCoord)) return false;
      return imageBackedStandardTextureSlots(material).every((slot) => faceUvSet(attributes, slot.texCoord) !== undefined);
    });
}

export function combinedMaterialSettings(materials: SourceMaterial[]): BakedMaterialSettings {
  const alphaMode = materials.some((material) => material.alphaMode === 'BLEND')
    ? 'BLEND'
    : materials.some((material) => material.alphaMode === 'MASK') ? 'MASK' : 'OPAQUE';
  const masked = materials.find((material) => material.alphaMode === 'MASK');
  return {
    alphaMode,
    alphaCutoff: masked?.alphaCutoff ?? 0.5,
    doubleSided: materials.some((material) => material.doubleSided),
    metallicFactor: materials[0]?.metallicFactor ?? 1,
    roughnessFactor: materials[0]?.roughnessFactor ?? 1,
    emissiveFactor: [...(materials[0]?.emissiveFactor ?? [0, 0, 0])] as [number, number, number],
    normalScale: materials[0]?.normalScale ?? 1,
    occlusionStrength: materials[0]?.occlusionStrength ?? 1,
  };
}

function cloneSamplerWithFilter(sampler: TextureSampler, filter: TextureSampler['filter']): TextureSampler {
  return { ...sampler, filter };
}

function cloneTextureWithFilter(texture: SourceTexture, filter: TextureSampler['filter']): SourceTexture {
  return {
    ...(texture.image ? { image: texture.image } : {}),
    sampler: cloneSamplerWithFilter(texture.sampler, filter),
    texCoord: texture.texCoord,
    ...(texture.name !== undefined ? { name: texture.name } : {}),
    ...(texture.mimeType !== undefined ? { mimeType: texture.mimeType } : {}),
  };
}

function cloneTextureSlot(slot: SourceMaterialTextureInfo, filter: TextureSampler['filter']): SourceMaterialTextureInfo {
  return {
    slot: slot.slot,
    texCoord: slot.texCoord,
    sampler: cloneSamplerWithFilter(slot.sampler, filter),
    hasImage: slot.hasImage,
    ...(slot.image ? { image: slot.image } : {}),
    ...(slot.name !== undefined ? { name: slot.name } : {}),
    ...(slot.mimeType !== undefined ? { mimeType: slot.mimeType } : {}),
  };
}

function cloneMaterialWithFilter(material: SourceMaterial, filter: TextureSampler['filter']): SourceMaterial {
  return {
    name: material.name,
    baseColorFactor: [...material.baseColorFactor],
    ...(material.baseColorTexture ? { baseColorTexture: cloneTextureWithFilter(material.baseColorTexture, filter) } : {}),
    textureSlots: material.textureSlots.map((slot) => cloneTextureSlot(slot, filter)),
    alphaMode: material.alphaMode,
    alphaCutoff: material.alphaCutoff,
    doubleSided: material.doubleSided,
    emissiveFactor: [...material.emissiveFactor],
    metallicFactor: material.metallicFactor,
    roughnessFactor: material.roughnessFactor,
    normalScale: material.normalScale,
    occlusionStrength: material.occlusionStrength,
  };
}

function cloneSourceWithFilter(source: TexturedRawMesh, filter: TextureSampler['filter']): TexturedRawMesh {
  return {
    rawMesh: source.rawMesh,
    faceAttributes: source.faceAttributes,
    materials: source.materials.map((material) => cloneMaterialWithFilter(material, filter)),
  };
}

function withProgress(options: SimplifyOptions, onProgress?: (progress: SimplificationProgress) => void): SimplifyOptions {
  return onProgress ? { ...options, onProgress } : options;
}

export function processGeometryOnly(
  rawMesh: RawMesh,
  options: ProcessingOptions,
  onProgress?: (progress: SimplificationProgress) => void,
  settings: GeometryProcessingSettings = {},
): GeometryProcessingResult {
  const started = nowMs();
  const raw = simplifyRawMesh(
    rawMesh,
    withProgress({
      ...toSimplifyOptions(options),
      ...(settings.onVirtualEdgeProgress ? { onVirtualEdgeProgress: settings.onVirtualEdgeProgress } : {}),
    }, onProgress),
  );
  return { raw, elapsedSeconds: (nowMs() - started) / 1000 };
}

export async function bakeTextureForSimplifiedGeometry(
  source: TexturedRawMesh,
  raw: RawSimplificationResult,
  options: ProcessingOptions,
  onBakeProgress?: (progress: BakeTextureProgress) => void,
  transferredAttributesOrSettings?: TransferredMeshAttributes | TextureBakeProcessingSettings,
  maybeSettings: TextureBakeProcessingSettings = {},
): Promise<TexturedProcessingResult> {
  const started = nowMs();
  const transferredAttributes = isTransferredMeshAttributes(transferredAttributesOrSettings)
    ? transferredAttributesOrSettings
    : undefined;
  const settings = isTransferredMeshAttributes(transferredAttributesOrSettings)
    ? maybeSettings
    : transferredAttributesOrSettings ?? maybeSettings;
  const filteredSource = cloneSourceWithFilter(source, options.textureFilter);
  if (!hasUsableTextureData(filteredSource.materials)) {
    throw new Error(NO_USABLE_TEXTURE_DATA_MESSAGE);
  }
  if (raw.rawMesh.faces.length === 0) throw new Error('Texture transfer cannot write an output mesh with zero faces.');

  const materialSettings = combinedMaterialSettings(filteredSource.materials);
  const baked = await bakeStandardMaterialTextures({
    source: filteredSource,
    outputRawMesh: raw.rawMesh,
    outputFaceIds: raw.outputFaceIds,
    history: raw.history,
    ...(transferredAttributes ? { transferredAttributes } : {}),
    textureSize: options.textureSize,
    padding: options.texturePadding,
    outputNormalScale: materialSettings.normalScale,
    ...(onBakeProgress ? { onProgress: onBakeProgress } : {}),
    ...(settings.batchRunner ? { batchRunner: settings.batchRunner } : {}),
    ...(settings.maxOutputTextureBytes !== undefined ? { maxOutputTextureBytes: settings.maxOutputTextureBytes } : {}),
  });

  return {
    raw,
    baked,
    materialSettings,
    elapsedSeconds: (nowMs() - started) / 1000,
  };
}

export async function processTextured(
  source: TexturedRawMesh,
  options: ProcessingOptions,
  onProgress?: (progress: SimplificationProgress) => void,
  onGeometryComplete?: (raw: RawSimplificationResult) => void,
  onBakeProgress?: (progress: BakeTextureProgress) => void,
  settings: TextureBakeProcessingSettings = {},
): Promise<TexturedProcessingResult> {
  const started = nowMs();
  if (!hasUsableTextureData(source.materials)) {
    throw new Error(NO_USABLE_TEXTURE_DATA_MESSAGE);
  }

  const raw = simplifyRawMesh(
    source.rawMesh,
    withProgress(toSimplifyOptions(options), onProgress),
  );
  onGeometryComplete?.(raw);

  const result = await bakeTextureForSimplifiedGeometry(source, raw, options, onBakeProgress, settings);
  return { ...result, elapsedSeconds: (nowMs() - started) / 1000 };
}
