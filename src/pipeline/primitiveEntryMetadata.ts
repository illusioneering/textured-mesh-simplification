import type { RawMesh } from '../simplification/types';
import type { SourceMaterial, TexturedRawMesh } from '../texture/types';
import { faceUvSet, hasMaterialTextures } from '../texture/types';
import {
  hasPreservableMaterialData,
} from './process';
import type { ProcessablePrimitiveEntry } from './sceneProcessing';

function imageBackedTextureSlots(material: SourceMaterial) {
  return material.textureSlots.filter((slot) => slot.hasImage);
}

function metadataBackedStandardTextureSlots(material: SourceMaterial) {
  return imageBackedTextureSlots(material).filter((slot) => (
    slot.slot === 'baseColor'
    || slot.slot === 'normal'
    || slot.slot === 'metallicRoughness'
    || slot.slot === 'occlusion'
    || slot.slot === 'emissive'
  ));
}

function metadataBackedBakeTextureSlots(material: SourceMaterial) {
  return metadataBackedStandardTextureSlots(material);
}

function metadataBackedTransferTextureSlots(material: SourceMaterial) {
  return metadataBackedBakeTextureSlots(material);
}

function hasAllRequiredTextureUvs(
  source: TexturedRawMesh,
  requiredSlots: (material: SourceMaterial) => ReturnType<typeof metadataBackedBakeTextureSlots>,
): boolean {
  return source.faceAttributes.length === source.rawMesh.faces.length
    && source.faceAttributes.every((attributes) => {
      const material = source.materials[attributes.materialId];
      if (!material) return false;
      return requiredSlots(material).every((slot) => faceUvSet(attributes, slot.texCoord) !== undefined);
    });
}

export function hasEntryPreservableMaterialData(source: TexturedRawMesh): boolean {
  return source.materials.some(hasPreservableMaterialData);
}

export function hasEntryImageBackedTextureBakeData(source: TexturedRawMesh): boolean {
  return source.materials.some((material) => metadataBackedBakeTextureSlots(material).length > 0)
    && hasAllRequiredTextureUvs(source, metadataBackedBakeTextureSlots);
}

export function hasEntryImageBackedTextureTransferData(source: TexturedRawMesh): boolean {
  return source.materials.some((material) => metadataBackedTransferTextureSlots(material).length > 0)
    && hasAllRequiredTextureUvs(source, metadataBackedTransferTextureSlots);
}

function hasSupportedSourceVertexAttributes(source: TexturedRawMesh): boolean {
  return source.faceAttributes.length === source.rawMesh.faces.length
    && source.faceAttributes.some((attributes) => (
      attributes.uvSets.length > 0
      || attributes.normalCorners !== undefined
      || attributes.tangentCorners !== undefined
      || attributes.colorCorners !== undefined
    ));
}

export const hasMetadataBakeableTextureMapData = hasEntryImageBackedTextureBakeData;
export const hasPreservableMaterialTextureData = hasEntryImageBackedTextureTransferData;

export interface ToProcessablePrimitiveEntryOptions {
  id: string;
  label?: string;
  rawMesh: RawMesh;
  texturedRawMesh?: TexturedRawMesh;
}

export function toProcessablePrimitiveEntry(options: ToProcessablePrimitiveEntryOptions): ProcessablePrimitiveEntry {
  const textured = options.texturedRawMesh;
  const bakeable = textured ? hasEntryImageBackedTextureBakeData(textured) : false;
  const hasTexturedMaterial = textured ? textured.materials.some(hasMaterialTextures) : false;
  const hasPreservableMaterial = textured ? hasEntryPreservableMaterialData(textured) : false;
  const requiresAttributeTransfer = textured ? hasSupportedSourceVertexAttributes(textured) : false;
  return {
    id: options.id,
    ...(options.label !== undefined ? { label: options.label } : {}),
    rawMesh: options.rawMesh,
    ...(textured ? { texturedRawMesh: textured } : {}),
    bakeable,
    hasPreservableMaterialData: hasPreservableMaterial,
    hasTexturedMaterial,
    ...(requiresAttributeTransfer ? { requiresAttributeTransfer: true } : {}),
  };
}
