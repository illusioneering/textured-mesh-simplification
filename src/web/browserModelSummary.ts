import { DoubleSide, type BufferGeometry, type Material, type Mesh, type Object3D } from 'three';
import type { PrimitiveExtractionSummary } from '../pipeline/primitiveExtraction';
import type { SourceMaterialTextureSlot } from '../texture/types';
import {
  browserMaterialTextureSlots,
  browserTextureChannel,
  browserTextureDimensions,
  browserTextureHasImageSource,
  type BrowserMaterialTextureSlot,
  type BrowserTextureSlotMaterial,
} from './browserMaterialTextures';

export interface BrowserAssetSummary extends PrimitiveExtractionSummary {
  hasPreservableMaterialData: boolean;
  hasImageBackedTextureTransferData: boolean;
  hasImageBackedTextureBakeData: boolean;
  materials: number;
  materialsWithTextures: number;
  materialsWithBaseColorImages: number;
  facesWithUvs: number;
  textureSlotKinds: string[];
  textureDimensions: string[];
}

type MaterialKey = Material | 'default';

const STANDARD_BAKED_TEXTURE_SLOTS = new Set<SourceMaterialTextureSlot>([
  'normal',
  'metallicRoughness',
  'occlusion',
  'emissive',
]);

interface MaterialParentGroupSummary {
  hasImageBackedTextureData: boolean;
  allFacesHaveRequiredUvs: boolean;
  hasAnyFaceWithRequiredUvs: boolean;
}

function uvTexCoordFromAttributeName(name: string): number | undefined {
  if (name === 'uv') return 0;
  const match = /^uv(\d+)$/.exec(name);
  if (!match) return undefined;
  const texCoord = Number(match[1]);
  return Number.isInteger(texCoord) && texCoord >= 0 ? texCoord : undefined;
}

function geometryUvTexCoords(geometry: BufferGeometry): Set<number> {
  const texCoords = new Set<number>();
  for (const [name, attribute] of Object.entries(geometry.attributes)) {
    const texCoord = uvTexCoordFromAttributeName(name);
    if (texCoord !== undefined && attribute.itemSize >= 2) texCoords.add(texCoord);
  }
  return texCoords;
}

function meshMaterials(mesh: Mesh): (Material | null)[] {
  if (Array.isArray(mesh.material)) return mesh.material;
  return [mesh.material ?? null];
}

function groupRangeIndex(geometry: BufferGeometry, indexOffset: number): number {
  if (geometry.groups.length === 0) return 0;
  const groupIndex = geometry.groups.findIndex((candidate) => indexOffset >= candidate.start && indexOffset < candidate.start + candidate.count);
  return groupIndex >= 0 ? groupIndex : 0;
}

function groupMaterialIndex(geometry: BufferGeometry, indexOffset: number): number {
  return geometry.groups[groupRangeIndex(geometry, indexOffset)]?.materialIndex ?? 0;
}

function materialKey(material: Material | null): MaterialKey {
  return material ?? 'default';
}

function materialTextureSlots(material: MaterialKey): BrowserMaterialTextureSlot[] {
  return material === 'default' ? [] : browserMaterialTextureSlots(material);
}

function materialBaseColorTexture(material: MaterialKey) {
  return material === 'default' ? null : (material as BrowserTextureSlotMaterial).map;
}

function imageBackedStandardTextureSlots(material: MaterialKey): BrowserMaterialTextureSlot[] {
  return materialTextureSlots(material).filter((slot) => (
    STANDARD_BAKED_TEXTURE_SLOTS.has(slot.slot)
    && browserTextureHasImageSource(slot.texture)
  ));
}

function materialHasBakeableTextureMapData(material: MaterialKey): boolean {
  return browserTextureHasImageSource(materialBaseColorTexture(material))
    || imageBackedStandardTextureSlots(material).length > 0;
}

function hasNonDefaultBaseColorFactor(material: MaterialKey): boolean {
  if (material === 'default') return false;
  const typed = material as Material & { color?: { r: number; g: number; b: number }; opacity?: number };
  return (typed.color?.r ?? 1) !== 1
    || (typed.color?.g ?? 1) !== 1
    || (typed.color?.b ?? 1) !== 1
    || (typed.opacity ?? 1) !== 1;
}

function hasNonDefaultEmissiveFactor(material: MaterialKey): boolean {
  if (material === 'default') return false;
  const emissive = (material as Material & { emissive?: { r: number; g: number; b: number } }).emissive;
  return (emissive?.r ?? 0) !== 0
    || (emissive?.g ?? 0) !== 0
    || (emissive?.b ?? 0) !== 0;
}

function materialHasPreservableData(material: MaterialKey): boolean {
  if (material === 'default') return false;
  const typed = material as Material & {
    alphaTest?: number;
    aoMapIntensity?: number;
    metalness?: number;
    normalScale?: { x: number };
    roughness?: number;
    transparent?: boolean;
  };
  return hasNonDefaultBaseColorFactor(material)
    || typed.transparent === true
    || (typed.alphaTest ?? 0) > 0
    || material.side === DoubleSide
    || (typed.metalness ?? 1) !== 1
    || (typed.roughness ?? 1) !== 1
    || hasNonDefaultEmissiveFactor(material)
    || (typed.normalScale?.x ?? 1) !== 1
    || (typed.aoMapIntensity ?? 1) !== 1
    || materialTextureSlots(material).some((slot) => browserTextureHasImageSource(slot.texture));
}

function requiredBakeUvChannels(material: MaterialKey): Set<number> {
  const channels = new Set<number>();
  const baseColorTexture = materialBaseColorTexture(material);
  if (browserTextureHasImageSource(baseColorTexture)) channels.add(browserTextureChannel(baseColorTexture));
  for (const slot of imageBackedStandardTextureSlots(material)) {
    channels.add(browserTextureChannel(slot.texture));
  }
  return channels;
}

function geometryFaceIndexOffsets(geometry: BufferGeometry, positionCount: number): number[] {
  const index = geometry.getIndex();
  const count = index?.count ?? positionCount;
  if (count % 3 !== 0) throw new Error(`${index ? 'Indexed' : 'Non-indexed'} geometry has ${count} ${index ? 'indices' : 'vertices'}, which is not divisible by 3.`);
  const offsets: number[] = [];
  for (let offset = 0; offset < count; offset += 3) offsets.push(offset);
  return offsets;
}

function objectOrdinalMap(root: Object3D): Map<Object3D, number> {
  const ordinals = new Map<Object3D, number>();
  root.traverse((object) => {
    ordinals.set(object, ordinals.size);
  });
  return ordinals;
}

function groupHasRequiredUvs(requiredChannels: Set<number>, uvTexCoords: Set<number>): boolean {
  for (const channel of requiredChannels) {
    if (!uvTexCoords.has(channel)) return false;
  }
  return true;
}

export async function summarizeBrowserObject(root: Object3D): Promise<BrowserAssetSummary> {
  root.updateMatrixWorld(true);
  const objectOrdinals = objectOrdinalMap(root);
  const rootOrdinal = objectOrdinals.get(root) ?? 0;
  let inputVertices = 0;
  let inputFaces = 0;
  let facesWithUvs = 0;
  const materials = new Set<MaterialKey>();
  const materialIds = new Map<MaterialKey, number>();
  const materialParentGroups = new Map<string, MaterialParentGroupSummary>();
  const textureSlotKinds = new Set<string>();
  const textureDimensionSet = new Set<string>();
  const warnings: string[] = [];
  let hasTransferableTextureData = false;
  let hasImageBackedTextureTransferData = false;

  const getMaterialId = (material: MaterialKey): number => {
    const existing = materialIds.get(material);
    if (existing !== undefined) return existing;
    const id = materialIds.size;
    materialIds.set(material, id);
    return id;
  };

  root.traverse((object) => {
    const mesh = object as Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const position = mesh.geometry.getAttribute('position');
    if (!position || position.itemSize !== 3) {
      warnings.push(`Mesh ${object.name || '<unnamed>'} is missing a VEC3 position attribute and was skipped.`);
      return;
    }
    inputVertices += position.count;
    const faceOffsets = geometryFaceIndexOffsets(mesh.geometry, position.count);
    const faceCount = faceOffsets.length;
    inputFaces += faceCount;
    const uvTexCoords = geometryUvTexCoords(mesh.geometry);
    if (uvTexCoords.size > 0) facesWithUvs += faceCount;
    const materialList = meshMaterials(mesh);
    const parent = mesh.parent ?? root;
    const parentObjectOrdinal = objectOrdinals.get(parent) ?? rootOrdinal;
    for (const faceOffset of faceOffsets) {
      const material = materialKey(materialList[groupMaterialIndex(mesh.geometry, faceOffset)] ?? materialList[0] ?? null);
      materials.add(material);
      const materialId = getMaterialId(material);
      for (const slot of materialTextureSlots(material)) {
        if (!browserTextureHasImageSource(slot.texture)) continue;
        textureSlotKinds.add(slot.slot);
        const dimensions = browserTextureDimensions(slot.texture);
        if (dimensions) textureDimensionSet.add(dimensions);
      }
      const groupKey = `${parentObjectOrdinal}/${materialId}`;
      const group = materialParentGroups.get(groupKey) ?? {
        hasImageBackedTextureData: false,
        allFacesHaveRequiredUvs: true,
        hasAnyFaceWithRequiredUvs: false,
      };
      if (materialHasBakeableTextureMapData(material)) {
        const faceHasRequiredUvs = groupHasRequiredUvs(requiredBakeUvChannels(material), uvTexCoords);
        if (faceHasRequiredUvs) {
          hasTransferableTextureData = true;
          hasImageBackedTextureTransferData = true;
        }
        group.hasImageBackedTextureData = true;
        group.hasAnyFaceWithRequiredUvs = group.hasAnyFaceWithRequiredUvs || faceHasRequiredUvs;
        group.allFacesHaveRequiredUvs = group.allFacesHaveRequiredUvs
          && faceHasRequiredUvs;
      }
      materialParentGroups.set(groupKey, group);
    }
  });

  const materialArray = Array.from(materials);
  const bakeableEntryCount = Array.from(materialParentGroups.values()).filter((group) => (
    group.hasImageBackedTextureData && group.allFacesHaveRequiredUvs
  )).length;
  return {
    inputVertices,
    inputFaces,
    bakeableEntryCount,
    hasTransferableTextureData,
    hasPreservableMaterialData: materialArray.some(materialHasPreservableData),
    hasImageBackedTextureTransferData,
    hasImageBackedTextureBakeData: bakeableEntryCount > 0,
    warnings,
    materials: materialArray.length,
    materialsWithTextures: materialArray.filter((material) => materialTextureSlots(material).some((slot) => browserTextureHasImageSource(slot.texture))).length,
    materialsWithBaseColorImages: materialArray.filter((material) => browserTextureHasImageSource(materialBaseColorTexture(material))).length,
    facesWithUvs,
    textureSlotKinds: Array.from(textureSlotKinds).sort(),
    textureDimensions: Array.from(textureDimensionSet).sort(),
  };
}
