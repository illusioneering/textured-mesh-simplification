import {
  BufferAttribute,
  BufferGeometry,
  ClampToEdgeWrapping,
  DataTexture,
  DoubleSide,
  Float32BufferAttribute,
  FrontSide,
  LinearFilter,
  LoadingManager,
  Material,
  Matrix3,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  MirroredRepeatWrapping,
  NearestFilter,
  Object3D,
  RGBAFormat,
  RepeatWrapping,
  Scene,
  SRGBColorSpace,
  Texture,
  Vector2,
  Vector3,
  Vector4,
} from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { SourceFaceColorCorners, TransferredMeshAttributes, VertexColorItemSize } from '../simplification/attributes';
import type { FaceIndices, RawMesh } from '../simplification/types';
import { appendRawMeshVertex, createWeldedVertexAppendState, type WeldedVertexAppendState } from '../simplification/weld';
import type {
  AtlasLayout,
  BakedMaterialTexture,
  RgbaImage,
  SourceFaceAttributes,
  SourceFaceUvSet,
  SourceMaterial,
  SourceMaterialTextureInfo,
  SourceMaterialTextureSlot,
  TexturedRawMesh,
  TextureSampler,
  WrapMode,
} from '../texture/types';
import { faceUvSet } from '../texture/types';
import { transformAuthoredTangent } from '../texture/tangentSpace';
import { type BakedMaterialSettings } from '../pipeline/process';
import type { PrimitiveGroupingMode } from '../pipeline/options';
import {
  buildAttributeTransferredPrimitiveGeometryData,
  buildAtlasPrimitiveGeometryData,
  buildIndexedPrimitiveGeometryData,
  computePrimitiveGeometryDataTangents,
  type PrimitiveGeometryData,
} from '../pipeline/primitiveOutputGeometry';
import {
  createPrimitiveExtractionResult,
  type PrimitiveExtractionMode,
  type PrimitiveExtractionOptions,
  type PrimitiveExtractionResult,
  type PrimitiveSourceAdapter,
} from '../pipeline/primitiveExtraction';
import {
  hasEntryImageBackedTextureBakeData,
  hasEntryImageBackedTextureTransferData,
  hasEntryPreservableMaterialData,
  toProcessablePrimitiveEntry,
} from '../pipeline/primitiveEntryMetadata';
import type { ProcessedPrimitiveEntry } from '../pipeline/sceneProcessing';
import {
  attachmentLocalMatrix,
  createSceneWithPreservedTransform,
  sceneTransformContextFromObject,
  type BrowserSceneTransformContext,
} from './sceneContext';
import {
  classifyExternalGltfResourceFileName,
  requestedExternalResourceFileName,
  type ExternalGltfResourceKind,
} from './externalGltfResources';
import {
  browserMaterialTextureSlots,
  browserTextureChannel,
  browserTextureHasImageSource,
  type BrowserTextureSlotMaterial,
} from './browserMaterialTextures';
import { normalizeBrowserSceneMaterialsToCorePbr } from './corePbrMaterialNormalization';
import { summarizeBrowserObject, type BrowserAssetSummary } from './browserModelSummary';

export type { BrowserAssetSummary } from './browserModelSummary';
export { summarizeBrowserObject } from './browserModelSummary';

export interface ExtractedBrowserModel {
  rawMesh: RawMesh;
  textured: TexturedRawMesh;
  primitiveEntries: BrowserPrimitiveEntry[];
  primitiveEntryGroups: Record<PrimitiveGroupingMode, BrowserPrimitiveEntry[]>;
  sceneContext: BrowserSceneTransformContext;
  warnings: string[];
}

export interface LoadedBrowserModel extends ExtractedBrowserModel {
  scene: Object3D;
  matchedExternalTextureFiles?: string[];
  matchedExternalBinaryBufferFiles?: string[];
}

export interface BrowserPrimitiveApplyMetadata {
  meshOrdinal: number;
  sourceMeshOrdinals: number[];
  parentObjectOrdinal?: number;
  sourceMaterial: Material | null;
  preserveSourceMeshTransform?: boolean;
}

export interface BrowserLoadedAsset extends PrimitiveSourceAdapter<BrowserPrimitiveApplyMetadata, Object3D, PrimitiveExtractionOptions> {
  assetRevision: number;
  scene: Object3D;
  summary: BrowserAssetSummary;
  warnings: string[];
  matchedExternalTextureFiles?: string[];
  matchedExternalBinaryBufferFiles?: string[];
  buildOutputScene(
    extraction: PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>,
    results: readonly PrimitiveOutputReplacement[],
  ): Promise<Object3D>;
  disposeIntermediateData(): void;
}

export interface TexturedOutputSceneOptions {
  rawMesh: RawMesh;
  atlas: AtlasLayout;
  image: RgbaImage;
  additionalTextures?: readonly BakedMaterialTexture[];
  materialSettings: BakedMaterialSettings;
  sceneContext?: BrowserSceneTransformContext;
  transferredAttributes?: TransferredMeshAttributes;
}

export interface BrowserPrimitiveEntry {
  id: string;
  label: string;
  meshOrdinal: number;
  sourceMeshOrdinals: number[];
  parentObjectOrdinal: number;
  sourceMaterial: Material | null;
  rawMesh: RawMesh;
  texturedRawMesh: TexturedRawMesh;
  bakeable: boolean;
  hasPreservableMaterialData?: boolean;
  hasTexturedMaterial: boolean;
  preserveSourceMeshTransform?: boolean;
  requiresAttributeTransfer?: boolean;
}

export type PrimitiveOutputReplacement = {
  id: string;
  meshOrdinal: number;
  sourceMeshOrdinals?: readonly number[];
  parentObjectOrdinal?: number;
  sourceMaterial?: Material | null;
  preserveSourceMeshTransform?: boolean;
  rawMesh: RawMesh;
  transferredAttributes?: TransferredMeshAttributes;
  materialMode: 'preserve' | 'neutral';
} | {
  id: string;
  meshOrdinal: number;
  sourceMeshOrdinals?: readonly number[];
  parentObjectOrdinal?: number;
  sourceMaterial?: Material | null;
  preserveSourceMeshTransform?: boolean;
  rawMesh: RawMesh;
  transferredAttributes?: TransferredMeshAttributes;
  materialMode: 'baked';
  atlas: AtlasLayout;
  image: RgbaImage;
  additionalTextures?: readonly BakedMaterialTexture[];
  materialSettings: BakedMaterialSettings;
};

type PreservedMaterialCloneCache = Map<Material, {
  normal?: Material;
  vertexColors?: Material;
}>;

export interface ExternalGltfResource {
  fileName: string;
  objectUrl: string;
  kind: ExternalGltfResourceKind;
}

export interface ParseGlbArrayBufferOptions {
  externalResourceFiles?: readonly File[];
}

function cloneImageBytes(data: ArrayLike<number>): Uint8Array {
  const output = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i += 1) output[i] = Math.max(0, Math.min(255, Math.round(data[i] ?? 0)));
  return output;
}

export function matchExternalGltfResource(uri: string, resources: readonly ExternalGltfResource[]): ExternalGltfResource | undefined {
  const fileName = requestedExternalResourceFileName(uri);
  if (!fileName) return undefined;
  return resources.find((resource) => resource.fileName === fileName);
}

function createExternalGltfResources(files: readonly File[]): ExternalGltfResource[] {
  const resources: ExternalGltfResource[] = [];
  for (const file of files) {
    const kind = classifyExternalGltfResourceFileName(file.name);
    if (!kind) continue;
    resources.push({
      fileName: file.name,
      objectUrl: URL.createObjectURL(file),
      kind,
    });
  }
  return resources;
}

function defaultMaterial(): SourceMaterial {
  return {
    name: 'default-material',
    baseColorFactor: [1, 1, 1, 1],
    textureSlots: [],
    alphaMode: 'OPAQUE',
    alphaCutoff: 0.5,
    doubleSided: false,
    emissiveFactor: [0, 0, 0],
    metallicFactor: 1,
    roughnessFactor: 1,
    normalScale: 1,
    occlusionStrength: 1,
  };
}

function materialName(material: Material | null): string {
  return material?.name && material.name.length > 0 ? material.name : 'material';
}

function wrapMode(value: Texture['wrapS']): WrapMode {
  if (value === ClampToEdgeWrapping) return 'clamp';
  if (value === MirroredRepeatWrapping) return 'mirrored-repeat';
  return 'repeat';
}

function textureFilter(texture: Texture | null | undefined): TextureSampler['filter'] {
  if (!texture) return 'linear';
  return texture.magFilter === NearestFilter || texture.minFilter === NearestFilter ? 'nearest' : 'linear';
}

function baseColorFactor(material: Material): [number, number, number, number] {
  const typed = material as Material & { color?: { r: number; g: number; b: number }; opacity?: number };
  return [typed.color?.r ?? 1, typed.color?.g ?? 1, typed.color?.b ?? 1, typed.opacity ?? 1];
}

function textureSampler(texture: Texture | null | undefined): TextureSampler {
  return {
    wrapS: wrapMode(texture?.wrapS ?? RepeatWrapping),
    wrapT: wrapMode(texture?.wrapT ?? RepeatWrapping),
    filter: textureFilter(texture),
  };
}

function textureName(texture: Texture | null | undefined): string | undefined {
  return texture?.name && texture.name.length > 0 ? texture.name : undefined;
}

function textureMimeType(texture: Texture | null | undefined): string | undefined {
  const mimeType = (texture?.userData as { mimeType?: unknown } | undefined)?.mimeType;
  return typeof mimeType === 'string' && mimeType.length > 0 ? mimeType : undefined;
}

function textureSlot(slot: SourceMaterialTextureSlot, texture: Texture | null | undefined): SourceMaterialTextureInfo | null {
  if (!texture) return null;
  const name = textureName(texture);
  const mimeType = textureMimeType(texture);
  return {
    slot,
    texCoord: browserTextureChannel(texture),
    sampler: textureSampler(texture),
    hasImage: browserTextureHasImageSource(texture),
    ...(name ? { name } : {}),
    ...(mimeType ? { mimeType } : {}),
  };
}

function materialMap(material: Material | null): Texture | null {
  return ((material as (Material & { map?: Texture | null }) | null)?.map) ?? null;
}

function materialTextureSlots(material: Material): SourceMaterialTextureInfo[] {
  return browserMaterialTextureSlots(material)
    .map(({ slot, texture }) => textureSlot(slot, texture))
    .filter((slot): slot is SourceMaterialTextureInfo => slot !== null);
}

function standardTextureForSlot(material: Material, slot: SourceMaterialTextureSlot): Texture | null | undefined {
  const typed = material as BrowserTextureSlotMaterial;
  if (slot === 'baseColor') return typed.map;
  if (slot === 'normal') return typed.normalMap;
  if (slot === 'metallicRoughness') return typed.metalnessMap ?? typed.roughnessMap;
  if (slot === 'occlusion') return typed.aoMap;
  if (slot === 'emissive') return typed.emissiveMap;
  return undefined;
}

async function materialTextureSlotsWithImages(
  material: Material,
  baseColorImage?: RgbaImage,
): Promise<SourceMaterialTextureInfo[]> {
  const imageCache = new Map<Texture, RgbaImage | undefined>();
  const imageForTexture = async (texture: Texture): Promise<RgbaImage | undefined> => {
    if (texture === materialMap(material) && baseColorImage) return baseColorImage;
    if (imageCache.has(texture)) return imageCache.get(texture);
    const image = await textureToRgbaImage(texture);
    imageCache.set(texture, image);
    return image;
  };

  const slots: SourceMaterialTextureInfo[] = [];
  for (const slot of materialTextureSlots(material)) {
    const texture = standardTextureForSlot(material, slot.slot);
    if (!texture || !slot.hasImage) {
      slots.push(slot);
      continue;
    }
    if (slot.slot === 'baseColor') {
      slots.push(slot);
      continue;
    }
    const image = await imageForTexture(texture);
    slots.push(image ? { ...slot, image } : slot);
  }
  return slots;
}

function alphaMode(material: Material): SourceMaterial['alphaMode'] {
  const typed = material as Material & { transparent?: boolean; alphaTest?: number };
  if (typed.transparent) return 'BLEND';
  if ((typed.alphaTest ?? 0) > 0) return 'MASK';
  return 'OPAQUE';
}

async function sourceMaterialFromThree(material: Material | null, mode: PrimitiveExtractionMode): Promise<SourceMaterial> {
  if (!material) return defaultMaterial();
  const includeTextureMetadata = mode !== 'geometry';
  const includeImages = mode === 'bake';
  const texture = includeTextureMetadata ? materialMap(material) : null;
  const image = includeImages ? await textureToRgbaImage(texture) : undefined;
  const typed = material as MeshStandardMaterial & MeshBasicMaterial;
  const texCoord = browserTextureChannel(texture);
  const sampler = textureSampler(texture);
  const baseColorTextureName = textureName(texture);
  const baseColorTextureMimeType = textureMimeType(texture);
  const textureSlots = includeTextureMetadata
    ? includeImages ? await materialTextureSlotsWithImages(material, image) : materialTextureSlots(material)
    : [];
  const emissive = (typed as MeshStandardMaterial).emissive;
  const normalScale = (typed as MeshStandardMaterial).normalScale;
  const aoMapIntensity = (typed as MeshStandardMaterial).aoMapIntensity;
  return {
    name: materialName(material),
    baseColorFactor: baseColorFactor(material),
    ...(texture ? {
      baseColorTexture: {
        ...(image ? { image } : {}),
        sampler,
        texCoord,
        ...(baseColorTextureName ? { name: baseColorTextureName } : {}),
        ...(baseColorTextureMimeType ? { mimeType: baseColorTextureMimeType } : {}),
      },
    } : {}),
    textureSlots,
    alphaMode: alphaMode(material),
    alphaCutoff: typed.alphaTest && typed.alphaTest > 0 ? typed.alphaTest : 0.5,
    doubleSided: material.side === DoubleSide,
    emissiveFactor: emissive ? [emissive.r, emissive.g, emissive.b] : [0, 0, 0],
    metallicFactor: typeof typed.metalness === 'number' ? typed.metalness : 1,
    roughnessFactor: typeof typed.roughness === 'number' ? typed.roughness : 1,
    normalScale: normalScale ? normalScale.x : 1,
    occlusionStrength: typeof aoMapIntensity === 'number' ? aoMapIntensity : 1,
  };
}

function sourceNormalMapYScaleFromThree(material: Material | null): number | undefined {
  if (!material) return undefined;
  const typed = material as MeshStandardMaterial & BrowserTextureSlotMaterial;
  if (!typed.normalMap || !browserTextureHasImageSource(typed.normalMap)) return undefined;
  const normalScale = typed.normalScale;
  if (!normalScale || Math.abs(normalScale.x) <= 1e-8) return undefined;
  const yScale = normalScale.y / normalScale.x;
  return Number.isFinite(yScale) && Math.abs(yScale - 1) > 1e-8 ? yScale : undefined;
}

function dataLikeImageToRgba(image: { data: ArrayLike<number>; width: number; height: number }): RgbaImage {
  return { width: image.width, height: image.height, data: cloneImageBytes(image.data) };
}

async function canvasImageToRgba(image: CanvasImageSource & { width?: number; height?: number }): Promise<RgbaImage | undefined> {
  const width = image.width ?? ('videoWidth' in image ? image.videoWidth : undefined);
  const height = image.height ?? ('videoHeight' in image ? image.videoHeight : undefined);
  if (!width || !height) return undefined;
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(width, height)
    : Object.assign(document.createElement('canvas'), { width, height });
  const context = canvas.getContext('2d');
  if (!context) return undefined;
  context.drawImage(image, 0, 0, width, height);
  const imageData = context.getImageData(0, 0, width, height);
  return { width, height, data: new Uint8ClampedArray(imageData.data) };
}

async function textureToRgbaImage(texture: Texture | null): Promise<RgbaImage | undefined> {
  if (!texture) return undefined;
  const source = (texture as Texture & { source?: { data?: unknown } }).source?.data ?? texture.image;
  if (!source) return undefined;
  if (typeof ImageData !== 'undefined' && source instanceof ImageData) {
    return { width: source.width, height: source.height, data: new Uint8ClampedArray(source.data) };
  }
  if (typeof source === 'object' && 'data' in source && 'width' in source && 'height' in source) {
    const candidate = source as { data: ArrayLike<number>; width: number; height: number };
    if (candidate.data && candidate.width && candidate.height) return dataLikeImageToRgba(candidate);
  }
  if (typeof document !== 'undefined' || typeof OffscreenCanvas !== 'undefined') {
    return canvasImageToRgba(source as CanvasImageSource & { width?: number; height?: number });
  }
  return undefined;
}

function groupMaterialIndex(geometry: BufferGeometry, indexOffset: number): number {
  return geometry.groups[groupRangeIndex(geometry, indexOffset)]?.materialIndex ?? 0;
}

function groupRangeIndex(geometry: BufferGeometry, indexOffset: number): number {
  if (geometry.groups.length === 0) return 0;
  const groupIndex = geometry.groups.findIndex((candidate) => indexOffset >= candidate.start && indexOffset < candidate.start + candidate.count);
  return groupIndex >= 0 ? groupIndex : 0;
}

function meshMaterials(mesh: Mesh): (Material | null)[] {
  if (Array.isArray(mesh.material)) return mesh.material;
  return [mesh.material ?? null];
}

function uvTexCoordFromAttributeName(name: string): number | undefined {
  if (name === 'uv') return 0;
  const match = /^uv(\d+)$/.exec(name);
  if (!match) return undefined;
  const texCoord = Number(match[1]);
  return Number.isInteger(texCoord) && texCoord >= 0 ? texCoord : undefined;
}

interface ReadableUvAttribute {
  itemSize: number;
  count: number;
  getX(index: number): number;
  getY(index: number): number;
}

function isReadableUvAttribute(attribute: unknown): attribute is ReadableUvAttribute {
  return typeof attribute === 'object'
    && attribute !== null
    && 'itemSize' in attribute
    && 'count' in attribute
    && 'getX' in attribute
    && 'getY' in attribute
    && typeof (attribute as ReadableUvAttribute).itemSize === 'number'
    && typeof (attribute as ReadableUvAttribute).count === 'number'
    && typeof (attribute as ReadableUvAttribute).getX === 'function'
    && typeof (attribute as ReadableUvAttribute).getY === 'function';
}

function geometryUvAttributes(geometry: BufferGeometry): Array<{ texCoord: number; attribute: ReadableUvAttribute }> {
  const uvAttributes: Array<{ texCoord: number; attribute: ReadableUvAttribute }> = [];
  for (const [name, attribute] of Object.entries(geometry.attributes)) {
    const texCoord = uvTexCoordFromAttributeName(name);
    if (texCoord === undefined || !isReadableUvAttribute(attribute) || attribute.itemSize < 2) continue;
    uvAttributes.push({ texCoord, attribute });
  }
  return uvAttributes.sort((a, b) => a.texCoord - b.texCoord);
}

function objectOrdinalMap(root: Object3D): Map<Object3D, number> {
  const ordinals = new Map<Object3D, number>();
  root.traverse((object) => {
    ordinals.set(object, ordinals.size);
  });
  return ordinals;
}

function objectDisplayLabel(object: Object3D, fallback: string): string {
  const name = object.name.trim();
  return name.length > 0 ? name : fallback;
}

function transformFaceNormals(
  normalCorners: [Vector3, Vector3, Vector3] | undefined,
  transform: Matrix4,
): [Vector3, Vector3, Vector3] | undefined {
  if (!normalCorners) return undefined;
  const normalMatrix = new Matrix3().getNormalMatrix(transform);
  return normalCorners.map((normal) => normal.clone().applyNormalMatrix(normalMatrix).normalize()) as [Vector3, Vector3, Vector3];
}

function cloneFaceNormals(
  normalCorners: [Vector3, Vector3, Vector3] | undefined,
): [Vector3, Vector3, Vector3] | undefined {
  if (!normalCorners) return undefined;
  return normalCorners.map((normal) => normal.clone()) as [Vector3, Vector3, Vector3];
}

function transformFaceTangents(
  tangentCorners: [Vector4, Vector4, Vector4] | undefined,
  transform: Matrix4,
): [Vector4, Vector4, Vector4] | undefined {
  if (!tangentCorners) return undefined;
  return tangentCorners.map((tangent) => transformAuthoredTangent(tangent, transform)) as [Vector4, Vector4, Vector4];
}

function cloneFaceTangents(
  tangentCorners: [Vector4, Vector4, Vector4] | undefined,
): [Vector4, Vector4, Vector4] | undefined {
  if (!tangentCorners) return undefined;
  return tangentCorners.map((tangent) => tangent.clone()) as [Vector4, Vector4, Vector4];
}

function cloneFaceColors(
  colorCorners: SourceFaceColorCorners,
): SourceFaceColorCorners {
  return colorCorners.map((color) => color.clone()) as SourceFaceColorCorners;
}

function colorSchemaKey(
  colorCorners: SourceFaceColorCorners | undefined,
  colorItemSize: VertexColorItemSize | undefined,
): string {
  return colorCorners && colorItemSize ? `color-${colorItemSize}` : 'color-none';
}

function groupedEntryLabel(mode: PrimitiveGroupingMode, object: Object3D, material: Material | null, index: number): string {
  const materialLabel = materialName(material);
  if (mode === 'material') return `${materialLabel} group`;
  if (mode === 'none') return `${objectDisplayLabel(object, `Mesh ${index + 1}`)} / ${materialLabel}`;
  return `${objectDisplayLabel(object, `Parent ${index + 1}`)} / ${materialLabel}`;
}

function uvAttributeName(texCoord: number): string {
  return texCoord === 0 ? 'uv' : `uv${texCoord}`;
}

function requiredTextureTexCoords(material: Material | null | undefined): number[] {
  if (!material) return [];
  return Array.from(new Set(materialTextureSlots(material)
    .filter((slot) => slot.hasImage)
    .map((slot) => slot.texCoord)))
    .sort((a, b) => a - b);
}

function materialReferencesNormalMap(material: Material | null | undefined): boolean {
  return material !== null
    && material !== undefined
    && Boolean((material as BrowserTextureSlotMaterial).normalMap);
}

function createBufferGeometryFromPrimitiveData(data: PrimitiveGeometryData): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(data.positions, 3));
  geometry.setAttribute('normal', new BufferAttribute(data.normals, 3));
  for (const [texCoord, texCoords] of data.texCoordsBySet) {
    geometry.setAttribute(uvAttributeName(texCoord), new Float32BufferAttribute(texCoords, 2));
  }
  if (data.tangents) geometry.setAttribute('tangent', new BufferAttribute(data.tangents, 4));
  if (data.colors && data.colorItemSize) geometry.setAttribute('color', new BufferAttribute(data.colors, data.colorItemSize));
  geometry.setIndex(new BufferAttribute(data.indices, 1));
  return geometry;
}

function transformPrimitiveGeometryData(
  data: PrimitiveGeometryData,
  positionMatrix: Matrix4,
  normalMatrix: Matrix3,
  tangentHandednessScale?: 1 | -1,
): PrimitiveGeometryData {
  const positions = data.positions.slice();
  const normals = data.normals.slice();
  const position = new Vector3();
  const normal = new Vector3();
  for (let offset = 0; offset + 2 < positions.length; offset += 3) {
    position.fromArray(positions, offset).applyMatrix4(positionMatrix).toArray(positions, offset);
    normal.fromArray(normals, offset).applyNormalMatrix(normalMatrix).normalize().toArray(normals, offset);
  }
  const transformed: PrimitiveGeometryData = {
    positions,
    normals,
    indices: data.indices,
    texCoordsBySet: data.texCoordsBySet,
  };
  if (data.tangents) {
    const tangents = computePrimitiveGeometryDataTangents(transformed);
    if (tangents) {
      if (tangentHandednessScale !== undefined) {
        for (let offset = 3; offset < tangents.length; offset += 4) {
          const handedness = tangents[offset];
          if (handedness !== undefined) {
            tangents[offset] = handedness * tangentHandednessScale;
          }
        }
      }
      transformed.tangents = tangents;
    }
  }
  if (data.colors && data.colorItemSize) {
    transformed.colors = data.colors;
    transformed.colorItemSize = data.colorItemSize;
  }
  return transformed;
}

function createRawMeshGeometry(
  rawMesh: RawMesh,
  transferredAttributes?: TransferredMeshAttributes,
  requiredTexCoords: readonly number[] = [],
  emitTangents = false,
  tangentHandednessScale?: 1 | -1,
): BufferGeometry {
  if (transferredAttributes) {
    return createBufferGeometryFromPrimitiveData(buildAttributeTransferredPrimitiveGeometryData(rawMesh, transferredAttributes, {
      requiredTexCoords,
      emitTangents,
      ...(emitTangents && tangentHandednessScale !== undefined ? { tangentHandednessScale } : {}),
    }));
  }

  return createBufferGeometryFromPrimitiveData(buildIndexedPrimitiveGeometryData(rawMesh));
}

function createBakedGeometry(
  rawMesh: RawMesh,
  atlas: AtlasLayout,
  additionalTextures?: readonly BakedMaterialTexture[],
  transferredAttributes?: TransferredMeshAttributes,
): BufferGeometry {
  return createBufferGeometryFromPrimitiveData(buildAtlasPrimitiveGeometryData(
    rawMesh,
    atlas,
    additionalTextures,
    transferredAttributes,
  ));
}

function createBakedMaterial(
  image: RgbaImage,
  settings: BakedMaterialSettings,
  additionalTextures: readonly BakedMaterialTexture[] = [],
  vertexColors = false,
): MeshStandardMaterial {
  const material = new MeshStandardMaterial({
    name: 'baked-base-color',
    map: imageToTexture(image, true),
    metalness: settings.metallicFactor,
    roughness: settings.roughnessFactor,
    side: settings.doubleSided ? DoubleSide : FrontSide,
    transparent: settings.alphaMode === 'BLEND',
    alphaTest: settings.alphaMode === 'MASK' ? settings.alphaCutoff : 0,
    vertexColors,
  });
  const emissiveFactor = settings.emissiveFactor ?? [0, 0, 0];
  material.emissive.setRGB(emissiveFactor[0], emissiveFactor[1], emissiveFactor[2]);
  material.normalScale.setScalar(settings.normalScale ?? 1);
  material.aoMapIntensity = settings.occlusionStrength ?? 1;
  for (const texture of additionalTextures) {
    if (texture.slot === 'normal') {
      material.normalMap = imageToTexture(texture.image);
    } else if (texture.slot === 'metallicRoughness') {
      const metalnessRoughnessMap = imageToTexture(texture.image);
      material.metalnessMap = metalnessRoughnessMap;
      material.roughnessMap = metalnessRoughnessMap;
    } else if (texture.slot === 'occlusion') {
      material.aoMap = imageToTexture(texture.image);
    } else if (texture.slot === 'emissive') {
      material.emissiveMap = imageToTexture(texture.image, true);
    }
  }
  return material;
}

function createNeutralMaterial(): MeshStandardMaterial {
  return new MeshStandardMaterial({ color: 0xb8c2cc, roughness: 0.8, metalness: 0 });
}

function geometryHasColorAttribute(geometry: BufferGeometry): boolean {
  return geometry.getAttribute('color') !== undefined;
}

function setMaterialVertexColors(material: Material, vertexColors: boolean): void {
  if ('vertexColors' in material) {
    (material as Material & { vertexColors: boolean }).vertexColors = vertexColors;
  }
}

function clonePreservedMaterial(
  sourceMaterial: Material | null | undefined,
  cloneCache: PreservedMaterialCloneCache,
  vertexColors: boolean = false,
): Material {
  if (!sourceMaterial) {
    const material = createNeutralMaterial();
    if (vertexColors) material.vertexColors = true;
    return material;
  }
  const cached = cloneCache.get(sourceMaterial);
  const existing = vertexColors ? cached?.vertexColors : cached?.normal;
  if (existing) return existing;
  const cloned = sourceMaterial.clone();
  setMaterialVertexColors(cloned, vertexColors);
  const entry = cached ?? {};
  if (vertexColors) {
    entry.vertexColors = cloned;
  } else {
    entry.normal = cloned;
  }
  cloneCache.set(sourceMaterial, entry);
  return cloned;
}

function clonePreservedMaterials(
  sourceMaterial: Material | Material[] | null | undefined,
  cloneCache: PreservedMaterialCloneCache,
  vertexColors: boolean = false,
): Material | Material[] {
  if (Array.isArray(sourceMaterial)) {
    return sourceMaterial.map((material) => clonePreservedMaterial(material, cloneCache, vertexColors));
  }
  return clonePreservedMaterial(sourceMaterial, cloneCache, vertexColors);
}

function cloneClonedOutputMaterials(root: Object3D, cloneCache: PreservedMaterialCloneCache): void {
  root.traverse((object) => {
    const mesh = object as Mesh;
    if (!mesh.isMesh || !mesh.material) return;
    mesh.material = clonePreservedMaterials(
      mesh.material,
      cloneCache,
      geometryHasColorAttribute(mesh.geometry as BufferGeometry),
    );
  });
}

function clonePreservedMaterialForGeometry(
  sourceMaterial: Material | null | undefined,
  cloneCache: PreservedMaterialCloneCache,
  geometry: BufferGeometry,
): Material {
  return clonePreservedMaterial(sourceMaterial, cloneCache, geometryHasColorAttribute(geometry));
}

export async function extractBrowserPrimitiveGroupsFromObject(
  root: Object3D,
  options: PrimitiveExtractionOptions,
): Promise<PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>> {
  root.updateMatrixWorld(true);
  const sceneContext = sceneTransformContextFromObject(root);
  const warnings = sceneContext.warning ? [sceneContext.warning] : [];
  const objectOrdinals = objectOrdinalMap(root);
  const rootOrdinal = objectOrdinals.get(root) ?? 0;
  const rootToLocal = root.matrixWorld.clone().invert();
  const materials: SourceMaterial[] = [];
  const materialIds = new Map<Material | 'default', number>();
  let inputVertices = 0;
  let inputFaces = 0;
  const weldVertices = options.weldVertices ?? true;

  interface GroupBuildEntry {
    mode: PrimitiveGroupingMode;
    material: Material | null;
    materialId: number;
    sourceMaterial: SourceMaterial;
    parent: Object3D;
    labelObject: Object3D;
    parentObjectOrdinal: number;
    rawMesh: RawMesh;
    faceAttributes: TexturedRawMesh['faceAttributes'];
    sourceMeshOrdinals: Set<number>;
    appendState: WeldedVertexAppendState;
    sourceVertexIds: Map<string, number>;
    firstMeshOrdinal: number;
    materialGroupIndex: number;
  }

  const groupBuilders = new Map<string, GroupBuildEntry>();

  const getMaterialId = async (material: Material | null): Promise<number> => {
    const key = material ?? 'default';
    const existing = materialIds.get(key);
    if (existing !== undefined) return existing;
    const sourceMaterial = material ? await sourceMaterialFromThree(material, options.mode) : defaultMaterial();
    const id = materials.length;
    materials.push(sourceMaterial);
    materialIds.set(key, id);
    return id;
  };

  const getGroup = (
    material: Material | null,
    materialId: number,
    parent: Object3D,
    parentObjectOrdinal: number,
    meshOrdinal: number,
    materialGroupIndex: number,
    colorSchema: string,
    labelObject = parent,
  ): GroupBuildEntry => {
    const key = options.groupingMode === 'material-parent'
      ? `${parentObjectOrdinal}/${materialId}/${colorSchema}`
      : options.groupingMode === 'material' ? `${materialId}/${colorSchema}` : `${meshOrdinal}/${materialGroupIndex}/${colorSchema}`;
    const existing = groupBuilders.get(key);
    if (existing) return existing;
    const sourceMaterial = materials[materialId] ?? defaultMaterial();
    const group: GroupBuildEntry = {
      mode: options.groupingMode,
      material,
      materialId,
      sourceMaterial,
      parent,
      labelObject,
      parentObjectOrdinal,
      rawMesh: { positions: [], faces: [] },
      faceAttributes: [],
      sourceMeshOrdinals: new Set(),
      appendState: createWeldedVertexAppendState(),
      sourceVertexIds: new Map(),
      firstMeshOrdinal: meshOrdinal,
      materialGroupIndex,
    };
    groupBuilders.set(key, group);
    return group;
  };

  const appendGroupedFace = (
    group: GroupBuildEntry,
    meshOrdinal: number,
    sourceIndices: FaceIndices,
    sourcePositions: readonly Vector3[],
    transform: Matrix4,
    uvSets: SourceFaceUvSet[],
    normalCorners?: [Vector3, Vector3, Vector3],
    tangentCorners?: [Vector4, Vector4, Vector4],
    colorCorners?: SourceFaceColorCorners,
    colorItemSize?: VertexColorItemSize,
    normalMapYScale?: number,
  ): void => {
    group.sourceMeshOrdinals.add(meshOrdinal);
    const face = sourceIndices.map((sourceVertexId) => {
      const sourcePosition = sourcePositions[sourceVertexId];
      if (!sourcePosition) throw new Error(`Face references missing vertex ${sourceVertexId}.`);
      if (!weldVertices) {
        const sourceKey = `${meshOrdinal}/${sourceVertexId}`;
        const existing = group.sourceVertexIds.get(sourceKey);
        if (existing !== undefined) return existing;
        const targetVertexId = appendRawMeshVertex(
          group.rawMesh,
          sourcePosition.clone().applyMatrix4(transform),
          group.appendState,
          { weldVertices },
        );
        group.sourceVertexIds.set(sourceKey, targetVertexId);
        return targetVertexId;
      }
      return appendRawMeshVertex(
        group.rawMesh,
        sourcePosition.clone().applyMatrix4(transform),
        group.appendState,
        { weldVertices },
      );
    }) as FaceIndices;
    group.rawMesh.faces.push(face);
    const transformedNormals = transformFaceNormals(normalCorners, transform);
    const transformedTangents = transformFaceTangents(tangentCorners, transform);
    group.faceAttributes.push({
      materialId: 0,
      uvSets,
      ...(transformedNormals ? { normalCorners: transformedNormals } : {}),
      ...(transformedTangents ? { tangentCorners: transformedTangents } : {}),
      ...(colorCorners && colorItemSize ? { colorCorners: cloneFaceColors(colorCorners), colorItemSize } : {}),
      ...(normalMapYScale !== undefined ? { normalMapYScale } : {}),
    });
  };

  const appendUngroupedFace = (
    group: GroupBuildEntry,
    meshOrdinal: number,
    sourceIndices: FaceIndices,
    sourcePositions: readonly Vector3[],
    uvSets: SourceFaceUvSet[],
    normalCorners?: [Vector3, Vector3, Vector3],
    tangentCorners?: [Vector4, Vector4, Vector4],
    colorCorners?: SourceFaceColorCorners,
    colorItemSize?: VertexColorItemSize,
    normalMapYScale?: number,
  ): void => {
    group.sourceMeshOrdinals.add(meshOrdinal);
    const face = sourceIndices.map((sourceVertexId) => {
      const sourcePosition = sourcePositions[sourceVertexId];
      if (!sourcePosition) throw new Error(`Face references missing vertex ${sourceVertexId}.`);
      if (!weldVertices) {
        const sourceKey = `${meshOrdinal}/${sourceVertexId}`;
        const existing = group.sourceVertexIds.get(sourceKey);
        if (existing !== undefined) return existing;
        const targetVertexId = appendRawMeshVertex(
          group.rawMesh,
          sourcePosition,
          group.appendState,
          { weldVertices },
        );
        group.sourceVertexIds.set(sourceKey, targetVertexId);
        return targetVertexId;
      }
      return appendRawMeshVertex(group.rawMesh, sourcePosition, group.appendState, { weldVertices });
    }) as FaceIndices;
    group.rawMesh.faces.push(face);
    const clonedNormals = cloneFaceNormals(normalCorners);
    const clonedTangents = cloneFaceTangents(tangentCorners);
    group.faceAttributes.push({
      materialId: 0,
      uvSets,
      ...(clonedNormals ? { normalCorners: clonedNormals } : {}),
      ...(clonedTangents ? { tangentCorners: clonedTangents } : {}),
      ...(colorCorners && colorItemSize ? { colorCorners: cloneFaceColors(colorCorners), colorItemSize } : {}),
      ...(normalMapYScale !== undefined ? { normalMapYScale } : {}),
    });
  };

  const meshes: Mesh[] = [];
  root.traverse((object) => {
    if ((object as Mesh).isMesh && (object as Mesh).geometry) meshes.push(object as Mesh);
  });

  for (let meshOrdinal = 0; meshOrdinal < meshes.length; meshOrdinal += 1) {
    const mesh = meshes[meshOrdinal]!;
    const geometry = mesh.geometry as BufferGeometry;
    const position = geometry.getAttribute('position');
    if (!position || position.itemSize !== 3) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} is missing a VEC3 position attribute and was skipped.`);
      continue;
    }
    inputVertices += position.count;
    const normal = geometry.getAttribute('normal');
    const tangent = geometry.getAttribute('tangent');
    const color = geometry.getAttribute('color');
    const uvAttributes = options.mode === 'geometry' ? [] : geometryUvAttributes(geometry);
    const transform = options.groupingMode === 'material' ? rootToLocal.clone().multiply(mesh.matrixWorld) : mesh.matrix.clone();
    const parent = mesh.parent ?? root;
    const parentObjectOrdinal = options.groupingMode === 'material'
      ? rootOrdinal
      : objectOrdinals.get(parent) ?? rootOrdinal;
    const meshPositions: Vector3[] = [];
    for (let i = 0; i < position.count; i += 1) {
      meshPositions.push(new Vector3(position.getX(i), position.getY(i), position.getZ(i)));
    }
    const meshNormals: Vector3[] = [];
    if (normal?.itemSize === 3 && normal.count !== position.count) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has NORMAL count ${normal.count} but POSITION count ${position.count}; source normals were ignored.`);
    } else if (normal?.itemSize === 3) {
      for (let i = 0; i < normal.count; i += 1) {
        meshNormals.push(new Vector3(normal.getX(i), normal.getY(i), normal.getZ(i)).normalize());
      }
    }
    const meshTangents: Vector4[] = [];
    if (tangent?.itemSize === 4 && tangent.count !== position.count) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has TANGENT count ${tangent.count} but POSITION count ${position.count}; source tangents were ignored.`);
    } else if (tangent?.itemSize === 4) {
      for (let i = 0; i < tangent.count; i += 1) {
        const xyz = new Vector3(tangent.getX(i), tangent.getY(i), tangent.getZ(i)).normalize();
        meshTangents.push(new Vector4(xyz.x, xyz.y, xyz.z, tangent.getW(i) < 0 ? -1 : 1));
      }
    }
    const meshColors: Vector4[] = [];
    const colorItemSize = color?.itemSize === 3 || color?.itemSize === 4 ? color.itemSize : undefined;
    if (color && colorItemSize === undefined) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has COLOR_0 item size ${color.itemSize}; source colors were ignored.`);
    } else if (color && color.count !== position.count) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has COLOR_0 count ${color.count} but POSITION count ${position.count}; source colors were ignored.`);
    } else if (color && colorItemSize !== undefined) {
      for (let i = 0; i < color.count; i += 1) {
        meshColors.push(new Vector4(
          color.getX(i),
          color.getY(i),
          color.getZ(i),
          colorItemSize === 4 ? color.getW(i) : 1,
        ));
      }
    }

    const localIndices: number[] = [];
    const index = geometry.getIndex();
    if (index) {
      if (index.count % 3 !== 0) throw new Error(`Indexed geometry has ${index.count} indices, which is not divisible by 3.`);
      for (let i = 0; i < index.count; i += 1) localIndices.push(index.getX(i));
    } else {
      if (position.count % 3 !== 0) throw new Error(`Non-indexed geometry has ${position.count} vertices, which is not divisible by 3.`);
      for (let i = 0; i < position.count; i += 1) localIndices.push(i);
    }

    const materialList = meshMaterials(mesh);
    for (let i = 0; i < localIndices.length; i += 3) {
      inputFaces += 1;
      const a = localIndices[i]!;
      const b = localIndices[i + 1]!;
      const c = localIndices[i + 2]!;
      const materialGroupIndex = groupRangeIndex(geometry, i);
      const material = materialList[groupMaterialIndex(geometry, i)] ?? materialList[0] ?? null;
      const materialId = await getMaterialId(material);
      const normalMapYScale = options.mode === 'geometry' ? undefined : sourceNormalMapYScaleFromThree(material);
      const uvSets = uvAttributes.map(({ texCoord, attribute }) => ({
        texCoord,
        uvs: [
          new Vector2(attribute.getX(a), attribute.getY(a)),
          new Vector2(attribute.getX(b), attribute.getY(b)),
          new Vector2(attribute.getX(c), attribute.getY(c)),
        ] as [Vector2, Vector2, Vector2],
      }));
      const localNormalCorners = meshNormals.length === position.count
        ? [meshNormals[a]!.clone(), meshNormals[b]!.clone(), meshNormals[c]!.clone()] as [Vector3, Vector3, Vector3]
        : undefined;
      const localTangentCorners = meshTangents.length === position.count
        ? [meshTangents[a]!.clone(), meshTangents[b]!.clone(), meshTangents[c]!.clone()] as [Vector4, Vector4, Vector4]
        : undefined;
      const localColorCorners = meshColors.length === position.count
        ? [meshColors[a]!.clone(), meshColors[b]!.clone(), meshColors[c]!.clone()] as SourceFaceColorCorners
        : undefined;
      const colorSchema = colorSchemaKey(localColorCorners, colorItemSize);
      const group = getGroup(
        material,
        materialId,
        options.groupingMode === 'material' ? root : parent,
        parentObjectOrdinal,
        meshOrdinal,
        materialGroupIndex,
        colorSchema,
        options.groupingMode === 'none' ? mesh : options.groupingMode === 'material' ? root : parent,
      );
      if (options.groupingMode === 'none') {
        appendUngroupedFace(
          group,
          meshOrdinal,
          [a, b, c],
          meshPositions,
          uvSets,
          localNormalCorners,
          localTangentCorners,
          localColorCorners,
          colorItemSize,
          normalMapYScale,
        );
      } else {
        appendGroupedFace(
          group,
          meshOrdinal,
          [a, b, c],
          meshPositions,
          transform,
          uvSets,
          localNormalCorners,
          localTangentCorners,
          localColorCorners,
          colorItemSize,
          normalMapYScale,
        );
      }
    }
  }

  if (inputFaces === 0) throw new Error('No triangle geometry found in the loaded model.');

  const applyMetadataByEntryId = new Map<string, BrowserPrimitiveApplyMetadata>();
  const entries = Array.from(groupBuilders.values()).map((group, index) => {
    const sourceMeshOrdinals = Array.from(group.sourceMeshOrdinals);
    const materialKey = String(group.materialId);
    const id = group.mode === 'material-parent'
      ? `${group.parentObjectOrdinal}/${materialKey}/${sourceMeshOrdinals.join(',')}`
      : group.mode === 'material'
        ? `${materialKey}/${sourceMeshOrdinals.join(',')}`
        : `${group.firstMeshOrdinal}/${group.materialGroupIndex}`;
    const label = groupedEntryLabel(group.mode, group.labelObject, group.material, index);
    const textured = { rawMesh: group.rawMesh, faceAttributes: group.faceAttributes, materials: [group.sourceMaterial] };
    const processable = toProcessablePrimitiveEntry({
      id,
      label,
      rawMesh: group.rawMesh,
      texturedRawMesh: textured,
    });
    applyMetadataByEntryId.set(id, {
      meshOrdinal: group.firstMeshOrdinal,
      sourceMeshOrdinals,
      parentObjectOrdinal: group.parentObjectOrdinal,
      sourceMaterial: group.material,
      ...(group.mode === 'none' ? { preserveSourceMeshTransform: true } : {}),
    });
    return processable;
  });

  const metadataBakeableEntries = entries.filter((entry) => (
    entry.texturedRawMesh ? hasEntryImageBackedTextureBakeData(entry.texturedRawMesh) : false
  ));
  const imageBackedTextureTransferEntries = entries.filter((entry) => (
    entry.texturedRawMesh ? hasEntryImageBackedTextureTransferData(entry.texturedRawMesh) : false
  ));

  return createPrimitiveExtractionResult({
    entries,
    applyMetadataByEntryId,
    extractionApplyState: options,
    summary: {
      inputVertices,
      inputFaces,
      bakeableEntryCount: metadataBakeableEntries.length,
      hasTransferableTextureData: imageBackedTextureTransferEntries.length > 0,
      hasPreservableMaterialData: entries.some((entry) => (
        entry.hasPreservableMaterialData === true
        || (entry.texturedRawMesh ? hasEntryPreservableMaterialData(entry.texturedRawMesh) : false)
      )),
      hasImageBackedTextureTransferData: imageBackedTextureTransferEntries.length > 0,
      hasImageBackedTextureBakeData: metadataBakeableEntries.length > 0,
      hasTransferableVertexAttributes: entries.some((entry) => entry.requiresAttributeTransfer === true),
      warnings,
    },
  });
}

// Compatibility helper for tests during the lazy-asset migration. Browser load must not call this.
export async function extractBrowserModelFromObject(
  root: Object3D,
  options: { weldVertices?: boolean } = {},
): Promise<ExtractedBrowserModel> {
  const weldVertices = options.weldVertices ?? true;
  const materialParent = await extractBrowserPrimitiveGroupsFromObject(root, { groupingMode: 'material-parent', mode: 'bake', weldVertices });
  const material = await extractBrowserPrimitiveGroupsFromObject(root, { groupingMode: 'material', mode: 'bake', weldVertices });
  const none = await extractBrowserPrimitiveGroupsFromObject(root, { groupingMode: 'none', mode: 'bake', weldVertices });
  root.updateMatrixWorld(true);
  const sceneContext = sceneTransformContextFromObject(root);
  const warnings = sceneContext.warning ? [sceneContext.warning] : [];
  const objectOrdinals = objectOrdinalMap(root);
  const rootOrdinal = objectOrdinals.get(root) ?? 0;
  const rootToLocal = root.matrixWorld.clone().invert();
  const rawMesh: RawMesh = { positions: [], faces: [] };
  const faceAttributes: TexturedRawMesh['faceAttributes'] = [];
  const materials: SourceMaterial[] = [];
  const materialIds = new Map<Material | 'default', number>();

  interface GroupBuildEntry {
    mode: PrimitiveGroupingMode;
    material: Material | null;
    materialId: number;
    sourceMaterial: SourceMaterial;
    parent: Object3D;
    labelObject: Object3D;
    parentObjectOrdinal: number;
    rawMesh: RawMesh;
    faceAttributes: TexturedRawMesh['faceAttributes'];
    sourceMeshOrdinals: Set<number>;
    appendState: WeldedVertexAppendState;
    sourceVertexIds: Map<string, number>;
    firstMeshOrdinal: number;
  }

  const groupBuilders: Record<PrimitiveGroupingMode, Map<string, GroupBuildEntry>> = {
    'material-parent': new Map(),
    material: new Map(),
    none: new Map(),
  };

  const getMaterialId = async (material: Material | null): Promise<number> => {
    const key = material ?? 'default';
    const existing = materialIds.get(key);
    if (existing !== undefined) return existing;
    const sourceMaterial = material ? await sourceMaterialFromThree(material, 'bake') : defaultMaterial();
    const id = materials.length;
    materials.push(sourceMaterial);
    materialIds.set(key, id);
    return id;
  };

  const getGroup = (
    mode: PrimitiveGroupingMode,
    material: Material | null,
    materialId: number,
    parent: Object3D,
    parentObjectOrdinal: number,
    meshOrdinal: number,
    materialGroupIndex: number,
    colorSchema: string,
    labelObject = parent,
  ): GroupBuildEntry => {
    const key = mode === 'material-parent'
      ? `${parentObjectOrdinal}/${materialId}/${colorSchema}`
      : mode === 'material' ? `${materialId}/${colorSchema}` : `${meshOrdinal}/${materialGroupIndex}/${colorSchema}`;
    const groups = groupBuilders[mode];
    const existing = groups.get(key);
    if (existing) return existing;
    const sourceMaterial = materials[materialId] ?? defaultMaterial();
    const group: GroupBuildEntry = {
      mode,
      material,
      materialId,
      sourceMaterial,
      parent,
      labelObject,
      parentObjectOrdinal,
      rawMesh: { positions: [], faces: [] },
      faceAttributes: [],
      sourceMeshOrdinals: new Set(),
      appendState: createWeldedVertexAppendState(),
      sourceVertexIds: new Map(),
      firstMeshOrdinal: meshOrdinal,
    };
    groups.set(key, group);
    return group;
  };

  const appendGroupedFace = (
    group: GroupBuildEntry,
    meshOrdinal: number,
    sourceIndices: FaceIndices,
    sourcePositions: readonly Vector3[],
    transform: Matrix4,
    uvSets: SourceFaceUvSet[],
    normalCorners?: [Vector3, Vector3, Vector3],
    tangentCorners?: [Vector4, Vector4, Vector4],
    colorCorners?: SourceFaceColorCorners,
    colorItemSize?: VertexColorItemSize,
    normalMapYScale?: number,
  ): void => {
    group.sourceMeshOrdinals.add(meshOrdinal);
    const face = sourceIndices.map((sourceVertexId) => {
      const sourcePosition = sourcePositions[sourceVertexId];
      if (!sourcePosition) throw new Error(`Face references missing vertex ${sourceVertexId}.`);
      if (!weldVertices) {
        const sourceKey = `${meshOrdinal}/${sourceVertexId}`;
        const existing = group.sourceVertexIds.get(sourceKey);
        if (existing !== undefined) return existing;
        const targetVertexId = appendRawMeshVertex(
          group.rawMesh,
          sourcePosition.clone().applyMatrix4(transform),
          group.appendState,
          { weldVertices },
        );
        group.sourceVertexIds.set(sourceKey, targetVertexId);
        return targetVertexId;
      }
      return appendRawMeshVertex(
        group.rawMesh,
        sourcePosition.clone().applyMatrix4(transform),
        group.appendState,
        { weldVertices },
      );
    }) as FaceIndices;
    group.rawMesh.faces.push(face);
    const transformedNormals = transformFaceNormals(normalCorners, transform);
    const transformedTangents = transformFaceTangents(tangentCorners, transform);
    group.faceAttributes.push({
      materialId: 0,
      uvSets,
      ...(transformedNormals ? { normalCorners: transformedNormals } : {}),
      ...(transformedTangents ? { tangentCorners: transformedTangents } : {}),
      ...(colorCorners && colorItemSize ? { colorCorners: cloneFaceColors(colorCorners), colorItemSize } : {}),
      ...(normalMapYScale !== undefined ? { normalMapYScale } : {}),
    });
  };

  const appendUngroupedFace = (
    group: GroupBuildEntry,
    meshOrdinal: number,
    sourceIndices: FaceIndices,
    sourcePositions: readonly Vector3[],
    uvSets: SourceFaceUvSet[],
    normalCorners?: [Vector3, Vector3, Vector3],
    tangentCorners?: [Vector4, Vector4, Vector4],
    colorCorners?: SourceFaceColorCorners,
    colorItemSize?: VertexColorItemSize,
    normalMapYScale?: number,
  ): void => {
    group.sourceMeshOrdinals.add(meshOrdinal);
    const face = sourceIndices.map((sourceVertexId) => {
      const sourcePosition = sourcePositions[sourceVertexId];
      if (!sourcePosition) throw new Error(`Face references missing vertex ${sourceVertexId}.`);
      if (!weldVertices) {
        const sourceKey = `${meshOrdinal}/${sourceVertexId}`;
        const existing = group.sourceVertexIds.get(sourceKey);
        if (existing !== undefined) return existing;
        const targetVertexId = appendRawMeshVertex(
          group.rawMesh,
          sourcePosition,
          group.appendState,
          { weldVertices },
        );
        group.sourceVertexIds.set(sourceKey, targetVertexId);
        return targetVertexId;
      }
      return appendRawMeshVertex(group.rawMesh, sourcePosition, group.appendState, { weldVertices });
    }) as FaceIndices;
    group.rawMesh.faces.push(face);
    const clonedNormals = cloneFaceNormals(normalCorners);
    const clonedTangents = cloneFaceTangents(tangentCorners);
    group.faceAttributes.push({
      materialId: 0,
      uvSets,
      ...(clonedNormals ? { normalCorners: clonedNormals } : {}),
      ...(clonedTangents ? { tangentCorners: clonedTangents } : {}),
      ...(colorCorners && colorItemSize ? { colorCorners: cloneFaceColors(colorCorners), colorItemSize } : {}),
      ...(normalMapYScale !== undefined ? { normalMapYScale } : {}),
    });
  };

  const meshes: Mesh[] = [];
  root.traverse((object) => {
    if ((object as Mesh).isMesh && (object as Mesh).geometry) meshes.push(object as Mesh);
  });

  for (let meshOrdinal = 0; meshOrdinal < meshes.length; meshOrdinal += 1) {
    const mesh = meshes[meshOrdinal]!;
    const geometry = mesh.geometry as BufferGeometry;
    const position = geometry.getAttribute('position');
    if (!position || position.itemSize !== 3) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} is missing a VEC3 position attribute and was skipped.`);
      continue;
    }
    const normal = geometry.getAttribute('normal');
    const tangent = geometry.getAttribute('tangent');
    const color = geometry.getAttribute('color');
    const uvAttributes = geometryUvAttributes(geometry);
    const baseVertex = rawMesh.positions.length;
    const worldMatrix = mesh.matrixWorld;
    const parent = mesh.parent ?? root;
    const parentObjectOrdinal = objectOrdinals.get(parent) ?? rootOrdinal;
    const parentLocalMatrix = mesh.matrix.clone();
    const rootLocalMatrix = rootToLocal.clone().multiply(mesh.matrixWorld);
    const localPosition = new Vector3();
    const meshPositions: Vector3[] = [];
    for (let i = 0; i < position.count; i += 1) {
      const meshPosition = new Vector3(position.getX(i), position.getY(i), position.getZ(i));
      meshPositions.push(meshPosition);
      localPosition.copy(meshPosition).applyMatrix4(worldMatrix);
      rawMesh.positions.push(localPosition.clone());
    }
    const meshNormals: Vector3[] = [];
    if (normal?.itemSize === 3 && normal.count !== position.count) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has NORMAL count ${normal.count} but POSITION count ${position.count}; source normals were ignored.`);
    } else if (normal?.itemSize === 3) {
      for (let i = 0; i < normal.count; i += 1) {
        meshNormals.push(new Vector3(normal.getX(i), normal.getY(i), normal.getZ(i)).normalize());
      }
    }
    const meshTangents: Vector4[] = [];
    if (tangent?.itemSize === 4 && tangent.count !== position.count) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has TANGENT count ${tangent.count} but POSITION count ${position.count}; source tangents were ignored.`);
    } else if (tangent?.itemSize === 4) {
      for (let i = 0; i < tangent.count; i += 1) {
        const xyz = new Vector3(tangent.getX(i), tangent.getY(i), tangent.getZ(i)).normalize();
        meshTangents.push(new Vector4(xyz.x, xyz.y, xyz.z, tangent.getW(i) < 0 ? -1 : 1));
      }
    }
    const meshColors: Vector4[] = [];
    const colorItemSize = color?.itemSize === 3 || color?.itemSize === 4 ? color.itemSize : undefined;
    if (color && colorItemSize === undefined) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has COLOR_0 item size ${color.itemSize}; source colors were ignored.`);
    } else if (color && color.count !== position.count) {
      warnings.push(`Mesh ${mesh.name || '<unnamed>'} has COLOR_0 count ${color.count} but POSITION count ${position.count}; source colors were ignored.`);
    } else if (color && colorItemSize !== undefined) {
      for (let i = 0; i < color.count; i += 1) {
        meshColors.push(new Vector4(
          color.getX(i),
          color.getY(i),
          color.getZ(i),
          colorItemSize === 4 ? color.getW(i) : 1,
        ));
      }
    }

    const localIndices: number[] = [];
    const index = geometry.getIndex();
    if (index) {
      if (index.count % 3 !== 0) throw new Error(`Indexed geometry has ${index.count} indices, which is not divisible by 3.`);
      for (let i = 0; i < index.count; i += 1) localIndices.push(index.getX(i));
    } else {
      if (position.count % 3 !== 0) throw new Error(`Non-indexed geometry has ${position.count} vertices, which is not divisible by 3.`);
      for (let i = 0; i < position.count; i += 1) localIndices.push(i);
    }

    const materialList = meshMaterials(mesh);
    for (let i = 0; i < localIndices.length; i += 3) {
      const a = localIndices[i]!;
      const b = localIndices[i + 1]!;
      const c = localIndices[i + 2]!;
      rawMesh.faces.push([baseVertex + a, baseVertex + b, baseVertex + c]);
      const materialGroupIndex = groupRangeIndex(geometry, i);
      const material = materialList[groupMaterialIndex(geometry, i)] ?? materialList[0] ?? null;
      const materialId = await getMaterialId(material);
      const normalMapYScale = sourceNormalMapYScaleFromThree(material);
      const uvSets = uvAttributes.map(({ texCoord, attribute }) => ({
        texCoord,
        uvs: [
          new Vector2(attribute.getX(a), attribute.getY(a)),
          new Vector2(attribute.getX(b), attribute.getY(b)),
          new Vector2(attribute.getX(c), attribute.getY(c)),
        ] as [Vector2, Vector2, Vector2],
      }));
      const localNormalCorners = meshNormals.length === position.count
        ? [meshNormals[a]!.clone(), meshNormals[b]!.clone(), meshNormals[c]!.clone()] as [Vector3, Vector3, Vector3]
        : undefined;
      const localTangentCorners = meshTangents.length === position.count
        ? [meshTangents[a]!.clone(), meshTangents[b]!.clone(), meshTangents[c]!.clone()] as [Vector4, Vector4, Vector4]
        : undefined;
      const localColorCorners = meshColors.length === position.count
        ? [meshColors[a]!.clone(), meshColors[b]!.clone(), meshColors[c]!.clone()] as SourceFaceColorCorners
        : undefined;
      const colorSchema = colorSchemaKey(localColorCorners, colorItemSize);
      const normalCorners = transformFaceNormals(localNormalCorners, worldMatrix);
      const tangentCorners = transformFaceTangents(localTangentCorners, worldMatrix);
      faceAttributes.push({
        materialId,
        uvSets,
        ...(normalCorners ? { normalCorners } : {}),
        ...(tangentCorners ? { tangentCorners } : {}),
        ...(localColorCorners && colorItemSize ? { colorCorners: cloneFaceColors(localColorCorners), colorItemSize } : {}),
        ...(normalMapYScale !== undefined ? { normalMapYScale } : {}),
      });
      appendGroupedFace(
        getGroup('material-parent', material, materialId, parent, parentObjectOrdinal, meshOrdinal, materialGroupIndex, colorSchema),
        meshOrdinal,
        [a, b, c],
        meshPositions,
        parentLocalMatrix,
        uvSets,
        localNormalCorners,
        localTangentCorners,
        localColorCorners,
        colorItemSize,
        normalMapYScale,
      );
      appendGroupedFace(
        getGroup('material', material, materialId, root, rootOrdinal, meshOrdinal, materialGroupIndex, colorSchema),
        meshOrdinal,
        [a, b, c],
        meshPositions,
        rootLocalMatrix,
        uvSets,
        localNormalCorners,
        localTangentCorners,
        localColorCorners,
        colorItemSize,
        normalMapYScale,
      );
      appendUngroupedFace(
        getGroup('none', material, materialId, parent, parentObjectOrdinal, meshOrdinal, materialGroupIndex, colorSchema, mesh),
        meshOrdinal,
        [a, b, c],
        meshPositions,
        uvSets,
        localNormalCorners,
        localTangentCorners,
        localColorCorners,
        colorItemSize,
        normalMapYScale,
      );
    }
  }

  if (rawMesh.positions.length === 0 || rawMesh.faces.length === 0) throw new Error('No triangle geometry found in the loaded model.');

  const buildEntries = (extraction: PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>): BrowserPrimitiveEntry[] => (
    extraction.entries.map((entry) => {
      const metadata = extraction.applyMetadataByEntryId.get(entry.id);
      if (!metadata) throw new Error(`Missing browser apply metadata for primitive entry "${entry.id}".`);
      return {
        id: entry.id,
        label: entry.label ?? entry.id,
        meshOrdinal: metadata.meshOrdinal,
        sourceMeshOrdinals: metadata.sourceMeshOrdinals,
        parentObjectOrdinal: metadata.parentObjectOrdinal ?? 0,
        sourceMaterial: metadata.sourceMaterial,
        rawMesh: entry.rawMesh,
        texturedRawMesh: entry.texturedRawMesh ?? { rawMesh: entry.rawMesh, faceAttributes: [], materials: [] },
        bakeable: entry.bakeable,
        ...(entry.hasPreservableMaterialData === true ? { hasPreservableMaterialData: true } : {}),
        hasTexturedMaterial: entry.hasTexturedMaterial === true,
        ...(metadata.preserveSourceMeshTransform === true ? { preserveSourceMeshTransform: true } : {}),
        ...(entry.requiresAttributeTransfer === true ? { requiresAttributeTransfer: true } : {}),
      };
    })
  );
  const primitiveEntryGroups: Record<PrimitiveGroupingMode, BrowserPrimitiveEntry[]> = {
    'material-parent': buildEntries(materialParent),
    material: buildEntries(material),
    none: buildEntries(none),
  };
  return {
    rawMesh,
    textured: { rawMesh, faceAttributes, materials },
    primitiveEntries: primitiveEntryGroups['material-parent'],
    primitiveEntryGroups,
    sceneContext,
    warnings,
  };
}

function primitiveReplacementFromProcessedEntry(
  extraction: PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>,
  result: ProcessedPrimitiveEntry,
): PrimitiveOutputReplacement {
  const metadata = extraction.applyMetadataByEntryId.get(result.id);
  if (!metadata) throw new Error(`Missing browser apply metadata for primitive entry "${result.id}".`);
  const common = {
    id: result.id,
    meshOrdinal: metadata.meshOrdinal,
    sourceMeshOrdinals: metadata.sourceMeshOrdinals,
    sourceMaterial: metadata.sourceMaterial,
    ...(metadata.parentObjectOrdinal !== undefined ? { parentObjectOrdinal: metadata.parentObjectOrdinal } : {}),
    ...(metadata.preserveSourceMeshTransform === true ? { preserveSourceMeshTransform: true } : {}),
    rawMesh: result.raw.rawMesh,
    ...(result.transferredAttributes ? { transferredAttributes: result.transferredAttributes } : {}),
  };
  if (result.baked) {
    return {
      ...common,
      materialMode: 'baked',
      atlas: result.baked.baked.atlas,
      image: result.baked.baked.image,
      additionalTextures: result.baked.baked.additionalTextures,
      materialSettings: result.baked.materialSettings,
    };
  }
  const geometryModeWithoutTransferredUvs = extraction.extractionApplyState.mode === 'geometry'
    && result.transferredAttributes === undefined;
  return {
    ...common,
    materialMode: geometryModeWithoutTransferredUvs ? 'neutral' : 'preserve',
  };
}

export async function parseGlbArrayBuffer(buffer: ArrayBuffer, options: ParseGlbArrayBufferOptions = {}): Promise<BrowserLoadedAsset> {
  const resources = createExternalGltfResources(options.externalResourceFiles ?? []);
  const matchedExternalTextureFiles = new Set<string>();
  const matchedExternalBinaryBufferFiles = new Set<string>();
  const manager = resources.length > 0 ? new LoadingManager() : undefined;
  manager?.setURLModifier((url: string) => {
    const matched = matchExternalGltfResource(url, resources);
    if (!matched) return url;
    if (matched.kind === 'texture-image') {
      matchedExternalTextureFiles.add(matched.fileName);
    } else {
      matchedExternalBinaryBufferFiles.add(matched.fileName);
    }
    return matched.objectUrl;
  });
  const loader = new GLTFLoader(manager);
  try {
    const gltf = await new Promise<{ scene: Object3D }>((resolve, reject) => {
      loader.parse(buffer, '', (result) => resolve(result as { scene: Object3D }), (error) => reject(error));
    });
    normalizeBrowserSceneMaterialsToCorePbr(gltf.scene);
    const summary = await summarizeBrowserObject(gltf.scene);
    const assetRevision = 1;
    const asset: BrowserLoadedAsset = {
      assetRevision,
      scene: gltf.scene,
      summary,
      warnings: summary.warnings,
      matchedExternalTextureFiles: Array.from(matchedExternalTextureFiles),
      matchedExternalBinaryBufferFiles: Array.from(matchedExternalBinaryBufferFiles),
      summarize: async () => summary,
      extractGroups: async (extractOptions) => await extractBrowserPrimitiveGroupsFromObject(gltf.scene, extractOptions),
      applyResults: async (extraction, results) => {
        return createPrimitiveOutputScene(
          gltf.scene,
          results.map((result) => primitiveReplacementFromProcessedEntry(extraction, result)),
        );
      },
      buildOutputScene: async (_extraction, replacements) => createPrimitiveOutputScene(gltf.scene, replacements),
      disposeIntermediateData: () => {},
    };
    return asset;
  } finally {
    for (const resource of resources) URL.revokeObjectURL(resource.objectUrl);
  }
}

export async function parseLoadedBrowserModelArrayBuffer(
  buffer: ArrayBuffer,
  options: ParseGlbArrayBufferOptions = {},
): Promise<LoadedBrowserModel> {
  const asset = await parseGlbArrayBuffer(buffer, options);
  const extracted = await extractBrowserModelFromObject(asset.scene);
  return {
    ...extracted,
    scene: asset.scene,
    ...(asset.matchedExternalTextureFiles !== undefined ? { matchedExternalTextureFiles: asset.matchedExternalTextureFiles } : {}),
    ...(asset.matchedExternalBinaryBufferFiles !== undefined ? { matchedExternalBinaryBufferFiles: asset.matchedExternalBinaryBufferFiles } : {}),
  };
}

export function createGeometryOutputScene(rawMesh: RawMesh, sceneContext?: BrowserSceneTransformContext): Scene {
  const toLocal = attachmentLocalMatrix(sceneContext);
  const normalMatrix = new Matrix3().getNormalMatrix(toLocal);
  const geometry = createBufferGeometryFromPrimitiveData(transformPrimitiveGeometryData(
    buildIndexedPrimitiveGeometryData(rawMesh),
    toLocal,
    normalMatrix,
  ));
  const mesh = new Mesh(geometry, new MeshStandardMaterial({ color: 0xb8c2cc, roughness: 0.8, metalness: 0 }));
  return createSceneWithPreservedTransform(mesh, sceneContext, 'simplified-geometry');
}

function imageToTexture(image: RgbaImage, colorTexture = false): DataTexture {
  const bytes = new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);
  const texture = new DataTexture(bytes, image.width, image.height, RGBAFormat);
  if (colorTexture) texture.colorSpace = SRGBColorSpace;
  texture.flipY = false;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

export function createTexturedOutputScene(options: TexturedOutputSceneOptions): Scene {
  const toLocal = attachmentLocalMatrix(options.sceneContext);
  const normalMatrix = new Matrix3().getNormalMatrix(toLocal);
  const geometry = createBufferGeometryFromPrimitiveData(transformPrimitiveGeometryData(
    buildAtlasPrimitiveGeometryData(
      options.rawMesh,
      options.atlas,
      options.additionalTextures,
      options.transferredAttributes,
    ),
    toLocal,
    normalMatrix,
  ));
  const material = createBakedMaterial(
    options.image,
    options.materialSettings,
    options.additionalTextures,
    Boolean(geometry.getAttribute('color')),
  );
  const mesh = new Mesh(geometry, material);
  return createSceneWithPreservedTransform(mesh, options.sceneContext, 'simplified-textured-geometry');
}

export function createPrimitiveOutputScene(source: Object3D, replacements: readonly PrimitiveOutputReplacement[]): Object3D {
  const materialCloneCache: PreservedMaterialCloneCache = new Map();
  const output = source.clone(true);
  cloneClonedOutputMaterials(output, materialCloneCache);
  const meshes: Mesh[] = [];
  const objects: Object3D[] = [];
  output.traverse((object) => {
    objects.push(object);
    if ((object as Mesh).isMesh && (object as Mesh).geometry) meshes.push(object as Mesh);
  });
  const hasGroupedReplacement = replacements.some((replacement) => replacement.sourceMeshOrdinals !== undefined || replacement.parentObjectOrdinal !== undefined);
  if (!hasGroupedReplacement) {
    const byOrdinal = new Map(replacements.map((replacement) => [replacement.meshOrdinal, replacement]));
    for (let meshOrdinal = 0; meshOrdinal < meshes.length; meshOrdinal += 1) {
      const replacement = byOrdinal.get(meshOrdinal);
      if (!replacement) continue;
      const mesh = meshes[meshOrdinal]!;
      if (replacement.materialMode === 'baked') {
        mesh.geometry = createBakedGeometry(
          replacement.rawMesh,
          replacement.atlas,
          replacement.additionalTextures,
          replacement.transferredAttributes,
        );
        mesh.material = createBakedMaterial(
          replacement.image,
          replacement.materialSettings,
          replacement.additionalTextures,
          Boolean((mesh.geometry as BufferGeometry).getAttribute('color')),
        );
      } else {
        const existingMaterial = Array.isArray(mesh.material) ? (mesh.material[0] ?? null) : mesh.material;
        const sourceMaterial = replacement.sourceMaterial ?? existingMaterial;
        mesh.geometry = createRawMeshGeometry(
          replacement.rawMesh,
          replacement.transferredAttributes,
          requiredTextureTexCoords(sourceMaterial),
          materialReferencesNormalMap(sourceMaterial) || replacement.transferredAttributes?.hasSourceTangents === true,
          -1,
        );
        if (replacement.materialMode === 'neutral') {
          mesh.material = createNeutralMaterial();
        } else {
          mesh.material = clonePreservedMaterialForGeometry(sourceMaterial, materialCloneCache, mesh.geometry as BufferGeometry);
        }
      }
    }
    output.updateMatrixWorld(true);
    return output;
  }

  const removeOrdinals = new Set<number>();
  for (const replacement of replacements) {
    for (const ordinal of replacement.sourceMeshOrdinals ?? [replacement.meshOrdinal]) removeOrdinals.add(ordinal);
  }
  Array.from(removeOrdinals)
    .sort((a, b) => b - a)
    .forEach((meshOrdinal) => {
      const mesh = meshes[meshOrdinal];
      if (mesh?.parent) mesh.parent.remove(mesh);
    });

  for (const replacement of replacements) {
    const parent = replacement.parentObjectOrdinal !== undefined
      ? objects[replacement.parentObjectOrdinal] ?? output
      : output;
    const sourceMeshOrdinal = replacement.sourceMeshOrdinals?.length === 1 ? replacement.sourceMeshOrdinals[0] : undefined;
    const sourceMesh = sourceMeshOrdinal !== undefined ? meshes[sourceMeshOrdinal] : undefined;
    const mesh = new Mesh();
    const sourceName = sourceMesh?.name.trim() ?? '';
    mesh.name = sourceName.length > 0 ? `simplified-${sourceName}-${replacement.id}` : `simplified-${replacement.id}`;
    if (sourceMesh && replacement.preserveSourceMeshTransform === true) {
      mesh.position.copy(sourceMesh.position);
      mesh.quaternion.copy(sourceMesh.quaternion);
      mesh.scale.copy(sourceMesh.scale);
    }
    if (replacement.materialMode === 'baked') {
      mesh.geometry = createBakedGeometry(
        replacement.rawMesh,
        replacement.atlas,
        replacement.additionalTextures,
        replacement.transferredAttributes,
      );
      mesh.material = createBakedMaterial(
        replacement.image,
        replacement.materialSettings,
        replacement.additionalTextures,
        Boolean((mesh.geometry as BufferGeometry).getAttribute('color')),
      );
    } else {
      mesh.geometry = createRawMeshGeometry(
        replacement.rawMesh,
        replacement.transferredAttributes,
        requiredTextureTexCoords(replacement.sourceMaterial),
        materialReferencesNormalMap(replacement.sourceMaterial) || replacement.transferredAttributes?.hasSourceTangents === true,
        -1,
      );
      mesh.material = replacement.materialMode === 'neutral'
        ? createNeutralMaterial()
        : clonePreservedMaterialForGeometry(replacement.sourceMaterial, materialCloneCache, mesh.geometry as BufferGeometry);
    }
    parent.add(mesh);
  }
  output.updateMatrixWorld(true);
  return output;
}

export async function exportSceneToGlb(scene: Object3D): Promise<ArrayBuffer> {
  const exporter = new GLTFExporter();
  const result = await exporter.parseAsync(scene, { binary: true, embedImages: true, forceIndices: true });
  if (result instanceof ArrayBuffer) return result;
  throw new Error('GLTFExporter returned JSON despite binary export mode.');
}
