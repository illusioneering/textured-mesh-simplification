import type { Vector2 } from 'three';
import type { SourceFaceAttributes } from '../simplification/attributes';
import type { RawMesh } from '../simplification/types';

export type { Barycentric } from '../simplification/types';
export type {
  SourceFaceColorCorners,
  SourceFaceAttributes,
  SourceFaceUvSet,
  TransferredMeshAttributes,
  TransferredVertexAttributes,
  TransferredVertexUvSet,
  VertexColorItemSize,
} from '../simplification/attributes';
export { faceUvSet, transferredVertexColor, transferredVertexUv } from '../simplification/attributes';

export type Rgba = [number, number, number, number];

export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

export type WrapMode = 'clamp' | 'repeat' | 'mirrored-repeat';

export interface TextureSampler {
  wrapS: WrapMode;
  wrapT: WrapMode;
  filter: 'nearest' | 'linear';
}

export type SourceMaterialTextureSlot =
  | 'baseColor'
  | 'normal'
  | 'metallicRoughness'
  | 'occlusion'
  | 'emissive';

export type StandardBakedTextureSlot =
  | 'normal'
  | 'metallicRoughness'
  | 'occlusion'
  | 'emissive';

export interface SourceTexture {
  image?: RgbaImage;
  sampler: TextureSampler;
  texCoord: number;
  name?: string;
  mimeType?: string;
}

export interface SourceMaterialTextureInfo {
  slot: SourceMaterialTextureSlot;
  texCoord: number;
  sampler: TextureSampler;
  hasImage: boolean;
  image?: RgbaImage;
  name?: string;
  mimeType?: string;
}

export interface SourceMaterial {
  name: string;
  baseColorFactor: [number, number, number, number];
  baseColorTexture?: SourceTexture;
  textureSlots: SourceMaterialTextureInfo[];
  alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff: number;
  doubleSided: boolean;
  emissiveFactor: [number, number, number];
  metallicFactor: number;
  roughnessFactor: number;
  normalScale: number;
  occlusionStrength: number;
}

export function hasMaterialTextures(material: SourceMaterial): boolean {
  return Boolean(material.baseColorTexture?.image) || material.textureSlots.some((slot) => slot.hasImage);
}

export interface TexturedRawMesh {
  rawMesh: RawMesh;
  faceAttributes: SourceFaceAttributes[];
  materials: SourceMaterial[];
}

export interface AtlasLayout {
  textureSize: number;
  padding: number;
  faceUvs: [Vector2, Vector2, Vector2][];
  facePixelTriangles: [[number, number], [number, number], [number, number]][];
  islandCount?: number;
}

export interface BakedTextureResult {
  image: RgbaImage;
  additionalTextures: BakedMaterialTexture[];
  atlas: AtlasLayout;
  stats: {
    filledPixels: number;
    mappedPixels: number;
    unmappedPixels: number;
  };
}

export interface BakedMaterialTexture {
  slot: StandardBakedTextureSlot;
  image: RgbaImage;
}

export type BakeTextureProgress =
  | { stage: 'atlas-created'; totalFaces: number; islandCount?: number }
  | {
      stage: 'resampling';
      completedBatches: number;
      totalBatches: number;
      processedFaces: number;
      totalFaces: number;
      processedSamples: number;
      totalSamples: number;
      mappedPixels: number;
      unmappedPixels: number;
    }
  | { stage: 'dilating'; gutterPass: number; gutterPasses: number }
  | { stage: 'complete'; filledPixels: number; mappedPixels: number; unmappedPixels: number };
