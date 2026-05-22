import type { Material, Texture } from 'three';
import type { SourceMaterialTextureSlot } from '../texture/types';

export type BrowserTextureSlotMaterial = Material & {
  map?: Texture | null;
  normalMap?: Texture | null;
  metalnessMap?: Texture | null;
  roughnessMap?: Texture | null;
  aoMap?: Texture | null;
  emissiveMap?: Texture | null;
} & Record<string, unknown>;

export interface BrowserMaterialTextureSlot {
  slot: SourceMaterialTextureSlot;
  texture: Texture | null | undefined;
}

export function browserTextureChannel(texture: Texture | null | undefined): number {
  return texture?.channel ?? 0;
}

export function browserTextureHasImageSource(texture: Texture | null | undefined): boolean {
  if (!texture) return false;
  const source = (texture as Texture & { source?: { data?: unknown } }).source?.data ?? texture.image;
  if (!source) return false;
  if (typeof ImageData !== 'undefined' && source instanceof ImageData) return true;
  if (typeof source === 'object' && 'data' in source && 'width' in source && 'height' in source) {
    const candidate = source as { data?: unknown; width?: unknown; height?: unknown };
    return Boolean(candidate.data && candidate.width && candidate.height);
  }
  if (typeof source === 'object' && 'width' in source && 'height' in source) {
    const candidate = source as { width?: unknown; height?: unknown };
    return Boolean(candidate.width && candidate.height);
  }
  return false;
}

export function browserTextureDimensions(texture: Texture | null | undefined): string | null {
  if (!browserTextureHasImageSource(texture)) return null;
  const source = ((texture as Texture & { source?: { data?: unknown } }).source?.data ?? texture?.image) as
    | { width?: number; height?: number }
    | undefined;
  if (!source?.width || !source.height) return null;
  return `${source.width}x${source.height}`;
}

export function browserMaterialTextureSlots(material: Material): BrowserMaterialTextureSlot[] {
  const typed = material as BrowserTextureSlotMaterial;
  return [
    { slot: 'baseColor', texture: typed.map },
    { slot: 'normal', texture: typed.normalMap },
    { slot: 'metallicRoughness', texture: typed.metalnessMap ?? typed.roughnessMap },
    { slot: 'occlusion', texture: typed.aoMap },
    { slot: 'emissive', texture: typed.emissiveMap },
  ];
}
