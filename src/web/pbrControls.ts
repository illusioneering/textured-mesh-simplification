import {
  Color,
  DoubleSide,
  FrontSide,
  type Material,
} from 'three';

export const textureSizeOptions = [1024, 2048, 4096, 8192] as const;
export type TextureSizeOption = typeof textureSizeOptions[number];

export const allPbrMaterialProperties = [
  { id: 'baseColor', label: 'Base color' },
  { id: 'metallic', label: 'Metallic' },
  { id: 'roughness', label: 'Roughness' },
  { id: 'normal', label: 'Normal' },
  { id: 'occlusion', label: 'Occlusion' },
  { id: 'emissive', label: 'Emissive' },
  { id: 'alpha', label: 'Alpha' },
  { id: 'doubleSided', label: 'Double-sided' },
] as const;

export type PbrMaterialPropertyId = typeof allPbrMaterialProperties[number]['id'];
export type PbrMaterialPropertyState = Record<PbrMaterialPropertyId, boolean>;

export interface ImageDimensions {
  width: number;
  height: number;
}

type PbrMaterialLike = Material & {
  color?: Color;
  map?: unknown | null;
  metalness?: number;
  metalnessMap?: unknown | null;
  roughness?: number;
  roughnessMap?: unknown | null;
  normalMap?: unknown | null;
  aoMap?: unknown | null;
  emissive?: Color;
  emissiveMap?: unknown | null;
};

const white = new Color(0xffffff);
const black = new Color(0x000000);
const defaultTextureSizeOption: TextureSizeOption = 1024;
const maxTextureSizeOption: TextureSizeOption = 8192;

export function defaultPbrMaterialPropertyState(value = false): PbrMaterialPropertyState {
  return {
    baseColor: value,
    metallic: value,
    roughness: value,
    normal: value,
    occlusion: value,
    emissive: value,
    alpha: value,
    doubleSided: value,
  };
}

function hasTexture(value: unknown | null | undefined): boolean {
  return value !== null && value !== undefined;
}

function hasNonDefaultColor(value: Color | undefined, defaultColor: Color): boolean {
  return value !== undefined && !value.equals(defaultColor);
}

export function detectPbrMaterialPropertyAvailability(materials: readonly Material[]): PbrMaterialPropertyState {
  const state = defaultPbrMaterialPropertyState(false);
  for (const material of materials) {
    const typed = material as PbrMaterialLike;
    state.baseColor ||= hasTexture(typed.map) || hasNonDefaultColor(typed.color, white);
    state.metallic ||= hasTexture(typed.metalnessMap) || (typed.metalness !== undefined && typed.metalness !== 1);
    state.roughness ||= hasTexture(typed.roughnessMap) || (typed.roughness !== undefined && typed.roughness !== 1);
    state.normal ||= hasTexture(typed.normalMap);
    state.occlusion ||= hasTexture(typed.aoMap);
    state.emissive ||= hasTexture(typed.emissiveMap) || hasNonDefaultColor(typed.emissive, black);
    state.alpha ||= material.transparent || material.opacity !== 1 || material.alphaTest !== 0;
    state.doubleSided ||= material.side === DoubleSide;
  }
  return state;
}

export function cloneMaterialForPbrPreview(material: Material, state: PbrMaterialPropertyState): Material {
  const clone = material.clone() as PbrMaterialLike;
  if (!state.baseColor) {
    if (clone.color) clone.color.set(white);
    if ('map' in clone) clone.map = null;
  }
  if (!state.metallic) {
    if ('metalness' in clone) clone.metalness = 1;
    if ('metalnessMap' in clone) clone.metalnessMap = null;
  }
  if (!state.roughness) {
    if ('roughness' in clone) clone.roughness = 1;
    if ('roughnessMap' in clone) clone.roughnessMap = null;
  }
  if (!state.normal && 'normalMap' in clone) clone.normalMap = null;
  if (!state.occlusion && 'aoMap' in clone) clone.aoMap = null;
  if (!state.emissive) {
    if (clone.emissive) clone.emissive.set(black);
    if ('emissiveMap' in clone) clone.emissiveMap = null;
  }
  if (!state.alpha) {
    clone.transparent = false;
    clone.opacity = 1;
    clone.alphaTest = 0;
  }
  if (!state.doubleSided) clone.side = FrontSide;
  clone.needsUpdate = true;
  return clone;
}

export function textureSizeForLargestBaseColorMap(
  dimensions: readonly ImageDimensions[],
  currentValue: number,
): TextureSizeOption {
  let largestDimension = 0;
  for (const image of dimensions) {
    largestDimension = Math.max(largestDimension, image.width, image.height);
  }
  if (largestDimension === 0) {
    return textureSizeOptions.includes(currentValue as TextureSizeOption)
      ? currentValue as TextureSizeOption
      : defaultTextureSizeOption;
  }
  for (const option of textureSizeOptions) {
    if (largestDimension <= option) return option;
  }
  return maxTextureSizeOption;
}

export function collectMaterialsForPbrControls(root: { traverse(callback: (object: unknown) => void): void }): Material[] {
  const materials: Material[] = [];
  root.traverse((object) => {
    const candidate = object as { material?: Material | Material[] };
    const objectMaterials = Array.isArray(candidate.material)
      ? candidate.material
      : candidate.material ? [candidate.material] : [];
    for (const material of objectMaterials) {
      materials.push(material);
    }
  });
  return materials;
}
