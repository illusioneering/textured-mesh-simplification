import {
  DoubleSide,
  FrontSide,
  Material,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Texture,
} from 'three';
import { browserTextureHasImageSource, type BrowserTextureSlotMaterial } from './browserMaterialTextures';

type CorePbrSourceMaterial = Material & {
  color?: { r: number; g: number; b: number };
  opacity?: number;
  transparent?: boolean;
  alphaTest?: number;
  metalness?: number;
  roughness?: number;
  normalScale?: { x: number; y: number };
  aoMapIntensity?: number;
  emissive?: { r: number; g: number; b: number };
} & BrowserTextureSlotMaterial;

function imageBackedTexture(texture: Texture | null | undefined): Texture | null {
  return browserTextureHasImageSource(texture) ? texture ?? null : null;
}

function normalizedMaterialFrom(source: Material): MeshStandardMaterial {
  const typed = source as CorePbrSourceMaterial;
  const material = new MeshStandardMaterial({
    name: source.name,
    opacity: typed.opacity ?? 1,
    map: imageBackedTexture(typed.map),
    metalness: typed.metalness ?? 1,
    roughness: typed.roughness ?? 1,
    metalnessMap: imageBackedTexture(typed.metalnessMap),
    roughnessMap: imageBackedTexture(typed.roughnessMap),
    normalMap: imageBackedTexture(typed.normalMap),
    aoMap: imageBackedTexture(typed.aoMap),
    emissiveMap: imageBackedTexture(typed.emissiveMap),
    transparent: typed.transparent ?? false,
    depthWrite: source.depthWrite,
    alphaTest: typed.alphaTest ?? 0,
    side: source.side === DoubleSide ? DoubleSide : FrontSide,
    vertexColors: source.vertexColors,
  });
  if (typed.color) material.color.setRGB(typed.color.r, typed.color.g, typed.color.b);
  if (typed.emissive) material.emissive.setRGB(typed.emissive.r, typed.emissive.g, typed.emissive.b);
  if (typed.normalScale) material.normalScale.copy(typed.normalScale);
  material.aoMapIntensity = typed.aoMapIntensity ?? 1;
  return material;
}

function normalizeMaterial(
  material: Material,
  cache: Map<Material, MeshStandardMaterial>,
): MeshStandardMaterial {
  const existing = cache.get(material);
  if (existing) return existing;
  const normalized = normalizedMaterialFrom(material);
  cache.set(material, normalized);
  return normalized;
}

export function normalizeBrowserSceneMaterialsToCorePbr(root: Object3D): void {
  const cache = new Map<Material, MeshStandardMaterial>();
  root.traverse((object) => {
    if (!(object as Mesh).isMesh) return;
    const mesh = object as Mesh;
    if (Array.isArray(mesh.material)) {
      mesh.material = mesh.material.map((material) => normalizeMaterial(material, cache));
      return;
    }
    if (mesh.material) mesh.material = normalizeMaterial(mesh.material, cache);
  });
}
