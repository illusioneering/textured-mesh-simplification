import type { Barycentric } from '../simplification/types';
import type { Rgba, RgbaImage, StandardBakedTextureSlot, TextureSampler, TexturedRawMesh, WrapMode } from './types';
import { faceUvSet } from './types';
import { interpolateVector2 } from '../simplification/barycentric';

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

export function applyWrap(value: number, mode: WrapMode): number {
  if (mode === 'clamp') return Math.max(0, Math.min(1, value));
  const repeated = value - Math.floor(value);
  if (mode === 'repeat') return repeated;
  const period = Math.floor(value);
  return Math.abs(period % 2) === 0 ? repeated : 1 - repeated;
}

function pixelOffset(image: RgbaImage, x: number, y: number): number {
  return (y * image.width + x) * 4;
}

function getPixel(image: RgbaImage, x: number, y: number): Rgba {
  const clampedX = Math.max(0, Math.min(image.width - 1, x));
  const clampedY = Math.max(0, Math.min(image.height - 1, y));
  const offset = pixelOffset(image, clampedX, clampedY);
  return [
    image.data[offset] ?? 0,
    image.data[offset + 1] ?? 0,
    image.data[offset + 2] ?? 0,
    image.data[offset + 3] ?? 255,
  ];
}

function wrapPixelCoord(coord: number, size: number, mode: WrapMode): number {
  if (mode === 'clamp') return Math.max(0, Math.min(size - 1, coord));
  if (mode === 'repeat') return ((coord % size) + size) % size;
  const period = size * 2;
  const mirrored = ((coord % period) + period) % period;
  return mirrored < size ? mirrored : period - 1 - mirrored;
}

function getWrappedPixel(image: RgbaImage, x: number, y: number, sampler: TextureSampler): Rgba {
  return getPixel(image, wrapPixelCoord(x, image.width, sampler.wrapS), wrapPixelCoord(y, image.height, sampler.wrapT));
}

export function sampleImageNearest(image: RgbaImage, u: number, v: number, sampler: TextureSampler): Rgba {
  if (image.width <= 0 || image.height <= 0) throw new Error('Cannot sample an empty image.');
  const wrappedU = applyWrap(u, sampler.wrapS);
  const wrappedV = applyWrap(v, sampler.wrapT);
  const x = Math.min(image.width - 1, Math.max(0, Math.floor(wrappedU * image.width)));
  const y = Math.min(image.height - 1, Math.max(0, Math.floor(wrappedV * image.height)));
  return getPixel(image, x, y);
}

export function sampleImageBilinear(image: RgbaImage, u: number, v: number, sampler: TextureSampler): Rgba {
  if (image.width <= 0 || image.height <= 0) throw new Error('Cannot sample an empty image.');
  const wrappedU = applyWrap(u, sampler.wrapS);
  const wrappedV = applyWrap(v, sampler.wrapT);
  const x = wrappedU * image.width - 0.5;
  const y = wrappedV * image.height - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const c00 = getWrappedPixel(image, x0, y0, sampler);
  const c10 = getWrappedPixel(image, x0 + 1, y0, sampler);
  const c01 = getWrappedPixel(image, x0, y0 + 1, sampler);
  const c11 = getWrappedPixel(image, x0 + 1, y0 + 1, sampler);
  const out: Rgba = [0, 0, 0, 0];
  for (let channel = 0; channel < 4; channel += 1) {
    const top = c00[channel]! * (1 - tx) + c10[channel]! * tx;
    const bottom = c01[channel]! * (1 - tx) + c11[channel]! * tx;
    out[channel] = clampByte(top * (1 - ty) + bottom * ty);
  }
  return out;
}

export function srgbByteToLinear(byte: number): number {
  const value = Math.max(0, Math.min(1, byte / 255));
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

export function linearToSrgbByte(linear: number): number {
  const value = Math.max(0, Math.min(1, linear));
  const srgb = value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;
  return clampByte(srgb * 255);
}

function factorToRgba(factor: [number, number, number, number]): Rgba {
  return [clampByte(factor[0] * 255), clampByte(factor[1] * 255), clampByte(factor[2] * 255), clampByte(factor[3] * 255)];
}

const DEFAULT_TEXTURE_COLORS: Record<StandardBakedTextureSlot, Rgba> = {
  normal: [128, 128, 255, 255],
  metallicRoughness: [255, 255, 255, 255],
  occlusion: [255, 255, 255, 255],
  emissive: [255, 255, 255, 255],
};
const BYTE_NORMAL_DECODE_STEP = Math.sqrt(3) / 255;

function sampleTextureImage(
  image: RgbaImage,
  sampler: TextureSampler,
  texCoord: number,
  mesh: TexturedRawMesh,
  faceId: number,
  barycentric: Barycentric,
): Rgba {
  const attributes = mesh.faceAttributes[faceId];
  if (!attributes) throw new Error(`Missing face attributes for source face ${faceId}.`);
  const uvSet = faceUvSet(attributes, texCoord);
  if (!uvSet) throw new Error(`Missing TEXCOORD_${texCoord} coordinates for textured source face ${faceId}.`);
  const uv = interpolateVector2(uvSet.uvs[0], uvSet.uvs[1], uvSet.uvs[2], barycentric);
  return sampler.filter === 'nearest'
    ? sampleImageNearest(image, uv.x, uv.y, sampler)
    : sampleImageBilinear(image, uv.x, uv.y, sampler);
}

function normalizeNormalSample(color: Rgba): Rgba {
  const x = color[0] / 255 * 2 - 1;
  const y = color[1] / 255 * 2 - 1;
  const z = color[2] / 255 * 2 - 1;
  const length = Math.hypot(x, y, z);
  if (length <= BYTE_NORMAL_DECODE_STEP) return [128, 128, 255, color[3]];
  return [
    clampByte(((x / length) * 0.5 + 0.5) * 255),
    clampByte(((y / length) * 0.5 + 0.5) * 255),
    clampByte(((z / length) * 0.5 + 0.5) * 255),
    color[3],
  ];
}

export function sampleSourceBaseColor(mesh: TexturedRawMesh, faceId: number, barycentric: Barycentric): Rgba {
  const attributes = mesh.faceAttributes[faceId];
  if (!attributes) throw new Error(`Missing face attributes for source face ${faceId}.`);
  const material = mesh.materials[attributes.materialId];
  if (!material) throw new Error(`Missing material ${attributes.materialId} for source face ${faceId}.`);

  let color: Rgba;
  const texture = material.baseColorTexture;
  if (texture?.image) {
    color = sampleTextureImage(texture.image, texture.sampler, texture.texCoord, mesh, faceId, barycentric);
  } else {
    color = factorToRgba([1, 1, 1, 1]);
  }

  return [
    linearToSrgbByte(srgbByteToLinear(color[0]) * material.baseColorFactor[0]),
    linearToSrgbByte(srgbByteToLinear(color[1]) * material.baseColorFactor[1]),
    linearToSrgbByte(srgbByteToLinear(color[2]) * material.baseColorFactor[2]),
    clampByte((color[3] / 255) * material.baseColorFactor[3] * 255),
  ];
}

export function sampleSourceMaterialTexture(
  mesh: TexturedRawMesh,
  faceId: number,
  barycentric: Barycentric,
  slot: StandardBakedTextureSlot,
): Rgba {
  const attributes = mesh.faceAttributes[faceId];
  if (!attributes) throw new Error(`Missing face attributes for source face ${faceId}.`);
  const material = mesh.materials[attributes.materialId];
  if (!material) throw new Error(`Missing material ${attributes.materialId} for source face ${faceId}.`);
  const texture = material.textureSlots.find((candidate) => candidate.slot === slot && candidate.hasImage && candidate.image);
  const color = sampleSourceMaterialTextureColor(mesh, faceId, barycentric, slot);

  return slot === 'normal' && texture?.sampler.filter === 'linear' ? normalizeNormalSample(color) : color;
}

export function sampleSourceMaterialTextureColor(
  mesh: TexturedRawMesh,
  faceId: number,
  barycentric: Barycentric,
  slot: StandardBakedTextureSlot,
): Rgba {
  const attributes = mesh.faceAttributes[faceId];
  if (!attributes) throw new Error(`Missing face attributes for source face ${faceId}.`);
  const material = mesh.materials[attributes.materialId];
  if (!material) throw new Error(`Missing material ${attributes.materialId} for source face ${faceId}.`);
  const texture = material.textureSlots.find((candidate) => candidate.slot === slot && candidate.hasImage && candidate.image);
  if (!texture?.image) return DEFAULT_TEXTURE_COLORS[slot];

  return sampleTextureImage(texture.image, texture.sampler, texture.texCoord, mesh, faceId, barycentric);
}
