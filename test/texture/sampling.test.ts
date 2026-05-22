import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import type { TexturedRawMesh } from '../../src/texture/types';
import {
  applyWrap,
  sampleImageBilinear,
  sampleImageNearest,
  sampleSourceBaseColor,
  sampleSourceMaterialTexture,
  sampleSourceMaterialTextureColor,
  srgbByteToLinear,
  linearToSrgbByte,
} from '../../src/texture/sampling';
import type { SourceMaterial } from '../../src';

const image = {
  width: 2,
  height: 2,
  data: new Uint8Array([
    255, 0, 0, 255, 0, 255, 0, 255,
    0, 0, 255, 255, 255, 255, 255, 255,
  ]),
};
const sampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'nearest' as const };

function material(overrides: Partial<SourceMaterial> = {}): SourceMaterial {
  return {
    name: 'mat',
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
    ...overrides,
  };
}

describe('texture sampling', () => {
  it('wraps UV coordinates with clamp, repeat, and mirrored-repeat', () => {
    expect(applyWrap(-0.25, 'clamp')).toBe(0);
    expect(applyWrap(1.25, 'clamp')).toBe(1);
    expect(applyWrap(1.25, 'repeat')).toBeCloseTo(0.25);
    expect(applyWrap(-0.25, 'repeat')).toBeCloseTo(0.75);
    expect(applyWrap(1.25, 'mirrored-repeat')).toBeCloseTo(0.75);
    expect(applyWrap(2.25, 'mirrored-repeat')).toBeCloseTo(0.25);
  });

  it('samples nearest texels from a 2x2 image with glTF UV orientation', () => {
    expect(sampleImageNearest(image, 0.25, 0.25, sampler)).toEqual([255, 0, 0, 255]);
    expect(sampleImageNearest(image, 0.75, 0.25, sampler)).toEqual([0, 255, 0, 255]);
    expect(sampleImageNearest(image, 0.25, 0.75, sampler)).toEqual([0, 0, 255, 255]);
  });

  it('bilinearly blends image colors', () => {
    const color = sampleImageBilinear(image, 0.5, 0.5, { ...sampler, filter: 'linear' });
    for (const channel of color.slice(0, 3)) expect(channel).toBeGreaterThan(100);
    expect(color[3]).toBe(255);
  });

  it('wraps bilinear taps across repeat texture borders', () => {
    const color = sampleImageBilinear(image, 0, 0.25, {
      wrapS: 'repeat',
      wrapT: 'clamp',
      filter: 'linear',
    });
    expect(color[0]).toBeGreaterThan(100);
    expect(color[1]).toBeGreaterThan(100);
    expect(color[2]).toBe(0);
    expect(color[3]).toBe(255);
  });

  it('uses constant material color when no image or UVs are available', () => {
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{ materialId: 0, uvSets: [] }],
      materials: [material({
        name: 'constant',
        baseColorFactor: [0.25, 0.5, 1, 0.5],
        alphaMode: 'BLEND',
        doubleSided: true,
      })],
    };
    expect(sampleSourceBaseColor(mesh, 0, [1, 0, 0])).toEqual([137, 188, 255, 128]);
  });

  it('throws when sampling an unmapped source face', () => {
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [],
      materials: [material({ name: 'red', baseColorFactor: [1, 0, 0, 1] })],
    };
    expect(() => sampleSourceBaseColor(mesh, 0, [1, 0, 0])).toThrow(/face attributes/i);
  });

  it('throws when sampling a face with a missing material', () => {
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{ materialId: 99, uvSets: [] }],
      materials: [],
    };
    expect(() => sampleSourceBaseColor(mesh, 0, [1, 0, 0])).toThrow(/material/i);
  });

  it('throws when sampling a textured material without UV coordinates', () => {
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{ materialId: 0, uvSets: [] }],
      materials: [material({
        name: 'textured',
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: { image, sampler, texCoord: 1 },
        textureSlots: [{ slot: 'baseColor', texCoord: 1, sampler, hasImage: true }],
      })],
    };
    expect(() => sampleSourceBaseColor(mesh, 0, [1, 0, 0])).toThrow(/TEXCOORD_1/i);
  });

  it('multiplies sampled sRGB colors by linear baseColorFactor', () => {
    expect(linearToSrgbByte(srgbByteToLinear(255) * 0.25)).toBe(137);
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0.25, 0.25), new Vector2(0.25, 0.25), new Vector2(0.25, 0.25)] }],
      }],
      materials: [material({
        name: 'textured',
        baseColorFactor: [0.25, 1, 1, 0.5],
        baseColorTexture: { image, sampler, texCoord: 0 },
        textureSlots: [{ slot: 'baseColor', texCoord: 0, sampler, hasImage: true }],
        alphaMode: 'BLEND',
        doubleSided: true,
      })],
    };
    expect(sampleSourceBaseColor(mesh, 0, [1, 0, 0])).toEqual([137, 0, 0, 128]);
  });

  it('samples base-color textures from their declared TEXCOORD set', () => {
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{
        materialId: 0,
        uvSets: [
          { texCoord: 0, uvs: [new Vector2(0.75, 0.25), new Vector2(0.75, 0.25), new Vector2(0.75, 0.25)] },
          { texCoord: 1, uvs: [new Vector2(0.25, 0.25), new Vector2(0.25, 0.25), new Vector2(0.25, 0.25)] },
        ],
      }],
      materials: [material({
        name: 'textured-texcoord-1',
        baseColorTexture: { image, sampler, texCoord: 1 },
        textureSlots: [{ slot: 'baseColor', texCoord: 1, sampler, hasImage: true }],
      })],
    };
    expect(sampleSourceBaseColor(mesh, 0, [1, 0, 0])).toEqual([255, 0, 0, 255]);
  });

  it('samples image-backed standard material texture slots from their declared TEXCOORD set', () => {
    const normalImage = {
      width: 2,
      height: 1,
      data: new Uint8Array([
        128, 128, 255, 255,
        200, 140, 80, 255,
      ]),
    };
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{
        materialId: 0,
        uvSets: [
          { texCoord: 0, uvs: [new Vector2(0.75, 0.5), new Vector2(0.75, 0.5), new Vector2(0.75, 0.5)] },
          { texCoord: 1, uvs: [new Vector2(0.25, 0.5), new Vector2(0.25, 0.5), new Vector2(0.25, 0.5)] },
        ],
      }],
      materials: [material({
        textureSlots: [{
          slot: 'normal',
          texCoord: 1,
          sampler,
          hasImage: true,
          image: normalImage,
        }],
      })],
    };

    expect(sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'normal')).toEqual([128, 128, 255, 255]);
  });

  it('returns neutral defaults for missing standard texture slots', () => {
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{ materialId: 0, uvSets: [] }],
      materials: [material()],
    };

    expect(sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'normal')).toEqual([128, 128, 255, 255]);
    expect(sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'metallicRoughness')).toEqual([255, 255, 255, 255]);
    expect(sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'occlusion')).toEqual([255, 255, 255, 255]);
    expect(sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'emissive')).toEqual([255, 255, 255, 255]);
  });

  it('normalizes linearly sampled normal-map values', () => {
    const linearSampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'linear' as const };
    const normalImage = {
      width: 2,
      height: 1,
      data: new Uint8Array([
        255, 128, 128, 255,
        128, 255, 128, 255,
      ]),
    };
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0.5, 0.5), new Vector2(0.5, 0.5), new Vector2(0.5, 0.5)] }],
      }],
      materials: [material({
        textureSlots: [{
          slot: 'normal',
          texCoord: 0,
          sampler: linearSampler,
          hasImage: true,
          image: normalImage,
        }],
      })],
    };

    const sampled = sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'normal');
    const nx = sampled[0] / 255 * 2 - 1;
    const ny = sampled[1] / 255 * 2 - 1;
    const nz = sampled[2] / 255 * 2 - 1;
    expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1, 2);
  });

  it('exposes raw linearly sampled normal-map colors before normalization', () => {
    const linearSampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'linear' as const };
    const normalImage = {
      width: 2,
      height: 1,
      data: new Uint8Array([
        255, 128, 128, 255,
        128, 255, 128, 255,
      ]),
    };
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0.5, 0.5), new Vector2(0.5, 0.5), new Vector2(0.5, 0.5)] }],
      }],
      materials: [material({
        textureSlots: [{
          slot: 'normal',
          texCoord: 0,
          sampler: linearSampler,
          hasImage: true,
          image: normalImage,
        }],
      })],
    };

    expect(sampleSourceMaterialTextureColor(mesh, 0, [1, 0, 0], 'normal')).toEqual([192, 192, 128, 255]);
  });

  it('falls back to neutral when linear normal-map samples cancel to near zero', () => {
    const linearSampler = { wrapS: 'clamp' as const, wrapT: 'clamp' as const, filter: 'linear' as const };
    const normalImage = {
      width: 2,
      height: 1,
      data: new Uint8Array([
        255, 128, 128, 255,
        1, 128, 128, 255,
      ]),
    };
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0.5, 0.5), new Vector2(0.5, 0.5), new Vector2(0.5, 0.5)] }],
      }],
      materials: [material({
        textureSlots: [{
          slot: 'normal',
          texCoord: 0,
          sampler: linearSampler,
          hasImage: true,
          image: normalImage,
        }],
      })],
    };

    expect(sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'normal')).toEqual([128, 128, 255, 255]);
  });

  it('copies nearest-filtered normal-map samples without renormalizing', () => {
    const normalImage = {
      width: 1,
      height: 1,
      data: new Uint8Array([200, 140, 80, 255]),
    };
    const mesh: TexturedRawMesh = {
      rawMesh: { positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] },
      faceAttributes: [{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0.5, 0.5), new Vector2(0.5, 0.5), new Vector2(0.5, 0.5)] }],
      }],
      materials: [material({
        textureSlots: [{
          slot: 'normal',
          texCoord: 0,
          sampler,
          hasImage: true,
          image: normalImage,
        }],
      })],
    };

    expect(sampleSourceMaterialTexture(mesh, 0, [1, 0, 0], 'normal')).toEqual([200, 140, 80, 255]);
  });
});
