import { Vector2, Vector3, Vector4 } from 'three';
import { describe, expect, it } from 'vitest';
import {
  hasEntryImageBackedTextureBakeData,
  hasEntryImageBackedTextureTransferData,
  hasEntryPreservableMaterialData,
  toProcessablePrimitiveEntry,
} from '../../src/pipeline/primitiveEntryMetadata';
import type { RawMesh } from '../../src/simplification/types';
import type { SourceMaterial, TexturedRawMesh } from '../../src/texture/types';

const rawMesh: RawMesh = {
  positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
  faces: [[0, 1, 2]],
};

const material: SourceMaterial = {
  name: 'm',
  baseColorFactor: [1, 1, 1, 1],
  baseColorTexture: {
    image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) },
    sampler: { wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' },
    texCoord: 1,
  },
  textureSlots: [{
    slot: 'baseColor',
    texCoord: 1,
    sampler: { wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' },
    hasImage: true,
  }],
  alphaMode: 'OPAQUE',
  alphaCutoff: 0.5,
  doubleSided: false,
  emissiveFactor: [0, 0, 0],
  metallicFactor: 1,
  roughnessFactor: 1,
  normalScale: 1,
  occlusionStrength: 1,
};

function defaultMaterial(): SourceMaterial {
  return {
    name: 'default',
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

function source(withUv: boolean): TexturedRawMesh {
  return {
    rawMesh,
    materials: [{
      ...material,
      baseColorFactor: [...material.baseColorFactor],
      ...(material.baseColorTexture ? { baseColorTexture: { ...material.baseColorTexture } } : {}),
      textureSlots: material.textureSlots.map((slot) => ({ ...slot })),
      emissiveFactor: [...material.emissiveFactor],
    }],
    faceAttributes: [{
      materialId: 0,
      uvSets: withUv
        ? [{ texCoord: 1, uvs: [new Vector2(), new Vector2(1, 0), new Vector2(0, 1)] }]
        : [],
    }],
  };
}

function attributeOnlySource(faceAttributes: TexturedRawMesh['faceAttributes']): TexturedRawMesh {
  return {
    rawMesh,
    materials: [defaultMaterial()],
    faceAttributes,
  };
}

describe('primitive entry metadata', () => {
  it('classifies base-color texture metadata only when required UV sets exist', () => {
    expect(hasEntryImageBackedTextureTransferData(source(true))).toBe(true);
    expect(hasEntryImageBackedTextureTransferData(source(false))).toBe(false);
    expect(hasEntryImageBackedTextureBakeData(source(true))).toBe(true);
    expect(hasEntryImageBackedTextureBakeData(source(false))).toBe(false);
  });

  it('does not request attribute transfer for image-backed textures without supported source attributes', () => {
    const missingUv = source(false);

    const entry = toProcessablePrimitiveEntry({
      id: 'missing-uv',
      rawMesh,
      texturedRawMesh: missingUv,
    });

    expect(entry.hasTexturedMaterial).toBe(true);
    expect(entry.hasPreservableMaterialData).toBe(true);
    expect(entry.requiresAttributeTransfer).toBeUndefined();
    expect(entry.bakeable).toBe(false);
  });

  it('keeps factor-only material data preservable but not bakeable or attribute-transferable', () => {
    const factorOnly = source(false);
    delete factorOnly.materials[0]!.baseColorTexture;
    factorOnly.materials[0]!.textureSlots = [];
    factorOnly.materials[0]!.baseColorFactor = [1, 0, 0, 1];

    expect(hasEntryPreservableMaterialData(factorOnly)).toBe(true);
    expect(hasEntryImageBackedTextureTransferData(factorOnly)).toBe(false);
    expect(hasEntryImageBackedTextureBakeData(factorOnly)).toBe(false);

    const entry = toProcessablePrimitiveEntry({
      id: 'factor',
      rawMesh,
      texturedRawMesh: factorOnly,
    });

    expect(entry.hasPreservableMaterialData).toBe(true);
    expect(entry.requiresAttributeTransfer).toBeUndefined();
    expect(entry.bakeable).toBe(false);
  });

  it('requests attribute transfer when source normals exist without image-backed textures', () => {
    const entry = toProcessablePrimitiveEntry({
      id: 'normal-only',
      rawMesh,
      texturedRawMesh: attributeOnlySource([{
        materialId: 0,
        uvSets: [],
        normalCorners: [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)],
      }]),
    });

    expect(entry.requiresAttributeTransfer).toBe(true);
  });

  it('requests attribute transfer when source UVs exist without image-backed textures', () => {
    const entry = toProcessablePrimitiveEntry({
      id: 'uv-only',
      rawMesh,
      texturedRawMesh: attributeOnlySource([{
        materialId: 0,
        uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
      }]),
    });

    expect(entry.requiresAttributeTransfer).toBe(true);
  });

  it('requests attribute transfer when source tangents exist without image-backed textures', () => {
    const entry = toProcessablePrimitiveEntry({
      id: 'tangent-only',
      rawMesh,
      texturedRawMesh: attributeOnlySource([{
        materialId: 0,
        uvSets: [],
        tangentCorners: [new Vector4(1, 0, 0, 1), new Vector4(0, 1, 0, 1), new Vector4(0, 0, 1, -1)],
      }]),
    });

    expect(entry.requiresAttributeTransfer).toBe(true);
  });

  it('requests attribute transfer when source vertex colors exist without image-backed textures', () => {
    const entry = toProcessablePrimitiveEntry({
      id: 'color-only',
      rawMesh,
      texturedRawMesh: attributeOnlySource([{
        materialId: 0,
        uvSets: [],
        colorCorners: [new Vector4(1, 0, 0, 1), new Vector4(0, 1, 0, 1), new Vector4(0, 0, 1, 1)],
        colorItemSize: 3,
      }]),
    });

    expect(entry.requiresAttributeTransfer).toBe(true);
  });

  it('creates processable entries using texturedRawMesh', () => {
    const entry = toProcessablePrimitiveEntry({
      id: 'entry-1',
      label: 'Entry 1',
      rawMesh,
      texturedRawMesh: source(true),
    });

    expect(entry.id).toBe('entry-1');
    expect(entry.texturedRawMesh).toBeDefined();
    expect(entry.bakeable).toBe(true);
    expect(entry.hasTexturedMaterial).toBe(true);
    expect(entry.requiresAttributeTransfer).toBe(true);
  });
});
