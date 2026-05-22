import { describe, expect, it } from 'vitest';
import { Vector2, Vector3, Vector4 } from 'three';
import { createInjectiveAtlas } from '../../src/texture/atlas';
import type { TransferredMeshAttributes } from '../../src/simplification/attributes';
import type { RawMesh, RawSimplificationResult } from '../../src/simplification/types';
import type { SourceMaterial, TexturedRawMesh } from '../../src/texture/types';
import {
  collectTransferables,
  deserializeAtlas,
  deserializeBakedMaterialTexture,
  deserializePrimitiveEntries,
  deserializeRawMesh,
  deserializeRgbaImage,
  deserializeSourceFaceAttributes,
  deserializeTexturedRawMesh,
  deserializeTransferredMeshAttributes,
  serializePrimitiveGeometryProcessingResult,
  serializePrimitiveSceneProcessingResult,
  serializeAtlas,
  serializeRawMesh,
  serializeSimplifiedGeometryResult,
  serializeRgbaImage,
  serializePrimitiveEntries,
  serializeTransferredMeshAttributes,
  serializeTexturedProcessingResult,
  serializeTexturedRawMesh,
  deserializeFullRawSimplificationResult,
  serializeFullRawSimplificationResult,
} from '../../src/web/serialization';

function rawMesh(): RawMesh {
  return { positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)], faces: [[0, 1, 2]] };
}

function rawResult(): RawSimplificationResult {
  return {
    rawMesh: rawMesh(),
    outputFaceIds: [0],
    history: [],
    stats: {
      inputVertices: 3,
      inputFaces: 1,
      outputVertices: 3,
      outputFaces: 1,
      physicalEdges: 3,
      virtualEdges: 0,
      collapses: 0,
      stoppedReason: 'target-reached',
    },
  };
}

function textured(): TexturedRawMesh {
  const sampler = { wrapS: 'repeat' as const, wrapT: 'mirrored-repeat' as const, filter: 'nearest' as const };
  const normalSampler = { wrapS: 'repeat' as const, wrapT: 'repeat' as const, filter: 'linear' as const };
  const material: SourceMaterial = {
    name: 'mat',
    baseColorFactor: [0.1, 0.2, 0.3, 0.4],
    baseColorTexture: {
      image: { width: 1, height: 1, data: new Uint8Array([1, 2, 3, 4]) },
      sampler,
      texCoord: 0,
    },
    textureSlots: [
      { slot: 'baseColor', texCoord: 0, sampler, hasImage: true },
      {
        slot: 'normal',
        texCoord: 1,
        sampler: normalSampler,
        hasImage: true,
        image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) },
      },
    ],
    alphaMode: 'BLEND',
    alphaCutoff: 0.33,
    doubleSided: true,
    emissiveFactor: [0.1, 0.2, 0.3],
    metallicFactor: 0,
    roughnessFactor: 0.5,
    normalScale: 0.75,
    occlusionStrength: 0.6,
  };
  return {
    rawMesh: rawMesh(),
    faceAttributes: [{
      materialId: 0,
      uvSets: [
        { texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] },
        { texCoord: 1, uvs: [new Vector2(0.5, 0), new Vector2(1, 0.5), new Vector2(0, 0.5)] },
      ],
      colorCorners: [
        new Vector4(1, 0, 0, 1),
        new Vector4(0, 1, 0, 0.75),
        new Vector4(0, 0, 1, 0.5),
      ],
      colorItemSize: 4,
    }],
    materials: [material],
  };
}

function primitiveEntry() {
  const texturedRawMesh = textured();
  return {
    id: 'primitive-0',
    label: 'Primitive 0',
    meshOrdinal: 0,
    rawMesh: rawMesh(),
    texturedRawMesh,
    bakeable: true,
    hasTexturedMaterial: true,
    requiresAttributeTransfer: true,
  };
}

function transferredAttributes(): TransferredMeshAttributes {
  return {
    vertices: [
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(0, 0) },
          { texCoord: 1, uv: new Vector2(0.5, 0.25) },
        ],
        normal: new Vector3(0, 0, 1),
        tangent: new Vector4(1, 0, 0, -1),
        color: new Vector4(1, 0, 0, 1),
      },
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(1, 0) }],
        color: new Vector4(0, 1, 0, 0.75),
      },
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(0, 1) }],
        normal: new Vector3(0, 1, 0),
        color: new Vector4(0, 0, 1, 0.5),
      },
    ],
    colorItemSize: 4,
    normalMapYScale: -1,
    hasSourceTangents: true,
  };
}

describe('worker serialization', () => {
  it('round-trips raw meshes', () => {
    const serialized = serializeRawMesh(rawMesh());
    const restored = deserializeRawMesh(serialized);
    expect(restored.positions[2]!.y).toBe(1);
    expect(restored.faces).toEqual([[0, 1, 2]]);
  });

  it('round-trips images, materials, and UVs', () => {
    const serialized = serializeTexturedRawMesh(textured());
    expect(serialized.faceAttributes.materialIds).toBeInstanceOf(Int32Array);
    expect(serialized.faceAttributes.uvSetCounts).toBeInstanceOf(Uint16Array);
    expect(serialized.faceAttributes.uvSetTexCoords).toBeInstanceOf(Uint16Array);
    expect(serialized.faceAttributes.uvs).toBeInstanceOf(Float32Array);
    expect(serialized.faceAttributes.uvs).toHaveLength(12);
    expect(serialized.faceAttributes.colorCornerFlags).toEqual(new Uint8Array([1]));
    expect(serialized.faceAttributes.colorItemSizes).toEqual(new Uint8Array([4]));
    expect(serialized.faceAttributes.colorCorners).toHaveLength(12);

    const restored = deserializeTexturedRawMesh(serialized);
    expect(restored.faceAttributes[0]!.uvSets[0]?.uvs[1].x).toBe(1);
    expect(restored.faceAttributes[0]!.uvSets[1]?.texCoord).toBe(1);
    expect(restored.faceAttributes[0]!.colorItemSize).toBe(4);
    expect(restored.faceAttributes[0]!.colorCorners?.[1].w).toBeCloseTo(0.75);
    expect(restored.materials[0]!.baseColorFactor).toEqual([0.1, 0.2, 0.3, 0.4]);
    expect(Array.from(restored.materials[0]!.baseColorTexture!.image!.data)).toEqual([1, 2, 3, 4]);
    expect(restored.materials[0]!.baseColorTexture!.sampler.wrapT).toBe('mirrored-repeat');
    expect(restored.materials[0]!.textureSlots.map((slot) => slot.slot)).toEqual(['baseColor', 'normal']);
    expect(restored.materials[0]!.textureSlots[0]!.hasImage).toBe(true);
    expect(restored.materials[0]!.textureSlots[0]!.image).toBeUndefined();
    expect(restored.materials[0]!.textureSlots[1]!.texCoord).toBe(1);
    expect(Array.from(restored.materials[0]!.textureSlots[1]!.image!.data)).toEqual([128, 128, 255, 255]);
    expect(restored.materials[0]!.emissiveFactor).toEqual([0.1, 0.2, 0.3]);
    expect(restored.materials[0]!.normalScale).toBe(0.75);
    expect(restored.materials[0]!.occlusionStrength).toBe(0.6);
  });

  it('round-trips every UV set on each face', () => {
    const source = textured();
    source.materials[0]!.baseColorTexture!.texCoord = 1;
    source.materials[0]!.textureSlots[0]!.texCoord = 1;
    source.faceAttributes[0]!.uvSets = [
      { texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] },
      { texCoord: 1, uvs: [new Vector2(0.25, 0.25), new Vector2(0.75, 0.25), new Vector2(0.25, 0.75)] },
    ];

    const serialized = serializeTexturedRawMesh(source);
    expect(serialized.faceAttributes.uvSetCounts).toEqual(new Uint16Array([2]));
    expect(serialized.faceAttributes.uvSetTexCoords).toEqual(new Uint16Array([0, 1]));
    expect(serialized.faceAttributes.uvs).toHaveLength(12);

    const restored = deserializeTexturedRawMesh(serialized);
    expect(restored.materials[0]!.baseColorTexture!.texCoord).toBe(1);
    expect(restored.faceAttributes[0]!.uvSets).toHaveLength(2);
    expect(restored.faceAttributes[0]!.uvSets[1]!.texCoord).toBe(1);
    expect(restored.faceAttributes[0]!.uvSets[1]!.uvs[1].x).toBeCloseTo(0.75);
  });

  it('round-trips optional source face normal corners', () => {
    const source = textured();
    source.rawMesh.faces.push([0, 2, 1]);
    source.faceAttributes[0]!.normalCorners = [
      new Vector3(0, 0, 1),
      new Vector3(0, 1, 0),
      new Vector3(1, 0, 0),
    ];
    source.faceAttributes.push({ materialId: 0, uvSets: [] });

    const serialized = serializeTexturedRawMesh(source);
    expect(serialized.faceAttributes.normalCornerFlags).toEqual(new Uint8Array([1, 0]));
    expect(serialized.faceAttributes.normalCorners).toBeInstanceOf(Float32Array);
    expect(serialized.faceAttributes.normalCorners).toHaveLength(9);

    const restored = deserializeTexturedRawMesh(serialized);

    expect(restored.faceAttributes[0]!.normalCorners).toHaveLength(3);
    expect(restored.faceAttributes[0]!.normalCorners?.[0].z).toBeCloseTo(1);
    expect(restored.faceAttributes[0]!.normalCorners?.[1].y).toBeCloseTo(1);
    expect(restored.faceAttributes[0]!.normalCorners?.[2].x).toBeCloseTo(1);
    expect('normalCorners' in restored.faceAttributes[1]!).toBe(false);
  });

  it('round-trips per-face source normal-map Y scale', () => {
    const source = textured();
    source.rawMesh.faces.push([0, 2, 1]);
    source.faceAttributes[0]!.normalMapYScale = -1;
    source.faceAttributes.push({ materialId: 0, uvSets: [] });

    const serialized = serializeTexturedRawMesh(source);
    expect(serialized.faceAttributes.normalMapYScales).toEqual(new Float32Array([-1, 1]));

    const restored = deserializeTexturedRawMesh(serialized);
    expect(restored.faceAttributes[0]!.normalMapYScale).toBe(-1);
    expect('normalMapYScale' in restored.faceAttributes[1]!).toBe(false);
  });

  it('round-trips optional source face tangent corners', () => {
    const source = textured();
    source.rawMesh.faces.push([0, 2, 1]);
    source.faceAttributes[0]!.tangentCorners = [
      new Vector4(1, 0, 0, -1),
      new Vector4(0, 1, 0, 1),
      new Vector4(0, 0, 1, -1),
    ];
    source.faceAttributes.push({ materialId: 0, uvSets: [] });

    const serialized = serializeTexturedRawMesh(source);
    expect(serialized.faceAttributes.tangentCornerFlags).toEqual(new Uint8Array([1, 0]));
    expect(serialized.faceAttributes.tangentCorners).toBeInstanceOf(Float32Array);
    expect(serialized.faceAttributes.tangentCorners).toHaveLength(12);

    const restored = deserializeTexturedRawMesh(serialized);
    expect(restored.faceAttributes[0]!.tangentCorners).toHaveLength(3);
    expect(restored.faceAttributes[0]!.tangentCorners?.[0].w).toBeCloseTo(-1);
    expect(restored.faceAttributes[0]!.tangentCorners?.[1].y).toBeCloseTo(1);
    expect(restored.faceAttributes[0]!.tangentCorners?.[2].z).toBeCloseTo(1);
    expect('tangentCorners' in restored.faceAttributes[1]!).toBe(false);
  });

  it('round-trips transferred vertex attributes', () => {
    const serialized = serializeTransferredMeshAttributes(transferredAttributes());

    expect(serialized.uvSetCounts).toEqual(new Uint16Array([2, 1, 1]));
    expect(serialized.uvSetTexCoords).toEqual(new Uint16Array([0, 1, 0, 0]));
    expect(serialized.uvs).toHaveLength(8);
    expect(serialized.normalFlags).toEqual(new Uint8Array([1, 0, 1]));
    expect(serialized.normals).toHaveLength(6);
    expect(serialized.tangentFlags).toEqual(new Uint8Array([1, 0, 0]));
    expect(serialized.tangents).toHaveLength(4);
    expect(serialized.colorFlags).toEqual(new Uint8Array([1, 1, 1]));
    expect(serialized.colors).toHaveLength(12);
    expect(serialized.colorItemSize).toBe(4);
    expect(serialized.normalMapYScale).toBe(-1);
    expect(serialized.hasSourceTangents).toBe(true);

    const restored = deserializeTransferredMeshAttributes(serialized);
    expect(restored.vertices).toHaveLength(3);
    expect(restored.vertices[0]!.uvSets[1]!.texCoord).toBe(1);
    expect(restored.vertices[0]!.uvSets[1]!.uv.x).toBeCloseTo(0.5);
    expect(restored.vertices[0]!.normal?.z).toBeCloseTo(1);
    expect(restored.vertices[0]!.tangent?.w).toBeCloseTo(-1);
    expect(restored.vertices[1]!.normal).toBeUndefined();
    expect(restored.colorItemSize).toBe(4);
    expect(restored.vertices[1]!.color?.y).toBeCloseTo(1);
    expect(restored.vertices[1]!.color?.w).toBeCloseTo(0.75);
    expect(restored.vertices[2]!.normal?.y).toBeCloseTo(1);
    expect(restored.normalMapYScale).toBe(-1);
    expect(restored.hasSourceTangents).toBe(true);
  });

  it('rejects inconsistent packed source face normal corners', () => {
    expect(() => deserializeSourceFaceAttributes({
      materialIds: new Int32Array([0, 0]),
      uvSetCounts: new Uint16Array([0, 0]),
      uvSetTexCoords: new Uint16Array(),
      uvs: new Float32Array(),
      normalCornerFlags: new Uint8Array([1]),
      normalCorners: new Float32Array(9),
    })).toThrow(/normal corner flag count 1 does not match face count 2/);

    expect(() => deserializeSourceFaceAttributes({
      materialIds: new Int32Array([0, 0]),
      uvSetCounts: new Uint16Array([0, 0]),
      uvSetTexCoords: new Uint16Array(),
      uvs: new Float32Array(),
      normalCornerFlags: new Uint8Array([1, 0]),
      normalCorners: new Float32Array(8),
    })).toThrow(/normal corner value count 8 does not match expected count 9/);
  });

  it('rejects inconsistent packed transferred vertex attributes', () => {
    expect(() => deserializeTransferredMeshAttributes({
      uvSetCounts: new Uint16Array([1]),
      uvSetTexCoords: new Uint16Array([0]),
      uvs: new Float32Array([0]),
    })).toThrow(/transferred UV value count 1 does not match expected count 2/);

    expect(() => deserializeTransferredMeshAttributes({
      uvSetCounts: new Uint16Array([0, 0]),
      uvSetTexCoords: new Uint16Array(),
      uvs: new Float32Array(),
      normalFlags: new Uint8Array([1]),
      normals: new Float32Array(3),
    })).toThrow(/normal flag count 1 does not match vertex count 2/);

    expect(() => deserializeTransferredMeshAttributes({
      uvSetCounts: new Uint16Array([0, 0]),
      uvSetTexCoords: new Uint16Array(),
      uvs: new Float32Array(),
      colorFlags: new Uint8Array([1]),
      colors: new Float32Array(4),
      colorItemSize: 4,
    })).toThrow(/color flag count 1 does not match vertex count 2/);
  });

  it('omits texture image bytes from primitive entries when requested while preserving texture metadata', () => {
    const entry = primitiveEntry();
    entry.texturedRawMesh.materials[0]!.baseColorTexture!.name = 'base-color.png';
    entry.texturedRawMesh.materials[0]!.baseColorTexture!.mimeType = 'image/png';
    entry.texturedRawMesh.materials[0]!.textureSlots[1]!.name = 'normal.png';
    entry.texturedRawMesh.materials[0]!.textureSlots[1]!.mimeType = 'image/png';

    const serialized = serializePrimitiveEntries([entry], { includeImages: false });
    const deserialized = deserializePrimitiveEntries(serialized);
    const material = serialized[0]!.textured.materials[0]!;

    expect(material.baseColorTexture).toMatchObject({
      sampler: entry.texturedRawMesh.materials[0]!.baseColorTexture!.sampler,
      texCoord: 0,
      name: 'base-color.png',
      mimeType: 'image/png',
    });
    expect(material.baseColorTexture!.image).toBeUndefined();
    expect(material.textureSlots[0]).toMatchObject({
      slot: 'baseColor',
      texCoord: 0,
      hasImage: true,
    });
    expect(material.textureSlots[0]!.image).toBeUndefined();
    expect(material.textureSlots[1]).toMatchObject({
      slot: 'normal',
      texCoord: 1,
      hasImage: true,
      name: 'normal.png',
      mimeType: 'image/png',
    });
    expect(material.textureSlots[1]!.image).toBeUndefined();
    expect(deserialized[0]!.texturedRawMesh?.materials).toHaveLength(1);
    expect(deserialized[0]!.texturedRawMesh?.faceAttributes).toHaveLength(1);
    expect('textured' in deserialized[0]!).toBe(false);
  });

  it('includes texture image bytes in default primitive entry serialization for bake requests', () => {
    const serialized = serializePrimitiveEntries([primitiveEntry()]);
    const material = serialized[0]!.textured.materials[0]!;

    expect(Array.from(material.baseColorTexture!.image!.data)).toEqual([1, 2, 3, 4]);
    expect(Array.from(material.textureSlots[1]!.image!.data)).toEqual([128, 128, 255, 255]);
    expect(material.textureSlots[1]!.hasImage).toBe(true);
  });

  it('switches primitive entry image payloads between geometry metadata and bake serialization', () => {
    const entries = [primitiveEntry()];

    const geometrySerialized = serializePrimitiveEntries(entries, { includeImages: false });
    expect(geometrySerialized[0]!.textured.materials[0]!.baseColorTexture?.image).toBeUndefined();
    expect(geometrySerialized[0]!.textured.materials[0]!.textureSlots[1]!.image).toBeUndefined();

    const bakeSerialized = serializePrimitiveEntries(entries);
    expect(bakeSerialized[0]!.textured.materials[0]!.baseColorTexture?.image).toBeDefined();
    expect(bakeSerialized[0]!.textured.materials[0]!.textureSlots[1]!.image).toBeDefined();
  });

  it('packs textured face attributes into contiguous typed arrays', () => {
    const source = textured();
    source.rawMesh.faces.push([0, 2, 1]);
    source.faceAttributes.push({ materialId: 0, uvSets: [] });

    const serialized = serializeTexturedRawMesh(source);
    expect(serialized.faceAttributes.materialIds).toEqual(new Int32Array([0, 0]));
    expect(serialized.faceAttributes.uvSetCounts).toEqual(new Uint16Array([2, 0]));
    expect(serialized.faceAttributes.uvSetTexCoords).toEqual(new Uint16Array([0, 1]));
    expect(serialized.faceAttributes.uvs).toHaveLength(12);

    const restored = deserializeTexturedRawMesh(serialized);
    expect(restored.faceAttributes).toHaveLength(2);
    expect(restored.faceAttributes[0]!.uvSets[0]?.uvs[2].y).toBe(1);
    expect(restored.faceAttributes[1]!.uvSets).toEqual([]);
  });

  it('round-trips atlas layouts and rgba images', async () => {
    const atlas = await createInjectiveAtlas(rawMesh(), { textureSize: 16, padding: 1 });
    const restoredAtlas = deserializeAtlas(serializeAtlas(atlas));
    expect(restoredAtlas.islandCount).toBe(1);
    expect(restoredAtlas.faceUvs[0]![1].x).toBeCloseTo(atlas.faceUvs[0]![1].x);

    const image = deserializeRgbaImage(serializeRgbaImage({ width: 1, height: 1, data: new Uint8Array([9, 8, 7, 6]) }));
    expect(Array.from(image.data)).toEqual([9, 8, 7, 6]);
  });

  it('serializes additional baked material textures on textured and primitive results', async () => {
    const atlas = await createInjectiveAtlas(rawMesh(), { textureSize: 16, padding: 1 });
    const baked = {
      image: { width: 1, height: 1, data: new Uint8Array([1, 2, 3, 4]) },
      additionalTextures: [{
        slot: 'normal' as const,
        image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) },
      }],
      atlas,
      stats: { filledPixels: 1, mappedPixels: 1, unmappedPixels: 0 },
    };
    const materialSettings = {
      alphaMode: 'OPAQUE' as const,
      alphaCutoff: 0.5,
      doubleSided: false,
      metallicFactor: 1,
      roughnessFactor: 1,
      emissiveFactor: [0, 0, 0] as [number, number, number],
      normalScale: 1,
      occlusionStrength: 1,
    };
    const texturedResult = {
      raw: rawResult(),
      baked,
      materialSettings,
      elapsedSeconds: 0,
    };

    const serializedTextured = serializeTexturedProcessingResult(texturedResult);
    expect(serializedTextured.baked.additionalTextures).toHaveLength(1);
    expect(serializedTextured.baked.additionalTextures[0]!.slot).toBe('normal');
    expect(Array.from(serializedTextured.baked.additionalTextures[0]!.image.data)).toEqual([128, 128, 255, 255]);
    const restoredTexture = deserializeBakedMaterialTexture(serializedTextured.baked.additionalTextures[0]!);
    expect(restoredTexture.slot).toBe('normal');
    expect(Array.from(restoredTexture.image.data)).toEqual([128, 128, 255, 255]);

    const primitiveResult = {
      entries: [{
        id: 'primitive-0',
        source: {
          id: 'primitive-0',
          rawMesh: rawMesh(),
          texturedRawMesh: textured(),
          bakeable: true,
        },
        geometry: {
          raw: rawResult(),
          elapsedSeconds: 0,
        },
        raw: rawResult(),
        baked: texturedResult,
        transferredAttributes: transferredAttributes(),
      }],
      stats: rawResult().stats,
      elapsedSeconds: 0,
    };
    const serializedPrimitive = serializePrimitiveSceneProcessingResult(primitiveResult, [{ id: 'primitive-0', meshOrdinal: 7 }]);
    expect(serializedPrimitive.entries[0]!.transferredAttributes?.colorFlags).toEqual(new Uint8Array([1, 1, 1]));
    expect(serializedPrimitive.entries[0]!.transferredAttributes?.colorItemSize).toBe(4);
    const restoredTransferred = deserializeTransferredMeshAttributes(serializedPrimitive.entries[0]!.transferredAttributes!);
    const restoredHasColor = restoredTransferred.vertices.map((vertex) => Boolean(vertex.color));
    const restoredColors = restoredTransferred.vertices.flatMap((vertex) => (
      vertex.color ? [vertex.color.x, vertex.color.y, vertex.color.z, vertex.color.w] : []
    ));
    expect(restoredHasColor).toEqual([true, true, true]);
    expect(restoredTransferred.colorItemSize).toBe(4);
    expect(restoredColors).toHaveLength(12);
    expect(restoredColors.slice(0, 4)).toEqual([1, 0, 0, 1]);
    expect(restoredTransferred.vertices[1]!.color?.y).toBeCloseTo(1);
    expect(restoredTransferred.vertices[1]!.color?.w).toBeCloseTo(0.75);
    expect(serializedPrimitive.entries[0]!.baked?.additionalTextures).toHaveLength(1);
    expect(serializedPrimitive.entries[0]!.baked?.additionalTextures[0]!.slot).toBe('normal');
    expect(Array.from(serializedPrimitive.entries[0]!.baked!.additionalTextures[0]!.image.data)).toEqual([128, 128, 255, 255]);
  });

  it('collects ArrayBuffers from nested serializable payloads', () => {
    const source = textured();
    source.faceAttributes[0]!.normalCorners = [
      new Vector3(0, 0, 1),
      new Vector3(0, 1, 0),
      new Vector3(1, 0, 0),
    ];
    const serialized = serializeTexturedRawMesh(source);
    const buffers = collectTransferables(serialized);
    const expectedBuffers = [
      serialized.rawMesh.positions.buffer,
      serialized.rawMesh.indices.buffer,
      serialized.faceAttributes.materialIds.buffer,
      serialized.faceAttributes.uvSetCounts.buffer,
      serialized.faceAttributes.uvSetTexCoords.buffer,
      serialized.faceAttributes.uvs.buffer,
      serialized.faceAttributes.normalCornerFlags!.buffer,
      serialized.faceAttributes.normalCorners!.buffer,
      serialized.faceAttributes.colorCornerFlags!.buffer,
      serialized.faceAttributes.colorCorners!.buffer,
      serialized.faceAttributes.colorItemSizes!.buffer,
      serialized.materials[0]!.baseColorTexture!.image!.data.buffer,
      serialized.materials[0]!.textureSlots[1]!.image!.data.buffer,
    ];
    expect(new Set(buffers)).toEqual(new Set(expectedBuffers));
  });

  it('serializes simplified geometry previews without provenance', () => {
    const result: RawSimplificationResult = {
      rawMesh: rawMesh(),
      outputFaceIds: [123],
      history: [{
        keepVertexId: 0,
        removedVertexId: 1,
        beforeFaces: [{ faceId: 123, vertices: [0, 1, 2], positions: rawMesh().positions as [Vector3, Vector3, Vector3] }],
        afterFaceIds: [456],
      }],
      stats: {
        inputVertices: 3,
        inputFaces: 1,
        outputVertices: 3,
        outputFaces: 1,
        physicalEdges: 3,
        virtualEdges: 0,
        collapses: 0,
        stoppedReason: 'target-reached',
      },
    };

    const serialized = serializeSimplifiedGeometryResult(result);
    expect('history' in serialized).toBe(false);
    expect('outputFaceIds' in serialized).toBe(false);
    expect(serialized.rawMesh.indices).toEqual(new Uint32Array([0, 1, 2]));
  });

  it('packs full raw simplification result history for primitive processing', () => {
    const originalRawMesh = rawMesh();
    const result: RawSimplificationResult = {
      rawMesh: originalRawMesh,
      outputFaceIds: [7],
      history: [{
        keepVertexId: 0,
        removedVertexId: 1,
        beforeFaces: [{
          faceId: 3,
          vertices: [0, 1, 2],
          positions: originalRawMesh.positions as [Vector3, Vector3, Vector3],
        }],
        afterFaceIds: [7],
      }],
      stats: {
        inputVertices: 3,
        inputFaces: 1,
        outputVertices: 3,
        outputFaces: 1,
        physicalEdges: 3,
        virtualEdges: 0,
        collapses: 1,
        stoppedReason: 'target-reached',
      },
    };

    const serialized = serializeFullRawSimplificationResult(result);
    expect(serialized.history.keepVertexIds).toEqual(new Uint32Array([0]));
    expect(serialized.history.beforeFacePositions).toHaveLength(9);
    expect(new Set(collectTransferables(serialized))).toEqual(new Set([
      serialized.rawMesh.positions.buffer,
      serialized.rawMesh.indices.buffer,
      serialized.outputFaceIds.buffer,
      serialized.history.keepVertexIds.buffer,
      serialized.history.removedVertexIds.buffer,
      serialized.history.beforeFaceOffsets.buffer,
      serialized.history.beforeFaceIds.buffer,
      serialized.history.beforeFaceVertices.buffer,
      serialized.history.beforeFacePositions.buffer,
      serialized.history.afterFaceOffsets.buffer,
      serialized.history.afterFaceIds.buffer,
    ]));

    const restored = deserializeFullRawSimplificationResult(serialized);
    expect(restored.outputFaceIds).toEqual([7]);
    expect(restored.history[0]!.beforeFaces[0]!.positions[1].x).toBe(1);
    expect(restored.rawMesh.faces).toEqual(originalRawMesh.faces);
  });

  it('serializes transferred vertex attributes on primitive geometry results', () => {
    const source = textured();
    const result = {
      entries: [{
        id: 'primitive-0',
        source: {
          id: 'primitive-0',
          rawMesh: source.rawMesh,
          texturedRawMesh: source,
          bakeable: false,
          requiresAttributeTransfer: true,
        },
        geometry: {
          raw: {
            rawMesh: source.rawMesh,
            outputFaceIds: [0],
            history: [],
            stats: {
              inputVertices: 3,
              inputFaces: 1,
              outputVertices: 3,
              outputFaces: 1,
              physicalEdges: 3,
              virtualEdges: 0,
              collapses: 0,
              stoppedReason: 'target-reached' as const,
            },
          },
          elapsedSeconds: 0,
        },
        raw: {
          rawMesh: source.rawMesh,
          outputFaceIds: [0],
          history: [],
          stats: {
            inputVertices: 3,
            inputFaces: 1,
            outputVertices: 3,
            outputFaces: 1,
            physicalEdges: 3,
            virtualEdges: 0,
            collapses: 0,
            stoppedReason: 'target-reached' as const,
          },
        },
        transferredAttributes: transferredAttributes(),
      }],
      stats: {
        inputVertices: 3,
        inputFaces: 1,
        outputVertices: 3,
        outputFaces: 1,
        physicalEdges: 3,
        virtualEdges: 0,
        collapses: 0,
        stoppedReason: 'target-reached' as const,
      },
      elapsedSeconds: 0,
    };

    const serialized = serializePrimitiveGeometryProcessingResult(result, [{ id: 'primitive-0', meshOrdinal: 7 }]);

    expect(serialized.entries[0]!.transferredAttributes?.uvSetCounts).toEqual(new Uint16Array([2, 1, 1]));
    expect(serialized.entries[0]!.transferredAttributes?.uvSetTexCoords).toEqual(new Uint16Array([0, 1, 0, 0]));
  });
});
