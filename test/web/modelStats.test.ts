import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import { formatBytes, formatProcessingCompleteStatus, formatScalar, inputModelStatItems, processedOutputStatItems, summarizeProcessingResult, summarizeRawMesh, summarizeTexturedRawMesh } from '../../src/web/modelStats';
import type { TexturedRawMesh } from '../../src/texture/types';

function source(): TexturedRawMesh {
  return {
    rawMesh: {
      positions: [new Vector3(), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2]],
    },
    faceAttributes: [{ materialId: 0, uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }] }],
    materials: [{
      name: 'mat',
      baseColorFactor: [1, 1, 1, 1],
      baseColorTexture: {
        image: { width: 4, height: 8, data: new Uint8Array(4 * 8 * 4) },
        sampler: { wrapS: 'clamp', wrapT: 'clamp', filter: 'linear' },
        texCoord: 0,
      },
      textureSlots: [{ slot: 'baseColor', texCoord: 0, sampler: { wrapS: 'clamp', wrapT: 'clamp', filter: 'linear' }, hasImage: true }],
      alphaMode: 'OPAQUE',
      alphaCutoff: 0.5,
      doubleSided: false,
      emissiveFactor: [0, 0, 0],
      metallicFactor: 1,
      roughnessFactor: 1,
      normalScale: 1,
      occlusionStrength: 1,
    }],
  };
}

describe('model stats', () => {
  it('formats scalar values as copyable numeric strings without grouping separators', () => {
    expect(formatScalar(1234.5)).toBe('1234.5');
    expect(formatScalar(0.0003426009354227017)).toBe('0.000342601');
  });

  it('summarizes raw and textured meshes', () => {
    expect(summarizeRawMesh(source().rawMesh)).toEqual({
      vertices: 3,
      faces: 1,
    });
    expect(summarizeTexturedRawMesh(source())).toEqual({
      vertices: 3,
      faces: 1,
      materials: 1,
      materialsWithTextures: 1,
      materialsWithBaseColorImages: 1,
      facesWithUvs: 1,
      textureSlotKinds: ['baseColor'],
      textureDimensions: ['4×8'],
    });
  });

  it('includes grid spacing in input model stat items', () => {
    const summary = summarizeTexturedRawMesh(source());

    expect(inputModelStatItems(summary, 'single mesh-node path captured', 100).map(([label, value]) => [label, value])).toEqual([
      ['Vertices', 3],
      ['Faces', 1],
      ['Grid spacing', '100'],
      ['Materials', 1],
      ['Materials with textures', 1],
      ['Materials with base-color textures', 1],
      ['Material texture slots', 'baseColor'],
      ['Faces with UVs', 1],
      ['Base-color texture dimensions', '4×8'],
      ['Transform preservation', 'single mesh-node path captured'],
    ]);
  });

  it('counts only base-color images in base-color image stats', () => {
    const mesh = source();
    mesh.materials.push({
      name: 'normal-only',
      baseColorFactor: [1, 1, 1, 1],
      textureSlots: [{
        slot: 'normal',
        texCoord: 0,
        sampler: { wrapS: 'clamp', wrapT: 'clamp', filter: 'linear' },
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
    });

    expect(summarizeTexturedRawMesh(mesh)).toMatchObject({
      materialsWithTextures: 2,
      materialsWithBaseColorImages: 1,
      textureSlotKinds: ['baseColor', 'normal'],
    });
  });

  it('summarizes processing outputs and byte sizes', () => {
    expect(formatBytes(1536)).toBe('1.5 KB');
    const summary = summarizeProcessingResult({
      stats: {
        inputVertices: 5,
        inputFaces: 3,
        outputVertices: 3,
        outputFaces: 1,
        physicalEdges: 4,
        virtualEdges: 2,
        collapses: 2,
        stoppedReason: 'target-reached',
      },
      bake: { filledPixels: 10, mappedPixels: 10, unmappedPixels: 0, islandCount: 1, outputVertices: 6 },
      exportedBytes: 2048,
      elapsedSeconds: 0.25,
    });

    expect(summary).toMatchObject({
      inputFaces: 3,
      outputFaces: 1,
      bakeIslandCount: 1,
      bakeOutputVertices: 6,
      exportedBytesLabel: '2.0 KB',
    });
    expect(processedOutputStatItems(summary).map(([label]) => label)).toEqual([
      'Output vertices',
      'Output faces',
      'Baked output vertices',
      'Collapses',
      'Physical edges',
      'Virtual edges',
      'Stopped reason',
      'Elapsed seconds',
      'Atlas islands',
      'Baked mapped pixels',
      'Baked unmapped pixels',
      'Exported GLB size',
    ]);
  });

  it('formats bake completion status with serialized baked vertex counts', () => {
    const stats = {
      inputVertices: 5,
      inputFaces: 3,
      outputVertices: 3,
      outputFaces: 1,
      physicalEdges: 4,
      virtualEdges: 2,
      collapses: 2,
      stoppedReason: 'target-reached' as const,
    };

    expect(formatProcessingCompleteStatus('bake', stats, { outputVertices: 6 }))
      .toBe('Texture atlas baking complete: 1 faces, 6 vertices.');
    expect(formatProcessingCompleteStatus('simplify', stats, { outputVertices: 6 }))
      .toBe('Geometry simplification complete: 1 faces, 3 vertices.');
  });
});
