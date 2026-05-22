import { describe, expect, it } from 'vitest';
import { Vector2, Vector3, Vector4 } from 'three';
import type { TransferredMeshAttributes } from '../../src/simplification/attributes';
import type { RawMesh } from '../../src/simplification/types';
import type { AtlasLayout } from '../../src/texture/types';
import {
  buildAttributeTransferredPrimitiveGeometryData,
  buildAtlasPrimitiveGeometryData,
  buildIndexedPrimitiveGeometryData,
  indexArrayForVertexCount,
} from '../../src/pipeline/primitiveOutputGeometry';

function angledSeamMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(0, 1, 0),
      new Vector3(0, 0, 1),
    ],
    faces: [[0, 1, 2], [0, 2, 3]],
  };
}

function weldedTransferredAttributes(): TransferredMeshAttributes {
  return {
    colorItemSize: 4,
    vertices: [
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(0.1, 0.2) }],
        normal: new Vector3(1, 0, 0),
        color: new Vector4(1, 0, 0, 1),
      },
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(0.9, 0.2) }],
        normal: new Vector3(0, 1, 0),
        color: new Vector4(0, 1, 0, 0.75),
      },
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(0.1, 0.9) }],
        normal: new Vector3(0, 0, 1),
        color: new Vector4(0, 0, 1, 0.5),
      },
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(0.8, 0.8) }],
        normal: new Vector3(0, -1, 0),
        color: new Vector4(1, 1, 1, 0.25),
      },
    ],
  };
}

function transferredTangentsWithoutUv0(): TransferredMeshAttributes {
  return {
    vertices: [
      {
        uvSets: [{ texCoord: 1, uv: new Vector2(0.1, 0.2) }],
        tangent: new Vector4(1, 0, 0, 0.5),
      },
      {
        uvSets: [{ texCoord: 1, uv: new Vector2(0.9, 0.2) }],
        tangent: new Vector4(0, 1, 0, -0.25),
      },
      {
        uvSets: [{ texCoord: 1, uv: new Vector2(0.1, 0.9) }],
        tangent: new Vector4(0, 0, 1, 1),
      },
      {
        uvSets: [{ texCoord: 1, uv: new Vector2(0.8, 0.8) }],
        tangent: new Vector4(-1, 0, 0, -1),
      },
    ],
  };
}

function transferredWrongTangentsWithUv0(): TransferredMeshAttributes {
  return {
    hasSourceTangents: true,
    vertices: [
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(0, 0) }],
        normal: new Vector3(0, 0, 1),
        tangent: new Vector4(0, 1, 0, -1),
      },
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(1, 0) }],
        normal: new Vector3(0, 0, 1),
        tangent: new Vector4(0, 1, 0, -1),
      },
      {
        uvSets: [{ texCoord: 0, uv: new Vector2(0, 1) }],
        normal: new Vector3(0, 0, 1),
        tangent: new Vector4(0, 1, 0, -1),
      },
    ],
  };
}

function seamAtlas(): AtlasLayout {
  return {
    textureSize: 64,
    padding: 2,
    faceUvs: [
      [new Vector2(0, 0), new Vector2(0.5, 0), new Vector2(0, 0.5)],
      [new Vector2(0.75, 0.75), new Vector2(0.75, 1), new Vector2(1, 0.75)],
    ],
    facePixelTriangles: [
      [[0, 0], [32, 0], [0, 32]],
      [[48, 48], [48, 63], [63, 48]],
    ],
  };
}

describe('primitive output geometry builders', () => {
  it('indexed geometry returns one position and normal tuple per raw vertex', () => {
    const rawMesh: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2]],
    };

    const data = buildIndexedPrimitiveGeometryData(rawMesh);

    expect(Array.from(data.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(data.normals)).toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    expect(Array.from(data.indices)).toEqual([0, 1, 2]);
    expect(data.texCoordsBySet.size).toBe(0);
  });

  it('attribute-transferred geometry keeps one serialized vertex per welded raw vertex', () => {
    const data = buildAttributeTransferredPrimitiveGeometryData(angledSeamMesh(), weldedTransferredAttributes(), {
      requiredTexCoords: [0],
    });

    expect(data.positions.length / 3).toBe(4);
    expect(Array.from(data.indices)).toEqual([0, 1, 2, 0, 2, 3]);
    expect(Array.from(data.texCoordsBySet.keys())).toEqual([0]);
    expect(data.texCoordsBySet.get(0)?.length).toBe(8);
    expect(Array.from(data.normals.slice(0, 12))).toEqual([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      0, -1, 0,
    ]);
    expect(data.colorItemSize).toBe(4);
    expect(Array.from(data.colors ?? [])).toEqual([
      1, 0, 0, 1,
      0, 1, 0, 0.75,
      0, 0, 1, 0.5,
      1, 1, 1, 0.25,
    ]);
  });

  it('drops transferred colors unless every welded output vertex has COLOR_0', () => {
    const attributes = weldedTransferredAttributes();
    delete attributes.vertices[3]!.color;

    const data = buildAttributeTransferredPrimitiveGeometryData(angledSeamMesh(), attributes, {
      requiredTexCoords: [0],
    });

    expect(data.colors).toBeUndefined();
    expect(data.colorItemSize).toBeUndefined();
  });

  it('does not copy transferred source tangents into final geometry', () => {
    const data = buildAttributeTransferredPrimitiveGeometryData(angledSeamMesh(), transferredTangentsWithoutUv0());

    expect(Array.from(data.texCoordsBySet.keys())).toEqual([1]);
    expect(data.tangents).toBeUndefined();
  });

  it('omits requested derived tangents when TEXCOORD_0 is unavailable', () => {
    const data = buildAttributeTransferredPrimitiveGeometryData(angledSeamMesh(), transferredTangentsWithoutUv0(), {
      emitTangents: true,
    });

    expect(Array.from(data.texCoordsBySet.keys())).toEqual([1]);
    expect(data.tangents).toBeUndefined();
  });

  it('derives final tangents from output geometry instead of transferred source tangent values', () => {
    const rawMesh: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2]],
    };

    const data = buildAttributeTransferredPrimitiveGeometryData(rawMesh, transferredWrongTangentsWithUv0(), {
      emitTangents: true,
    });

    expect(data.tangents).toBeDefined();
    expect(data.tangents![0]).toBeCloseTo(1);
    expect(data.tangents![1]).toBeCloseTo(0);
    expect(data.tangents![2]).toBeCloseTo(0);
    expect(data.tangents![3]).toBeCloseTo(1);
  });

  it('scales only derived tangent handedness when requested for attribute-transferred geometry', () => {
    const rawMesh: RawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2]],
    };

    const unscaled = buildAttributeTransferredPrimitiveGeometryData(rawMesh, transferredWrongTangentsWithUv0(), {
      emitTangents: true,
    });
    const scaled = buildAttributeTransferredPrimitiveGeometryData(rawMesh, transferredWrongTangentsWithUv0(), {
      emitTangents: true,
      tangentHandednessScale: -1,
    });

    expect(unscaled.tangents).toBeDefined();
    expect(scaled.tangents).toBeDefined();
    expect(scaled.tangents?.length).toBe(unscaled.tangents?.length);
    for (let offset = 0; offset < scaled.tangents!.length; offset += 4) {
      expect(scaled.tangents![offset]).toBeCloseTo(unscaled.tangents![offset]!);
      expect(scaled.tangents![offset + 1]).toBeCloseTo(unscaled.tangents![offset + 1]!);
      expect(scaled.tangents![offset + 2]).toBeCloseTo(unscaled.tangents![offset + 2]!);
      expect(scaled.tangents![offset + 3]).toBeCloseTo(-unscaled.tangents![offset + 3]!);
    }
  });

  it('atlas geometry duplicates vertices by source vertex and atlas UV tuple', () => {
    const data = buildAtlasPrimitiveGeometryData(angledSeamMesh(), seamAtlas(), [
      { slot: 'normal', image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) } },
    ]);

    expect(data.positions.length / 3).toBe(6);
    expect(Array.from(data.indices)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(data.texCoordsBySet.keys())).toEqual([0]);
    expect(Array.from(data.texCoordsBySet.get(0) ?? [])).toEqual([
      0, 0,
      0.5, 0,
      0, 0.5,
      0.75, 0.75,
      0.75, 1,
      1, 0.75,
    ]);
    expect(data.tangents?.length).toBe((data.positions.length / 3) * 4);
  });

  it('atlas geometry copies transferred COLOR_0 onto atlas seam duplicates', () => {
    const data = buildAtlasPrimitiveGeometryData(
      angledSeamMesh(),
      seamAtlas(),
      undefined,
      weldedTransferredAttributes(),
    );

    expect(data.positions.length / 3).toBe(6);
    expect(data.colorItemSize).toBe(4);
    expect(Array.from(data.colors ?? [])).toEqual([
      1, 0, 0, 1,
      0, 1, 0, 0.75,
      0, 0, 1, 0.5,
      1, 0, 0, 1,
      0, 0, 1, 0.5,
      1, 1, 1, 0.25,
    ]);
  });

  it('atlas geometry emits generated tangents when source tangent provenance exists without a baked normal map', () => {
    const attributes = weldedTransferredAttributes();
    attributes.hasSourceTangents = true;
    const data = buildAtlasPrimitiveGeometryData(
      angledSeamMesh(),
      seamAtlas(),
      undefined,
      attributes,
    );

    expect(data.tangents?.length).toBe((data.positions.length / 3) * 4);
  });

  it('atlas geometry omits COLOR_0 when transferred colors are incomplete', () => {
    const attributes = weldedTransferredAttributes();
    delete attributes.vertices[3]!.color;

    const data = buildAtlasPrimitiveGeometryData(
      angledSeamMesh(),
      seamAtlas(),
      undefined,
      attributes,
    );

    expect(data.colors).toBeUndefined();
    expect(data.colorItemSize).toBeUndefined();
  });

  it('atlas geometry omits COLOR_0 when transferred vertex count does not match output mesh', () => {
    const attributes = weldedTransferredAttributes();
    attributes.vertices.pop();

    const data = buildAtlasPrimitiveGeometryData(
      angledSeamMesh(),
      seamAtlas(),
      undefined,
      attributes,
    );

    expect(data.colors).toBeUndefined();
    expect(data.colorItemSize).toBeUndefined();
  });

  it('atlas geometry copies prepared normals onto atlas seam duplicates and emits generated tangents for normal maps', () => {
    const data = buildAtlasPrimitiveGeometryData(angledSeamMesh(), seamAtlas(), [
      { slot: 'normal', image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) } },
    ], weldedTransferredAttributes());

    const originVertexIds: number[] = [];
    for (let vertexId = 0; vertexId < data.positions.length / 3; vertexId += 1) {
      const offset = vertexId * 3;
      if (data.positions[offset] === 0 && data.positions[offset + 1] === 0 && data.positions[offset + 2] === 0) {
        originVertexIds.push(vertexId);
      }
    }

    expect(originVertexIds).toHaveLength(2);
    for (const vertexId of originVertexIds) {
      expect(data.normals[vertexId * 3]).toBeCloseTo(1);
      expect(data.normals[vertexId * 3 + 1]).toBeCloseTo(0);
      expect(data.normals[vertexId * 3 + 2]).toBeCloseTo(0);
    }
    expect(data.tangents?.length).toBe((data.positions.length / 3) * 4);
  });

  it('uses Uint16 indices up to 65535 vertices and Uint32 indices above the threshold', () => {
    expect(indexArrayForVertexCount(65535, [0, 1, 2])).toBeInstanceOf(Uint16Array);
    expect(indexArrayForVertexCount(65536, [0, 1, 65535])).toBeInstanceOf(Uint32Array);
  });
});
