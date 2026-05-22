import { Vector2, Vector3 } from 'three';
import type { RawSimplificationResult } from '../simplification/types';
import type { TextureBakeBatch, TextureBakeBatchRunInput } from '../texture/bakeBatch';
import type { AtlasLayout, StandardBakedTextureSlot } from '../texture/types';
import {
  deserializeFullRawSimplificationResult,
  deserializeTexturedRawMesh,
  serializeFullRawSimplificationResult,
  serializeTexturedRawMesh,
  type SerializedFullRawSimplificationResult,
  type SerializedTexturedRawMesh,
} from './primitiveSerialization';

export interface SerializedAtlasLayout {
  textureSize: number;
  padding: number;
  faceUvs: Float32Array;
  facePixelTriangles: Float32Array;
  islandCount?: number;
}

export interface SerializedTextureBakeBatchRunInput {
  source: SerializedTexturedRawMesh;
  output: SerializedFullRawSimplificationResult;
  atlas: SerializedAtlasLayout;
  activeSlots: StandardBakedTextureSlot[];
  outputNormalScale?: number;
  outputVertexNormals?: Float32Array;
  batches: TextureBakeBatch[];
  totalFaces: number;
  totalSamples: number;
}

export function serializeAtlas(atlas: AtlasLayout): SerializedAtlasLayout {
  const faceUvs = new Float32Array(atlas.faceUvs.length * 6);
  const facePixelTriangles = new Float32Array(atlas.facePixelTriangles.length * 6);
  atlas.faceUvs.forEach((uvs, faceIndex) => {
    const base = faceIndex * 6;
    faceUvs[base] = uvs[0].x;
    faceUvs[base + 1] = uvs[0].y;
    faceUvs[base + 2] = uvs[1].x;
    faceUvs[base + 3] = uvs[1].y;
    faceUvs[base + 4] = uvs[2].x;
    faceUvs[base + 5] = uvs[2].y;
  });
  atlas.facePixelTriangles.forEach((triangle, faceIndex) => {
    const base = faceIndex * 6;
    facePixelTriangles[base] = triangle[0][0];
    facePixelTriangles[base + 1] = triangle[0][1];
    facePixelTriangles[base + 2] = triangle[1][0];
    facePixelTriangles[base + 3] = triangle[1][1];
    facePixelTriangles[base + 4] = triangle[2][0];
    facePixelTriangles[base + 5] = triangle[2][1];
  });
  return {
    textureSize: atlas.textureSize,
    padding: atlas.padding,
    faceUvs,
    facePixelTriangles,
    ...(atlas.islandCount !== undefined ? { islandCount: atlas.islandCount } : {}),
  };
}

export function deserializeAtlas(serialized: SerializedAtlasLayout): AtlasLayout {
  const faceUvs: AtlasLayout['faceUvs'] = [];
  const facePixelTriangles: AtlasLayout['facePixelTriangles'] = [];
  for (let i = 0; i < serialized.faceUvs.length; i += 6) {
    faceUvs.push([
      new Vector2(serialized.faceUvs[i]!, serialized.faceUvs[i + 1]!),
      new Vector2(serialized.faceUvs[i + 2]!, serialized.faceUvs[i + 3]!),
      new Vector2(serialized.faceUvs[i + 4]!, serialized.faceUvs[i + 5]!),
    ]);
  }
  for (let i = 0; i < serialized.facePixelTriangles.length; i += 6) {
    facePixelTriangles.push([
      [serialized.facePixelTriangles[i]!, serialized.facePixelTriangles[i + 1]!],
      [serialized.facePixelTriangles[i + 2]!, serialized.facePixelTriangles[i + 3]!],
      [serialized.facePixelTriangles[i + 4]!, serialized.facePixelTriangles[i + 5]!],
    ]);
  }
  return {
    textureSize: serialized.textureSize,
    padding: serialized.padding,
    faceUvs,
    facePixelTriangles,
    ...(serialized.islandCount !== undefined ? { islandCount: serialized.islandCount } : {}),
  };
}

function serializeVector3Array(values: readonly Vector3[]): Float32Array {
  const serialized = new Float32Array(values.length * 3);
  values.forEach((value, index) => {
    const offset = index * 3;
    serialized[offset] = value.x;
    serialized[offset + 1] = value.y;
    serialized[offset + 2] = value.z;
  });
  return serialized;
}

function deserializeVector3Array(values: Float32Array): Vector3[] {
  const vectors: Vector3[] = [];
  for (let index = 0; index + 2 < values.length; index += 3) {
    vectors.push(new Vector3(values[index]!, values[index + 1]!, values[index + 2]!));
  }
  return vectors;
}

function textureBakeOutputResult(input: TextureBakeBatchRunInput): RawSimplificationResult {
  return {
    rawMesh: input.outputRawMesh,
    outputFaceIds: input.outputFaceIds,
    history: input.history,
    stats: {
      inputVertices: input.source.rawMesh.positions.length,
      inputFaces: input.source.rawMesh.faces.length,
      outputVertices: input.outputRawMesh.positions.length,
      outputFaces: input.outputRawMesh.faces.length,
      physicalEdges: 0,
      virtualEdges: 0,
      collapses: input.history.length,
      stoppedReason: 'target-reached',
    },
  };
}

export function serializeTextureBakeBatchRunInput(input: TextureBakeBatchRunInput): SerializedTextureBakeBatchRunInput {
  return {
    source: serializeTexturedRawMesh(input.source),
    output: serializeFullRawSimplificationResult(textureBakeOutputResult(input)),
    atlas: serializeAtlas(input.atlas),
    activeSlots: [...input.activeSlots],
    ...(input.outputNormalScale !== undefined ? { outputNormalScale: input.outputNormalScale } : {}),
    ...(input.outputVertexNormals !== undefined ? { outputVertexNormals: serializeVector3Array(input.outputVertexNormals) } : {}),
    batches: input.batches.map((batch) => ({ ...batch })),
    totalFaces: input.totalFaces,
    totalSamples: input.totalSamples,
  };
}

export function deserializeTextureBakeBatchRunInput(input: SerializedTextureBakeBatchRunInput): TextureBakeBatchRunInput {
  const output = deserializeFullRawSimplificationResult(input.output);
  return {
    source: deserializeTexturedRawMesh(input.source),
    outputRawMesh: output.rawMesh,
    outputFaceIds: output.outputFaceIds,
    history: output.history,
    atlas: deserializeAtlas(input.atlas),
    activeSlots: [...input.activeSlots],
    ...(input.outputNormalScale !== undefined ? { outputNormalScale: input.outputNormalScale } : {}),
    ...(input.outputVertexNormals !== undefined ? { outputVertexNormals: deserializeVector3Array(input.outputVertexNormals) } : {}),
    batches: input.batches.map((batch) => ({ ...batch })),
    totalFaces: input.totalFaces,
    totalSamples: input.totalSamples,
  };
}
