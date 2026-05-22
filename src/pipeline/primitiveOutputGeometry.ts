import { Vector2, Vector3, Vector4 } from 'three';
import type { TransferredMeshAttributes, TransferredVertexAttributes, VertexColorItemSize } from '../simplification/attributes';
import { transferredVertexUv } from '../simplification/attributes';
import type { AtlasLayout, BakedMaterialTexture, SourceFaceAttributes, SourceFaceUvSet } from '../texture/types';
import { computeSerializedTangents } from '../texture/tangentSpace';
import { computeAreaWeightedVertexNormals } from '../simplification/normals';
import type { RawMesh } from '../simplification/types';

export interface PrimitiveGeometryData {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint16Array | Uint32Array;
  texCoordsBySet: Map<number, Float32Array>;
  tangents?: Float32Array;
  colors?: Float32Array;
  colorItemSize?: VertexColorItemSize;
}

export interface UvSplitPrimitiveGeometryOptions {
  texCoords?: readonly number[];
}

export interface AttributeTransferredPrimitiveGeometryOptions {
  requiredTexCoords?: readonly number[];
  emitTangents?: boolean;
  tangentHandednessScale?: 1 | -1;
}

export function indexArrayForVertexCount(vertexCount: number, indices: readonly number[]): Uint16Array | Uint32Array {
  return vertexCount > 65535 ? new Uint32Array(indices) : new Uint16Array(indices);
}

function uvKey(value: number): string {
  return String(Math.round(value * 1e9));
}

function commonTexCoords(faceAttributes: readonly SourceFaceAttributes[]): number[] {
  if (faceAttributes.length === 0) return [];
  const common = new Set(faceAttributes[0]!.uvSets.map((set) => set.texCoord));
  for (const attributes of faceAttributes.slice(1)) {
    const available = new Set(attributes.uvSets.map((set) => set.texCoord));
    for (const texCoord of Array.from(common)) {
      if (!available.has(texCoord)) common.delete(texCoord);
    }
  }
  return Array.from(common).sort((a, b) => a - b);
}

function sortedUniqueTexCoords(texCoords: readonly number[]): number[] {
  return Array.from(new Set(texCoords)).sort((a, b) => a - b);
}

function sourceUvSet(attributes: SourceFaceAttributes, texCoord: number): SourceFaceUvSet | undefined {
  return attributes.uvSets.find((set) => set.texCoord === texCoord);
}

function transferredTexCoords(attributes: TransferredMeshAttributes): number[] {
  if (attributes.vertices.length === 0) return [];
  const common = new Set(attributes.vertices[0]!.uvSets.map((set) => set.texCoord));
  for (const vertex of attributes.vertices.slice(1)) {
    const available = new Set(vertex.uvSets.map((set) => set.texCoord));
    for (const texCoord of Array.from(common)) {
      if (!available.has(texCoord)) common.delete(texCoord);
    }
  }
  return Array.from(common).sort((a, b) => a - b);
}

function missingTransferredUvVertex(
  vertices: readonly TransferredVertexAttributes[],
  texCoord: number,
): number {
  const index = vertices.findIndex((vertex) => !transferredVertexUv(vertex, texCoord));
  return index >= 0 ? index : 0;
}

function vector3Values(values: ArrayLike<number>): Vector3[] {
  const vectors: Vector3[] = [];
  for (let i = 0; i + 2 < values.length; i += 3) {
    vectors.push(new Vector3(values[i]!, values[i + 1]!, values[i + 2]!));
  }
  return vectors;
}

function vector2Values(values: ArrayLike<number>): Vector2[] {
  const vectors: Vector2[] = [];
  for (let i = 0; i + 1 < values.length; i += 2) {
    vectors.push(new Vector2(values[i]!, values[i + 1]!));
  }
  return vectors;
}

function hasBakedNormalMap(textures: readonly BakedMaterialTexture[] | undefined): boolean {
  return textures?.some((texture) => texture.slot === 'normal') ?? false;
}

function geometryDataFromArrays(
  positions: readonly number[],
  normals: readonly number[],
  indices: readonly number[],
  texCoordsBySet: Map<number, readonly number[]>,
  tangents?: Float32Array,
  colors?: Float32Array,
  colorItemSize?: VertexColorItemSize,
): PrimitiveGeometryData {
  const outputTexCoords = new Map<number, Float32Array>();
  for (const [texCoord, values] of texCoordsBySet) {
    outputTexCoords.set(texCoord, new Float32Array(values));
  }
  const data: PrimitiveGeometryData = {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: indexArrayForVertexCount(positions.length / 3, indices),
    texCoordsBySet: outputTexCoords,
  };
  if (tangents) data.tangents = tangents;
  if (colors && colorItemSize) {
    data.colors = colors;
    data.colorItemSize = colorItemSize;
  }
  return data;
}

export function buildIndexedPrimitiveGeometryData(rawMesh: RawMesh): PrimitiveGeometryData {
  const vertexNormals = computeAreaWeightedVertexNormals(rawMesh);
  const positions = new Float32Array(rawMesh.positions.length * 3);
  const normals = new Float32Array(rawMesh.positions.length * 3);
  for (let index = 0; index < rawMesh.positions.length; index += 1) {
    const position = rawMesh.positions[index]!;
    const normal = vertexNormals[index] ?? new Vector3(0, 0, 1);
    positions[index * 3] = position.x;
    positions[index * 3 + 1] = position.y;
    positions[index * 3 + 2] = position.z;
    normals[index * 3] = normal.x;
    normals[index * 3 + 1] = normal.y;
    normals[index * 3 + 2] = normal.z;
  }
  const indices: number[] = [];
  for (const face of rawMesh.faces) indices.push(face[0], face[1], face[2]);
  return {
    positions,
    normals,
    indices: indexArrayForVertexCount(rawMesh.positions.length, indices),
    texCoordsBySet: new Map(),
  };
}

export function buildUvSplitPrimitiveGeometryData(
  rawMesh: RawMesh,
  faceAttributes: readonly SourceFaceAttributes[],
  options: UvSplitPrimitiveGeometryOptions = {},
): PrimitiveGeometryData {
  if (faceAttributes.length !== rawMesh.faces.length) {
    throw new Error('Transferred face attribute count must match output mesh face count.');
  }

  const texCoords = options.texCoords === undefined
    ? commonTexCoords(faceAttributes)
    : sortedUniqueTexCoords(options.texCoords);
  const sourceVertexNormals = computeAreaWeightedVertexNormals(rawMesh);
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const texCoordsBySet = new Map(texCoords.map((texCoord) => [texCoord, [] as number[]]));
  const vertexMap = new Map<string, number>();

  for (let faceIndex = 0; faceIndex < rawMesh.faces.length; faceIndex += 1) {
    const face = rawMesh.faces[faceIndex]!;
    const attributes = faceAttributes[faceIndex]!;
    for (let corner = 0; corner < 3; corner += 1) {
      const sourceVertexId = face[corner]!;
      const uvValues = texCoords.map((texCoord) => {
        const uv = sourceUvSet(attributes, texCoord)?.uvs[corner];
        if (!uv) throw new Error(`Missing TEXCOORD_${texCoord} for output face ${faceIndex}.`);
        return { texCoord, uv };
      });
      const key = [
        String(sourceVertexId),
        ...uvValues.map(({ texCoord, uv }) => `${texCoord}:${uvKey(uv.x)}:${uvKey(uv.y)}`),
      ].join('/');
      let outputVertexId = vertexMap.get(key);
      if (outputVertexId === undefined) {
        outputVertexId = vertexMap.size;
        vertexMap.set(key, outputVertexId);
        const position = rawMesh.positions[sourceVertexId];
        if (!position) throw new Error(`Output face ${faceIndex} references missing vertex ${sourceVertexId}.`);
        const normal = sourceVertexNormals[sourceVertexId] ?? new Vector3(0, 0, 1);
        positions.push(position.x, position.y, position.z);
        normals.push(normal.x, normal.y, normal.z);
        for (const { texCoord, uv } of uvValues) texCoordsBySet.get(texCoord)!.push(uv.x, uv.y);
      }
      indices.push(outputVertexId);
    }
  }

  return geometryDataFromArrays(positions, normals, indices, texCoordsBySet);
}

export function buildAttributeTransferredPrimitiveGeometryData(
  rawMesh: RawMesh,
  attributes: TransferredMeshAttributes,
  options: AttributeTransferredPrimitiveGeometryOptions = {},
): PrimitiveGeometryData {
  if (attributes.vertices.length !== rawMesh.positions.length) {
    throw new Error('Transferred vertex attribute count must match output vertex count.');
  }

  const commonTexCoords = transferredTexCoords(attributes);
  const commonTexCoordSet = new Set(commonTexCoords);
  for (const texCoord of options.requiredTexCoords ?? []) {
    if (!commonTexCoordSet.has(texCoord)) {
      const missingVertex = missingTransferredUvVertex(attributes.vertices, texCoord);
      throw new Error(`Missing transferred TEXCOORD_${texCoord} coordinates for output vertex ${missingVertex}.`);
    }
  }
  const texCoords = sortedUniqueTexCoords([
    ...commonTexCoords,
    ...(options.requiredTexCoords ?? []),
  ]);
  const texCoordsBySet = new Map(texCoords.map((texCoord) => [texCoord, [] as number[]]));
  const fallbackNormals = computeAreaWeightedVertexNormals(rawMesh);
  const positions: number[] = [];
  const normals: number[] = [];
  const includeColors = attributes.vertices.every((vertex) => vertex.color);
  const colorItemSize: VertexColorItemSize = attributes.colorItemSize ?? 3;
  const colors: number[] = [];

  for (let vertexIndex = 0; vertexIndex < rawMesh.positions.length; vertexIndex += 1) {
    const position = rawMesh.positions[vertexIndex]!;
    const transferred = attributes.vertices[vertexIndex]!;
    const normal = transferred.normal ?? fallbackNormals[vertexIndex] ?? new Vector3(0, 0, 1);
    positions.push(position.x, position.y, position.z);
    normals.push(normal.x, normal.y, normal.z);
    for (const texCoord of texCoords) {
      const uv = transferredVertexUv(transferred, texCoord);
      if (!uv) {
        throw new Error(`Missing transferred TEXCOORD_${texCoord} coordinates for output vertex ${vertexIndex}.`);
      }
      texCoordsBySet.get(texCoord)!.push(uv.x, uv.y);
    }
    if (includeColors) {
      const color = transferred.color!;
      colors.push(color.x, color.y, color.z);
      if (colorItemSize === 4) colors.push(color.w);
    }
  }

  const indices: number[] = [];
  for (const face of rawMesh.faces) indices.push(face[0], face[1], face[2]);
  const data = geometryDataFromArrays(
    positions,
    normals,
    indices,
    texCoordsBySet,
    undefined,
    includeColors ? new Float32Array(colors) : undefined,
    includeColors ? colorItemSize : undefined,
  );
  if (options.emitTangents === true) {
    const tangents = computePrimitiveGeometryDataTangents(data);
    if (tangents) {
      if (options.tangentHandednessScale !== undefined) {
        for (let offset = 3; offset < tangents.length; offset += 4) {
          const handedness = tangents[offset];
          if (handedness !== undefined) {
            tangents[offset] = handedness * options.tangentHandednessScale;
          }
        }
      }
      data.tangents = tangents;
    }
  }
  return data;
}

export function computePrimitiveGeometryDataTangents(data: PrimitiveGeometryData): Float32Array | undefined {
  const uvs = data.texCoordsBySet.get(0);
  if (!uvs) return undefined;
  return computeSerializedTangents({
    positions: vector3Values(data.positions),
    normals: vector3Values(data.normals),
    uvs: vector2Values(uvs),
    indices: Array.from(data.indices),
  });
}

function atlasColorForVertex(
  attributes: TransferredMeshAttributes | undefined,
  rawMesh: RawMesh,
  sourceVertexId: number,
): Vector4 | undefined {
  if (!attributes || attributes.vertices.length !== rawMesh.positions.length) return undefined;
  if (!attributes.vertices.every((vertex) => vertex.color)) return undefined;
  return attributes.vertices[sourceVertexId]?.color;
}

function atlasNormalForVertex(
  attributes: TransferredMeshAttributes | undefined,
  rawMesh: RawMesh,
  fallbackNormals: readonly Vector3[],
  sourceVertexId: number,
): Vector3 {
  if (attributes && attributes.vertices.length === rawMesh.positions.length) {
    const normal = attributes.vertices[sourceVertexId]?.normal;
    if (normal && normal.lengthSq() > 0) return normal;
  }
  return fallbackNormals[sourceVertexId] ?? new Vector3(0, 0, 1);
}

export function buildAtlasPrimitiveGeometryData(
  rawMesh: RawMesh,
  atlas: AtlasLayout,
  additionalTextures?: readonly BakedMaterialTexture[],
  transferredAttributes?: TransferredMeshAttributes,
): PrimitiveGeometryData {
  if (atlas.faceUvs.length !== rawMesh.faces.length) throw new Error('Atlas face UV count must match output mesh face count.');

  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const vertexMap = new Map<string, number>();
  const fallbackNormals = computeAreaWeightedVertexNormals(rawMesh);
  const includeColors = transferredAttributes !== undefined
    && transferredAttributes.vertices.length === rawMesh.positions.length
    && transferredAttributes.vertices.every((vertex) => vertex.color);
  const colorItemSize: VertexColorItemSize = transferredAttributes?.colorItemSize ?? 3;

  for (let faceIndex = 0; faceIndex < rawMesh.faces.length; faceIndex += 1) {
    const face = rawMesh.faces[faceIndex]!;
    const faceUvs = atlas.faceUvs[faceIndex]!;
    for (let corner = 0; corner < 3; corner += 1) {
      const sourceVertexId = face[corner]!;
      const uv = faceUvs[corner]!;
      const key = `${sourceVertexId}/${uvKey(uv.x)}/${uvKey(uv.y)}`;
      let outputVertexId = vertexMap.get(key);
      if (outputVertexId === undefined) {
        outputVertexId = vertexMap.size;
        vertexMap.set(key, outputVertexId);
        const position = rawMesh.positions[sourceVertexId];
        if (!position) throw new Error(`Output face ${faceIndex} references missing vertex ${sourceVertexId}.`);
        const normal = atlasNormalForVertex(transferredAttributes, rawMesh, fallbackNormals, sourceVertexId);
        positions.push(position.x, position.y, position.z);
        normals.push(normal.x, normal.y, normal.z);
        uvs.push(uv.x, uv.y);
        if (includeColors) {
          const color = atlasColorForVertex(transferredAttributes, rawMesh, sourceVertexId)!;
          colors.push(color.x, color.y, color.z);
          if (colorItemSize === 4) colors.push(color.w);
        }
      }
      indices.push(outputVertexId);
    }
  }

  const data = geometryDataFromArrays(
    positions,
    normals,
    indices,
    new Map([[0, uvs]]),
    undefined,
    includeColors ? new Float32Array(colors) : undefined,
    includeColors ? colorItemSize : undefined,
  );
  const shouldEmitTangents = hasBakedNormalMap(additionalTextures)
    || transferredAttributes?.hasSourceTangents === true;
  const tangents = shouldEmitTangents
    ? computePrimitiveGeometryDataTangents(data)
    : undefined;
  if (tangents) data.tangents = tangents;
  return data;
}
