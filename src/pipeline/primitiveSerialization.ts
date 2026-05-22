import { Vector2, Vector3, Vector4 } from 'three';
import type {
  CollapseHistoryRecord,
  FaceSnapshot,
  RawMesh,
  RawSimplificationResult,
} from '../simplification/types';
import type { TransferredMeshAttributes } from '../simplification/attributes';
import type {
  RgbaImage,
  SourceFaceAttributes,
  SourceMaterial,
  SourceMaterialTextureInfo,
  SourceTexture,
  TextureSampler,
  TexturedRawMesh,
} from '../texture/types';
import type { ProcessablePrimitiveEntry } from './sceneProcessing';

export interface SerializedRawMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

export interface SerializedRgbaImage {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface SerializedSourceMaterial {
  name: string;
  baseColorFactor: [number, number, number, number];
  baseColorTexture?: SerializedSourceTexture;
  textureSlots: SerializedSourceMaterialTextureInfo[];
  alphaMode: SourceMaterial['alphaMode'];
  alphaCutoff: number;
  doubleSided: boolean;
  emissiveFactor: [number, number, number];
  metallicFactor: number;
  roughnessFactor: number;
  normalScale: number;
  occlusionStrength: number;
}

export interface SerializedSourceMaterialTextureInfo {
  slot: SourceMaterialTextureInfo['slot'];
  texCoord: number;
  sampler: TextureSampler;
  hasImage: boolean;
  image?: SerializedRgbaImage;
  name?: string;
  mimeType?: string;
}

export interface SerializedSourceTexture {
  image?: SerializedRgbaImage;
  sampler: TextureSampler;
  texCoord: number;
  name?: string;
  mimeType?: string;
}

export interface SerializedSourceFaceAttributes {
  materialIds: Int32Array;
  uvSetCounts: Uint16Array;
  uvSetTexCoords: Uint16Array;
  uvs: Float32Array;
  normalCornerFlags?: Uint8Array;
  normalCorners?: Float32Array;
  tangentCornerFlags?: Uint8Array;
  tangentCorners?: Float32Array;
  colorCornerFlags?: Uint8Array;
  colorCorners?: Float32Array;
  colorItemSizes?: Uint8Array;
  normalMapYScales?: Float32Array;
}

export interface SerializedTransferredMeshAttributes {
  uvSetCounts: Uint16Array;
  uvSetTexCoords: Uint16Array;
  uvs: Float32Array;
  normalFlags?: Uint8Array;
  normals?: Float32Array;
  tangentFlags?: Uint8Array;
  tangents?: Float32Array;
  colorFlags?: Uint8Array;
  colors?: Float32Array;
  colorItemSize?: 3 | 4;
  normalMapYScale?: number;
  hasSourceTangents?: boolean;
}

export interface SerializedTexturedRawMesh {
  rawMesh: SerializedRawMesh;
  faceAttributes: SerializedSourceFaceAttributes;
  materials: SerializedSourceMaterial[];
}

export interface SerializedPrimitiveProcessingEntry {
  id: string;
  label: string;
  meshOrdinal: number;
  rawMesh: SerializedRawMesh;
  textured: SerializedTexturedRawMesh;
  bakeable: boolean;
  hasTexturedMaterial: boolean;
  requiresAttributeTransfer?: boolean;
}

export interface SerializedFullCollapseHistory {
  keepVertexIds: Uint32Array;
  removedVertexIds: Uint32Array;
  beforeFaceOffsets: Uint32Array;
  beforeFaceIds: Uint32Array;
  beforeFaceVertices: Uint32Array;
  beforeFacePositions: Float64Array;
  afterFaceOffsets: Uint32Array;
  afterFaceIds: Uint32Array;
}

export interface SerializedFullRawSimplificationResult {
  rawMesh: SerializedRawMesh;
  outputFaceIds: Uint32Array;
  history: SerializedFullCollapseHistory;
  stats: RawSimplificationResult['stats'];
}

export interface SerializationOptions {
  includeImages?: boolean;
}

function toUint8Array(data: RgbaImage['data']): Uint8Array {
  return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

function includesImages(options?: SerializationOptions): boolean {
  return options?.includeImages !== false;
}

export function serializeRawMesh(rawMesh: RawMesh): SerializedRawMesh {
  const positions = new Float32Array(rawMesh.positions.length * 3);
  rawMesh.positions.forEach((position, index) => {
    positions[index * 3] = position.x;
    positions[index * 3 + 1] = position.y;
    positions[index * 3 + 2] = position.z;
  });
  const indices = new Uint32Array(rawMesh.faces.length * 3);
  rawMesh.faces.forEach((face, index) => {
    indices[index * 3] = face[0];
    indices[index * 3 + 1] = face[1];
    indices[index * 3 + 2] = face[2];
  });
  return { positions, indices };
}

export function deserializeRawMesh(serialized: SerializedRawMesh): RawMesh {
  const positions: Vector3[] = [];
  for (let i = 0; i < serialized.positions.length; i += 3) {
    positions.push(new Vector3(serialized.positions[i]!, serialized.positions[i + 1]!, serialized.positions[i + 2]!));
  }
  const faces: RawMesh['faces'] = [];
  for (let i = 0; i < serialized.indices.length; i += 3) {
    faces.push([serialized.indices[i]!, serialized.indices[i + 1]!, serialized.indices[i + 2]!]);
  }
  return { positions, faces };
}

export function serializeRgbaImage(image: RgbaImage): SerializedRgbaImage {
  return { width: image.width, height: image.height, data: toUint8Array(image.data) };
}

export function deserializeRgbaImage(image: SerializedRgbaImage): RgbaImage {
  return { width: image.width, height: image.height, data: image.data };
}

export function serializeSourceTexture(texture: SourceTexture, options?: SerializationOptions): SerializedSourceTexture {
  return {
    ...(includesImages(options) && texture.image ? { image: serializeRgbaImage(texture.image) } : {}),
    sampler: { ...texture.sampler },
    texCoord: texture.texCoord,
    ...(texture.name !== undefined ? { name: texture.name } : {}),
    ...(texture.mimeType !== undefined ? { mimeType: texture.mimeType } : {}),
  };
}

export function deserializeSourceTexture(texture: SerializedSourceTexture): SourceTexture {
  return {
    ...(texture.image ? { image: deserializeRgbaImage(texture.image) } : {}),
    sampler: { ...texture.sampler },
    texCoord: texture.texCoord,
    ...(texture.name !== undefined ? { name: texture.name } : {}),
    ...(texture.mimeType !== undefined ? { mimeType: texture.mimeType } : {}),
  };
}

export function serializeSourceMaterial(material: SourceMaterial, options?: SerializationOptions): SerializedSourceMaterial {
  return {
    name: material.name,
    baseColorFactor: [...material.baseColorFactor],
    ...(material.baseColorTexture ? { baseColorTexture: serializeSourceTexture(material.baseColorTexture, options) } : {}),
    textureSlots: material.textureSlots.map((slot) => ({
      slot: slot.slot,
      texCoord: slot.texCoord,
      sampler: { ...slot.sampler },
      hasImage: slot.hasImage,
      ...(includesImages(options) && slot.image && slot.slot !== 'baseColor' ? { image: serializeRgbaImage(slot.image) } : {}),
      ...(slot.name !== undefined ? { name: slot.name } : {}),
      ...(slot.mimeType !== undefined ? { mimeType: slot.mimeType } : {}),
    })),
    alphaMode: material.alphaMode,
    alphaCutoff: material.alphaCutoff,
    doubleSided: material.doubleSided,
    emissiveFactor: [...material.emissiveFactor],
    metallicFactor: material.metallicFactor,
    roughnessFactor: material.roughnessFactor,
    normalScale: material.normalScale,
    occlusionStrength: material.occlusionStrength,
  };
}

export function deserializeSourceMaterial(material: SerializedSourceMaterial): SourceMaterial {
  return {
    name: material.name,
    baseColorFactor: [...material.baseColorFactor],
    ...(material.baseColorTexture ? { baseColorTexture: deserializeSourceTexture(material.baseColorTexture) } : {}),
    textureSlots: material.textureSlots.map((slot) => ({
      slot: slot.slot,
      texCoord: slot.texCoord,
      sampler: { ...slot.sampler },
      hasImage: slot.hasImage,
      ...(slot.image && slot.slot !== 'baseColor' ? { image: deserializeRgbaImage(slot.image) } : {}),
      ...(slot.name !== undefined ? { name: slot.name } : {}),
      ...(slot.mimeType !== undefined ? { mimeType: slot.mimeType } : {}),
    })),
    alphaMode: material.alphaMode,
    alphaCutoff: material.alphaCutoff,
    doubleSided: material.doubleSided,
    emissiveFactor: [...material.emissiveFactor],
    metallicFactor: material.metallicFactor,
    roughnessFactor: material.roughnessFactor,
    normalScale: material.normalScale,
    occlusionStrength: material.occlusionStrength,
  };
}

export function serializeSourceFaceAttributes(faceAttributes: readonly SourceFaceAttributes[]): SerializedSourceFaceAttributes {
  const materialIds = new Int32Array(faceAttributes.length);
  const uvSetCounts = new Uint16Array(faceAttributes.length);
  const uvSetCount = faceAttributes.reduce((sum, attributes) => sum + attributes.uvSets.length, 0);
  const uvSetTexCoords = new Uint16Array(uvSetCount);
  const uvs = new Float32Array(uvSetCount * 6);
  const normalCornerCount = faceAttributes.reduce((sum, attributes) => sum + (attributes.normalCorners ? 1 : 0), 0);
  const normalCornerFlags = normalCornerCount > 0 ? new Uint8Array(faceAttributes.length) : undefined;
  const normalCorners = normalCornerCount > 0 ? new Float32Array(normalCornerCount * 9) : undefined;
  const tangentCornerCount = faceAttributes.reduce((sum, attributes) => sum + (attributes.tangentCorners ? 1 : 0), 0);
  const tangentCornerFlags = tangentCornerCount > 0 ? new Uint8Array(faceAttributes.length) : undefined;
  const tangentCorners = tangentCornerCount > 0 ? new Float32Array(tangentCornerCount * 12) : undefined;
  const colorCornerCount = faceAttributes.reduce((sum, attributes) => sum + (attributes.colorCorners ? 1 : 0), 0);
  const colorCornerFlags = colorCornerCount > 0 ? new Uint8Array(faceAttributes.length) : undefined;
  const colorCorners = colorCornerCount > 0 ? new Float32Array(colorCornerCount * 12) : undefined;
  const colorItemSizes = colorCornerCount > 0 ? new Uint8Array(faceAttributes.length) : undefined;
  const normalMapYScales = faceAttributes.some((attributes) => attributes.normalMapYScale !== undefined)
    ? new Float32Array(faceAttributes.length)
    : undefined;
  let uvSetIndex = 0;
  let normalCornerIndex = 0;
  let tangentCornerIndex = 0;
  let colorCornerIndex = 0;
  faceAttributes.forEach((attributes, faceIndex) => {
    materialIds[faceIndex] = attributes.materialId;
    uvSetCounts[faceIndex] = attributes.uvSets.length;
    if (normalMapYScales) normalMapYScales[faceIndex] = attributes.normalMapYScale ?? 1;
    for (const uvSet of attributes.uvSets) {
      uvSetTexCoords[uvSetIndex] = uvSet.texCoord;
      const base = uvSetIndex * 6;
      uvs[base] = uvSet.uvs[0].x;
      uvs[base + 1] = uvSet.uvs[0].y;
      uvs[base + 2] = uvSet.uvs[1].x;
      uvs[base + 3] = uvSet.uvs[1].y;
      uvs[base + 4] = uvSet.uvs[2].x;
      uvs[base + 5] = uvSet.uvs[2].y;
      uvSetIndex += 1;
    }
    if (attributes.normalCorners && normalCornerFlags && normalCorners) {
      normalCornerFlags[faceIndex] = 1;
      const base = normalCornerIndex * 9;
      normalCorners[base] = attributes.normalCorners[0].x;
      normalCorners[base + 1] = attributes.normalCorners[0].y;
      normalCorners[base + 2] = attributes.normalCorners[0].z;
      normalCorners[base + 3] = attributes.normalCorners[1].x;
      normalCorners[base + 4] = attributes.normalCorners[1].y;
      normalCorners[base + 5] = attributes.normalCorners[1].z;
      normalCorners[base + 6] = attributes.normalCorners[2].x;
      normalCorners[base + 7] = attributes.normalCorners[2].y;
      normalCorners[base + 8] = attributes.normalCorners[2].z;
      normalCornerIndex += 1;
    }
    if (attributes.tangentCorners && tangentCornerFlags && tangentCorners) {
      tangentCornerFlags[faceIndex] = 1;
      const base = tangentCornerIndex * 12;
      tangentCorners[base] = attributes.tangentCorners[0].x;
      tangentCorners[base + 1] = attributes.tangentCorners[0].y;
      tangentCorners[base + 2] = attributes.tangentCorners[0].z;
      tangentCorners[base + 3] = attributes.tangentCorners[0].w;
      tangentCorners[base + 4] = attributes.tangentCorners[1].x;
      tangentCorners[base + 5] = attributes.tangentCorners[1].y;
      tangentCorners[base + 6] = attributes.tangentCorners[1].z;
      tangentCorners[base + 7] = attributes.tangentCorners[1].w;
      tangentCorners[base + 8] = attributes.tangentCorners[2].x;
      tangentCorners[base + 9] = attributes.tangentCorners[2].y;
      tangentCorners[base + 10] = attributes.tangentCorners[2].z;
      tangentCorners[base + 11] = attributes.tangentCorners[2].w;
      tangentCornerIndex += 1;
    }
    if (attributes.colorCorners && colorCornerFlags && colorCorners && colorItemSizes) {
      colorCornerFlags[faceIndex] = 1;
      colorItemSizes[faceIndex] = attributes.colorItemSize ?? 3;
      const base = colorCornerIndex * 12;
      colorCorners[base] = attributes.colorCorners[0].x;
      colorCorners[base + 1] = attributes.colorCorners[0].y;
      colorCorners[base + 2] = attributes.colorCorners[0].z;
      colorCorners[base + 3] = attributes.colorCorners[0].w;
      colorCorners[base + 4] = attributes.colorCorners[1].x;
      colorCorners[base + 5] = attributes.colorCorners[1].y;
      colorCorners[base + 6] = attributes.colorCorners[1].z;
      colorCorners[base + 7] = attributes.colorCorners[1].w;
      colorCorners[base + 8] = attributes.colorCorners[2].x;
      colorCorners[base + 9] = attributes.colorCorners[2].y;
      colorCorners[base + 10] = attributes.colorCorners[2].z;
      colorCorners[base + 11] = attributes.colorCorners[2].w;
      colorCornerIndex += 1;
    }
  });
  return {
    materialIds,
    uvSetCounts,
    uvSetTexCoords,
    uvs,
    ...(normalCornerFlags && normalCorners ? { normalCornerFlags, normalCorners } : {}),
    ...(tangentCornerFlags && tangentCorners ? { tangentCornerFlags, tangentCorners } : {}),
    ...(colorCornerFlags && colorCorners && colorItemSizes ? { colorCornerFlags, colorCorners, colorItemSizes } : {}),
    ...(normalMapYScales ? { normalMapYScales } : {}),
  };
}

export function deserializeSourceFaceAttributes(serialized: SerializedSourceFaceAttributes): SourceFaceAttributes[] {
  if (serialized.normalCornerFlags && serialized.normalCornerFlags.length !== serialized.materialIds.length) {
    throw new Error(`Serialized normal corner flag count ${serialized.normalCornerFlags.length} does not match face count ${serialized.materialIds.length}.`);
  }
  if (serialized.normalCorners && !serialized.normalCornerFlags) {
    throw new Error('Serialized normal corner values are present without normal corner flags.');
  }
  if (serialized.tangentCornerFlags && serialized.tangentCornerFlags.length !== serialized.materialIds.length) {
    throw new Error(`Serialized tangent corner flag count ${serialized.tangentCornerFlags.length} does not match face count ${serialized.materialIds.length}.`);
  }
  if (serialized.tangentCorners && !serialized.tangentCornerFlags) {
    throw new Error('Serialized tangent corner values are present without tangent corner flags.');
  }
  if (serialized.colorCornerFlags && serialized.colorCornerFlags.length !== serialized.materialIds.length) {
    throw new Error(`Serialized color corner flag count ${serialized.colorCornerFlags.length} does not match face count ${serialized.materialIds.length}.`);
  }
  if (serialized.colorCorners && !serialized.colorCornerFlags) {
    throw new Error('Serialized color corner values are present without color corner flags.');
  }
  if (serialized.colorItemSizes && !serialized.colorCornerFlags) {
    throw new Error('Serialized color item sizes are present without color corner flags.');
  }
  if (serialized.colorCornerFlags && !serialized.colorItemSizes) {
    throw new Error('Serialized color corner flags are present without color item sizes.');
  }
  if (serialized.colorItemSizes && serialized.colorItemSizes.length !== serialized.materialIds.length) {
    throw new Error(`Serialized color item size count ${serialized.colorItemSizes.length} does not match face count ${serialized.materialIds.length}.`);
  }
  if (serialized.normalMapYScales && serialized.normalMapYScales.length !== serialized.materialIds.length) {
    throw new Error(`Serialized normal-map Y scale count ${serialized.normalMapYScales.length} does not match face count ${serialized.materialIds.length}.`);
  }
  let expectedNormalCornerValueCount = 0;
  for (const flag of serialized.normalCornerFlags ?? []) {
    if (flag !== 0 && flag !== 1) throw new Error(`Unsupported serialized normal corner flag value ${flag}.`);
    if (flag === 1) expectedNormalCornerValueCount += 9;
  }
  if ((serialized.normalCorners?.length ?? 0) !== expectedNormalCornerValueCount) {
    throw new Error(`Serialized normal corner value count ${serialized.normalCorners?.length ?? 0} does not match expected count ${expectedNormalCornerValueCount}.`);
  }
  let expectedTangentCornerValueCount = 0;
  for (const flag of serialized.tangentCornerFlags ?? []) {
    if (flag !== 0 && flag !== 1) throw new Error(`Unsupported serialized tangent corner flag value ${flag}.`);
    if (flag === 1) expectedTangentCornerValueCount += 12;
  }
  if ((serialized.tangentCorners?.length ?? 0) !== expectedTangentCornerValueCount) {
    throw new Error(`Serialized tangent corner value count ${serialized.tangentCorners?.length ?? 0} does not match expected count ${expectedTangentCornerValueCount}.`);
  }
  let expectedColorCornerValueCount = 0;
  for (let faceIndex = 0; faceIndex < (serialized.colorCornerFlags?.length ?? 0); faceIndex += 1) {
    const flag = serialized.colorCornerFlags![faceIndex]!;
    if (flag !== 0 && flag !== 1) throw new Error(`Unsupported serialized color corner flag value ${flag}.`);
    const itemSize = serialized.colorItemSizes?.[faceIndex] ?? 0;
    if (flag === 1) {
      if (itemSize !== 3 && itemSize !== 4) throw new Error(`Unsupported serialized color item size ${itemSize}.`);
      expectedColorCornerValueCount += 12;
    } else if (itemSize !== 0) {
      throw new Error(`Serialized color item size ${itemSize} is present for uncolored face ${faceIndex}.`);
    }
  }
  if ((serialized.colorCorners?.length ?? 0) !== expectedColorCornerValueCount) {
    throw new Error(`Serialized color corner value count ${serialized.colorCorners?.length ?? 0} does not match expected count ${expectedColorCornerValueCount}.`);
  }

  const faceAttributes: SourceFaceAttributes[] = [];
  let uvSetIndex = 0;
  let normalCornerIndex = 0;
  let tangentCornerIndex = 0;
  let colorCornerIndex = 0;
  for (let faceIndex = 0; faceIndex < serialized.materialIds.length; faceIndex += 1) {
    const uvSets: SourceFaceAttributes['uvSets'] = [];
    const uvSetCount = serialized.uvSetCounts[faceIndex] ?? 0;
    for (let faceUvSetIndex = 0; faceUvSetIndex < uvSetCount; faceUvSetIndex += 1) {
      const base = uvSetIndex * 6;
      uvSets.push({
        texCoord: serialized.uvSetTexCoords[uvSetIndex] ?? 0,
        uvs: [
          new Vector2(serialized.uvs[base]!, serialized.uvs[base + 1]!),
          new Vector2(serialized.uvs[base + 2]!, serialized.uvs[base + 3]!),
          new Vector2(serialized.uvs[base + 4]!, serialized.uvs[base + 5]!),
        ],
      });
      uvSetIndex += 1;
    }
    const serializedNormalCorners = serialized.normalCorners;
    const hasNormalCorners = serialized.normalCornerFlags?.[faceIndex] === 1 && serializedNormalCorners !== undefined;
    const normalCornerBase = normalCornerIndex * 9;
    const normalCorners = hasNormalCorners
      ? [
          new Vector3(
            serializedNormalCorners[normalCornerBase]!,
            serializedNormalCorners[normalCornerBase + 1]!,
            serializedNormalCorners[normalCornerBase + 2]!,
          ),
          new Vector3(
            serializedNormalCorners[normalCornerBase + 3]!,
            serializedNormalCorners[normalCornerBase + 4]!,
            serializedNormalCorners[normalCornerBase + 5]!,
          ),
          new Vector3(
            serializedNormalCorners[normalCornerBase + 6]!,
            serializedNormalCorners[normalCornerBase + 7]!,
            serializedNormalCorners[normalCornerBase + 8]!,
          ),
        ] as [Vector3, Vector3, Vector3]
      : undefined;
    if (hasNormalCorners) normalCornerIndex += 1;
    const serializedTangentCorners = serialized.tangentCorners;
    const hasTangentCorners = serialized.tangentCornerFlags?.[faceIndex] === 1 && serializedTangentCorners !== undefined;
    const tangentCornerBase = tangentCornerIndex * 12;
    const tangentCorners = hasTangentCorners
      ? [
          new Vector4(
            serializedTangentCorners[tangentCornerBase]!,
            serializedTangentCorners[tangentCornerBase + 1]!,
            serializedTangentCorners[tangentCornerBase + 2]!,
            serializedTangentCorners[tangentCornerBase + 3]!,
          ),
          new Vector4(
            serializedTangentCorners[tangentCornerBase + 4]!,
            serializedTangentCorners[tangentCornerBase + 5]!,
            serializedTangentCorners[tangentCornerBase + 6]!,
            serializedTangentCorners[tangentCornerBase + 7]!,
          ),
          new Vector4(
            serializedTangentCorners[tangentCornerBase + 8]!,
            serializedTangentCorners[tangentCornerBase + 9]!,
            serializedTangentCorners[tangentCornerBase + 10]!,
            serializedTangentCorners[tangentCornerBase + 11]!,
          ),
        ] as [Vector4, Vector4, Vector4]
      : undefined;
    if (hasTangentCorners) tangentCornerIndex += 1;
    const serializedColorCorners = serialized.colorCorners;
    const hasColorCorners = serialized.colorCornerFlags?.[faceIndex] === 1 && serializedColorCorners !== undefined;
    const colorCornerBase = colorCornerIndex * 12;
    const colorCorners = hasColorCorners
      ? [
          new Vector4(
            serializedColorCorners[colorCornerBase]!,
            serializedColorCorners[colorCornerBase + 1]!,
            serializedColorCorners[colorCornerBase + 2]!,
            serializedColorCorners[colorCornerBase + 3]!,
          ),
          new Vector4(
            serializedColorCorners[colorCornerBase + 4]!,
            serializedColorCorners[colorCornerBase + 5]!,
            serializedColorCorners[colorCornerBase + 6]!,
            serializedColorCorners[colorCornerBase + 7]!,
          ),
          new Vector4(
            serializedColorCorners[colorCornerBase + 8]!,
            serializedColorCorners[colorCornerBase + 9]!,
            serializedColorCorners[colorCornerBase + 10]!,
            serializedColorCorners[colorCornerBase + 11]!,
          ),
        ] as [Vector4, Vector4, Vector4]
      : undefined;
    const colorItemSize = hasColorCorners ? serialized.colorItemSizes![faceIndex] as 3 | 4 : undefined;
    if (hasColorCorners) colorCornerIndex += 1;
    const normalMapYScale = serialized.normalMapYScales?.[faceIndex];
    faceAttributes.push({
      materialId: serialized.materialIds[faceIndex]!,
      uvSets,
      ...(normalCorners ? { normalCorners } : {}),
      ...(tangentCorners ? { tangentCorners } : {}),
      ...(colorCorners && colorItemSize !== undefined ? { colorCorners, colorItemSize } : {}),
      ...(normalMapYScale !== undefined && Math.abs(normalMapYScale - 1) > 1e-8 ? { normalMapYScale } : {}),
    });
  }
  return faceAttributes;
}

export function serializeTransferredMeshAttributes(attributes: TransferredMeshAttributes): SerializedTransferredMeshAttributes {
  const uvSetCounts = new Uint16Array(attributes.vertices.length);
  const uvSetCount = attributes.vertices.reduce((sum, vertex) => sum + vertex.uvSets.length, 0);
  const uvSetTexCoords = new Uint16Array(uvSetCount);
  const uvs = new Float32Array(uvSetCount * 2);
  const normalCount = attributes.vertices.reduce((sum, vertex) => sum + (vertex.normal ? 1 : 0), 0);
  const normalFlags = normalCount > 0 ? new Uint8Array(attributes.vertices.length) : undefined;
  const normals = normalCount > 0 ? new Float32Array(normalCount * 3) : undefined;
  const tangentCount = attributes.vertices.reduce((sum, vertex) => sum + (vertex.tangent ? 1 : 0), 0);
  const tangentFlags = tangentCount > 0 ? new Uint8Array(attributes.vertices.length) : undefined;
  const tangents = tangentCount > 0 ? new Float32Array(tangentCount * 4) : undefined;
  const colorCount = attributes.vertices.reduce((sum, vertex) => sum + (vertex.color ? 1 : 0), 0);
  const colorFlags = colorCount > 0 ? new Uint8Array(attributes.vertices.length) : undefined;
  const colors = colorCount > 0 ? new Float32Array(colorCount * 4) : undefined;
  let uvSetIndex = 0;
  let normalIndex = 0;
  let tangentIndex = 0;
  let colorIndex = 0;

  attributes.vertices.forEach((vertex, vertexIndex) => {
    uvSetCounts[vertexIndex] = vertex.uvSets.length;
    for (const uvSet of vertex.uvSets) {
      uvSetTexCoords[uvSetIndex] = uvSet.texCoord;
      const uvBase = uvSetIndex * 2;
      uvs[uvBase] = uvSet.uv.x;
      uvs[uvBase + 1] = uvSet.uv.y;
      uvSetIndex += 1;
    }
    if (vertex.normal && normalFlags && normals) {
      normalFlags[vertexIndex] = 1;
      const normalBase = normalIndex * 3;
      normals[normalBase] = vertex.normal.x;
      normals[normalBase + 1] = vertex.normal.y;
      normals[normalBase + 2] = vertex.normal.z;
      normalIndex += 1;
    }
    if (vertex.tangent && tangentFlags && tangents) {
      tangentFlags[vertexIndex] = 1;
      const tangentBase = tangentIndex * 4;
      tangents[tangentBase] = vertex.tangent.x;
      tangents[tangentBase + 1] = vertex.tangent.y;
      tangents[tangentBase + 2] = vertex.tangent.z;
      tangents[tangentBase + 3] = vertex.tangent.w;
      tangentIndex += 1;
    }
    if (vertex.color && colorFlags && colors) {
      colorFlags[vertexIndex] = 1;
      const colorBase = colorIndex * 4;
      colors[colorBase] = vertex.color.x;
      colors[colorBase + 1] = vertex.color.y;
      colors[colorBase + 2] = vertex.color.z;
      colors[colorBase + 3] = vertex.color.w;
      colorIndex += 1;
    }
  });

  return {
    uvSetCounts,
    uvSetTexCoords,
    uvs,
    ...(normalFlags && normals ? { normalFlags, normals } : {}),
    ...(tangentFlags && tangents ? { tangentFlags, tangents } : {}),
    ...(colorFlags && colors ? { colorFlags, colors, colorItemSize: attributes.colorItemSize ?? 3 } : {}),
    ...(attributes.normalMapYScale !== undefined ? { normalMapYScale: attributes.normalMapYScale } : {}),
    ...(attributes.hasSourceTangents ? { hasSourceTangents: true } : {}),
  };
}

function validateColorItemSize(name: string, itemSize: number | undefined): asserts itemSize is 3 | 4 {
  if (itemSize !== 3 && itemSize !== 4) {
    throw new Error(`Unsupported serialized ${name} item size ${itemSize ?? 'missing'}.`);
  }
}

function validateTransferredAttributeFlags(
  name: string,
  flags: Uint8Array | undefined,
  values: Float32Array | undefined,
  vertexCount: number,
  valuesPerItem: number,
): void {
  if (flags && flags.length !== vertexCount) {
    throw new Error(`Serialized ${name} flag count ${flags.length} does not match vertex count ${vertexCount}.`);
  }
  if (values && !flags) {
    throw new Error(`Serialized ${name} values are present without ${name} flags.`);
  }
  let expectedValueCount = 0;
  for (const flag of flags ?? []) {
    if (flag !== 0 && flag !== 1) throw new Error(`Unsupported serialized ${name} flag value ${flag}.`);
    if (flag === 1) expectedValueCount += valuesPerItem;
  }
  if ((values?.length ?? 0) !== expectedValueCount) {
    throw new Error(`Serialized ${name} value count ${values?.length ?? 0} does not match expected count ${expectedValueCount}.`);
  }
}

export function deserializeTransferredMeshAttributes(
  serialized: SerializedTransferredMeshAttributes,
): TransferredMeshAttributes {
  const vertexCount = serialized.uvSetCounts.length;
  const uvSetCount = Array.from(serialized.uvSetCounts).reduce((sum, count) => sum + count, 0);
  if (serialized.uvSetTexCoords.length !== uvSetCount) {
    throw new Error(`Serialized transferred UV texcoord count ${serialized.uvSetTexCoords.length} does not match expected count ${uvSetCount}.`);
  }
  if (serialized.uvs.length !== uvSetCount * 2) {
    throw new Error(`Serialized transferred UV value count ${serialized.uvs.length} does not match expected count ${uvSetCount * 2}.`);
  }
  validateTransferredAttributeFlags('normal', serialized.normalFlags, serialized.normals, vertexCount, 3);
  validateTransferredAttributeFlags('tangent', serialized.tangentFlags, serialized.tangents, vertexCount, 4);
  validateTransferredAttributeFlags('color', serialized.colorFlags, serialized.colors, vertexCount, 4);
  if (serialized.colorFlags || serialized.colors || serialized.colorItemSize !== undefined) {
    validateColorItemSize('color', serialized.colorItemSize);
  }

  const vertices: TransferredMeshAttributes['vertices'] = [];
  let uvSetIndex = 0;
  let normalIndex = 0;
  let tangentIndex = 0;
  let colorIndex = 0;
  for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 1) {
    const uvSets: TransferredMeshAttributes['vertices'][number]['uvSets'] = [];
    const uvSetCountForVertex = serialized.uvSetCounts[vertexIndex] ?? 0;
    for (let vertexUvSetIndex = 0; vertexUvSetIndex < uvSetCountForVertex; vertexUvSetIndex += 1) {
      const uvBase = uvSetIndex * 2;
      uvSets.push({
        texCoord: serialized.uvSetTexCoords[uvSetIndex] ?? 0,
        uv: new Vector2(serialized.uvs[uvBase]!, serialized.uvs[uvBase + 1]!),
      });
      uvSetIndex += 1;
    }
    const hasNormal = serialized.normalFlags?.[vertexIndex] === 1 && serialized.normals !== undefined;
    const normalBase = normalIndex * 3;
    const normal = hasNormal
      ? new Vector3(
          serialized.normals![normalBase]!,
          serialized.normals![normalBase + 1]!,
          serialized.normals![normalBase + 2]!,
        )
      : undefined;
    if (hasNormal) normalIndex += 1;
    const hasTangent = serialized.tangentFlags?.[vertexIndex] === 1 && serialized.tangents !== undefined;
    const tangentBase = tangentIndex * 4;
    const tangent = hasTangent
      ? new Vector4(
          serialized.tangents![tangentBase]!,
          serialized.tangents![tangentBase + 1]!,
          serialized.tangents![tangentBase + 2]!,
          serialized.tangents![tangentBase + 3]!,
        )
      : undefined;
    if (hasTangent) tangentIndex += 1;
    const hasColor = serialized.colorFlags?.[vertexIndex] === 1 && serialized.colors !== undefined;
    const colorBase = colorIndex * 4;
    const color = hasColor
      ? new Vector4(
          serialized.colors![colorBase]!,
          serialized.colors![colorBase + 1]!,
          serialized.colors![colorBase + 2]!,
          serialized.colors![colorBase + 3]!,
        )
      : undefined;
    if (hasColor) colorIndex += 1;
    vertices.push({
      uvSets,
      ...(normal ? { normal } : {}),
      ...(tangent ? { tangent } : {}),
      ...(color ? { color } : {}),
    });
  }

  return {
    vertices,
    ...(serialized.colorFlags ? { colorItemSize: serialized.colorItemSize } : {}),
    ...(serialized.normalMapYScale !== undefined ? { normalMapYScale: serialized.normalMapYScale } : {}),
    ...(serialized.hasSourceTangents ? { hasSourceTangents: true } : {}),
  };
}

export function serializeTexturedRawMesh(source: TexturedRawMesh, options?: SerializationOptions): SerializedTexturedRawMesh {
  return {
    rawMesh: serializeRawMesh(source.rawMesh),
    faceAttributes: serializeSourceFaceAttributes(source.faceAttributes),
    materials: source.materials.map((material) => serializeSourceMaterial(material, options)),
  };
}

export function deserializeTexturedRawMesh(serialized: SerializedTexturedRawMesh): TexturedRawMesh {
  return {
    rawMesh: deserializeRawMesh(serialized.rawMesh),
    faceAttributes: deserializeSourceFaceAttributes(serialized.faceAttributes),
    materials: serialized.materials.map(deserializeSourceMaterial),
  };
}

export function serializePrimitiveEntries(entries: ReadonlyArray<{
  id: string;
  label: string;
  meshOrdinal: number;
  rawMesh: RawMesh;
  texturedRawMesh: TexturedRawMesh;
  bakeable: boolean;
  hasTexturedMaterial: boolean;
  requiresAttributeTransfer?: boolean;
}>, options?: SerializationOptions): SerializedPrimitiveProcessingEntry[] {
  return entries.map((entry) => ({
    id: entry.id,
    label: entry.label,
    meshOrdinal: entry.meshOrdinal,
    rawMesh: serializeRawMesh(entry.rawMesh),
    textured: serializeTexturedRawMesh(entry.texturedRawMesh, options),
    bakeable: entry.bakeable,
    hasTexturedMaterial: entry.hasTexturedMaterial,
    ...(entry.requiresAttributeTransfer === true ? { requiresAttributeTransfer: true } : {}),
  }));
}

export function deserializePrimitiveEntries(entries: readonly SerializedPrimitiveProcessingEntry[]): Array<ProcessablePrimitiveEntry & { meshOrdinal: number }> {
  return entries.map((entry) => ({
    id: entry.id,
    label: entry.label,
    meshOrdinal: entry.meshOrdinal,
    rawMesh: deserializeRawMesh(entry.rawMesh),
    texturedRawMesh: deserializeTexturedRawMesh(entry.textured),
    bakeable: entry.bakeable,
    hasTexturedMaterial: entry.hasTexturedMaterial,
    ...(entry.requiresAttributeTransfer === true ? { requiresAttributeTransfer: true } : {}),
  }));
}

function serializeFullCollapseHistory(history: readonly CollapseHistoryRecord[]): SerializedFullCollapseHistory {
  const beforeFaceCount = history.reduce((sum, record) => sum + record.beforeFaces.length, 0);
  const afterFaceCount = history.reduce((sum, record) => sum + record.afterFaceIds.length, 0);
  const keepVertexIds = new Uint32Array(history.length);
  const removedVertexIds = new Uint32Array(history.length);
  const beforeFaceOffsets = new Uint32Array(history.length + 1);
  const beforeFaceIds = new Uint32Array(beforeFaceCount);
  const beforeFaceVertices = new Uint32Array(beforeFaceCount * 3);
  const beforeFacePositions = new Float64Array(beforeFaceCount * 9);
  const afterFaceOffsets = new Uint32Array(history.length + 1);
  const afterFaceIds = new Uint32Array(afterFaceCount);
  let beforeFaceIndex = 0;
  let afterFaceIndex = 0;

  history.forEach((record, recordIndex) => {
    keepVertexIds[recordIndex] = record.keepVertexId;
    removedVertexIds[recordIndex] = record.removedVertexId;
    beforeFaceOffsets[recordIndex] = beforeFaceIndex;
    afterFaceOffsets[recordIndex] = afterFaceIndex;

    for (const face of record.beforeFaces) {
      beforeFaceIds[beforeFaceIndex] = face.faceId;
      const vertexBase = beforeFaceIndex * 3;
      beforeFaceVertices[vertexBase] = face.vertices[0];
      beforeFaceVertices[vertexBase + 1] = face.vertices[1];
      beforeFaceVertices[vertexBase + 2] = face.vertices[2];
      const positionBase = beforeFaceIndex * 9;
      face.positions.forEach((position, cornerIndex) => {
        const base = positionBase + cornerIndex * 3;
        beforeFacePositions[base] = position.x;
        beforeFacePositions[base + 1] = position.y;
        beforeFacePositions[base + 2] = position.z;
      });
      beforeFaceIndex += 1;
    }

    for (const faceId of record.afterFaceIds) {
      afterFaceIds[afterFaceIndex] = faceId;
      afterFaceIndex += 1;
    }
  });

  beforeFaceOffsets[history.length] = beforeFaceIndex;
  afterFaceOffsets[history.length] = afterFaceIndex;
  return {
    keepVertexIds,
    removedVertexIds,
    beforeFaceOffsets,
    beforeFaceIds,
    beforeFaceVertices,
    beforeFacePositions,
    afterFaceOffsets,
    afterFaceIds,
  };
}

function deserializeBeforeFace(serialized: SerializedFullCollapseHistory, beforeFaceIndex: number): FaceSnapshot {
  const vertexBase = beforeFaceIndex * 3;
  const positionBase = beforeFaceIndex * 9;
  return {
    faceId: serialized.beforeFaceIds[beforeFaceIndex]!,
    vertices: [
      serialized.beforeFaceVertices[vertexBase]!,
      serialized.beforeFaceVertices[vertexBase + 1]!,
      serialized.beforeFaceVertices[vertexBase + 2]!,
    ],
    positions: [
      new Vector3(
        serialized.beforeFacePositions[positionBase]!,
        serialized.beforeFacePositions[positionBase + 1]!,
        serialized.beforeFacePositions[positionBase + 2]!,
      ),
      new Vector3(
        serialized.beforeFacePositions[positionBase + 3]!,
        serialized.beforeFacePositions[positionBase + 4]!,
        serialized.beforeFacePositions[positionBase + 5]!,
      ),
      new Vector3(
        serialized.beforeFacePositions[positionBase + 6]!,
        serialized.beforeFacePositions[positionBase + 7]!,
        serialized.beforeFacePositions[positionBase + 8]!,
      ),
    ],
  };
}

function deserializeFullCollapseHistory(serialized: SerializedFullCollapseHistory): CollapseHistoryRecord[] {
  const history: CollapseHistoryRecord[] = [];
  for (let recordIndex = 0; recordIndex < serialized.keepVertexIds.length; recordIndex += 1) {
    const beforeFaceStart = serialized.beforeFaceOffsets[recordIndex] ?? 0;
    const beforeFaceEnd = serialized.beforeFaceOffsets[recordIndex + 1] ?? beforeFaceStart;
    const beforeFaces: FaceSnapshot[] = [];
    for (let beforeFaceIndex = beforeFaceStart; beforeFaceIndex < beforeFaceEnd; beforeFaceIndex += 1) {
      beforeFaces.push(deserializeBeforeFace(serialized, beforeFaceIndex));
    }

    const afterFaceStart = serialized.afterFaceOffsets[recordIndex] ?? 0;
    const afterFaceEnd = serialized.afterFaceOffsets[recordIndex + 1] ?? afterFaceStart;
    const afterFaceIds: number[] = [];
    for (let afterFaceIndex = afterFaceStart; afterFaceIndex < afterFaceEnd; afterFaceIndex += 1) {
      afterFaceIds.push(serialized.afterFaceIds[afterFaceIndex]!);
    }

    history.push({
      keepVertexId: serialized.keepVertexIds[recordIndex]!,
      removedVertexId: serialized.removedVertexIds[recordIndex]!,
      beforeFaces,
      afterFaceIds,
    });
  }
  return history;
}

export function serializeFullRawSimplificationResult(result: RawSimplificationResult): SerializedFullRawSimplificationResult {
  return {
    rawMesh: serializeRawMesh(result.rawMesh),
    outputFaceIds: new Uint32Array(result.outputFaceIds),
    history: serializeFullCollapseHistory(result.history),
    stats: result.stats,
  };
}

export function deserializeFullRawSimplificationResult(result: SerializedFullRawSimplificationResult): RawSimplificationResult {
  return {
    rawMesh: deserializeRawMesh(result.rawMesh),
    outputFaceIds: [...result.outputFaceIds],
    history: deserializeFullCollapseHistory(result.history),
    stats: result.stats,
  };
}
