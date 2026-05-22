import type { Vector2, Vector3, Vector4 } from 'three';

export type VertexColorItemSize = 3 | 4;
export type SourceFaceColorCorners = [Vector4, Vector4, Vector4];

export interface SourceFaceUvSet {
  texCoord: number;
  uvs: [Vector2, Vector2, Vector2];
}

export interface SourceFaceAttributes {
  materialId: number;
  uvSets: SourceFaceUvSet[];
  normalCorners?: [Vector3, Vector3, Vector3];
  tangentCorners?: [Vector4, Vector4, Vector4];
  colorCorners?: SourceFaceColorCorners;
  colorItemSize?: VertexColorItemSize;
  normalMapYScale?: number;
}

export interface TransferredVertexUvSet {
  texCoord: number;
  uv: Vector2;
}

export interface TransferredVertexAttributes {
  uvSets: TransferredVertexUvSet[];
  normal?: Vector3;
  tangent?: Vector4;
  color?: Vector4;
}

export interface TransferredMeshAttributes {
  vertices: TransferredVertexAttributes[];
  colorItemSize?: VertexColorItemSize;
  normalMapYScale?: number;
  hasSourceTangents?: boolean;
}

export function faceUvSet(attributes: SourceFaceAttributes, texCoord: number): SourceFaceUvSet | undefined;
export function faceUvSet(attributes: TransferredVertexAttributes, texCoord: number): Vector2 | undefined;
export function faceUvSet(
  attributes: SourceFaceAttributes | TransferredVertexAttributes,
  texCoord: number,
): SourceFaceUvSet | Vector2 | undefined {
  const uvSet = attributes.uvSets.find((candidate) => candidate.texCoord === texCoord);
  if (!uvSet) return undefined;
  return 'uv' in uvSet ? uvSet.uv : uvSet;
}

export function transferredVertexUv(attributes: TransferredVertexAttributes, texCoord: number): Vector2 | undefined {
  return faceUvSet(attributes, texCoord);
}

export function transferredVertexColor(attributes: TransferredVertexAttributes): Vector4 | undefined {
  return attributes.color;
}
