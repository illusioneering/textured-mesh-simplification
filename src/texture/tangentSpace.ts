import { Matrix4, Vector2, Vector3, Vector4 } from 'three';
import type { Barycentric } from '../simplification/types';
import type { Rgba } from './types';

const EPSILON = 1e-10;

export interface TangentFrame {
  tangent: Vector3;
  bitangent: Vector3;
  normal: Vector3;
  handedness: 1 | -1;
}

export interface FaceTangentFrameInput {
  positions: [Vector3, Vector3, Vector3];
  uvs: [Vector2, Vector2, Vector2];
  normal: Vector3;
}

export interface AuthoredTangentFrameInput {
  normals: [Vector3, Vector3, Vector3];
  tangents: [Vector4, Vector4, Vector4];
  barycentric: Barycentric;
}

export interface SerializedTangentsInput {
  positions: Vector3[];
  normals: Vector3[];
  uvs: Vector2[];
  indices: number[];
}

export function normalRgbToVector(color: Rgba): Vector3 {
  const vector = new Vector3(
    (color[0] / 255) * 2 - 1,
    (color[1] / 255) * 2 - 1,
    (color[2] / 255) * 2 - 1,
  );
  return normalizeOrDefault(vector, new Vector3(0, 0, 1));
}

export function vectorToNormalRgb(vector: Vector3, alpha = 255): Rgba {
  const normal = normalizeOrDefault(vector.clone(), new Vector3(0, 0, 1));
  return [
    vectorComponentToByte(normal.x),
    vectorComponentToByte(normal.y),
    vectorComponentToByte(normal.z),
    clampByte(alpha),
  ];
}

export function transformAuthoredTangent(tangent: Vector4, matrix: Matrix4): Vector4 {
  const direction = new Vector3(tangent.x, tangent.y, tangent.z);
  if (direction.lengthSq() <= EPSILON) return new Vector4(0, 0, 0, tangent.w < 0 ? -1 : 1);
  direction.transformDirection(matrix);
  return new Vector4(direction.x, direction.y, direction.z, tangent.w < 0 ? -1 : 1);
}

export function computeFaceTangentFrame(input: FaceTangentFrameInput): TangentFrame | null {
  const [p0, p1, p2] = input.positions;
  const [uv0, uv1, uv2] = input.uvs;
  const edge1 = p1.clone().sub(p0);
  const edge2 = p2.clone().sub(p0);
  const faceCross = edge1.clone().cross(edge2);
  if (faceCross.lengthSq() <= EPSILON) {
    return null;
  }

  const uvEdge1 = uv1.clone().sub(uv0);
  const uvEdge2 = uv2.clone().sub(uv0);
  const determinant = uvEdge1.x * uvEdge2.y - uvEdge2.x * uvEdge1.y;
  if (Math.abs(determinant) <= EPSILON) {
    return null;
  }

  const normal = normalizeOrDefault(input.normal.clone(), faceCross);
  const inverseDeterminant = 1 / determinant;
  const rawTangent = edge1.clone().multiplyScalar(uvEdge2.y)
    .sub(edge2.clone().multiplyScalar(uvEdge1.y))
    .multiplyScalar(inverseDeterminant);
  const rawBitangent = edge2.clone().multiplyScalar(uvEdge1.x)
    .sub(edge1.clone().multiplyScalar(uvEdge2.x))
    .multiplyScalar(inverseDeterminant);
  const tangent = orthogonalize(rawTangent, normal);
  if (tangent === null) {
    return null;
  }

  const referenceBitangent = normal.clone().cross(tangent);
  const handedness: 1 | -1 = referenceBitangent.dot(rawBitangent) < 0 ? -1 : 1;
  const bitangent = referenceBitangent.multiplyScalar(handedness);

  return { tangent, bitangent, normal, handedness };
}

export function computeAuthoredTangentFrame(input: AuthoredTangentFrameInput): TangentFrame | null {
  const cornerFrames = input.normals.map((normal, index) => (
    authoredCornerBasis(normal, input.tangents[index]!)
  ));
  if (cornerFrames.some((frame) => frame === null)) return null;
  const frames = cornerFrames as [
    { normal: Vector3; tangent: Vector3; bitangent: Vector3 },
    { normal: Vector3; tangent: Vector3; bitangent: Vector3 },
    { normal: Vector3; tangent: Vector3; bitangent: Vector3 },
  ];

  const normal = normalizeOrDefault(interpolateVector3ByBarycentric([
    frames[0].normal,
    frames[1].normal,
    frames[2].normal,
  ], input.barycentric), new Vector3(0, 0, 1));
  const tangent = orthogonalize(interpolateVector3ByBarycentric([
    frames[0].tangent,
    frames[1].tangent,
    frames[2].tangent,
  ], input.barycentric), normal);
  if (!tangent) return null;
  const interpolatedBitangent = interpolateVector3ByBarycentric([
    frames[0].bitangent,
    frames[1].bitangent,
    frames[2].bitangent,
  ], input.barycentric);
  const fallbackCross = normal.clone().cross(tangent);
  const fallbackBitangent = fallbackCross.clone()
    .multiplyScalar(fallbackCross.dot(interpolatedBitangent) < 0 ? -1 : 1);
  const bitangent = orthogonalizeBitangent(interpolatedBitangent, normal, tangent) ?? fallbackBitangent;
  const handedness: 1 | -1 = normal.clone().cross(tangent).dot(bitangent) < 0 ? -1 : 1;

  return { tangent, bitangent, normal, handedness };
}

export function tangentNormalToWorld(tangentNormal: Vector3, frame: TangentFrame): Vector3 {
  const normal = normalizeOrDefault(tangentNormal.clone(), new Vector3(0, 0, 1));
  return frame.tangent.clone().multiplyScalar(normal.x)
    .add(frame.bitangent.clone().multiplyScalar(normal.y))
    .add(frame.normal.clone().multiplyScalar(normal.z))
    .normalize();
}

export function worldNormalToTangent(worldNormal: Vector3, frame: TangentFrame): Vector3 {
  const normal = normalizeOrDefault(worldNormal.clone(), frame.normal);
  return normalizeOrDefault(new Vector3(
    normal.dot(frame.tangent),
    normal.dot(frame.bitangent),
    normal.dot(frame.normal),
  ), new Vector3(0, 0, 1));
}

export function transformTangentSpaceNormal(color: Rgba, sourceFrame: TangentFrame, outputFrame: TangentFrame): Rgba {
  const worldNormal = tangentNormalToWorld(normalRgbToVector(color), sourceFrame);
  return vectorToNormalRgb(worldNormalToTangent(worldNormal, outputFrame), color[3]);
}

export function computeSerializedTangents(input: SerializedTangentsInput): Float32Array {
  const tangents = Array.from({ length: input.positions.length }, () => new Vector3());
  const bitangents = Array.from({ length: input.positions.length }, () => new Vector3());

  for (let i = 0; i + 2 < input.indices.length; i += 3) {
    const i0 = input.indices[i];
    const i1 = input.indices[i + 1];
    const i2 = input.indices[i + 2];
    if (i0 === undefined || i1 === undefined || i2 === undefined) {
      continue;
    }

    const p0 = input.positions[i0];
    const p1 = input.positions[i1];
    const p2 = input.positions[i2];
    const uv0 = input.uvs[i0];
    const uv1 = input.uvs[i1];
    const uv2 = input.uvs[i2];
    if (!p0 || !p1 || !p2 || !uv0 || !uv1 || !uv2) {
      continue;
    }

    const vectors = computeTriangleTangentVectors([p0, p1, p2], [uv0, uv1, uv2]);
    if (vectors === null) {
      continue;
    }

    tangents[i0]?.add(vectors.tangent);
    tangents[i1]?.add(vectors.tangent);
    tangents[i2]?.add(vectors.tangent);
    bitangents[i0]?.add(vectors.bitangent);
    bitangents[i1]?.add(vectors.bitangent);
    bitangents[i2]?.add(vectors.bitangent);
  }

  const result = new Float32Array(input.positions.length * 4);
  for (let i = 0; i < input.positions.length; i += 1) {
    const normal = normalizeOrDefault(input.normals[i]?.clone() ?? new Vector3(0, 0, 1), new Vector3(0, 0, 1));
    const tangent = orthogonalize(tangents[i] ?? new Vector3(), normal) ?? fallbackTangent(normal);
    const bitangent = bitangents[i] ?? new Vector3();
    const handedness: 1 | -1 = normal.clone().cross(tangent).dot(bitangent) < 0 ? -1 : 1;
    const offset = i * 4;
    result[offset] = tangent.x;
    result[offset + 1] = tangent.y;
    result[offset + 2] = tangent.z;
    result[offset + 3] = handedness;
  }
  return result;
}

function computeTriangleTangentVectors(
  positions: [Vector3, Vector3, Vector3],
  uvs: [Vector2, Vector2, Vector2],
): { tangent: Vector3; bitangent: Vector3 } | null {
  const [p0, p1, p2] = positions;
  const [uv0, uv1, uv2] = uvs;
  const edge1 = p1.clone().sub(p0);
  const edge2 = p2.clone().sub(p0);
  if (edge1.clone().cross(edge2).lengthSq() <= EPSILON) {
    return null;
  }

  const uvEdge1 = uv1.clone().sub(uv0);
  const uvEdge2 = uv2.clone().sub(uv0);
  const determinant = uvEdge1.x * uvEdge2.y - uvEdge2.x * uvEdge1.y;
  if (Math.abs(determinant) <= EPSILON) {
    return null;
  }

  const inverseDeterminant = 1 / determinant;
  return {
    tangent: edge1.clone().multiplyScalar(uvEdge2.y)
      .sub(edge2.clone().multiplyScalar(uvEdge1.y))
      .multiplyScalar(inverseDeterminant),
    bitangent: edge2.clone().multiplyScalar(uvEdge1.x)
      .sub(edge1.clone().multiplyScalar(uvEdge2.x))
      .multiplyScalar(inverseDeterminant),
  };
}

function orthogonalize(vector: Vector3, normal: Vector3): Vector3 | null {
  const tangent = vector.clone().sub(normal.clone().multiplyScalar(normal.dot(vector)));
  if (tangent.lengthSq() <= EPSILON) {
    return null;
  }
  return tangent.normalize();
}

function interpolateVector3ByBarycentric(values: [Vector3, Vector3, Vector3], barycentric: Barycentric): Vector3 {
  return new Vector3()
    .addScaledVector(values[0], barycentric[0])
    .addScaledVector(values[1], barycentric[1])
    .addScaledVector(values[2], barycentric[2]);
}

function authoredCornerBasis(normal: Vector3, tangent: Vector4): { normal: Vector3; tangent: Vector3; bitangent: Vector3 } | null {
  const n = normalizeOrDefault(normal.clone(), new Vector3(0, 0, 1));
  const t = orthogonalize(new Vector3(tangent.x, tangent.y, tangent.z), n);
  if (!t) return null;
  const handedness = tangent.w < 0 ? -1 : 1;
  return {
    normal: n,
    tangent: t,
    bitangent: n.clone().cross(t).multiplyScalar(handedness),
  };
}

function orthogonalizeBitangent(vector: Vector3, normal: Vector3, tangent: Vector3): Vector3 | null {
  const bitangent = vector.clone()
    .sub(normal.clone().multiplyScalar(normal.dot(vector)))
    .sub(tangent.clone().multiplyScalar(tangent.dot(vector)));
  if (bitangent.lengthSq() <= EPSILON) return null;
  return bitangent.normalize();
}

function normalizeOrDefault(vector: Vector3, fallback: Vector3): Vector3 {
  if (vector.lengthSq() > EPSILON) {
    return vector.normalize();
  }
  if (fallback.lengthSq() > EPSILON) {
    return fallback.clone().normalize();
  }
  return new Vector3(0, 0, 1);
}

function fallbackTangent(normal: Vector3): Vector3 {
  const axis = Math.abs(normal.x) < 0.9 ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0);
  return orthogonalize(axis, normal) ?? new Vector3(1, 0, 0);
}

function vectorComponentToByte(value: number): number {
  return clampByte((value * 0.5 + 0.5) * 255);
}

function clampByte(value: number): number {
  return Math.min(255, Math.max(0, Math.round(value)));
}
