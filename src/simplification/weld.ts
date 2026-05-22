import { Matrix4, Vector3 } from 'three';
import type { FaceIndices, RawMesh } from './types';

export const GROUP_VERTEX_WELD_TOLERANCE = 1e-9;

export interface WeldedVertexAppendState {
  readonly vertexByPositionKey: Map<string, number>;
}

export interface RawMeshAppendOptions {
  weldVertices?: boolean;
}

export function createWeldedVertexAppendState(): WeldedVertexAppendState {
  return { vertexByPositionKey: new Map() };
}

export function quantizedPositionKey(position: Vector3, tolerance = GROUP_VERTEX_WELD_TOLERANCE): string {
  const scale = 1 / tolerance;
  return `${Math.round(position.x * scale)},${Math.round(position.y * scale)},${Math.round(position.z * scale)}`;
}

export function appendWeldedVertex(
  target: RawMesh,
  position: Vector3,
  state: WeldedVertexAppendState,
): number {
  const key = quantizedPositionKey(position);
  const existing = state.vertexByPositionKey.get(key);
  if (existing !== undefined) return existing;
  const vertexId = target.positions.length;
  target.positions.push(position.clone());
  state.vertexByPositionKey.set(key, vertexId);
  return vertexId;
}

export function appendRawMeshVertex(
  target: RawMesh,
  position: Vector3,
  state: WeldedVertexAppendState,
  options: RawMeshAppendOptions = {},
): number {
  if (options.weldVertices === false) {
    const vertexId = target.positions.length;
    target.positions.push(position.clone());
    return vertexId;
  }
  return appendWeldedVertex(target, position, state);
}

export function appendTransformedRawMesh(
  target: RawMesh,
  source: RawMesh,
  transform: Matrix4,
  state: WeldedVertexAppendState,
  options: RawMeshAppendOptions = {},
): void {
  const sourceToTarget = new Map<number, number>();
  for (let sourceVertexId = 0; sourceVertexId < source.positions.length; sourceVertexId += 1) {
    const sourcePosition = source.positions[sourceVertexId];
    if (!sourcePosition) continue;
    sourceToTarget.set(sourceVertexId, appendRawMeshVertex(
      target,
      sourcePosition.clone().applyMatrix4(transform),
      state,
      options,
    ));
  }
  for (const face of source.faces) {
    target.faces.push([
      sourceToTarget.get(face[0])!,
      sourceToTarget.get(face[1])!,
      sourceToTarget.get(face[2])!,
    ] as FaceIndices);
  }
}

export function appendTransformedRawMeshWelded(
  target: RawMesh,
  source: RawMesh,
  transform: Matrix4,
  state: WeldedVertexAppendState,
): void {
  appendTransformedRawMesh(target, source, transform, state, { weldVertices: true });
}
