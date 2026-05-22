import { Vector2, Vector3, Vector4 } from 'three';
import type { TransferredMeshAttributes, SourceFaceAttributes } from '../simplification/attributes';
import {
  transferVertexAttributesToSimplifiedMesh,
  type AttributeTransferProgress,
} from '../simplification/attributeTransfer';
import { computeAreaWeightedVertexNormals } from '../simplification/normals';
import { createHistoryTraceIndex, mapOutputSampleToInput } from '../simplification/successiveMapping';
import type { Barycentric, RawSimplificationResult } from '../simplification/types';

const NORMAL_EPSILON = 1e-20;
const CORNER_BARYCENTRICS = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const satisfies readonly Barycentric[];

export interface PrepareOutputTransferredAttributesOptions {
  sourceFaceAttributes?: readonly SourceFaceAttributes[];
  raw: RawSimplificationResult;
  recomputeNormals: boolean;
  onProgress?: (progress: AttributeTransferProgress) => void;
}

export function hasTransferableSourceAttributes(
  sourceFaceAttributes: readonly SourceFaceAttributes[] | undefined,
  raw: RawSimplificationResult,
): sourceFaceAttributes is readonly [SourceFaceAttributes, ...SourceFaceAttributes[]] {
  return sourceFaceAttributes !== undefined
    && sourceFaceAttributes.length > 0
    && mappedSourceFaceAttributesAreTransferable(sourceFaceAttributes, raw)
    && sourceFaceAttributes.some((attributes) => Array.isArray(attributes.uvSets) && attributes.uvSets.length > 0
      || attributes.normalCorners !== undefined
      || attributes.tangentCorners !== undefined
      || attributes.colorCorners !== undefined
      || attributes.normalMapYScale !== undefined);
}

function isFiniteVector2(value: unknown): value is Vector2 {
  return value instanceof Vector2
    && Number.isFinite(value.x)
    && Number.isFinite(value.y);
}

function isFiniteVector3(value: unknown): value is Vector3 {
  return value instanceof Vector3
    && Number.isFinite(value.x)
    && Number.isFinite(value.y)
    && Number.isFinite(value.z);
}

function isFiniteVector4(value: unknown): value is Vector4 {
  return value instanceof Vector4
    && Number.isFinite(value.x)
    && Number.isFinite(value.y)
    && Number.isFinite(value.z)
    && Number.isFinite(value.w);
}

function isFiniteVector2Tuple(value: unknown): value is [Vector2, Vector2, Vector2] {
  return Array.isArray(value)
    && value.length === 3
    && value.every(isFiniteVector2);
}

function isFiniteVector3Tuple(value: unknown): value is [Vector3, Vector3, Vector3] {
  return Array.isArray(value)
    && value.length === 3
    && value.every(isFiniteVector3);
}

function isFiniteVector4Tuple(value: unknown): value is [Vector4, Vector4, Vector4] {
  return Array.isArray(value)
    && value.length === 3
    && value.every(isFiniteVector4);
}

function sourceFaceAttributeIsWellFormed(attributes: SourceFaceAttributes): boolean {
  if (!Array.isArray(attributes.uvSets)) return false;
  for (const uvSet of attributes.uvSets) {
    if (!Number.isInteger(uvSet.texCoord) || uvSet.texCoord < 0 || !isFiniteVector2Tuple(uvSet.uvs)) {
      return false;
    }
  }
  if (attributes.normalCorners !== undefined && !isFiniteVector3Tuple(attributes.normalCorners)) return false;
  if (attributes.tangentCorners !== undefined && !isFiniteVector4Tuple(attributes.tangentCorners)) return false;
  if (attributes.colorCorners !== undefined && !isFiniteVector4Tuple(attributes.colorCorners)) return false;
  if (
    attributes.colorCorners !== undefined
    && attributes.colorItemSize !== undefined
    && attributes.colorItemSize !== 3
    && attributes.colorItemSize !== 4
  ) {
    return false;
  }
  if (attributes.normalMapYScale !== undefined && !Number.isFinite(attributes.normalMapYScale)) return false;
  return true;
}

function mappedSourceFaceAttributesAreTransferable(
  sourceFaceAttributes: readonly SourceFaceAttributes[],
  raw: RawSimplificationResult,
): boolean {
  if (raw.outputFaceIds.length < raw.rawMesh.faces.length) return false;
  const historyIndex = createHistoryTraceIndex(raw.history);
  const mappedFaceIds = new Set<number>();
  for (let outputFaceIndex = 0; outputFaceIndex < raw.rawMesh.faces.length; outputFaceIndex += 1) {
    for (const outputBarycentric of CORNER_BARYCENTRICS) {
      try {
        const mapped = mapOutputSampleToInput({
          outputRawMesh: raw.rawMesh,
          outputFaceIds: raw.outputFaceIds,
          outputFaceIndex,
          outputBarycentric,
          history: raw.history,
          historyIndex,
        });
        if (!Number.isInteger(mapped.faceId) || mapped.faceId < 0 || !sourceFaceAttributes[mapped.faceId]) {
          return false;
        }
        mappedFaceIds.add(mapped.faceId);
      } catch {
        return false;
      }
    }
  }
  for (const faceId of mappedFaceIds) {
    const attributes = sourceFaceAttributes[faceId];
    if (!attributes || !sourceFaceAttributeIsWellFormed(attributes)) return false;
  }
  return mappedFaceIds.size > 0 || raw.rawMesh.faces.length === 0;
}

function isValidNormal(normal: Vector3 | undefined): normal is Vector3 {
  return normal !== undefined
    && Number.isFinite(normal.x)
    && Number.isFinite(normal.y)
    && Number.isFinite(normal.z)
    && normal.lengthSq() > NORMAL_EPSILON;
}

export function shouldAttachPreparedOutputAttributes(options: {
  sourceFaceAttributes?: readonly SourceFaceAttributes[];
  raw: RawSimplificationResult;
  recomputeNormals: boolean;
  requiresAttributeTransfer?: boolean;
}): boolean {
  if (!hasTransferableSourceAttributes(options.sourceFaceAttributes, options.raw)) return false;
  return options.requiresAttributeTransfer === true
    || (!options.recomputeNormals && options.sourceFaceAttributes.some((attributes) => attributes.normalCorners));
}

export function prepareOutputTransferredAttributes(
  options: PrepareOutputTransferredAttributesOptions,
): TransferredMeshAttributes {
  const computedNormals = computeAreaWeightedVertexNormals(options.raw.rawMesh);
  const transferred: TransferredMeshAttributes = hasTransferableSourceAttributes(options.sourceFaceAttributes, options.raw)
    ? transferVertexAttributesToSimplifiedMesh({
        sourceFaceAttributes: options.sourceFaceAttributes,
        raw: options.raw,
        ...(options.onProgress ? { onProgress: options.onProgress } : {}),
      })
    : {
        vertices: options.raw.rawMesh.positions.map(() => ({ uvSets: [] })),
      };

  return {
    ...transferred,
    vertices: transferred.vertices.map((vertex, index) => {
      const computedNormal = computedNormals[index]?.clone() ?? new Vector3(0, 0, 1);
      const sourceNormal = vertex.normal;
      const normal = options.recomputeNormals || !isValidNormal(sourceNormal)
        ? computedNormal
        : sourceNormal.clone().normalize();
      return {
        ...vertex,
        normal,
      };
    }),
  };
}
