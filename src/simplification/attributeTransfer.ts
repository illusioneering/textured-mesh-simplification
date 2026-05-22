import { Vector2, Vector3, Vector4 } from 'three';
import type { Barycentric, RawSimplificationResult } from './types';
import type {
  SourceFaceAttributes,
  TransferredMeshAttributes,
  TransferredVertexAttributes,
  VertexColorItemSize,
} from './attributes';
import { interpolateVector2, interpolateVector3 } from './barycentric';
import { createHistoryTraceIndex, mapOutputSampleToInput } from './successiveMapping';

const CORNER_BARYCENTRICS = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const satisfies readonly Barycentric[];

const MIN_AREA = 1e-12;
const DEFAULT_NORMAL_MAP_Y_SCALE = 1;
const NORMAL_MAP_Y_SCALE_EPSILON = 1e-8;

export interface AttributeTransferProgress {
  processedFaces: number;
  totalFaces: number;
}

interface UvAccumulator {
  sum: Vector2;
  weight: number;
  sampleCount: number;
}

interface VertexAttributeAccumulator {
  sampleCount: number;
  uvSets: Map<number, UvAccumulator>;
  normalSum: Vector3;
  normalWeight: number;
  tangentSum: Vector3;
  tangentHandedness: number;
  tangentWeight: number;
  colorSum: Vector4;
  colorWeight: number;
  colorSampleCount: number;
  colorItemSize?: VertexColorItemSize;
}

interface NormalMapYScaleAccumulator {
  scale?: number;
  consistent: boolean;
  sampleCount: number;
}

function createVertexAccumulator(): VertexAttributeAccumulator {
  return {
    sampleCount: 0,
    uvSets: new Map(),
    normalSum: new Vector3(),
    normalWeight: 0,
    tangentSum: new Vector3(),
    tangentHandedness: 0,
    tangentWeight: 0,
    colorSum: new Vector4(0, 0, 0, 0),
    colorWeight: 0,
    colorSampleCount: 0,
  };
}

function outputTriangleWeight(raw: RawSimplificationResult, outputFaceIndex: number): number {
  const face = raw.rawMesh.faces[outputFaceIndex];
  if (!face) throw new Error(`Missing output face at index ${outputFaceIndex}.`);
  const a = raw.rawMesh.positions[face[0]];
  const b = raw.rawMesh.positions[face[1]];
  const c = raw.rawMesh.positions[face[2]];
  if (!a || !b || !c) throw new Error(`Output face ${outputFaceIndex} references a missing vertex.`);
  const area = b.clone().sub(a).cross(c.clone().sub(a)).length() * 0.5;
  return Number.isFinite(area) && area > MIN_AREA ? area : 1;
}

function interpolateNormal(attributes: SourceFaceAttributes, barycentric: Barycentric): Vector3 | undefined {
  const corners = attributes.normalCorners;
  if (!corners) return undefined;
  const normal = interpolateVector3(corners[0], corners[1], corners[2], barycentric);
  return normal.lengthSq() > 0 ? normal.normalize() : undefined;
}

function interpolateTangent(attributes: SourceFaceAttributes, barycentric: Barycentric): Vector4 | undefined {
  const corners = attributes.tangentCorners;
  if (!corners) return undefined;
  const tangent = new Vector4(0, 0, 0, 0)
    .addScaledVector(corners[0], barycentric[0])
    .addScaledVector(corners[1], barycentric[1])
    .addScaledVector(corners[2], barycentric[2]);
  const direction = new Vector3(tangent.x, tangent.y, tangent.z);
  if (direction.lengthSq() <= 0) return undefined;
  direction.normalize();
  return new Vector4(direction.x, direction.y, direction.z, tangent.w < 0 ? -1 : 1);
}

function interpolateColor(attributes: SourceFaceAttributes, barycentric: Barycentric): Vector4 | undefined {
  const corners = attributes.colorCorners;
  if (!corners) return undefined;
  return new Vector4(0, 0, 0, 0)
    .addScaledVector(corners[0], barycentric[0])
    .addScaledVector(corners[1], barycentric[1])
    .addScaledVector(corners[2], barycentric[2]);
}

function accumulateNormalMapYScale(
  accumulator: NormalMapYScaleAccumulator,
  attributes: SourceFaceAttributes,
): void {
  accumulator.sampleCount += 1;
  const scale = attributes.normalMapYScale;
  if (
    scale === undefined
    || !Number.isFinite(scale)
    || Math.abs(scale - DEFAULT_NORMAL_MAP_Y_SCALE) <= NORMAL_MAP_Y_SCALE_EPSILON
  ) {
    accumulator.consistent = false;
    return;
  }
  if (accumulator.scale === undefined) {
    accumulator.scale = scale;
    return;
  }
  if (Math.abs(accumulator.scale - scale) > NORMAL_MAP_Y_SCALE_EPSILON) {
    accumulator.consistent = false;
  }
}

function accumulateSample(
  accumulator: VertexAttributeAccumulator,
  attributes: SourceFaceAttributes,
  barycentric: Barycentric,
  weight: number,
): void {
  accumulator.sampleCount += 1;
  const seenTexCoords = new Set<number>();
  for (const uvSet of attributes.uvSets) {
    if (seenTexCoords.has(uvSet.texCoord)) continue;
    seenTexCoords.add(uvSet.texCoord);
    let uvAccumulator = accumulator.uvSets.get(uvSet.texCoord);
    if (!uvAccumulator) {
      uvAccumulator = { sum: new Vector2(), weight: 0, sampleCount: 0 };
      accumulator.uvSets.set(uvSet.texCoord, uvAccumulator);
    }
    uvAccumulator.sum.addScaledVector(
      interpolateVector2(uvSet.uvs[0], uvSet.uvs[1], uvSet.uvs[2], barycentric),
      weight,
    );
    uvAccumulator.weight += weight;
    uvAccumulator.sampleCount += 1;
  }

  const normal = interpolateNormal(attributes, barycentric);
  if (normal) {
    accumulator.normalSum.addScaledVector(normal, weight);
    accumulator.normalWeight += weight;
  }

  const tangent = interpolateTangent(attributes, barycentric);
  if (tangent) {
    accumulator.tangentSum.addScaledVector(new Vector3(tangent.x, tangent.y, tangent.z), weight);
    accumulator.tangentHandedness += tangent.w * weight;
    accumulator.tangentWeight += weight;
  }

  const color = interpolateColor(attributes, barycentric);
  if (color) {
    accumulator.colorSum.addScaledVector(color, weight);
    accumulator.colorWeight += weight;
    accumulator.colorSampleCount += 1;
    if (attributes.colorItemSize === 4) {
      accumulator.colorItemSize = 4;
    }
  }
}

function finalizeVertexAttributes(accumulator: VertexAttributeAccumulator): TransferredVertexAttributes {
  const uvSets = [...accumulator.uvSets.entries()]
    .filter(([, uvAccumulator]) => uvAccumulator.sampleCount === accumulator.sampleCount && uvAccumulator.weight > 0)
    .sort(([a], [b]) => a - b)
    .map(([texCoord, uvAccumulator]) => ({
      texCoord,
      uv: uvAccumulator.sum.clone().multiplyScalar(1 / uvAccumulator.weight),
    }));

  const normal = accumulator.normalWeight > 0 && accumulator.normalSum.lengthSq() > 0
    ? accumulator.normalSum.clone().normalize()
    : undefined;
  const tangentDirection = accumulator.tangentWeight > 0 && accumulator.tangentSum.lengthSq() > 0
    ? accumulator.tangentSum.clone().normalize()
    : undefined;
  const tangent = tangentDirection
    ? new Vector4(tangentDirection.x, tangentDirection.y, tangentDirection.z, accumulator.tangentHandedness < 0 ? -1 : 1)
    : undefined;
  const color = accumulator.colorSampleCount === accumulator.sampleCount && accumulator.colorWeight > 0
    ? accumulator.colorSum.clone().multiplyScalar(1 / accumulator.colorWeight)
    : undefined;

  return {
    uvSets,
    ...(normal ? { normal } : {}),
    ...(tangent ? { tangent } : {}),
    ...(color ? { color } : {}),
  };
}

export function transferVertexAttributesToSimplifiedMesh(options: {
  sourceFaceAttributes: readonly SourceFaceAttributes[];
  raw: RawSimplificationResult;
  onProgress?: (progress: AttributeTransferProgress) => void;
}): TransferredMeshAttributes {
  const historyIndex = createHistoryTraceIndex(options.raw.history);
  const totalFaces = options.raw.rawMesh.faces.length;
  const vertexAccumulators = options.raw.rawMesh.positions.map(() => createVertexAccumulator());
  const normalMapYScale: NormalMapYScaleAccumulator = { consistent: true, sampleCount: 0 };
  const hasSourceTangents = options.sourceFaceAttributes.some((attributes) => attributes.tangentCorners !== undefined);

  for (let outputFaceIndex = 0; outputFaceIndex < totalFaces; outputFaceIndex += 1) {
    const face = options.raw.rawMesh.faces[outputFaceIndex];
    if (!face) throw new Error(`Missing output face at index ${outputFaceIndex}.`);
    const weight = outputTriangleWeight(options.raw, outputFaceIndex);

    for (let cornerIndex = 0; cornerIndex < 3; cornerIndex += 1) {
      const vertexId = face[cornerIndex]!;
      const accumulator = vertexAccumulators[vertexId];
      if (!accumulator) throw new Error(`Output face ${outputFaceIndex} references missing vertex ${vertexId}.`);
      const mapped = mapOutputSampleToInput({
        outputRawMesh: options.raw.rawMesh,
        outputFaceIds: options.raw.outputFaceIds,
        outputFaceIndex,
        outputBarycentric: CORNER_BARYCENTRICS[cornerIndex]!,
        history: options.raw.history,
        historyIndex,
      });
      const attributes = options.sourceFaceAttributes[mapped.faceId];
      if (!attributes) throw new Error(`Missing source face attributes for mapped face ${mapped.faceId}.`);
      accumulateSample(accumulator, attributes, mapped.barycentric, weight);
      accumulateNormalMapYScale(normalMapYScale, attributes);
    }

    const processedFaces = outputFaceIndex + 1;
    if (processedFaces % 2048 === 0 || processedFaces === totalFaces) {
      options.onProgress?.({ processedFaces, totalFaces });
    }
  }

  const vertices = vertexAccumulators.map(finalizeVertexAttributes);
  const colorItemSize: VertexColorItemSize | undefined = vertices.every((vertex) => vertex.color)
    ? (vertexAccumulators.some((accumulator) => accumulator.colorItemSize === 4) ? 4 : 3)
    : undefined;
  const transferred = {
    vertices,
    ...(colorItemSize ? { colorItemSize } : {}),
    ...(normalMapYScale.consistent && normalMapYScale.sampleCount > 0 && normalMapYScale.scale !== undefined
      ? { normalMapYScale: normalMapYScale.scale }
      : {}),
    ...(hasSourceTangents ? { hasSourceTangents: true } : {}),
  };
  return transferred;
}
