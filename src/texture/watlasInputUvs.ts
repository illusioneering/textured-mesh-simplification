import type { Vector2 } from 'three';
import type { TransferredMeshAttributes } from '../simplification/attributes';
import { transferredVertexUv } from '../simplification/attributes';
import type { Barycentric, CollapseHistoryRecord, RawMesh } from '../simplification/types';
import { interpolateVector2 } from '../simplification/barycentric';
import { createHistoryTraceIndex, mapOutputSampleToInput } from '../simplification/successiveMapping';
import type { AtlasInputFaceUvs } from './atlas';
import { faceUvSet, type SourceFaceAttributes, type TexturedRawMesh } from './types';

const CORNER_BARYCENTRICS = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const satisfies readonly Barycentric[];

interface MappedCorner {
  attributes: SourceFaceAttributes;
  barycentric: Barycentric;
}

function commonTexCoordsForCorners(corners: readonly MappedCorner[]): Set<number> {
  const common = new Set(corners[0]?.attributes.uvSets.map((uvSet) => uvSet.texCoord) ?? []);
  for (const corner of corners.slice(1)) {
    const available = new Set(corner.attributes.uvSets.map((uvSet) => uvSet.texCoord));
    for (const texCoord of Array.from(common)) {
      if (!available.has(texCoord)) common.delete(texCoord);
    }
  }
  return common;
}

function chooseTexCoord(commonTexCoords: ReadonlySet<number>, preferredTexCoord: number): number | undefined {
  if (commonTexCoords.has(preferredTexCoord)) return preferredTexCoord;
  return Array.from(commonTexCoords).sort((a, b) => a - b)[0];
}

function transferredFaceUvs(
  rawMesh: RawMesh,
  attributes: TransferredMeshAttributes | undefined,
  texCoord: number,
): AtlasInputFaceUvs | undefined {
  if (!attributes || attributes.vertices.length !== rawMesh.positions.length) return undefined;

  const faceUvs: [Vector2, Vector2, Vector2][] = [];
  for (const face of rawMesh.faces) {
    const corners: Vector2[] = [];
    for (const vertexId of face) {
      const uv = transferredVertexUv(attributes.vertices[vertexId] ?? { uvSets: [] }, texCoord);
      if (!uv) return undefined;
      corners.push(uv.clone());
    }
    faceUvs.push(corners as [Vector2, Vector2, Vector2]);
  }
  return faceUvs;
}

export function deriveWatlasInputFaceUvs(options: {
  source: TexturedRawMesh;
  outputRawMesh: RawMesh;
  outputFaceIds: number[];
  history: CollapseHistoryRecord[];
  preferredTexCoord?: number;
  transferredAttributes?: TransferredMeshAttributes;
}): AtlasInputFaceUvs | undefined {
  const preferredTexCoord = options.preferredTexCoord ?? 0;
  const transferredUvs = transferredFaceUvs(options.outputRawMesh, options.transferredAttributes, preferredTexCoord);
  if (transferredUvs) return transferredUvs;

  const historyIndex = createHistoryTraceIndex(options.history);
  const mappedFaces: MappedCorner[][] = [];
  let commonTexCoords: Set<number> | undefined;

  for (let outputFaceIndex = 0; outputFaceIndex < options.outputRawMesh.faces.length; outputFaceIndex += 1) {
    const mappedCorners = CORNER_BARYCENTRICS.map((outputBarycentric): MappedCorner => {
      const mapped = mapOutputSampleToInput({
        outputRawMesh: options.outputRawMesh,
        outputFaceIds: options.outputFaceIds,
        outputFaceIndex,
        outputBarycentric,
        history: options.history,
        historyIndex,
      });
      const attributes = options.source.faceAttributes[mapped.faceId];
      if (!attributes) throw new Error(`Missing source face attributes for mapped face ${mapped.faceId}.`);
      return { attributes, barycentric: mapped.barycentric };
    });

    const faceCommonTexCoords = commonTexCoordsForCorners(mappedCorners);
    if (commonTexCoords === undefined) {
      commonTexCoords = faceCommonTexCoords;
    } else {
      for (const texCoord of Array.from(commonTexCoords)) {
        if (!faceCommonTexCoords.has(texCoord)) commonTexCoords.delete(texCoord);
      }
    }
    mappedFaces.push(mappedCorners);
  }

  const texCoord = chooseTexCoord(commonTexCoords ?? new Set(), preferredTexCoord);
  if (texCoord === undefined) return undefined;

  return mappedFaces.map((mappedCorners) => mappedCorners.map((corner) => {
    const uvSet = faceUvSet(corner.attributes, texCoord);
    if (!uvSet) throw new Error(`Missing TEXCOORD_${texCoord} while deriving watlas input UVs.`);
    return interpolateVector2(uvSet.uvs[0], uvSet.uvs[1], uvSet.uvs[2], corner.barycentric);
  }) as [Vector2, Vector2, Vector2]);
}
