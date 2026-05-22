import type { Barycentric, CollapseHistoryRecord, RawMesh } from './types';
import { pointFromBarycentric } from './barycentric';
import { closestPointOnTriangle } from './projection';

export interface MappedSample {
  faceId: number;
  barycentric: Barycentric;
  distanceSquared: number;
  projections: number;
}

interface IndexedHistoryRecord {
  historyIndex: number;
  record: CollapseHistoryRecord;
}

export interface HistoryTraceIndex {
  recordsByAfterFaceId: ReadonlyMap<number, readonly IndexedHistoryRecord[]>;
}

export function createHistoryTraceIndex(history: readonly CollapseHistoryRecord[]): HistoryTraceIndex {
  const recordsByAfterFaceId = new Map<number, IndexedHistoryRecord[]>();
  for (let historyIndex = history.length - 1; historyIndex >= 0; historyIndex -= 1) {
    const record = history[historyIndex]!;
    for (const faceId of record.afterFaceIds) {
      let records = recordsByAfterFaceId.get(faceId);
      if (!records) {
        records = [];
        recordsByAfterFaceId.set(faceId, records);
      }
      records.push({ historyIndex, record });
    }
  }
  return { recordsByAfterFaceId };
}

function nextIndexedRecordBefore(
  records: readonly IndexedHistoryRecord[],
  exclusiveUpperHistoryIndex: number,
): IndexedHistoryRecord | undefined {
  let low = 0;
  let high = records.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (records[middle]!.historyIndex < exclusiveUpperHistoryIndex) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return records[low];
}

export function mapOutputSampleToInput(options: {
  outputRawMesh: RawMesh;
  outputFaceIds: number[];
  outputFaceIndex: number;
  outputBarycentric: Barycentric;
  history: CollapseHistoryRecord[];
  historyIndex?: HistoryTraceIndex;
}): MappedSample {
  const face = options.outputRawMesh.faces[options.outputFaceIndex];
  const faceId = options.outputFaceIds[options.outputFaceIndex];
  if (!face) throw new Error(`Missing output face at index ${options.outputFaceIndex}.`);
  if (faceId === undefined) throw new Error(`Missing output face id at index ${options.outputFaceIndex}.`);
  const a = options.outputRawMesh.positions[face[0]];
  const b = options.outputRawMesh.positions[face[1]];
  const c = options.outputRawMesh.positions[face[2]];
  if (!a || !b || !c) throw new Error(`Output face ${options.outputFaceIndex} references a missing vertex.`);

  const finalPoint = pointFromBarycentric(a, b, c, options.outputBarycentric);
  let currentFaceId = faceId;
  let currentBarycentric: Barycentric = [...options.outputBarycentric] as Barycentric;
  let distanceSquared = 0;
  let projections = 0;

  const applyRecord = (record: CollapseHistoryRecord): void => {
    let best: { faceId: number; barycentric: Barycentric; distanceSquared: number } | undefined;
    for (const snapshot of record.beforeFaces) {
      const projected = closestPointOnTriangle(finalPoint, snapshot.positions[0], snapshot.positions[1], snapshot.positions[2]);
      if (!best || projected.distanceSquared < best.distanceSquared) {
        best = {
          faceId: snapshot.faceId,
          barycentric: projected.barycentric,
          distanceSquared: projected.distanceSquared,
        };
      }
    }
    if (best) {
      currentFaceId = best.faceId;
      currentBarycentric = best.barycentric;
      distanceSquared = best.distanceSquared;
      projections += 1;
    }
  };

  if (options.historyIndex) {
    let exclusiveUpperHistoryIndex = options.history.length;
    while (true) {
      const records = options.historyIndex.recordsByAfterFaceId.get(currentFaceId);
      if (!records) break;
      const indexed = nextIndexedRecordBefore(records, exclusiveUpperHistoryIndex);
      if (!indexed) break;
      exclusiveUpperHistoryIndex = indexed.historyIndex;
      applyRecord(indexed.record);
    }
  } else {
    for (let i = options.history.length - 1; i >= 0; i -= 1) {
      const record = options.history[i]!;
      if (!record.afterFaceIds.includes(currentFaceId)) continue;
      applyRecord(record);
    }
  }

  return { faceId: currentFaceId, barycentric: currentBarycentric, distanceSquared, projections };
}
