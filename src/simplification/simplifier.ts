import type {
  RawMesh,
  RawSimplificationResult,
  SimplificationProgress,
  SimplificationResult,
  SimplifyOptions,
  VirtualEdgeDiagnostics,
  VirtualEdgeOptions,
  VirtualEdgeProgress,
} from './types';
import { SimplificationMesh } from './mesh';
import { CandidateQueue } from './priorityQueue';
import { addVirtualEdges } from './virtualEdges';
import { MIN_GENERATED_FACE_QUALITY, triangleQualityFromCoordinates } from './faceQuality';

export interface NormalizedOptions {
  targetFaceCount: number;
  virtualEdges: VirtualEdgeOptions;
  maxIterations: number;
}

interface CandidateData {
  edgeId: number;
  edgeVersion: number;
  x: number;
  y: number;
  z: number;
  cost: number;
}

interface CollapseResult {
  collapsed: boolean;
  affectedEdges: number[];
}

export interface SimplificationState {
  mesh: SimplificationMesh;
  normalized: NormalizedOptions;
  physicalEdges: number;
  virtualEdges: number;
  virtualEdgeDiagnostics?: VirtualEdgeDiagnostics;
  history: RawSimplificationResult['history'];
  collapses: number;
  stoppedReason: SimplificationResult['stats']['stoppedReason'];
  virtualEdgeStageComplete: boolean;
  simplificationStageComplete: boolean;
}

export interface SimplificationStageResult {
  collapses: number;
  stoppedReason: SimplificationResult['stats']['stoppedReason'];
}

type CandidatePosition = readonly [number, number, number];

function requireFiniteNumber(name: string, value: number): void {
  if (!Number.isFinite(value)) throw new Error(`${name} must be a finite number.`);
}

function normalizeOptions(inputFaces: number, options: SimplifyOptions): NormalizedOptions {
  const hasTargetFaces = options.targetFaceCount !== undefined;
  const hasTargetRatio = options.targetRatio !== undefined;
  if (hasTargetFaces && hasTargetRatio) throw new Error('Specify either targetFaceCount or targetRatio, not both.');

  let targetFaceCount: number;
  if (hasTargetFaces) {
    requireFiniteNumber('targetFaceCount', options.targetFaceCount!);
    if (!Number.isInteger(options.targetFaceCount!) || options.targetFaceCount! < 0) {
      throw new Error('targetFaceCount must be a non-negative integer.');
    }
    targetFaceCount = options.targetFaceCount!;
  } else {
    const targetRatio = options.targetRatio ?? 0.5;
    requireFiniteNumber('targetRatio', targetRatio);
    if (targetRatio <= 0 || targetRatio > 1) throw new Error('targetRatio must be greater than 0 and less than or equal to 1.');
    targetFaceCount = Math.floor(inputFaces * targetRatio);
  }

  const virtualEdges = options.virtualEdges ?? { mode: 'auto-local-radius' };
  if (virtualEdges.mode === 'manual-global-radius') {
    requireFiniteNumber('virtual edge radius', virtualEdges.radius);
    if (virtualEdges.radius < 0) throw new Error('Virtual edge radius must be non-negative.');
  } else if (virtualEdges.mode !== 'auto-local-radius' && virtualEdges.mode !== 'auto-global-radius') {
    throw new Error('Unsupported virtual edge mode.');
  }

  const maxIterations = options.maxIterations ?? Number.POSITIVE_INFINITY;
  if (options.maxIterations !== undefined) {
    requireFiniteNumber('maxIterations', options.maxIterations);
    if (!Number.isInteger(options.maxIterations) || options.maxIterations <= 0) {
      throw new Error('maxIterations must be a positive integer when provided.');
    }
  }

  return {
    targetFaceCount,
    virtualEdges,
    maxIterations,
  };
}

function validateRawOutput(result: RawMesh, inputFaces: number): void {
  for (const [index, position] of result.positions.entries()) {
    if (!Number.isFinite(position.x) || !Number.isFinite(position.y) || !Number.isFinite(position.z)) {
      throw new Error(`Output position ${index} contains a non-finite coordinate.`);
    }
  }
  for (const [index, face] of result.faces.entries()) {
    if (new Set(face).size !== 3) throw new Error(`Output face ${index} is degenerate.`);
    for (const vertexId of face) {
      if (vertexId < 0 || vertexId >= result.positions.length) {
        throw new Error(`Output face ${index} references missing vertex ${vertexId}.`);
      }
    }
  }
  if (result.faces.length > inputFaces) throw new Error('Simplification increased the face count.');
}

function addQuadric(out: number[], other: readonly number[], scale = 1): void {
  for (let i = 0; i < 13; i += 1) out[i] = out[i]! + other[i]! * scale;
}

function vertexQuadric(mesh: SimplificationMesh, vertexId: number): number[] {
  const offset = vertexId * 13;
  const out = new Array<number>(13);
  for (let i = 0; i < 13; i += 1) out[i] = mesh.quadrics[offset + i]!;
  return out;
}

function evaluateQuadric(q: readonly number[], x: number, y: number, z: number): number {
  const ax = q[0]! * x + q[1]! * y + q[2]! * z;
  const ay = q[3]! * x + q[4]! * y + q[5]! * z;
  const az = q[6]! * x + q[7]! * y + q[8]! * z;
  const value = x * ax + y * ay + z * az + 2 * (q[9]! * x + q[10]! * y + q[11]! * z) + q[12]!;
  return Math.max(0, value);
}

function determinant3(a: readonly number[]): number {
  return (
    a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!)
    - a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!)
    + a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!)
  );
}

function solve3x3(q: readonly number[]): [number, number, number] | null {
  const det = determinant3(q);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  const inv = [
    (q[4]! * q[8]! - q[5]! * q[7]!) * invDet,
    (q[2]! * q[7]! - q[1]! * q[8]!) * invDet,
    (q[1]! * q[5]! - q[2]! * q[4]!) * invDet,
    (q[5]! * q[6]! - q[3]! * q[8]!) * invDet,
    (q[0]! * q[8]! - q[2]! * q[6]!) * invDet,
    (q[2]! * q[3]! - q[0]! * q[5]!) * invDet,
    (q[3]! * q[7]! - q[4]! * q[6]!) * invDet,
    (q[1]! * q[6]! - q[0]! * q[7]!) * invDet,
    (q[0]! * q[4]! - q[1]! * q[3]!) * invDet,
  ];
  const rhsX = -q[9]!;
  const rhsY = -q[10]!;
  const rhsZ = -q[11]!;
  const x = inv[0]! * rhsX + inv[1]! * rhsY + inv[2]! * rhsZ;
  const y = inv[3]! * rhsX + inv[4]! * rhsY + inv[5]! * rhsZ;
  const z = inv[6]! * rhsX + inv[7]! * rhsY + inv[8]! * rhsZ;
  return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) ? [x, y, z] : null;
}

function boundaryAreaQuadric(mesh: SimplificationMesh, edgeId: number): number[] {
  const a = mesh.edgeA[edgeId]!;
  const b = mesh.edgeB[edgeId]!;
  const [ax, ay, az] = mesh.position(a);
  const [bx, by, bz] = mesh.position(b);
  const sx = bx - ax;
  const sy = by - ay;
  const sz = bz - az;
  const tx = ay * bz - az * by;
  const ty = az * bx - ax * bz;
  const tz = ax * by - ay * bx;
  const m = [0, -sz, sy, sz, 0, -sx, -sy, sx, 0];
  const q = new Array<number>(13).fill(0);
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      q[row * 3 + col] = 0.5 * (
        m[row]! * m[col]!
        + m[3 + row]! * m[3 + col]!
        + m[6 + row]! * m[6 + col]!
      );
    }
  }
  const mt = [
    m[0]! * tx + m[1]! * ty + m[2]! * tz,
    m[3]! * tx + m[4]! * ty + m[5]! * tz,
    m[6]! * tx + m[7]! * ty + m[8]! * tz,
  ];
  q[9] = -0.5 * mt[0]!;
  q[10] = -0.5 * mt[1]!;
  q[11] = -0.5 * mt[2]!;
  q[12] = 0.5 * (tx * tx + ty * ty + tz * tz);
  return q;
}

function isBoundaryEdge(mesh: SimplificationMesh, edgeId: number): boolean {
  return Boolean(mesh.edgeActive[edgeId]) && !mesh.edgeVirtual[edgeId] && mesh.activeIncidentFaceCount(edgeId) !== 2;
}

function areaQuadricForEdge(mesh: SimplificationMesh, edgeId: number): number[] {
  const q = new Array<number>(13).fill(0);
  const edgeA = mesh.edgeA[edgeId]!;
  const edgeB = mesh.edgeB[edgeId]!;
  const seen = new Set<number>();
  for (const vertexId of [edgeA, edgeB]) {
    for (const incidentEdgeId of mesh.incidentEdgeIds(vertexId)) {
      if (incidentEdgeId === edgeId || seen.has(incidentEdgeId)) continue;
      seen.add(incidentEdgeId);
      if (isBoundaryEdge(mesh, incidentEdgeId)) addQuadric(q, boundaryAreaQuadric(mesh, incidentEdgeId));
    }
  }
  return q;
}

function computeCollapseCandidate(mesh: SimplificationMesh, edgeId: number): CandidateData | null {
  if (!mesh.edgeActive[edgeId]) return null;
  const a = mesh.edgeA[edgeId]!;
  const b = mesh.edgeB[edgeId]!;
  if (!mesh.vertexActive[a] || !mesh.vertexActive[b] || a === b) return null;

  const total = vertexQuadric(mesh, a);
  addQuadric(total, vertexQuadric(mesh, b));
  addQuadric(total, areaQuadricForEdge(mesh, edgeId));

  const [ax, ay, az] = mesh.position(a);
  const [bx, by, bz] = mesh.position(b);
  const midpoint: [number, number, number] = [(ax + bx) * 0.5, (ay + by) * 0.5, (az + bz) * 0.5];
  const solved = solve3x3(total);
  let best = solved ?? [ax, ay, az] as [number, number, number];
  let bestError = evaluateQuadric(total, best[0], best[1], best[2]);
  if (!solved) {
    for (const fallback of [[bx, by, bz] as [number, number, number], midpoint]) {
      const error = evaluateQuadric(total, fallback[0], fallback[1], fallback[2]);
      if (error < bestError) {
        best = fallback;
        bestError = error;
      }
    }
  }
  if (!Number.isFinite(bestError)) return null;
  const candidate = { edgeId, edgeVersion: mesh.edgeVersions[edgeId]!, x: best[0], y: best[1], z: best[2], cost: bestError };
  if (collapseWouldWorsenFaceIntoSliver(mesh, candidate, a, b)) return null;
  return candidate;
}

function hasRepeatedVertices(face: readonly [number, number, number]): boolean {
  return face[0] === face[1] || face[1] === face[2] || face[0] === face[2];
}

function vertexCoordinatesAfterCollapse(mesh: SimplificationMesh, vertexId: number, keep: number, candidatePosition: CandidatePosition): CandidatePosition {
  return vertexId === keep ? candidatePosition : mesh.position(vertexId);
}

function triangleQualityAfterCollapse(
  mesh: SimplificationMesh,
  vertices: readonly [number, number, number],
  keep: number,
  candidatePosition: CandidatePosition,
): number {
  const a = vertexCoordinatesAfterCollapse(mesh, vertices[0], keep, candidatePosition);
  const b = vertexCoordinatesAfterCollapse(mesh, vertices[1], keep, candidatePosition);
  const c = vertexCoordinatesAfterCollapse(mesh, vertices[2], keep, candidatePosition);
  return triangleQualityFromCoordinates(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}

function collapseWouldWorsenFaceIntoSliver(mesh: SimplificationMesh, candidate: CandidateData, keep: number, remove: number): boolean {
  const candidatePosition: CandidatePosition = [candidate.x, candidate.y, candidate.z];
  for (const faceId of mesh.activeIncidentFaceIds([keep, remove])) {
    const currentQuality = mesh.triangleQualityByFace(faceId);
    const oldVertices = mesh.faceVertices(faceId);
    const nextVertices: [number, number, number] = [
      oldVertices[0] === remove ? keep : oldVertices[0],
      oldVertices[1] === remove ? keep : oldVertices[1],
      oldVertices[2] === remove ? keep : oldVertices[2],
    ];
    if (hasRepeatedVertices(nextVertices)) continue;
    if (!mesh.canRegisterFaceVertices(faceId, nextVertices[0], nextVertices[1], nextVertices[2])) continue;

    const nextQuality = triangleQualityAfterCollapse(mesh, nextVertices, keep, candidatePosition);
    if (nextQuality < MIN_GENERATED_FACE_QUALITY && nextQuality < currentQuality) return true;
  }
  return false;
}

function collectAffectedVertices(mesh: SimplificationMesh, edgeIds: readonly number[], keep: number, remove: number): number[] {
  const seen = new Set<number>([keep, remove]);
  for (const edgeId of edgeIds) {
    if (edgeId < 0 || edgeId >= mesh.edgeCount()) continue;
    seen.add(mesh.edgeA[edgeId]!);
    seen.add(mesh.edgeB[edgeId]!);
  }
  return [...seen];
}

function redirectVirtualEdges(mesh: SimplificationMesh, remove: number, keep: number, incidentEdgeIds: readonly number[]): number[] {
  const redirected: number[] = [];
  for (const edgeId of incidentEdgeIds) {
    if (!mesh.edgeActive[edgeId] || !mesh.edgeVirtual[edgeId]) continue;
    const other = mesh.edgeA[edgeId] === remove ? mesh.edgeB[edgeId]! : mesh.edgeA[edgeId]!;
    mesh.deactivateEdge(edgeId);
    if (other !== keep && mesh.vertexActive[other]) {
      const redirectedEdgeId = mesh.addVirtualEdge(keep, other);
      if (redirectedEdgeId !== null) {
        mesh.edgeVersions[redirectedEdgeId] = mesh.edgeVersions[redirectedEdgeId]! + 1;
        redirected.push(redirectedEdgeId);
      }
    }
  }
  return redirected;
}

function collapseEdge(mesh: SimplificationMesh, candidate: CandidateData): CollapseResult {
  const edgeId = candidate.edgeId;
  if (!mesh.edgeActive[edgeId] || mesh.edgeVersions[edgeId] !== candidate.edgeVersion) return { collapsed: false, affectedEdges: [] };
  const keep = mesh.edgeA[edgeId]!;
  const remove = mesh.edgeB[edgeId]!;
  if (!mesh.vertexActive[keep] || !mesh.vertexActive[remove] || keep === remove) return { collapsed: false, affectedEdges: [] };

  const incidentEdgeIds = [...new Set([...mesh.incidentEdgeIds(keep), ...mesh.incidentEdgeIds(remove)])];
  const facesToUpdate = mesh.activeIncidentFaceIds([remove]);
  const affectedVertices = collectAffectedVertices(mesh, incidentEdgeIds, keep, remove);

  mesh.setPosition(keep, candidate.x, candidate.y, candidate.z);
  mesh.addVertexQuadricInPlace(keep, remove);
  mesh.vertexVersions[keep] = mesh.vertexVersions[keep]! + 1;
  mesh.deactivateVertex(remove);

  for (const faceId of facesToUpdate) {
    if (!mesh.faceActive[faceId]) continue;
    const oldVertices = mesh.faceVertices(faceId);
    mesh.unregisterFaceKey(faceId, oldVertices);
    mesh.removeFaceReferences(faceId, oldVertices);
    mesh.faceA[faceId] = oldVertices[0] === remove ? keep : oldVertices[0];
    mesh.faceB[faceId] = oldVertices[1] === remove ? keep : oldVertices[1];
    mesh.faceC[faceId] = oldVertices[2] === remove ? keep : oldVertices[2];
    if (mesh.isFaceDegenerate(faceId) || !mesh.canRegisterFace(faceId)) {
      mesh.deactivateFace(faceId);
      continue;
    }
    mesh.registerFace(faceId);
    mesh.addFaceReferences(faceId);
  }

  const redirected = redirectVirtualEdges(mesh, remove, keep, incidentEdgeIds);
  const affectedEdges = new Set<number>([...incidentEdgeIds, ...redirected]);
  for (const vertexId of affectedVertices) {
    if (!mesh.vertexActive[vertexId]) continue;
    for (const incidentEdgeId of mesh.incidentEdgeIds(vertexId)) {
      if (mesh.edgeActive[incidentEdgeId]) {
        mesh.edgeVersions[incidentEdgeId] = mesh.edgeVersions[incidentEdgeId]! + 1;
        affectedEdges.add(incidentEdgeId);
      }
    }
  }
  return { collapsed: true, affectedEdges: [...affectedEdges] };
}

function enqueue(mesh: SimplificationMesh, queue: CandidateQueue, edgeId: number): void {
  if (!mesh.edgeActive[edgeId]) {
    queue.remove(edgeId);
    return;
  }
  const candidate = computeCollapseCandidate(mesh, edgeId);
  if (!candidate) {
    queue.remove(edgeId);
    return;
  }
  queue.upsert(edgeId, candidate.cost, candidate.x, candidate.y, candidate.z);
}

function virtualEdgeDiagnosticsFromSearch(search: ReturnType<typeof addVirtualEdges>): VirtualEdgeDiagnostics {
  return {
    mode: search.mode,
    ...(search.radius !== undefined ? { radius: search.radius } : {}),
    ...(search.radiusScale !== undefined ? { radiusScale: search.radiusScale } : {}),
    ...(search.clampMin !== undefined ? { clampMin: search.clampMin } : {}),
    ...(search.clampMax !== undefined ? { clampMax: search.clampMax } : {}),
    ...(search.bboxDistanceLimit !== undefined ? { bboxDistanceLimit: search.bboxDistanceLimit } : {}),
    ...(search.maxPairsPerComponentPair !== undefined ? { maxPairsPerComponentPair: search.maxPairsPerComponentPair } : {}),
    componentCount: search.componentCount,
    faceCount: search.faceCount,
    searchStrategy: search.searchStrategy,
    candidateFacePairs: search.candidateFacePairs,
    exactDistanceTests: search.exactDistanceTests,
    aabbRejectedPairs: search.aabbRejectedPairs,
    distanceRejectedPairs: search.distanceRejectedPairs,
    duplicateVertexPairCandidates: search.duplicateVertexPairCandidates,
    cappedVirtualEdgeCandidates: search.cappedVirtualEdgeCandidates,
    generatedVirtualEdges: search.generatedVirtualEdges,
  };
}

export function createSimplificationState(rawMesh: RawMesh, options: SimplifyOptions = {}): SimplificationState {
  const normalized = normalizeOptions(rawMesh.faces.length, options);
  const mesh = SimplificationMesh.fromRawMesh(rawMesh);
  mesh.computeInitialQuadrics();

  const physicalEdges = mesh.activeEdgeCount(false);
  return {
    mesh,
    normalized,
    physicalEdges,
    virtualEdges: 0,
    history: [],
    collapses: 0,
    stoppedReason: 'target-reached',
    virtualEdgeStageComplete: false,
    simplificationStageComplete: false,
  };
}

export function runVirtualEdgeStage(
  state: SimplificationState,
  onProgress?: (progress: VirtualEdgeProgress) => void,
): VirtualEdgeDiagnostics {
  if (state.virtualEdgeStageComplete) {
    return state.virtualEdgeDiagnostics!;
  }

  const search = addVirtualEdges(state.mesh, state.normalized.virtualEdges, onProgress);
  const diagnostics = virtualEdgeDiagnosticsFromSearch(search);
  state.virtualEdges = search.added;
  state.virtualEdgeDiagnostics = diagnostics;
  state.virtualEdgeStageComplete = true;
  return diagnostics;
}

export function runSimplificationStage(
  state: SimplificationState,
  onProgress?: (stats: SimplificationProgress) => void,
): SimplificationStageResult {
  if (!state.virtualEdgeStageComplete) {
    throw new Error('Run virtual-edge search before simplification.');
  }
  if (state.simplificationStageComplete) {
    return { collapses: state.collapses, stoppedReason: state.stoppedReason };
  }

  const mesh = state.mesh;
  const queue = new CandidateQueue(mesh.edgeCapacity());
  for (let edgeId = 0; edgeId < mesh.edgeCount(); edgeId += 1) enqueue(mesh, queue, edgeId);

  let stoppedReason: SimplificationResult['stats']['stoppedReason'] = 'target-reached';
  while (mesh.activeFaceCount() > state.normalized.targetFaceCount) {
    if (state.collapses >= state.normalized.maxIterations) {
      stoppedReason = 'max-iterations';
      break;
    }
    const queued = queue.pop();
    if (!queued) {
      stoppedReason = 'queue-empty';
      break;
    }
    if (!mesh.edgeActive[queued.edgeId]) continue;
    const fresh = computeCollapseCandidate(mesh, queued.edgeId);
    if (!fresh) continue;

    const keepVertexId = mesh.edgeA[queued.edgeId]!;
    const removedVertexId = mesh.edgeB[queued.edgeId]!;
    const beforeFaces = mesh.snapshotIncidentFaces([keepVertexId, removedVertexId]);
    const result = collapseEdge(mesh, fresh);
    if (!result.collapsed) continue;
    state.collapses += 1;
    state.history.push({
      keepVertexId,
      removedVertexId,
      beforeFaces,
      afterFaceIds: mesh.activeIncidentFaceIds([keepVertexId]),
    });
    for (const edgeId of result.affectedEdges) enqueue(mesh, queue, edgeId);
    onProgress?.({
      iteration: state.collapses,
      activeFaces: mesh.activeFaceCount(),
      activeVertices: mesh.activeVertexCount(),
      activeEdges: mesh.activeEdgeCount(true),
      lastCost: fresh.cost,
    } satisfies SimplificationProgress);
  }

  if (mesh.activeFaceCount() <= state.normalized.targetFaceCount) stoppedReason = 'target-reached';
  state.stoppedReason = stoppedReason;
  state.simplificationStageComplete = true;
  return { collapses: state.collapses, stoppedReason: state.stoppedReason };
}

export function finalizeSimplification(state: SimplificationState): RawSimplificationResult {
  if (!state.virtualEdgeStageComplete) {
    throw new Error('Run virtual-edge search before finalizing simplification.');
  }
  if (!state.simplificationStageComplete) {
    throw new Error('Run simplification before finalizing simplification.');
  }

  const mesh = state.mesh;
  const output = mesh.toRawMeshWithMaps();
  validateRawOutput(output.rawMesh, mesh.inputFaceCount);
  return {
    rawMesh: output.rawMesh,
    outputFaceIds: output.outputFaceIds,
    history: state.history,
    stats: {
      inputVertices: mesh.vertexCount,
      inputFaces: mesh.inputFaceCount,
      outputVertices: output.rawMesh.positions.length,
      outputFaces: output.rawMesh.faces.length,
      physicalEdges: state.physicalEdges,
      virtualEdges: state.virtualEdges,
      ...(state.virtualEdgeDiagnostics ? { virtualEdgeDiagnostics: state.virtualEdgeDiagnostics } : {}),
      collapses: state.collapses,
      stoppedReason: state.stoppedReason,
    },
  };
}

export function simplifyRawMeshCore(rawMesh: RawMesh, options: SimplifyOptions = {}): RawSimplificationResult {
  const state = createSimplificationState(rawMesh, options);
  runVirtualEdgeStage(state, options.onVirtualEdgeProgress);
  runSimplificationStage(state, options.onProgress);
  return finalizeSimplification(state);
}
