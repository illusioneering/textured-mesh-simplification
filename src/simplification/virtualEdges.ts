import type { VirtualEdgeOptions, VirtualEdgeProgress } from './types';
import { SimplificationMesh } from './mesh';

const EMPTY = -1;
const MIN_INT32 = -2147483648;
const MAX_INT32 = 2147483647;

export interface ConnectedComponents {
  vertexComponents: Int32Array;
  faceComponents: Int32Array;
  count: number;
}

export interface VirtualEdgeSearchStats {
  added: number;
  mode: VirtualEdgeOptions['mode'];
  radius?: number;
  radiusScale?: number;
  clampMin?: number;
  clampMax?: number;
  bboxDistanceLimit?: number;
  maxPairsPerComponentPair?: number | null;
  componentCount: number;
  faceCount: number;
  searchStrategy: 'grid-exact';
  candidateFacePairs: number;
  exactDistanceTests: number;
  aabbRejectedPairs: number;
  distanceRejectedPairs: number;
  duplicateVertexPairCandidates: number;
  cappedVirtualEdgeCandidates: number;
  generatedVirtualEdges: number;
}

export interface LocalTriangleRadii {
  mode: 'auto-local-radius';
  faceRadii: Float64Array;
  radiusScale: number;
  clampMin: number;
  clampMax: number;
  bboxDistanceLimit: number;
  searchDistance: number;
}

export interface AutoGlobalRadius {
  mode: 'auto-global-radius';
  radius: number;
  medianEdgeLength: number;
  radiusScale: number;
  bboxDistanceLimit: number;
}

interface NormalizedVirtualEdgeOptions {
  mode: VirtualEdgeOptions['mode'];
  radius?: number;
  maxPairsPerComponentPair?: number | null;
}

interface VirtualEdgeCandidate {
  faceA: number;
  faceB: number;
  vertexA: number;
  vertexB: number;
  distanceSquared: number;
}

const AUTO_LOCAL_RADIUS_SCALE = 0.5;
const AUTO_LOCAL_CLAMP_LOW = 0.05;
const AUTO_LOCAL_CLAMP_HIGH = 0.95;
const AUTO_LOCAL_BBOX_DISTANCE_RATIO = 0.02;
const AUTO_GLOBAL_RADIUS_SCALE = 0.5;
const AUTO_GLOBAL_BBOX_DISTANCE_RATIO = 0.02;
interface Aabb {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
}

interface CellRange {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
}

function growInt32(source: Int32Array<ArrayBuffer>, required: number): Int32Array<ArrayBuffer> {
  if (required <= source.length) return source;
  const next = new Int32Array(Math.max(required, source.length * 2, 1));
  next.set(source);
  return next;
}

function createFilledInt32(length: number, value: number): Int32Array<ArrayBuffer> {
  const out = new Int32Array(length);
  out.fill(value);
  return out;
}

function cellCoordinate(value: number, cellSize: number): number {
  const coordinate = Math.floor(value / cellSize);
  if (coordinate < MIN_INT32 || coordinate > MAX_INT32) {
    throw new Error('Virtual-edge grid cell coordinate exceeds supported 32-bit range.');
  }
  return coordinate;
}

function cellHash(x: number, y: number, z: number): number {
  return Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791);
}

class CellBuckets {
  private readonly hashHeads = new Map<number, number>();
  private cellX = new Int32Array(1024);
  private cellY = new Int32Array(1024);
  private cellZ = new Int32Array(1024);
  private cellFaceHead = createFilledInt32(1024, EMPTY);
  private cellHashNext = createFilledInt32(1024, EMPTY);
  private refFace = new Int32Array(4096);
  private refNext = createFilledInt32(4096, EMPTY);
  private cellCount = 0;
  private refCount = 0;

  findCell(x: number, y: number, z: number): number {
    const hash = cellHash(x, y, z);
    let cellId = this.hashHeads.get(hash) ?? EMPTY;
    while (cellId !== EMPTY) {
      if (this.cellX[cellId] === x && this.cellY[cellId] === y && this.cellZ[cellId] === z) return cellId;
      cellId = this.cellHashNext[cellId]!;
    }
    return EMPTY;
  }

  ensureCell(x: number, y: number, z: number): number {
    const existing = this.findCell(x, y, z);
    if (existing !== EMPTY) return existing;
    const cellId = this.cellCount;
    this.ensureCellCapacity(cellId + 1);
    const hash = cellHash(x, y, z);
    this.cellX[cellId] = x;
    this.cellY[cellId] = y;
    this.cellZ[cellId] = z;
    this.cellFaceHead[cellId] = EMPTY;
    this.cellHashNext[cellId] = this.hashHeads.get(hash) ?? EMPTY;
    this.hashHeads.set(hash, cellId);
    this.cellCount += 1;
    return cellId;
  }

  addFace(cellId: number, faceId: number): void {
    const refId = this.refCount;
    this.ensureRefCapacity(refId + 1);
    this.refFace[refId] = faceId;
    this.refNext[refId] = this.cellFaceHead[cellId]!;
    this.cellFaceHead[cellId] = refId;
    this.refCount += 1;
  }

  forEachFace(cellId: number, visit: (faceId: number) => void): void {
    for (let refId = this.cellFaceHead[cellId]!; refId !== EMPTY; refId = this.refNext[refId]!) {
      visit(this.refFace[refId]!);
    }
  }

  private ensureCellCapacity(required: number): void {
    if (required <= this.cellX.length) return;
    const oldLength = this.cellX.length;
    this.cellX = growInt32(this.cellX, required);
    this.cellY = growInt32(this.cellY, required);
    this.cellZ = growInt32(this.cellZ, required);
    this.cellFaceHead = growInt32(this.cellFaceHead, required);
    this.cellHashNext = growInt32(this.cellHashNext, required);
    this.cellFaceHead.fill(EMPTY, oldLength);
    this.cellHashNext.fill(EMPTY, oldLength);
  }

  private ensureRefCapacity(required: number): void {
    if (required <= this.refFace.length) return;
    const oldLength = this.refFace.length;
    this.refFace = growInt32(this.refFace, required);
    this.refNext = growInt32(this.refNext, required);
    this.refNext.fill(EMPTY, oldLength);
  }
}

function faceAabb(mesh: SimplificationMesh, faceId: number): Aabb {
  const a = mesh.faceA[faceId]!;
  const b = mesh.faceB[faceId]!;
  const c = mesh.faceC[faceId]!;
  const ai = a * 3;
  const bi = b * 3;
  const ci = c * 3;
  const ax = mesh.positions[ai]!;
  const ay = mesh.positions[ai + 1]!;
  const az = mesh.positions[ai + 2]!;
  const bx = mesh.positions[bi]!;
  const by = mesh.positions[bi + 1]!;
  const bz = mesh.positions[bi + 2]!;
  const cx = mesh.positions[ci]!;
  const cy = mesh.positions[ci + 1]!;
  const cz = mesh.positions[ci + 2]!;
  return {
    minX: Math.min(ax, bx, cx),
    minY: Math.min(ay, by, cy),
    minZ: Math.min(az, bz, cz),
    maxX: Math.max(ax, bx, cx),
    maxY: Math.max(ay, by, cy),
    maxZ: Math.max(az, bz, cz),
  };
}

function writeFaceAabb(mesh: SimplificationMesh, faceId: number, out: Float64Array): void {
  const aabb = faceAabb(mesh, faceId);
  const offset = faceId * 6;
  out[offset] = aabb.minX;
  out[offset + 1] = aabb.minY;
  out[offset + 2] = aabb.minZ;
  out[offset + 3] = aabb.maxX;
  out[offset + 4] = aabb.maxY;
  out[offset + 5] = aabb.maxZ;
}

function readAabb(aabbs: Float64Array, faceId: number): Aabb {
  const offset = faceId * 6;
  return {
    minX: aabbs[offset]!,
    minY: aabbs[offset + 1]!,
    minZ: aabbs[offset + 2]!,
    maxX: aabbs[offset + 3]!,
    maxY: aabbs[offset + 4]!,
    maxZ: aabbs[offset + 5]!,
  };
}

function expandedCellRange(aabb: Aabb, amount: number, cellSize: number): CellRange {
  return {
    minX: cellCoordinate(aabb.minX - amount, cellSize),
    minY: cellCoordinate(aabb.minY - amount, cellSize),
    minZ: cellCoordinate(aabb.minZ - amount, cellSize),
    maxX: cellCoordinate(aabb.maxX + amount, cellSize),
    maxY: cellCoordinate(aabb.maxY + amount, cellSize),
    maxZ: cellCoordinate(aabb.maxZ + amount, cellSize),
  };
}

function cellReferenceCount(range: CellRange): number {
  return (range.maxX - range.minX + 1) * (range.maxY - range.minY + 1) * (range.maxZ - range.minZ + 1);
}

function aabbDistanceSquared(a: Aabb, b: Aabb): number {
  const dx = Math.max(0, Math.max(a.minX - b.maxX, b.minX - a.maxX));
  const dy = Math.max(0, Math.max(a.minY - b.maxY, b.minY - a.maxY));
  const dz = Math.max(0, Math.max(a.minZ - b.maxZ, b.minZ - a.maxZ));
  return dx * dx + dy * dy + dz * dz;
}

function distanceSquared(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  const dz = az - bz;
  return dx * dx + dy * dy + dz * dz;
}

function pointTriangleDistanceSquared(
  px: number,
  py: number,
  pz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const apx = px - ax;
  const apy = py - ay;
  const apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return distanceSquared(px, py, pz, ax, ay, az);

  const bpx = px - bx;
  const bpy = py - by;
  const bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return distanceSquared(px, py, pz, bx, by, bz);

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return distanceSquared(px, py, pz, ax + abx * v, ay + aby * v, az + abz * v);
  }

  const cpx = px - cx;
  const cpy = py - cy;
  const cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return distanceSquared(px, py, pz, cx, cy, cz);

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return distanceSquared(px, py, pz, ax + acx * w, ay + acy * w, az + acz * w);
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return distanceSquared(px, py, pz, bx + (cx - bx) * w, by + (cy - by) * w, bz + (cz - bz) * w);
  }

  const denominator = 1 / (va + vb + vc);
  const v = vb * denominator;
  const w = vc * denominator;
  return distanceSquared(
    px,
    py,
    pz,
    ax + abx * v + acx * w,
    ay + aby * v + acy * w,
    az + abz * v + acz * w,
  );
}

function segmentSegmentDistanceSquared(
  p1x: number,
  p1y: number,
  p1z: number,
  q1x: number,
  q1y: number,
  q1z: number,
  p2x: number,
  p2y: number,
  p2z: number,
  q2x: number,
  q2y: number,
  q2z: number,
): number {
  const d1x = q1x - p1x;
  const d1y = q1y - p1y;
  const d1z = q1z - p1z;
  const d2x = q2x - p2x;
  const d2y = q2y - p2y;
  const d2z = q2z - p2z;
  const rx = p1x - p2x;
  const ry = p1y - p2y;
  const rz = p1z - p2z;
  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;
  const epsilon = 1e-12;
  let s: number;
  let t: number;

  if (a <= epsilon && e <= epsilon) return distanceSquared(p1x, p1y, p1z, p2x, p2y, p2z);
  if (a <= epsilon) {
    s = 0;
    t = Math.min(1, Math.max(0, f / e));
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= epsilon) {
      t = 0;
      s = Math.min(1, Math.max(0, -c / a));
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const denominator = a * e - b * b;
      if (denominator !== 0) s = Math.min(1, Math.max(0, (b * f - c * e) / denominator));
      else s = 0;
      const tNom = b * s + f;
      if (tNom < 0) {
        t = 0;
        s = Math.min(1, Math.max(0, -c / a));
      } else if (tNom > e) {
        t = 1;
        s = Math.min(1, Math.max(0, (b - c) / a));
      } else {
        t = tNom / e;
      }
    }
  }

  return distanceSquared(
    p1x + d1x * s,
    p1y + d1y * s,
    p1z + d1z * s,
    p2x + d2x * t,
    p2y + d2y * t,
    p2z + d2z * t,
  );
}

function vertexPosition(mesh: SimplificationMesh, vertexId: number): [number, number, number] {
  const offset = vertexId * 3;
  return [mesh.positions[offset]!, mesh.positions[offset + 1]!, mesh.positions[offset + 2]!];
}

function triangleTriangleDistanceSquared(mesh: SimplificationMesh, faceA: number, faceB: number): number {
  const a0 = mesh.faceA[faceA]!;
  const a1 = mesh.faceB[faceA]!;
  const a2 = mesh.faceC[faceA]!;
  const b0 = mesh.faceA[faceB]!;
  const b1 = mesh.faceB[faceB]!;
  const b2 = mesh.faceC[faceB]!;
  const [a0x, a0y, a0z] = vertexPosition(mesh, a0);
  const [a1x, a1y, a1z] = vertexPosition(mesh, a1);
  const [a2x, a2y, a2z] = vertexPosition(mesh, a2);
  const [b0x, b0y, b0z] = vertexPosition(mesh, b0);
  const [b1x, b1y, b1z] = vertexPosition(mesh, b1);
  const [b2x, b2y, b2z] = vertexPosition(mesh, b2);

  let best = Number.POSITIVE_INFINITY;
  best = Math.min(best, pointTriangleDistanceSquared(a0x, a0y, a0z, b0x, b0y, b0z, b1x, b1y, b1z, b2x, b2y, b2z));
  best = Math.min(best, pointTriangleDistanceSquared(a1x, a1y, a1z, b0x, b0y, b0z, b1x, b1y, b1z, b2x, b2y, b2z));
  best = Math.min(best, pointTriangleDistanceSquared(a2x, a2y, a2z, b0x, b0y, b0z, b1x, b1y, b1z, b2x, b2y, b2z));
  best = Math.min(best, pointTriangleDistanceSquared(b0x, b0y, b0z, a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z));
  best = Math.min(best, pointTriangleDistanceSquared(b1x, b1y, b1z, a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z));
  best = Math.min(best, pointTriangleDistanceSquared(b2x, b2y, b2z, a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z));

  best = Math.min(best, segmentSegmentDistanceSquared(a0x, a0y, a0z, a1x, a1y, a1z, b0x, b0y, b0z, b1x, b1y, b1z));
  best = Math.min(best, segmentSegmentDistanceSquared(a0x, a0y, a0z, a1x, a1y, a1z, b1x, b1y, b1z, b2x, b2y, b2z));
  best = Math.min(best, segmentSegmentDistanceSquared(a0x, a0y, a0z, a1x, a1y, a1z, b2x, b2y, b2z, b0x, b0y, b0z));
  best = Math.min(best, segmentSegmentDistanceSquared(a1x, a1y, a1z, a2x, a2y, a2z, b0x, b0y, b0z, b1x, b1y, b1z));
  best = Math.min(best, segmentSegmentDistanceSquared(a1x, a1y, a1z, a2x, a2y, a2z, b1x, b1y, b1z, b2x, b2y, b2z));
  best = Math.min(best, segmentSegmentDistanceSquared(a1x, a1y, a1z, a2x, a2y, a2z, b2x, b2y, b2z, b0x, b0y, b0z));
  best = Math.min(best, segmentSegmentDistanceSquared(a2x, a2y, a2z, a0x, a0y, a0z, b0x, b0y, b0z, b1x, b1y, b1z));
  best = Math.min(best, segmentSegmentDistanceSquared(a2x, a2y, a2z, a0x, a0y, a0z, b1x, b1y, b1z, b2x, b2y, b2z));
  best = Math.min(best, segmentSegmentDistanceSquared(a2x, a2y, a2z, a0x, a0y, a0z, b2x, b2y, b2z, b0x, b0y, b0z));
  return best;
}

function closestVertexPair(mesh: SimplificationMesh, faceA: number, faceB: number): [number, number] {
  const aVertices = [mesh.faceA[faceA]!, mesh.faceB[faceA]!, mesh.faceC[faceA]!];
  const bVertices = [mesh.faceA[faceB]!, mesh.faceB[faceB]!, mesh.faceC[faceB]!];
  let bestA = aVertices[0]!;
  let bestB = bVertices[0]!;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const a of aVertices) {
    const ai = a * 3;
    const ax = mesh.positions[ai]!;
    const ay = mesh.positions[ai + 1]!;
    const az = mesh.positions[ai + 2]!;
    for (const b of bVertices) {
      const bi = b * 3;
      const distance = distanceSquared(ax, ay, az, mesh.positions[bi]!, mesh.positions[bi + 1]!, mesh.positions[bi + 2]!);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestA = a;
        bestB = b;
      }
    }
  }
  return [bestA, bestB];
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]!
    : (sorted[mid - 1]! + sorted[mid]!) * 0.5;
}

function percentile(sortedValues: readonly number[], fraction: number): number {
  if (sortedValues.length === 0) return 0;
  if (sortedValues.length === 1) return sortedValues[0]!;
  const index = (sortedValues.length - 1) * fraction;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) return sortedValues[low]!;
  const t = index - low;
  return sortedValues[low]! * (1 - t) + sortedValues[high]! * t;
}

function sortedPositivePhysicalEdgeLengths(mesh: SimplificationMesh): number[] {
  const lengths: number[] = [];
  for (let edgeId = 0; edgeId < mesh.edgeCount(); edgeId += 1) {
    if (!mesh.edgeActive[edgeId] || mesh.edgeVirtual[edgeId]) continue;
    const length = mesh.edgeLength(edgeId);
    if (length > 0 && Number.isFinite(length)) lengths.push(length);
  }
  return lengths.sort((a, b) => a - b);
}

function meshBoundingBoxDiagonal(mesh: SimplificationMesh): number {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let vertexId = 0; vertexId < mesh.vertexCount; vertexId += 1) {
    if (!mesh.vertexActive[vertexId]) continue;
    const offset = vertexId * 3;
    const x = mesh.positions[offset]!;
    const y = mesh.positions[offset + 1]!;
    const z = mesh.positions[offset + 2]!;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  if (!Number.isFinite(minX)) return 0;
  return Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
}

function facePositiveEdgeLengths(mesh: SimplificationMesh, faceId: number): number[] {
  const face = mesh.faceVertices(faceId);
  const lengths: number[] = [];
  for (const [a, b] of [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] as const) {
    const edgeId = mesh.getEdgeBetween(a, b);
    if (edgeId === undefined) continue;
    const length = mesh.edgeLength(edgeId);
    if (length > 0 && Number.isFinite(length)) lengths.push(length);
  }
  return lengths;
}

function oneRingFaceIds(mesh: SimplificationMesh, faceId: number): number[] {
  const face = mesh.faceVertices(faceId);
  const seen = new Set<number>([faceId]);
  for (const [a, b] of [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] as const) {
    const edgeId = mesh.getEdgeBetween(a, b);
    if (edgeId === undefined || mesh.edgeVirtual[edgeId]) continue;
    for (const incidentFaceId of mesh.edgeIncidentFaces(edgeId)) {
      if (mesh.faceActive[incidentFaceId]) seen.add(incidentFaceId);
    }
  }
  return [...seen];
}

export function computeLocalTriangleRadii(mesh: SimplificationMesh): LocalTriangleRadii {
  const sortedLengths = sortedPositivePhysicalEdgeLengths(mesh);
  const clampMin = percentile(sortedLengths, AUTO_LOCAL_CLAMP_LOW);
  const clampMax = percentile(sortedLengths, AUTO_LOCAL_CLAMP_HIGH);
  const fallbackScale = median(sortedLengths);
  const faceRadii = new Float64Array(mesh.inputFaceCount);
  let maxRadius = 0;

  for (let faceId = 0; faceId < mesh.inputFaceCount; faceId += 1) {
    if (!mesh.faceActive[faceId]) continue;
    const oneRingLengths: number[] = [];
    for (const neighborFaceId of oneRingFaceIds(mesh, faceId)) {
      oneRingLengths.push(...facePositiveEdgeLengths(mesh, neighborFaceId));
    }
    const ownLengths = facePositiveEdgeLengths(mesh, faceId);
    const rawScale = median(oneRingLengths.length > 0 ? oneRingLengths : ownLengths) || fallbackScale;
    const clampedScale = clampMax > 0
      ? Math.min(clampMax, Math.max(clampMin, rawScale))
      : rawScale;
    const radius = Math.max(0, clampedScale * AUTO_LOCAL_RADIUS_SCALE);
    faceRadii[faceId] = radius;
    maxRadius = Math.max(maxRadius, radius);
  }

  const bboxDistanceLimit = meshBoundingBoxDiagonal(mesh) * AUTO_LOCAL_BBOX_DISTANCE_RATIO;
  return {
    mode: 'auto-local-radius',
    faceRadii,
    radiusScale: AUTO_LOCAL_RADIUS_SCALE,
    clampMin,
    clampMax,
    bboxDistanceLimit,
    searchDistance: bboxDistanceLimit > 0 ? Math.min(maxRadius * 2, bboxDistanceLimit) : maxRadius * 2,
  };
}

export function computeAutoGlobalRadius(mesh: SimplificationMesh): AutoGlobalRadius {
  const medianEdgeLength = median(sortedPositivePhysicalEdgeLengths(mesh));
  const rawRadius = medianEdgeLength * AUTO_GLOBAL_RADIUS_SCALE;
  const bboxDistanceLimit = meshBoundingBoxDiagonal(mesh) * AUTO_GLOBAL_BBOX_DISTANCE_RATIO;
  const radius = bboxDistanceLimit > 0 ? Math.min(rawRadius, bboxDistanceLimit * 0.5) : rawRadius;
  return {
    mode: 'auto-global-radius',
    radius,
    medianEdgeLength,
    radiusScale: AUTO_GLOBAL_RADIUS_SCALE,
    bboxDistanceLimit,
  };
}

function normalizeVirtualEdgeOptions(options: VirtualEdgeOptions): NormalizedVirtualEdgeOptions {
  if (options.mode === 'auto-local-radius') {
    const maxPairsPerComponentPair = options.maxPairsPerComponentPair === undefined
      ? null
      : options.maxPairsPerComponentPair;
    if (maxPairsPerComponentPair !== null && (!Number.isInteger(maxPairsPerComponentPair) || maxPairsPerComponentPair < 0)) {
      throw new Error('maxPairsPerComponentPair must be a non-negative integer or null.');
    }
    return { mode: 'auto-local-radius', maxPairsPerComponentPair };
  }
  if (options.mode === 'auto-global-radius') {
    return { mode: 'auto-global-radius' };
  }
  if (options.mode === 'manual-global-radius') {
    if (!Number.isFinite(options.radius) || options.radius < 0) throw new Error('Virtual edge radius must be a non-negative finite number.');
    return { mode: 'manual-global-radius', radius: options.radius };
  }
  throw new Error('Unsupported virtual edge mode.');
}

function componentPairKey(a: number, b: number): string {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function vertexPairKey(a: number, b: number): string {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function recordCandidate(
  candidatesByComponentPair: Map<string, VirtualEdgeCandidate[]>,
  candidate: VirtualEdgeCandidate,
  componentA: number,
  componentB: number,
): void {
  const key = componentPairKey(componentA, componentB);
  const bucket = candidatesByComponentPair.get(key);
  if (bucket) bucket.push(candidate);
  else candidatesByComponentPair.set(key, [candidate]);
}

function reportProgress(
  onProgress: ((progress: VirtualEdgeProgress) => void) | undefined,
  phase: VirtualEdgeProgress['phase'],
  processedFaces: number,
  totalFaces: number,
  stats: VirtualEdgeSearchStats,
): void {
  onProgress?.({
    phase,
    processedFaces,
    totalFaces,
    candidateFacePairs: stats.candidateFacePairs,
    exactDistanceTests: stats.exactDistanceTests,
    generatedVirtualEdges: stats.generatedVirtualEdges,
  });
}

export function computeConnectedComponents(mesh: SimplificationMesh): ConnectedComponents {
  const vertexComponents = new Int32Array(mesh.vertexCount);
  vertexComponents.fill(-1);
  const queue = new Int32Array(mesh.vertexCount);
  let component = 0;

  for (let vertexId = 0; vertexId < mesh.vertexCount; vertexId += 1) {
    if (!mesh.vertexActive[vertexId] || vertexComponents[vertexId] !== -1) continue;
    let head = 0;
    let tail = 0;
    queue[tail] = vertexId;
    tail += 1;
    vertexComponents[vertexId] = component;
    while (head < tail) {
      const current = queue[head]!;
      head += 1;
      for (const edgeId of mesh.incidentEdgeIds(current)) {
        if (mesh.edgeVirtual[edgeId]) continue;
        const neighbor = mesh.edgeA[edgeId] === current ? mesh.edgeB[edgeId]! : mesh.edgeA[edgeId]!;
        if (!mesh.vertexActive[neighbor] || vertexComponents[neighbor] !== -1) continue;
        vertexComponents[neighbor] = component;
        queue[tail] = neighbor;
        tail += 1;
      }
    }
    component += 1;
  }

  const faceComponents = new Int32Array(mesh.inputFaceCount);
  faceComponents.fill(-1);
  for (let faceId = 0; faceId < mesh.inputFaceCount; faceId += 1) {
    if (mesh.faceActive[faceId]) faceComponents[faceId] = vertexComponents[mesh.faceA[faceId]!]!;
  }
  return { vertexComponents, faceComponents, count: component };
}

export function addVirtualEdges(
  mesh: SimplificationMesh,
  options: VirtualEdgeOptions,
  onProgress?: (progress: VirtualEdgeProgress) => void,
): VirtualEdgeSearchStats {
  const normalized = normalizeVirtualEdgeOptions(options);
  const localRadii = normalized.mode === 'auto-local-radius' ? computeLocalTriangleRadii(mesh) : null;
  const autoGlobal = normalized.mode === 'auto-global-radius' ? computeAutoGlobalRadius(mesh) : null;
  const threshold = normalized.mode === 'manual-global-radius'
    ? 2 * normalized.radius!
    : normalized.mode === 'auto-global-radius'
      ? 2 * autoGlobal!.radius
      : localRadii!.searchDistance;
  const stats: VirtualEdgeSearchStats = {
    added: 0,
    mode: normalized.mode,
    ...(normalized.radius !== undefined ? { radius: normalized.radius } : {}),
    ...(autoGlobal ? { radius: autoGlobal.radius } : {}),
    ...(localRadii ? {
      radiusScale: localRadii.radiusScale,
      clampMin: localRadii.clampMin,
      clampMax: localRadii.clampMax,
      bboxDistanceLimit: localRadii.bboxDistanceLimit,
    } : {}),
    ...(autoGlobal ? {
      radiusScale: autoGlobal.radiusScale,
      bboxDistanceLimit: autoGlobal.bboxDistanceLimit,
    } : {}),
    ...(normalized.maxPairsPerComponentPair !== undefined ? { maxPairsPerComponentPair: normalized.maxPairsPerComponentPair } : {}),
    componentCount: 0,
    faceCount: 0,
    searchStrategy: 'grid-exact',
    candidateFacePairs: 0,
    exactDistanceTests: 0,
    aabbRejectedPairs: 0,
    distanceRejectedPairs: 0,
    duplicateVertexPairCandidates: 0,
    cappedVirtualEdgeCandidates: 0,
    generatedVirtualEdges: 0,
  };
  if (!(threshold > 0)) return stats;
  const thresholdSquared = threshold * threshold;
  const components = computeConnectedComponents(mesh);
  stats.componentCount = components.count;
  if (components.count <= 1) return stats;

  const activeFaces = mesh.outputFaceIds();
  const faceCount = activeFaces.length;
  stats.faceCount = faceCount;
  const aabbs = new Float64Array(mesh.inputFaceCount * 6);
  reportProgress(onProgress, 'building-buckets', 0, faceCount, stats);
  for (const faceId of activeFaces) writeFaceAabb(mesh, faceId, aabbs);

  const cellSize = Math.max(threshold, 1e-9);
  const buckets = new CellBuckets();
  const candidateMarks = new Int32Array(mesh.inputFaceCount);
  candidateMarks.fill(EMPTY);
  const candidateFaces: number[] = [];
  const candidatesByComponentPair = new Map<string, VirtualEdgeCandidate[]>();

  reportProgress(onProgress, 'searching-pairs', 0, faceCount, stats);
  for (let index = 0; index < activeFaces.length; index += 1) {
    const faceId = activeFaces[index]!;
    const aabb = readAabb(aabbs, faceId);
    const range = expandedCellRange(aabb, threshold, cellSize);
    candidateFaces.length = 0;

    for (let x = range.minX; x <= range.maxX; x += 1) {
      for (let y = range.minY; y <= range.maxY; y += 1) {
        for (let z = range.minZ; z <= range.maxZ; z += 1) {
          const cellId = buckets.findCell(x, y, z);
          if (cellId === EMPTY) continue;
          buckets.forEachFace(cellId, (other) => {
            if (components.faceComponents[faceId] === components.faceComponents[other]) return;
            if (candidateMarks[other] === faceId) return;
            candidateMarks[other] = faceId;
            candidateFaces.push(other);
          });
        }
      }
    }

    candidateFaces.sort((a, b) => a - b);
    for (const other of candidateFaces) {
      stats.candidateFacePairs += 1;
      const otherAabb = readAabb(aabbs, other);
      if (aabbDistanceSquared(aabb, otherAabb) > thresholdSquared) {
        stats.aabbRejectedPairs += 1;
        continue;
      }
      stats.exactDistanceTests += 1;
      const distanceSquaredValue = triangleTriangleDistanceSquared(mesh, faceId, other);
      if (distanceSquaredValue >= thresholdSquared) {
        stats.distanceRejectedPairs += 1;
        continue;
      }
      if (localRadii) {
        const localThreshold = localRadii.faceRadii[faceId]! + localRadii.faceRadii[other]!;
        const localLimit = localRadii.bboxDistanceLimit > 0 ? Math.min(localThreshold, localRadii.bboxDistanceLimit) : localThreshold;
        if (!(localLimit > 0) || distanceSquaredValue >= localLimit * localLimit) {
          stats.distanceRejectedPairs += 1;
          continue;
        }
      }
      const [a, b] = closestVertexPair(mesh, faceId, other);
      recordCandidate(candidatesByComponentPair, {
        faceA: Math.min(faceId, other),
        faceB: Math.max(faceId, other),
        vertexA: a,
        vertexB: b,
        distanceSquared: distanceSquaredValue,
      }, components.faceComponents[faceId]!, components.faceComponents[other]!);
    }

    for (let x = range.minX; x <= range.maxX; x += 1) {
      for (let y = range.minY; y <= range.maxY; y += 1) {
        for (let z = range.minZ; z <= range.maxZ; z += 1) {
          buckets.addFace(buckets.ensureCell(x, y, z), faceId);
        }
      }
    }

    if ((index + 1) % 2048 === 0 || index + 1 === faceCount) {
      reportProgress(onProgress, 'searching-pairs', index + 1, faceCount, stats);
    }
  }

  for (const candidates of candidatesByComponentPair.values()) {
    candidates.sort((a, b) => (
      a.distanceSquared - b.distanceSquared
      || a.faceA - b.faceA
      || a.faceB - b.faceB
      || Math.min(a.vertexA, a.vertexB) - Math.min(b.vertexA, b.vertexB)
      || Math.max(a.vertexA, a.vertexB) - Math.max(b.vertexA, b.vertexB)
    ));

    const unique: VirtualEdgeCandidate[] = [];
    const seenVertexPairs = new Set<string>();
    for (const candidate of candidates) {
      const key = vertexPairKey(candidate.vertexA, candidate.vertexB);
      if (seenVertexPairs.has(key)) {
        stats.duplicateVertexPairCandidates += 1;
        continue;
      }
      seenVertexPairs.add(key);
      unique.push(candidate);
    }

    const selected = normalized.maxPairsPerComponentPair == null
      ? unique
      : unique.slice(0, normalized.maxPairsPerComponentPair);
    if (normalized.maxPairsPerComponentPair != null) {
      stats.cappedVirtualEdgeCandidates += Math.max(0, unique.length - selected.length);
    }
    for (const candidate of selected) {
      const existing = mesh.getEdgeBetween(candidate.vertexA, candidate.vertexB);
      const edgeId = mesh.addVirtualEdge(candidate.vertexA, candidate.vertexB);
      if (existing === undefined && edgeId !== null && mesh.edgeVirtual[edgeId]) {
        stats.added += 1;
        stats.generatedVirtualEdges += 1;
      }
    }
  }
  return stats;
}
