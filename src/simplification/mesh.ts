import { Vector3 } from 'three';
import { triangleQualityFromCoordinates } from './faceQuality';
import type { FaceIndices, FaceSnapshot, RawMesh } from './types';

const EMPTY = -1;

function growInt32(source: Int32Array, required: number): Int32Array {
  if (required <= source.length) return source;
  const next = new Int32Array(Math.max(required, source.length * 2, 1));
  next.set(source);
  return next;
}

function growUint8(source: Uint8Array, required: number): Uint8Array {
  if (required <= source.length) return source;
  const next = new Uint8Array(Math.max(required, source.length * 2, 1));
  next.set(source);
  return next;
}

function growFloat64(source: Float64Array, required: number): Float64Array {
  if (required <= source.length) return source;
  const next = new Float64Array(Math.max(required, source.length * 2, 1));
  next.set(source);
  return next;
}

function sortedFace(a: number, b: number, c: number): [number, number, number] {
  if (a > b) [a, b] = [b, a];
  if (b > c) [b, c] = [c, b];
  if (a > b) [a, b] = [b, a];
  return [a, b, c];
}

export class SimplificationMesh {
  readonly vertexCount: number;
  readonly inputFaceCount: number;
  readonly edgeKeyBase: number;

  readonly positions: Float64Array;
  readonly vertexActive: Uint8Array;
  readonly vertexVersions: Int32Array;
  readonly vertexFaceHead: Int32Array;
  readonly vertexEdgeHead: Int32Array;
  readonly quadrics: Float64Array;

  readonly faceA: Int32Array;
  readonly faceB: Int32Array;
  readonly faceC: Int32Array;
  readonly faceActive: Uint8Array;
  readonly faceSlotNext: Int32Array;
  readonly faceSlotPrev: Int32Array;
  readonly faceSlotVertex: Int32Array;

  edgeA: Int32Array;
  edgeB: Int32Array;
  edgeActive: Uint8Array;
  edgeVirtual: Uint8Array;
  edgeVersions: Int32Array;
  edgeFace0: Int32Array;
  edgeFace1: Int32Array;
  edgeFaceCount: Int32Array;
  edgeSlotNext: Int32Array;
  edgeSlotPrev: Int32Array;
  edgeSlotVertex: Int32Array;

  readonly edgeKeyToId = new Map<number, number>();
  readonly faceKeyToId = new Map<bigint, number>();
  private readonly extraEdgeFaces = new Map<number, number[]>();
  private readonly faceKeyBase: bigint;
  private edgeTotal = 0;
  private activeVertexTotal = 0;
  private activeFaceTotal = 0;
  private activePhysicalEdgeTotal = 0;
  private activeVirtualEdgeTotal = 0;

  private constructor(rawMesh: RawMesh) {
    this.vertexCount = rawMesh.positions.length;
    this.inputFaceCount = rawMesh.faces.length;
    this.edgeKeyBase = this.vertexCount;
    if (this.vertexCount > 0 && (this.vertexCount - 1) * this.edgeKeyBase + (this.vertexCount - 1) > Number.MAX_SAFE_INTEGER) {
      throw new Error('Mesh has too many vertices for numeric edge keys.');
    }
    this.faceKeyBase = BigInt(this.vertexCount + 1);

    this.positions = new Float64Array(this.vertexCount * 3);
    this.vertexActive = new Uint8Array(this.vertexCount);
    this.vertexVersions = new Int32Array(this.vertexCount);
    this.vertexFaceHead = new Int32Array(this.vertexCount);
    this.vertexEdgeHead = new Int32Array(this.vertexCount);
    this.quadrics = new Float64Array(this.vertexCount * 13);
    this.vertexFaceHead.fill(EMPTY);
    this.vertexEdgeHead.fill(EMPTY);
    this.vertexActive.fill(1);
    this.activeVertexTotal = this.vertexCount;

    rawMesh.positions.forEach((position, index) => {
      const offset = index * 3;
      this.positions[offset] = position.x;
      this.positions[offset + 1] = position.y;
      this.positions[offset + 2] = position.z;
    });

    this.faceA = new Int32Array(this.inputFaceCount);
    this.faceB = new Int32Array(this.inputFaceCount);
    this.faceC = new Int32Array(this.inputFaceCount);
    this.faceActive = new Uint8Array(this.inputFaceCount);
    this.faceSlotNext = new Int32Array(this.inputFaceCount * 3);
    this.faceSlotPrev = new Int32Array(this.inputFaceCount * 3);
    this.faceSlotVertex = new Int32Array(this.inputFaceCount * 3);
    this.faceSlotNext.fill(EMPTY);
    this.faceSlotPrev.fill(EMPTY);
    this.faceSlotVertex.fill(EMPTY);

    rawMesh.faces.forEach((face, index) => {
      this.faceA[index] = face[0];
      this.faceB[index] = face[1];
      this.faceC[index] = face[2];
      this.faceActive[index] = 1;
    });
    this.activeFaceTotal = this.inputFaceCount;

    const estimatedEdges = Math.max(1, this.inputFaceCount * 2);
    this.edgeA = new Int32Array(estimatedEdges);
    this.edgeB = new Int32Array(estimatedEdges);
    this.edgeActive = new Uint8Array(estimatedEdges);
    this.edgeVirtual = new Uint8Array(estimatedEdges);
    this.edgeVersions = new Int32Array(estimatedEdges);
    this.edgeFace0 = new Int32Array(estimatedEdges);
    this.edgeFace1 = new Int32Array(estimatedEdges);
    this.edgeFaceCount = new Int32Array(estimatedEdges);
    this.edgeSlotNext = new Int32Array(estimatedEdges * 2);
    this.edgeSlotPrev = new Int32Array(estimatedEdges * 2);
    this.edgeSlotVertex = new Int32Array(estimatedEdges * 2);
    this.edgeFace0.fill(EMPTY);
    this.edgeFace1.fill(EMPTY);
    this.edgeSlotNext.fill(EMPTY);
    this.edgeSlotPrev.fill(EMPTY);
    this.edgeSlotVertex.fill(EMPTY);
  }

  static fromRawMesh(rawMesh: RawMesh): SimplificationMesh {
    const mesh = new SimplificationMesh(rawMesh);
    mesh.removeDegenerateAndDuplicateFaces();
    mesh.rebuildInitialAdjacency();
    return mesh;
  }

  edgeCapacity(): number {
    return this.edgeA.length;
  }

  edgeCount(): number {
    return this.edgeTotal;
  }

  activeVertexCount(): number {
    return this.activeVertexTotal;
  }

  activeFaceCount(): number {
    return this.activeFaceTotal;
  }

  activeEdgeCount(includeVirtual = true): number {
    return includeVirtual ? this.activePhysicalEdgeTotal + this.activeVirtualEdgeTotal : this.activePhysicalEdgeTotal;
  }

  edgeKey(a0: number, b0: number): number {
    const a = Math.min(a0, b0);
    const b = Math.max(a0, b0);
    return a * this.edgeKeyBase + b;
  }

  faceKeyForVertices(a0: number, b0: number, c0: number): bigint {
    const [a, b, c] = sortedFace(a0, b0, c0);
    return (BigInt(a) * this.faceKeyBase + BigInt(b)) * this.faceKeyBase + BigInt(c);
  }

  outputFaceIds(): number[] {
    const ids: number[] = [];
    for (let faceId = 0; faceId < this.inputFaceCount; faceId += 1) {
      if (this.faceActive[faceId]) ids.push(faceId);
    }
    return ids;
  }

  faceVertices(faceId: number): FaceIndices {
    return [this.faceA[faceId]!, this.faceB[faceId]!, this.faceC[faceId]!];
  }

  positionVector(vertexId: number): Vector3 {
    const offset = vertexId * 3;
    return new Vector3(this.positions[offset]!, this.positions[offset + 1]!, this.positions[offset + 2]!);
  }

  setPosition(vertexId: number, x: number, y: number, z: number): void {
    const offset = vertexId * 3;
    this.positions[offset] = x;
    this.positions[offset + 1] = y;
    this.positions[offset + 2] = z;
  }

  position(vertexId: number): [number, number, number] {
    const offset = vertexId * 3;
    return [this.positions[offset]!, this.positions[offset + 1]!, this.positions[offset + 2]!];
  }

  activeEdgeIds(includeVirtual = true): number[] {
    const ids: number[] = [];
    for (let edgeId = 0; edgeId < this.edgeTotal; edgeId += 1) {
      if (this.edgeActive[edgeId] && (includeVirtual || !this.edgeVirtual[edgeId])) ids.push(edgeId);
    }
    return ids;
  }

  incidentEdgeIds(vertexId: number): number[] {
    const ids: number[] = [];
    for (let slot = this.vertexEdgeHead[vertexId]!; slot !== EMPTY; slot = this.edgeSlotNext[slot]!) {
      const edgeId = Math.floor(slot / 2);
      if (this.edgeActive[edgeId]) ids.push(edgeId);
    }
    return ids;
  }

  activeIncidentFaceIds(vertexIds: Iterable<number>): number[] {
    const seen = new Set<number>();
    for (const vertexId of vertexIds) {
      if (!this.vertexActive[vertexId]) continue;
      for (let slot = this.vertexFaceHead[vertexId]!; slot !== EMPTY; slot = this.faceSlotNext[slot]!) {
        const faceId = Math.floor(slot / 3);
        if (this.faceActive[faceId]) seen.add(faceId);
      }
    }
    return [...seen].sort((a, b) => a - b);
  }

  snapshotIncidentFaces(vertexIds: Iterable<number>): FaceSnapshot[] {
    return this.activeIncidentFaceIds(vertexIds).map((faceId) => {
      const vertices = this.faceVertices(faceId);
      return {
        faceId,
        vertices,
        positions: [
          this.positionVector(vertices[0]),
          this.positionVector(vertices[1]),
          this.positionVector(vertices[2]),
        ],
      };
    });
  }

  activeIncidentFaceCount(edgeId: number): number {
    let count = 0;
    for (const faceId of this.edgeIncidentFaces(edgeId)) {
      if (this.faceActive[faceId]) count += 1;
    }
    return count;
  }

  edgeIncidentFaces(edgeId: number): number[] {
    if (!this.edgeActive[edgeId]) return [];
    const faces: number[] = [];
    const first = this.edgeFace0[edgeId]!;
    const second = this.edgeFace1[edgeId]!;
    if (first !== EMPTY) faces.push(first);
    if (second !== EMPTY) faces.push(second);
    const extra = this.extraEdgeFaces.get(edgeId);
    if (extra) faces.push(...extra);
    return faces;
  }

  edgeLength(edgeId: number): number {
    const a = this.edgeA[edgeId]!;
    const b = this.edgeB[edgeId]!;
    const ax = this.positions[a * 3]!;
    const ay = this.positions[a * 3 + 1]!;
    const az = this.positions[a * 3 + 2]!;
    const bx = this.positions[b * 3]!;
    const by = this.positions[b * 3 + 1]!;
    const bz = this.positions[b * 3 + 2]!;
    return Math.hypot(ax - bx, ay - by, az - bz);
  }

  getEdgeBetween(a: number, b: number): number | undefined {
    const edgeId = this.edgeKeyToId.get(this.edgeKey(a, b));
    return edgeId === undefined || !this.edgeActive[edgeId] ? undefined : edgeId;
  }

  addVirtualEdge(a0: number, b0: number): number | null {
    if (a0 === b0) return null;
    if (!this.vertexActive[a0] || !this.vertexActive[b0]) return null;
    const a = Math.min(a0, b0);
    const b = Math.max(a0, b0);
    const key = this.edgeKey(a, b);
    const existing = this.edgeKeyToId.get(key);
    if (existing !== undefined && this.edgeActive[existing]) return existing;

    const edgeId = this.createEdge(a, b, true);
    this.edgeKeyToId.set(key, edgeId);
    this.activeVirtualEdgeTotal += 1;
    this.insertEdgeSlots(edgeId, a, b);
    return edgeId;
  }

  ensurePhysicalEdge(a0: number, b0: number, faceId: number): number {
    const a = Math.min(a0, b0);
    const b = Math.max(a0, b0);
    const key = this.edgeKey(a, b);
    const existing = this.edgeKeyToId.get(key);
    if (existing !== undefined && this.edgeActive[existing]) {
      if (this.edgeVirtual[existing]) {
        this.edgeVirtual[existing] = 0;
        this.activeVirtualEdgeTotal -= 1;
        this.activePhysicalEdgeTotal += 1;
        this.edgeVersions[existing] = this.edgeVersions[existing]! + 1;
      }
      this.addEdgeFace(existing, faceId);
      return existing;
    }

    const edgeId = this.createEdge(a, b, false);
    this.edgeKeyToId.set(key, edgeId);
    this.activePhysicalEdgeTotal += 1;
    this.addEdgeFace(edgeId, faceId);
    this.insertEdgeSlots(edgeId, a, b);
    return edgeId;
  }

  deactivateEdge(edgeId: number): void {
    if (!this.edgeActive[edgeId]) return;
    const key = this.edgeKey(this.edgeA[edgeId]!, this.edgeB[edgeId]!);
    if (this.edgeKeyToId.get(key) === edgeId) this.edgeKeyToId.delete(key);
    this.removeEdgeSlots(edgeId);
    this.edgeActive[edgeId] = 0;
    if (this.edgeVirtual[edgeId]) this.activeVirtualEdgeTotal -= 1;
    else this.activePhysicalEdgeTotal -= 1;
    this.edgeFace0[edgeId] = EMPTY;
    this.edgeFace1[edgeId] = EMPTY;
    this.edgeFaceCount[edgeId] = 0;
    this.extraEdgeFaces.delete(edgeId);
    this.edgeVersions[edgeId] = this.edgeVersions[edgeId]! + 1;
  }

  deactivateVertex(vertexId: number): void {
    if (!this.vertexActive[vertexId]) return;
    this.vertexActive[vertexId] = 0;
    this.activeVertexTotal -= 1;
    this.vertexVersions[vertexId] = this.vertexVersions[vertexId]! + 1;
  }

  deactivateFace(faceId: number): void {
    if (!this.faceActive[faceId]) return;
    this.faceActive[faceId] = 0;
    this.activeFaceTotal -= 1;
  }

  removeFaceReferences(faceId: number, vertices: FaceIndices): void {
    this.removeFaceSlots(faceId);
    this.removeFaceFromEdge(vertices[0], vertices[1], faceId);
    this.removeFaceFromEdge(vertices[1], vertices[2], faceId);
    this.removeFaceFromEdge(vertices[2], vertices[0], faceId);
  }

  addFaceReferences(faceId: number): void {
    const a = this.faceA[faceId]!;
    const b = this.faceB[faceId]!;
    const c = this.faceC[faceId]!;
    this.insertFaceSlots(faceId, [a, b, c]);
    this.ensurePhysicalEdge(a, b, faceId);
    this.ensurePhysicalEdge(b, c, faceId);
    this.ensurePhysicalEdge(c, a, faceId);
  }

  unregisterFaceKey(faceId: number, vertices: FaceIndices): void {
    const key = this.faceKeyForVertices(vertices[0], vertices[1], vertices[2]);
    if (this.faceKeyToId.get(key) === faceId) this.faceKeyToId.delete(key);
  }

  canRegisterFace(faceId: number): boolean {
    return this.canRegisterFaceVertices(faceId, this.faceA[faceId]!, this.faceB[faceId]!, this.faceC[faceId]!);
  }

  canRegisterFaceVertices(faceId: number, a: number, b: number, c: number): boolean {
    const key = this.faceKeyForVertices(a, b, c);
    const existing = this.faceKeyToId.get(key);
    return existing === undefined || existing === faceId || !this.faceActive[existing];
  }

  registerFace(faceId: number): void {
    this.faceKeyToId.set(this.faceKeyForVertices(this.faceA[faceId]!, this.faceB[faceId]!, this.faceC[faceId]!), faceId);
  }

  isFaceDegenerate(faceId: number): boolean {
    const a = this.faceA[faceId]!;
    const b = this.faceB[faceId]!;
    const c = this.faceC[faceId]!;
    if (a === b || b === c || a === c) return true;
    if (!this.vertexActive[a] || !this.vertexActive[b] || !this.vertexActive[c]) return true;
    return this.triangleAreaByVertices(a, b, c) <= 1e-20;
  }

  triangleAreaByVertices(a: number, b: number, c: number): number {
    const ax = this.positions[a * 3]!;
    const ay = this.positions[a * 3 + 1]!;
    const az = this.positions[a * 3 + 2]!;
    const bx = this.positions[b * 3]!;
    const by = this.positions[b * 3 + 1]!;
    const bz = this.positions[b * 3 + 2]!;
    const cx = this.positions[c * 3]!;
    const cy = this.positions[c * 3 + 1]!;
    const cz = this.positions[c * 3 + 2]!;
    const ux = bx - ax;
    const uy = by - ay;
    const uz = bz - az;
    const vx = cx - ax;
    const vy = cy - ay;
    const vz = cz - az;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    return Math.hypot(nx, ny, nz) * 0.5;
  }

  triangleQualityByVertices(a: number, b: number, c: number): number {
    return triangleQualityFromCoordinates(
      this.positions[a * 3]!,
      this.positions[a * 3 + 1]!,
      this.positions[a * 3 + 2]!,
      this.positions[b * 3]!,
      this.positions[b * 3 + 1]!,
      this.positions[b * 3 + 2]!,
      this.positions[c * 3]!,
      this.positions[c * 3 + 1]!,
      this.positions[c * 3 + 2]!,
    );
  }

  triangleQualityByFace(faceId: number): number {
    return this.triangleQualityByVertices(this.faceA[faceId]!, this.faceB[faceId]!, this.faceC[faceId]!);
  }

  triangleQualityWithVertexPosition(faceId: number, vertexId: number, position: readonly [number, number, number]): number {
    const a = this.faceA[faceId]!;
    const b = this.faceB[faceId]!;
    const c = this.faceC[faceId]!;
    const ax = a === vertexId ? position[0] : this.positions[a * 3]!;
    const ay = a === vertexId ? position[1] : this.positions[a * 3 + 1]!;
    const az = a === vertexId ? position[2] : this.positions[a * 3 + 2]!;
    const bx = b === vertexId ? position[0] : this.positions[b * 3]!;
    const by = b === vertexId ? position[1] : this.positions[b * 3 + 1]!;
    const bz = b === vertexId ? position[2] : this.positions[b * 3 + 2]!;
    const cx = c === vertexId ? position[0] : this.positions[c * 3]!;
    const cy = c === vertexId ? position[1] : this.positions[c * 3 + 1]!;
    const cz = c === vertexId ? position[2] : this.positions[c * 3 + 2]!;
    return triangleQualityFromCoordinates(ax, ay, az, bx, by, bz, cx, cy, cz);
  }

  computeInitialQuadrics(): void {
    this.quadrics.fill(0);
    for (let faceId = 0; faceId < this.inputFaceCount; faceId += 1) {
      if (!this.faceActive[faceId]) continue;
      const a = this.faceA[faceId]!;
      const b = this.faceB[faceId]!;
      const c = this.faceC[faceId]!;
      const area = this.triangleAreaByVertices(a, b, c);
      if (area <= 1e-20) continue;
      const quadric = this.planeQuadric(a, b, c, area / 3);
      this.addQuadricToVertex(a, quadric);
      this.addQuadricToVertex(b, quadric);
      this.addQuadricToVertex(c, quadric);
    }
  }

  addQuadricToVertex(vertexId: number, quadric: readonly number[]): void {
    const offset = vertexId * 13;
    for (let i = 0; i < 13; i += 1) this.quadrics[offset + i] = this.quadrics[offset + i]! + quadric[i]!;
  }

  addVertexQuadricInPlace(targetVertexId: number, sourceVertexId: number): void {
    const target = targetVertexId * 13;
    const source = sourceVertexId * 13;
    for (let i = 0; i < 13; i += 1) this.quadrics[target + i] = this.quadrics[target + i]! + this.quadrics[source + i]!;
  }

  toRawMeshWithMaps(): { rawMesh: RawMesh; outputFaceIds: number[]; oldToNewVertexIds: Map<number, number> } {
    const used = new Uint8Array(this.vertexCount);
    const outputFaceIds: number[] = [];
    for (let faceId = 0; faceId < this.inputFaceCount; faceId += 1) {
      if (!this.faceActive[faceId]) continue;
      used[this.faceA[faceId]!] = 1;
      used[this.faceB[faceId]!] = 1;
      used[this.faceC[faceId]!] = 1;
      outputFaceIds.push(faceId);
    }

    const oldToNewVertexIds = new Map<number, number>();
    const positions: Vector3[] = [];
    for (let vertexId = 0; vertexId < this.vertexCount; vertexId += 1) {
      if (!used[vertexId]) continue;
      oldToNewVertexIds.set(vertexId, positions.length);
      positions.push(this.positionVector(vertexId));
    }

    const faces: FaceIndices[] = [];
    for (const faceId of outputFaceIds) {
      faces.push([
        oldToNewVertexIds.get(this.faceA[faceId]!)!,
        oldToNewVertexIds.get(this.faceB[faceId]!)!,
        oldToNewVertexIds.get(this.faceC[faceId]!)!,
      ]);
    }

    return { rawMesh: { positions, faces }, outputFaceIds, oldToNewVertexIds };
  }

  private removeDegenerateAndDuplicateFaces(): void {
    this.faceKeyToId.clear();
    for (let faceId = 0; faceId < this.inputFaceCount; faceId += 1) {
      if (!this.faceActive[faceId]) continue;
      if (this.isFaceDegenerate(faceId)) {
        this.deactivateFace(faceId);
        continue;
      }
      const key = this.faceKeyForVertices(this.faceA[faceId]!, this.faceB[faceId]!, this.faceC[faceId]!);
      if (this.faceKeyToId.has(key)) {
        this.deactivateFace(faceId);
      } else {
        this.faceKeyToId.set(key, faceId);
      }
    }
  }

  private rebuildInitialAdjacency(): void {
    for (let faceId = 0; faceId < this.inputFaceCount; faceId += 1) {
      if (!this.faceActive[faceId]) continue;
      this.addFaceReferences(faceId);
    }
  }

  private createEdge(a: number, b: number, virtual: boolean): number {
    const edgeId = this.edgeTotal;
    this.ensureEdgeCapacity(edgeId + 1);
    this.edgeA[edgeId] = a;
    this.edgeB[edgeId] = b;
    this.edgeActive[edgeId] = 1;
    this.edgeVirtual[edgeId] = virtual ? 1 : 0;
    this.edgeVersions[edgeId] = 0;
    this.edgeFace0[edgeId] = EMPTY;
    this.edgeFace1[edgeId] = EMPTY;
    this.edgeFaceCount[edgeId] = 0;
    this.edgeSlotNext[edgeId * 2] = EMPTY;
    this.edgeSlotNext[edgeId * 2 + 1] = EMPTY;
    this.edgeSlotPrev[edgeId * 2] = EMPTY;
    this.edgeSlotPrev[edgeId * 2 + 1] = EMPTY;
    this.edgeSlotVertex[edgeId * 2] = EMPTY;
    this.edgeSlotVertex[edgeId * 2 + 1] = EMPTY;
    this.edgeTotal += 1;
    return edgeId;
  }

  private ensureEdgeCapacity(required: number): void {
    const oldEdgeLength = this.edgeA.length;
    this.edgeA = growInt32(this.edgeA, required);
    this.edgeB = growInt32(this.edgeB, required);
    this.edgeActive = growUint8(this.edgeActive, required);
    this.edgeVirtual = growUint8(this.edgeVirtual, required);
    this.edgeVersions = growInt32(this.edgeVersions, required);
    this.edgeFace0 = growInt32(this.edgeFace0, required);
    this.edgeFace1 = growInt32(this.edgeFace1, required);
    this.edgeFaceCount = growInt32(this.edgeFaceCount, required);
    this.edgeSlotNext = growInt32(this.edgeSlotNext, required * 2);
    this.edgeSlotPrev = growInt32(this.edgeSlotPrev, required * 2);
    this.edgeSlotVertex = growInt32(this.edgeSlotVertex, required * 2);
    if (this.edgeA.length !== oldEdgeLength) {
      this.edgeFace0.fill(EMPTY, oldEdgeLength);
      this.edgeFace1.fill(EMPTY, oldEdgeLength);
      this.edgeSlotNext.fill(EMPTY, oldEdgeLength * 2);
      this.edgeSlotPrev.fill(EMPTY, oldEdgeLength * 2);
      this.edgeSlotVertex.fill(EMPTY, oldEdgeLength * 2);
    }
  }

  private insertFaceSlots(faceId: number, vertices: FaceIndices): void {
    for (let corner = 0; corner < 3; corner += 1) {
      const slot = faceId * 3 + corner;
      const vertexId = vertices[corner]!;
      this.faceSlotVertex[slot] = vertexId;
      this.faceSlotPrev[slot] = EMPTY;
      this.faceSlotNext[slot] = this.vertexFaceHead[vertexId]!;
      if (this.vertexFaceHead[vertexId] !== EMPTY) this.faceSlotPrev[this.vertexFaceHead[vertexId]!] = slot;
      this.vertexFaceHead[vertexId] = slot;
    }
  }

  private removeFaceSlots(faceId: number): void {
    for (let corner = 0; corner < 3; corner += 1) {
      const slot = faceId * 3 + corner;
      const vertexId = this.faceSlotVertex[slot]!;
      if (vertexId === EMPTY) continue;
      const prev = this.faceSlotPrev[slot]!;
      const next = this.faceSlotNext[slot]!;
      if (prev !== EMPTY) this.faceSlotNext[prev] = next;
      else this.vertexFaceHead[vertexId] = next;
      if (next !== EMPTY) this.faceSlotPrev[next] = prev;
      this.faceSlotVertex[slot] = EMPTY;
      this.faceSlotPrev[slot] = EMPTY;
      this.faceSlotNext[slot] = EMPTY;
    }
  }

  private insertEdgeSlots(edgeId: number, a: number, b: number): void {
    this.insertEdgeSlot(edgeId * 2, a);
    this.insertEdgeSlot(edgeId * 2 + 1, b);
  }

  private insertEdgeSlot(slot: number, vertexId: number): void {
    this.edgeSlotVertex[slot] = vertexId;
    this.edgeSlotPrev[slot] = EMPTY;
    this.edgeSlotNext[slot] = this.vertexEdgeHead[vertexId]!;
    if (this.vertexEdgeHead[vertexId] !== EMPTY) this.edgeSlotPrev[this.vertexEdgeHead[vertexId]!] = slot;
    this.vertexEdgeHead[vertexId] = slot;
  }

  private removeEdgeSlots(edgeId: number): void {
    this.removeEdgeSlot(edgeId * 2);
    this.removeEdgeSlot(edgeId * 2 + 1);
  }

  private removeEdgeSlot(slot: number): void {
    const vertexId = this.edgeSlotVertex[slot]!;
    if (vertexId === EMPTY) return;
    const prev = this.edgeSlotPrev[slot]!;
    const next = this.edgeSlotNext[slot]!;
    if (prev !== EMPTY) this.edgeSlotNext[prev] = next;
    else this.vertexEdgeHead[vertexId] = next;
    if (next !== EMPTY) this.edgeSlotPrev[next] = prev;
    this.edgeSlotVertex[slot] = EMPTY;
    this.edgeSlotPrev[slot] = EMPTY;
    this.edgeSlotNext[slot] = EMPTY;
  }

  private addEdgeFace(edgeId: number, faceId: number): void {
    if (this.edgeFace0[edgeId] === faceId || this.edgeFace1[edgeId] === faceId) return;
    const extra = this.extraEdgeFaces.get(edgeId);
    if (extra?.includes(faceId)) return;
    if (this.edgeFace0[edgeId] === EMPTY) this.edgeFace0[edgeId] = faceId;
    else if (this.edgeFace1[edgeId] === EMPTY) this.edgeFace1[edgeId] = faceId;
    else if (extra) extra.push(faceId);
    else this.extraEdgeFaces.set(edgeId, [faceId]);
    this.edgeFaceCount[edgeId] = this.edgeFaceCount[edgeId]! + 1;
    this.edgeVersions[edgeId] = this.edgeVersions[edgeId]! + 1;
  }

  private removeFaceFromEdge(a: number, b: number, faceId: number): void {
    const edgeId = this.getEdgeBetween(a, b);
    if (edgeId === undefined || this.edgeVirtual[edgeId]) return;
    const faces = this.edgeIncidentFaces(edgeId).filter((id) => id !== faceId);
    if (faces.length === this.edgeFaceCount[edgeId]) return;
    this.edgeFace0[edgeId] = faces[0] ?? EMPTY;
    this.edgeFace1[edgeId] = faces[1] ?? EMPTY;
    if (faces.length > 2) this.extraEdgeFaces.set(edgeId, faces.slice(2));
    else this.extraEdgeFaces.delete(edgeId);
    this.edgeFaceCount[edgeId] = faces.length;
    this.edgeVersions[edgeId] = this.edgeVersions[edgeId]! + 1;
    if (faces.length === 0) this.deactivateEdge(edgeId);
  }

  private planeQuadric(a: number, b: number, c: number, weight: number): number[] {
    const ax = this.positions[a * 3]!;
    const ay = this.positions[a * 3 + 1]!;
    const az = this.positions[a * 3 + 2]!;
    const bx = this.positions[b * 3]!;
    const by = this.positions[b * 3 + 1]!;
    const bz = this.positions[b * 3 + 2]!;
    const cx = this.positions[c * 3]!;
    const cy = this.positions[c * 3 + 1]!;
    const cz = this.positions[c * 3 + 2]!;
    const ux = bx - ax;
    const uy = by - ay;
    const uz = bz - az;
    const vx = cx - ax;
    const vy = cy - ay;
    const vz = cz - az;
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const length = Math.hypot(nx, ny, nz);
    if (length <= 1e-20 || weight === 0) return new Array(13).fill(0);
    nx /= length;
    ny /= length;
    nz /= length;
    const d = nx * ax + ny * ay + nz * az;
    return [
      nx * nx * weight, nx * ny * weight, nx * nz * weight,
      ny * nx * weight, ny * ny * weight, ny * nz * weight,
      nz * nx * weight, nz * ny * weight, nz * nz * weight,
      -d * nx * weight, -d * ny * weight, -d * nz * weight,
      d * d * weight,
    ];
  }
}
