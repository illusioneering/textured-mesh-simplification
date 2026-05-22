export interface Candidate {
  edgeId: number;
  cost: number;
  x: number;
  y: number;
  z: number;
}

export class CandidateQueue {
  private heap: Int32Array;
  private heapIndex: Int32Array;
  private costs: Float64Array;
  private xs: Float64Array;
  private ys: Float64Array;
  private zs: Float64Array;
  private heapSize = 0;

  constructor(edgeCapacity: number) {
    this.heap = new Int32Array(Math.max(1, edgeCapacity));
    this.heapIndex = new Int32Array(edgeCapacity);
    this.heapIndex.fill(-1);
    this.costs = new Float64Array(edgeCapacity);
    this.xs = new Float64Array(edgeCapacity);
    this.ys = new Float64Array(edgeCapacity);
    this.zs = new Float64Array(edgeCapacity);
  }

  get size(): number {
    return this.heapSize;
  }

  liveCandidateCount(): number {
    return this.heapSize;
  }

  has(edgeId: number): boolean {
    return (this.heapIndex[edgeId] ?? -1) !== -1;
  }

  upsert(edgeId: number, cost: number, x: number, y: number, z: number): void {
    this.ensureCandidateCapacity(edgeId + 1);
    this.costs[edgeId] = cost;
    this.xs[edgeId] = x;
    this.ys[edgeId] = y;
    this.zs[edgeId] = z;

    const existing = this.heapIndex[edgeId]!;
    if (existing !== -1) {
      this.bubbleUp(existing);
      this.bubbleDown(this.heapIndex[edgeId]!);
      return;
    }

    this.ensureHeapCapacity(this.heapSize + 1);
    const index = this.heapSize;
    this.heap[index] = edgeId;
    this.heapIndex[edgeId] = index;
    this.heapSize += 1;
    this.bubbleUp(index);
  }

  remove(edgeId: number): void {
    const index = this.heapIndex[edgeId] ?? -1;
    if (index === -1) return;
    this.removeAt(index);
  }

  pop(): Candidate | undefined {
    if (this.heapSize === 0) return undefined;
    const edgeId = this.heap[0]!;
    const candidate = {
      edgeId,
      cost: this.costs[edgeId]!,
      x: this.xs[edgeId]!,
      y: this.ys[edgeId]!,
      z: this.zs[edgeId]!,
    };
    this.removeAt(0);
    return candidate;
  }

  clear(): void {
    for (let i = 0; i < this.heapSize; i += 1) this.heapIndex[this.heap[i]!] = -1;
    this.heapSize = 0;
  }

  private ensureHeapCapacity(required: number): void {
    if (required <= this.heap.length) return;
    const next = new Int32Array(Math.max(required, this.heap.length * 2));
    next.set(this.heap);
    this.heap = next;
  }

  private ensureCandidateCapacity(required: number): void {
    if (required <= this.heapIndex.length) return;
    const nextLength = Math.max(required, this.heapIndex.length * 2, 1);
    const nextHeapIndex = new Int32Array(nextLength);
    nextHeapIndex.fill(-1);
    nextHeapIndex.set(this.heapIndex);
    const nextCosts = new Float64Array(nextLength);
    nextCosts.set(this.costs);
    const nextXs = new Float64Array(nextLength);
    nextXs.set(this.xs);
    const nextYs = new Float64Array(nextLength);
    nextYs.set(this.ys);
    const nextZs = new Float64Array(nextLength);
    nextZs.set(this.zs);
    this.heapIndex = nextHeapIndex;
    this.costs = nextCosts;
    this.xs = nextXs;
    this.ys = nextYs;
    this.zs = nextZs;
  }

  private removeAt(index: number): void {
    const removedEdgeId = this.heap[index]!;
    this.heapIndex[removedEdgeId] = -1;
    this.heapSize -= 1;
    if (index === this.heapSize) return;

    const movedEdgeId = this.heap[this.heapSize]!;
    this.heap[index] = movedEdgeId;
    this.heapIndex[movedEdgeId] = index;
    this.bubbleUp(index);
    this.bubbleDown(this.heapIndex[movedEdgeId]!);
  }

  private less(leftIndex: number, rightIndex: number): boolean {
    return this.costs[this.heap[leftIndex]!]! < this.costs[this.heap[rightIndex]!]!;
  }

  private swap(left: number, right: number): void {
    const leftEdge = this.heap[left]!;
    const rightEdge = this.heap[right]!;
    this.heap[left] = rightEdge;
    this.heap[right] = leftEdge;
    this.heapIndex[leftEdge] = right;
    this.heapIndex[rightEdge] = left;
  }

  private bubbleUp(index: number): void {
    let current = index;
    while (current > 0) {
      const parent = Math.floor((current - 1) / 2);
      if (!this.less(current, parent)) break;
      this.swap(current, parent);
      current = parent;
    }
  }

  private bubbleDown(index: number): void {
    let current = index;
    while (true) {
      const left = current * 2 + 1;
      const right = left + 1;
      let smallest = current;
      if (left < this.heapSize && this.less(left, smallest)) smallest = left;
      if (right < this.heapSize && this.less(right, smallest)) smallest = right;
      if (smallest === current) break;
      this.swap(current, smallest);
      current = smallest;
    }
  }
}
