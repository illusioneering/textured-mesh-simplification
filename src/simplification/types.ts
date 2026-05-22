import type { BufferGeometry, Vector3 } from 'three';

export type FaceIndices = [number, number, number];
export type Barycentric = [number, number, number];

export interface RawMesh {
  positions: Vector3[];
  faces: FaceIndices[];
}

export interface FaceSnapshot {
  faceId: number;
  vertices: FaceIndices;
  positions: [Vector3, Vector3, Vector3];
}

export interface CollapseHistoryRecord {
  keepVertexId: number;
  removedVertexId: number;
  beforeFaces: FaceSnapshot[];
  afterFaceIds: number[];
}

export type VirtualEdgeOptions =
  | { mode: 'auto-local-radius'; maxPairsPerComponentPair?: number | null }
  | { mode: 'auto-global-radius' }
  | { mode: 'manual-global-radius'; radius: number };

export interface SimplifyOptions {
  targetFaceCount?: number;
  targetRatio?: number;
  maxIterations?: number;
  virtualEdges?: VirtualEdgeOptions;
  onProgress?: (stats: SimplificationProgress) => void;
  onVirtualEdgeProgress?: (progress: VirtualEdgeProgress) => void;
}

export interface SimplificationProgress {
  iteration: number;
  activeFaces: number;
  activeVertices: number;
  activeEdges: number;
  lastCost: number;
}

export type VirtualEdgeProgressPhase = 'building-buckets' | 'searching-pairs';

export interface VirtualEdgeProgress {
  phase: VirtualEdgeProgressPhase;
  processedFaces: number;
  totalFaces: number;
  candidateFacePairs: number;
  exactDistanceTests: number;
  generatedVirtualEdges: number;
}

export interface VirtualEdgeDiagnostics {
  mode: VirtualEdgeOptions['mode'];
  radius?: number;
  radiusScale?: number;
  clampMin?: number;
  clampMax?: number;
  bboxDistanceLimit?: number;
  maxPairsPerComponentPair?: number | null;
  componentCount: number;
  faceCount: number;
  searchStrategy?: 'grid-exact';
  candidateFacePairs?: number;
  exactDistanceTests?: number;
  aabbRejectedPairs?: number;
  distanceRejectedPairs?: number;
  duplicateVertexPairCandidates?: number;
  cappedVirtualEdgeCandidates?: number;
  generatedVirtualEdges?: number;
}

export interface SimplificationResult {
  geometry: BufferGeometry;
  stats: {
    inputVertices: number;
    inputFaces: number;
    outputVertices: number;
    outputFaces: number;
    physicalEdges: number;
    virtualEdges: number;
    virtualEdgeDiagnostics?: VirtualEdgeDiagnostics;
    collapses: number;
    stoppedReason: 'target-reached' | 'queue-empty' | 'max-iterations';
  };
}

export interface RawSimplificationResult {
  rawMesh: RawMesh;
  outputFaceIds: number[];
  history: CollapseHistoryRecord[];
  stats: SimplificationResult['stats'];
}
