import type { SimplifyOptions } from '../simplification/types';

export type TargetMode =
  | { kind: 'ratio'; ratio: number }
  | { kind: 'faces'; targetFaceCount: number };

export type VirtualEdgeOptions =
  | { mode: 'auto-local-radius'; maxPairsPerComponentPair?: number | null }
  | { mode: 'auto-global-radius' }
  | { mode: 'manual-global-radius'; radius: number };

export type PrimitiveGroupingMode = 'material-parent' | 'material' | 'none';

export interface ProcessingOptions {
  target: TargetMode;
  primitiveGrouping: PrimitiveGroupingMode;
  virtualEdges: VirtualEdgeOptions;
  weldVertices: boolean;
  recomputeNormals: boolean;
  maxIterations?: number;
  transferTextures: boolean;
  textureSize: number;
  texturePadding: number;
  textureFilter: 'nearest' | 'linear';
}

export type ProcessingOptionsInput = Partial<Omit<ProcessingOptions, 'target'>> & {
  target?: TargetMode;
};

function requireFiniteNumber(name: string, value: number): void {
  if (!Number.isFinite(value)) throw new Error(`${name} must be a finite number.`);
}

function requireNonNegativeNumber(name: string, value: number): void {
  requireFiniteNumber(name, value);
  if (value < 0) throw new Error(`${name} must be non-negative.`);
}

function requirePositiveInteger(name: string, value: number): void {
  requireFiniteNumber(name, value);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`);
}

function requireNonNegativeInteger(name: string, value: number): void {
  requireFiniteNumber(name, value);
  if (!Number.isInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer.`);
}

function normalizeTarget(target: TargetMode): TargetMode {
  if (target.kind === 'ratio') {
    requireFiniteNumber('Target ratio', target.ratio);
    if (target.ratio <= 0 || target.ratio > 1) throw new Error('Target ratio must be greater than 0 and less than or equal to 1.');
    return { kind: 'ratio', ratio: target.ratio };
  }
  if (target.kind === 'faces') {
    requireFiniteNumber('Target face count', target.targetFaceCount);
    if (!Number.isInteger(target.targetFaceCount) || target.targetFaceCount <= 0) {
      throw new Error('Target face count must be a positive integer.');
    }
    return { kind: 'faces', targetFaceCount: target.targetFaceCount };
  }
  throw new Error('Target mode must be ratio or faces.');
}

function normalizeVirtualEdges(virtualEdges: VirtualEdgeOptions): VirtualEdgeOptions {
  if (virtualEdges.mode === 'auto-local-radius') {
    if (virtualEdges.maxPairsPerComponentPair === undefined) return { mode: 'auto-local-radius' };
    if (virtualEdges.maxPairsPerComponentPair === null) return { mode: 'auto-local-radius', maxPairsPerComponentPair: null };
    requireNonNegativeInteger('Virtual edge candidate cap', virtualEdges.maxPairsPerComponentPair);
    return { mode: 'auto-local-radius', maxPairsPerComponentPair: virtualEdges.maxPairsPerComponentPair };
  }
  if (virtualEdges.mode === 'auto-global-radius') return { mode: 'auto-global-radius' };
  if (virtualEdges.mode === 'manual-global-radius') {
    requireNonNegativeNumber('Virtual edge radius', virtualEdges.radius);
    return { mode: 'manual-global-radius', radius: virtualEdges.radius };
  }
  throw new Error('Virtual edge mode must be auto-local-radius, auto-global-radius, or manual-global-radius.');
}

function normalizePrimitiveGrouping(value: PrimitiveGroupingMode): PrimitiveGroupingMode {
  if (value === 'material-parent' || value === 'material' || value === 'none') return value;
  throw new Error('Primitive grouping must be material-parent, material, or none.');
}

export function defaultProcessingOptions(): ProcessingOptions {
  return {
    target: { kind: 'ratio', ratio: 0.5 },
    primitiveGrouping: 'material-parent',
    virtualEdges: { mode: 'auto-local-radius' },
    weldVertices: true,
    recomputeNormals: true,
    transferTextures: false,
    textureSize: 1024,
    texturePadding: 2,
    textureFilter: 'linear',
  };
}

export function normalizeProcessingOptions(input: ProcessingOptionsInput = {}): ProcessingOptions {
  const defaults = defaultProcessingOptions();
  const target = normalizeTarget(input.target ?? defaults.target);
  const primitiveGrouping = normalizePrimitiveGrouping(input.primitiveGrouping ?? defaults.primitiveGrouping);
  const virtualEdges = normalizeVirtualEdges(input.virtualEdges ?? defaults.virtualEdges);
  const maxIterations = input.maxIterations;
  if (maxIterations !== undefined) requirePositiveInteger('Max iterations', maxIterations);
  const textureSize = input.textureSize ?? defaults.textureSize;
  requirePositiveInteger('Texture size', textureSize);
  const texturePadding = input.texturePadding ?? defaults.texturePadding;
  requireNonNegativeInteger('Texture padding', texturePadding);
  const textureFilter = input.textureFilter ?? defaults.textureFilter;
  if (textureFilter !== 'nearest' && textureFilter !== 'linear') throw new Error('Texture filter must be nearest or linear.');

  return {
    target,
    primitiveGrouping,
    virtualEdges,
    weldVertices: input.weldVertices ?? defaults.weldVertices,
    recomputeNormals: input.recomputeNormals ?? defaults.recomputeNormals,
    ...(maxIterations !== undefined ? { maxIterations } : {}),
    transferTextures: input.transferTextures ?? defaults.transferTextures,
    textureSize,
    texturePadding,
    textureFilter,
  };
}

export function toSimplifyOptions(options: ProcessingOptions): SimplifyOptions {
  return {
    ...(options.target.kind === 'faces'
      ? { targetFaceCount: options.target.targetFaceCount }
      : { targetRatio: options.target.ratio }),
    virtualEdges: options.virtualEdges,
    ...(options.maxIterations !== undefined ? { maxIterations: options.maxIterations } : {}),
  };
}
