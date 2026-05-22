import { normalizeProcessingOptions, type PrimitiveGroupingMode, type ProcessingOptions, type ProcessingOptionsInput } from '../pipeline/options';

export interface ProcessingFormValues {
  targetMode: 'ratio' | 'faces';
  targetRatio: string;
  targetFaceCount: string;
  primitiveGrouping: PrimitiveGroupingMode;
  virtualEdgeMode: 'auto-local-radius' | 'auto-global-radius' | 'manual-global-radius';
  virtualEdgeRadius: string;
  weldVertices: boolean;
  recomputeNormals: boolean;
  maxIterations: string;
  textureSize: string;
  texturePadding: string;
  textureFilter: 'nearest' | 'linear';
}

export interface TextureBakeFormValues {
  textureSize: string;
  texturePadding: string;
  textureFilter: 'nearest' | 'linear';
}

function parseNumber(name: string, value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be a finite number.`);
  return parsed;
}

function parseOptionalInteger(name: string, value: string): number | undefined {
  const trimmed = value.trim();
  if (trimmed === '') return undefined;
  const parsed = parseNumber(name, trimmed);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer.`);
  return parsed;
}

export function processingOptionsInputFromFormValues(
  values: ProcessingFormValues,
  transferTextures = false,
): ProcessingOptionsInput {
  const maxIterations = parseOptionalInteger('Max iterations', values.maxIterations);
  return {
    target: values.targetMode === 'ratio'
      ? { kind: 'ratio', ratio: parseNumber('Target ratio', values.targetRatio) }
      : { kind: 'faces', targetFaceCount: parseNumber('Target face count', values.targetFaceCount) },
    primitiveGrouping: values.primitiveGrouping,
    virtualEdges: values.virtualEdgeMode === 'manual-global-radius'
      ? { mode: 'manual-global-radius', radius: parseNumber('Virtual edge radius', values.virtualEdgeRadius) }
      : { mode: values.virtualEdgeMode },
    weldVertices: values.weldVertices,
    recomputeNormals: values.recomputeNormals,
    ...(maxIterations !== undefined ? { maxIterations } : {}),
    transferTextures,
    textureSize: parseNumber('Texture size', values.textureSize),
    texturePadding: parseNumber('Texture padding', values.texturePadding),
    textureFilter: values.textureFilter,
  };
}

export function parseProcessingOptionsValues(values: ProcessingFormValues, transferTextures = false): ProcessingOptions {
  return normalizeProcessingOptions(processingOptionsInputFromFormValues(values, transferTextures));
}

export function parseTextureBakeOptionsValues(
  values: TextureBakeFormValues,
  simplifyOptions: ProcessingOptions,
): ProcessingOptions {
  return normalizeProcessingOptions({
    ...simplifyOptions,
    transferTextures: true,
    textureSize: parseNumber('Texture size', values.textureSize),
    texturePadding: parseNumber('Texture padding', values.texturePadding),
    textureFilter: values.textureFilter,
  });
}
