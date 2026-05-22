import { describe, expect, it } from 'vitest';
import {
  processingOptionsInputFromFormValues,
  parseProcessingOptionsValues,
  parseTextureBakeOptionsValues,
  type ProcessingFormValues,
} from '../../src/web/formOptions';

const baseValues: ProcessingFormValues = {
  targetMode: 'ratio',
  targetRatio: '0.25',
  targetFaceCount: '120',
  primitiveGrouping: 'material-parent',
  virtualEdgeMode: 'auto-local-radius',
  virtualEdgeRadius: '2',
  weldVertices: true,
  recomputeNormals: true,
  maxIterations: '',
  textureSize: '512',
  texturePadding: '8',
  textureFilter: 'nearest',
};

describe('web form option parsing', () => {
  it('converts ratio-mode form strings into normalized processing options', () => {
    const options = parseProcessingOptionsValues(baseValues, true);

    expect(options.target).toEqual({ kind: 'ratio', ratio: 0.25 });
    expect(options.primitiveGrouping).toBe('material-parent');
    expect(options.virtualEdges).toEqual({ mode: 'auto-local-radius' });
    expect(options.weldVertices).toBe(true);
    expect(options.recomputeNormals).toBe(true);
    expect(options.maxIterations).toBeUndefined();
    expect(options.transferTextures).toBe(true);
    expect(options.textureSize).toBe(512);
    expect(options.texturePadding).toBe(8);
    expect(options.textureFilter).toBe('nearest');
  });

  it('sets texture transfer from the requested operation instead of a form checkbox', () => {
    expect(parseProcessingOptionsValues(baseValues).transferTextures).toBe(false);
    expect(parseProcessingOptionsValues(baseValues, true).transferTextures).toBe(true);
  });

  it('converts auto global virtual edge mode without reading the radius field', () => {
    expect(processingOptionsInputFromFormValues({
      ...baseValues,
      virtualEdgeMode: 'auto-global-radius',
      virtualEdgeRadius: 'not-used',
    }, false)).toMatchObject({
      virtualEdges: { mode: 'auto-global-radius' },
    });
  });

  it('converts face-count and manual global virtual edge fields', () => {
    expect(processingOptionsInputFromFormValues({
      ...baseValues,
      targetMode: 'faces',
      targetRatio: '0.5',
      targetFaceCount: '17',
      primitiveGrouping: 'none',
      virtualEdgeMode: 'manual-global-radius',
      virtualEdgeRadius: '0.125',
      weldVertices: false,
      recomputeNormals: false,
      maxIterations: '99',
      textureSize: '1024',
      texturePadding: '4',
      textureFilter: 'linear',
    }, false)).toMatchObject({
      target: { kind: 'faces', targetFaceCount: 17 },
      primitiveGrouping: 'none',
      virtualEdges: { mode: 'manual-global-radius', radius: 0.125 },
      weldVertices: false,
      recomputeNormals: false,
      maxIterations: 99,
      transferTextures: false,
    });
  });

  it('combines current texture controls with the active simplification options for baking', () => {
    const activeSimplifyOptions = parseProcessingOptionsValues({
      ...baseValues,
      targetMode: 'faces',
      targetFaceCount: '77',
      primitiveGrouping: 'none',
      virtualEdgeMode: 'manual-global-radius',
      virtualEdgeRadius: '0.25',
      weldVertices: false,
      recomputeNormals: false,
      maxIterations: '42',
      textureSize: '1024',
      texturePadding: '4',
      textureFilter: 'linear',
    }, false);

    const bakeOptions = parseTextureBakeOptionsValues({
      textureSize: '2048',
      texturePadding: '12',
      textureFilter: 'nearest',
    }, activeSimplifyOptions);

    expect(bakeOptions).toMatchObject({
      target: { kind: 'faces', targetFaceCount: 77 },
      primitiveGrouping: 'none',
      virtualEdges: { mode: 'manual-global-radius', radius: 0.25 },
      weldVertices: false,
      recomputeNormals: false,
      maxIterations: 42,
      transferTextures: true,
      textureSize: 2048,
      texturePadding: 12,
      textureFilter: 'nearest',
    });
  });

  it('does not read stale simplification panel values when parsing texture bake options', () => {
    const activeSimplifyOptions = parseProcessingOptionsValues(baseValues, false);
    const staleFormValues = {
      ...baseValues,
      targetRatio: 'not-a-number',
      targetFaceCount: 'not-an-integer',
      primitiveGrouping: 'invalid-grouping',
      virtualEdgeRadius: 'not-a-radius',
      maxIterations: 'not-an-integer',
      textureSize: '4096',
      texturePadding: '6',
      textureFilter: 'linear',
    } as const;

    const bakeOptions = parseTextureBakeOptionsValues(staleFormValues, activeSimplifyOptions);

    expect(bakeOptions).toMatchObject({
      target: activeSimplifyOptions.target,
      primitiveGrouping: activeSimplifyOptions.primitiveGrouping,
      virtualEdges: activeSimplifyOptions.virtualEdges,
      weldVertices: activeSimplifyOptions.weldVertices,
      recomputeNormals: activeSimplifyOptions.recomputeNormals,
      transferTextures: true,
      textureSize: 4096,
      texturePadding: 6,
      textureFilter: 'linear',
    });
  });
});
