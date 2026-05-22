import { describe, expect, it } from 'vitest';
import { defaultProcessingOptions, normalizeProcessingOptions, toSimplifyOptions } from '../../src/pipeline/options';

describe('processing options', () => {
  it('uses CLI-compatible defaults', () => {
    expect(defaultProcessingOptions()).toEqual({
      target: { kind: 'ratio', ratio: 0.5 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'auto-local-radius' },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 1024,
      texturePadding: 2,
      textureFilter: 'linear',
    });
  });

  it('normalizes partial options over defaults', () => {
    const normalized = normalizeProcessingOptions({
      target: { kind: 'faces', targetFaceCount: 42 },
      primitiveGrouping: 'none',
      virtualEdges: { mode: 'manual-global-radius', radius: 0.125 },
      weldVertices: false,
      recomputeNormals: false,
      transferTextures: true,
      textureFilter: 'nearest',
    });

    expect(normalized.target).toEqual({ kind: 'faces', targetFaceCount: 42 });
    expect(normalized.primitiveGrouping).toBe('none');
    expect(normalized.virtualEdges).toEqual({ mode: 'manual-global-radius', radius: 0.125 });
    expect(normalized.weldVertices).toBe(false);
    expect(normalized.recomputeNormals).toBe(false);
    expect(normalized.transferTextures).toBe(true);
    expect(normalized.textureFilter).toBe('nearest');
    expect(normalized.textureSize).toBe(1024);
    expect(normalized.texturePadding).toBe(2);
  });

  it('normalizes all virtual-edge radius modes', () => {
    expect(normalizeProcessingOptions({
      virtualEdges: { mode: 'auto-local-radius', maxPairsPerComponentPair: 4 },
    }).virtualEdges).toEqual({ mode: 'auto-local-radius', maxPairsPerComponentPair: 4 });
    expect(normalizeProcessingOptions({
      virtualEdges: { mode: 'auto-global-radius' },
    }).virtualEdges).toEqual({ mode: 'auto-global-radius' });
    expect(normalizeProcessingOptions({
      virtualEdges: { mode: 'manual-global-radius', radius: 0.25 },
    }).virtualEdges).toEqual({ mode: 'manual-global-radius', radius: 0.25 });
  });

  it('rejects invalid target ratios and face counts', () => {
    expect(() => normalizeProcessingOptions({ target: { kind: 'ratio', ratio: 0 } })).toThrow(/target ratio/i);
    expect(() => normalizeProcessingOptions({ target: { kind: 'ratio', ratio: 1.1 } })).toThrow(/target ratio/i);
    expect(() => normalizeProcessingOptions({ target: { kind: 'ratio', ratio: Number.NaN } })).toThrow(/target ratio/i);
    expect(() => normalizeProcessingOptions({ target: { kind: 'faces', targetFaceCount: 0 } })).toThrow(/target face count/i);
    expect(() => normalizeProcessingOptions({ target: { kind: 'faces', targetFaceCount: -1 } })).toThrow(/target face count/i);
    expect(() => normalizeProcessingOptions({ target: { kind: 'faces', targetFaceCount: 1.5 } })).toThrow(/target face count/i);
  });

  it('rejects invalid texture controls', () => {
    expect(() => normalizeProcessingOptions({ maxIterations: 0 })).toThrow(/max iterations/i);
    expect(() => normalizeProcessingOptions({ texturePadding: -1 })).toThrow(/texture padding/i);
    expect(() => normalizeProcessingOptions({ textureFilter: 'cubic' as never })).toThrow(/texture filter/i);
    expect(() => normalizeProcessingOptions({ textureSize: 0 })).toThrow(/texture size/i);
    expect(() => normalizeProcessingOptions({ primitiveGrouping: 'primitive' as never })).toThrow(/primitive grouping/i);
    expect(() => normalizeProcessingOptions({ virtualEdges: { mode: 'manual-global-radius', radius: -1 } })).toThrow(/virtual edge radius/i);
    expect(() => normalizeProcessingOptions({ virtualEdges: { mode: 'auto-local-radius', maxPairsPerComponentPair: -1 } })).toThrow(/candidate cap/i);
  });

  it('rejects legacy virtual-edge mode literals', () => {
    expect(() => normalizeProcessingOptions({ virtualEdges: { mode: 'auto-local' } as never })).toThrow(/virtual edge mode/i);
    expect(() => normalizeProcessingOptions({ virtualEdges: { mode: 'global-radius', radius: 0.5 } as never })).toThrow(/virtual edge mode/i);
  });

  it('converts UI options to simplification options', () => {
    expect(toSimplifyOptions(normalizeProcessingOptions({ target: { kind: 'ratio', ratio: 0.25 }, maxIterations: 10 }))).toEqual({
      targetRatio: 0.25,
      virtualEdges: { mode: 'auto-local-radius' },
      maxIterations: 10,
    });
    expect(toSimplifyOptions(normalizeProcessingOptions({
      target: { kind: 'faces', targetFaceCount: 12 },
      virtualEdges: { mode: 'manual-global-radius', radius: 0.5 },
    }))).toMatchObject({
      targetFaceCount: 12,
      virtualEdges: { mode: 'manual-global-radius', radius: 0.5 },
    });
    expect(toSimplifyOptions(normalizeProcessingOptions({
      target: { kind: 'ratio', ratio: 0.5 },
      virtualEdges: { mode: 'auto-local-radius', maxPairsPerComponentPair: null },
    }))).toMatchObject({
      targetRatio: 0.5,
      virtualEdges: { mode: 'auto-local-radius', maxPairsPerComponentPair: null },
    });
    expect(toSimplifyOptions(normalizeProcessingOptions({
      target: { kind: 'ratio', ratio: 0.5 },
      virtualEdges: { mode: 'auto-global-radius' },
    }))).toMatchObject({
      targetRatio: 0.5,
      virtualEdges: { mode: 'auto-global-radius' },
    });
  });
});
