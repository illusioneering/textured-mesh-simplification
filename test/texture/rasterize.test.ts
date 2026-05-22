import { describe, expect, it } from 'vitest';
import { rasterizeAtlasTriangle } from '../../src/texture/rasterize';

describe('rasterizeAtlasTriangle', () => {
  it('returns covered pixels with normalized barycentric coordinates', () => {
    const samples = rasterizeAtlasTriangle(2, [[2, 2], [14, 2], [2, 14]], 16);
    expect(samples.length).toBeGreaterThan(0);
    for (const sample of samples) {
      expect(sample.faceIndex).toBe(2);
      expect(sample.x).toBeGreaterThanOrEqual(0);
      expect(sample.x).toBeLessThan(16);
      expect(sample.y).toBeGreaterThanOrEqual(0);
      expect(sample.y).toBeLessThan(16);
      expect(sample.barycentric[0] + sample.barycentric[1] + sample.barycentric[2]).toBeCloseTo(1);
      for (const weight of sample.barycentric) expect(weight).toBeGreaterThanOrEqual(-1e-8);
    }
  });
});
