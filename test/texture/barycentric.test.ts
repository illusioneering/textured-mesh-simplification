import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import {
  barycentricForPoint,
  clampBarycentricToTriangle,
  interpolateVector2,
  pointFromBarycentric,
} from '../../src/texture/barycentric';

describe('barycentric helpers', () => {
  it('round-trips a point inside a 3D triangle', () => {
    const a = new Vector3(0, 0, 0);
    const b = new Vector3(2, 0, 0);
    const c = new Vector3(0, 2, 0);
    const point = pointFromBarycentric(a, b, c, [0.25, 0.25, 0.5]);
    const bary = barycentricForPoint(point, a, b, c);

    expect(bary[0]).toBeCloseTo(0.25);
    expect(bary[1]).toBeCloseTo(0.25);
    expect(bary[2]).toBeCloseTo(0.5);
    expect(bary[0] + bary[1] + bary[2]).toBeCloseTo(1);
  });

  it('interpolates UVs by barycentric weights', () => {
    const uv = interpolateVector2(
      new Vector2(0, 0),
      new Vector2(1, 0),
      new Vector2(0, 1),
      [0.25, 0.25, 0.5],
    );
    expect(uv.x).toBeCloseTo(0.25);
    expect(uv.y).toBeCloseTo(0.5);
  });

  it('clamps negative barycentric weights back to a triangle', () => {
    const bary = clampBarycentricToTriangle([-0.5, 0.5, 1]);
    expect(bary[0]).toBeGreaterThanOrEqual(0);
    expect(bary[1]).toBeGreaterThanOrEqual(0);
    expect(bary[2]).toBeGreaterThanOrEqual(0);
    expect(bary[0] + bary[1] + bary[2]).toBeCloseTo(1);
  });
});
