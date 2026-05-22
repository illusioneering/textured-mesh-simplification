import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { closestPointOnTriangle } from '../../src/texture/projection';

describe('closestPointOnTriangle', () => {
  it('returns the original point and barycentric coordinates when inside the triangle', () => {
    const result = closestPointOnTriangle(
      new Vector3(0.25, 0.25, 1),
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(0, 1, 0),
    );

    expect(result.point.x).toBeCloseTo(0.25);
    expect(result.point.y).toBeCloseTo(0.25);
    expect(result.point.z).toBeCloseTo(0);
    expect(result.barycentric[0]).toBeCloseTo(0.5);
    expect(result.barycentric[1]).toBeCloseTo(0.25);
    expect(result.barycentric[2]).toBeCloseTo(0.25);
    expect(result.distanceSquared).toBeCloseTo(1);
  });

  it('clamps an outside point to the nearest edge', () => {
    const result = closestPointOnTriangle(
      new Vector3(0.5, -1, 0),
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(0, 1, 0),
    );

    expect(result.point.x).toBeCloseTo(0.5);
    expect(result.point.y).toBeCloseTo(0);
    expect(result.barycentric[2]).toBeCloseTo(0);
    expect(result.barycentric[0] + result.barycentric[1] + result.barycentric[2]).toBeCloseTo(1);
  });

  it('handles degenerate triangles without NaN values', () => {
    const result = closestPointOnTriangle(
      new Vector3(2, 0, 0),
      new Vector3(0, 0, 0),
      new Vector3(0, 0, 0),
      new Vector3(0, 0, 0),
    );
    expect(Number.isFinite(result.point.x)).toBe(true);
    expect(Number.isFinite(result.barycentric[0])).toBe(true);
    expect(result.barycentric[0] + result.barycentric[1] + result.barycentric[2]).toBeCloseTo(1);
  });
});
