import { Vector3 } from 'three';
import type { Barycentric } from './types';
import { barycentricForPoint, pointFromBarycentric } from './barycentric';

export interface ClosestPointOnTriangleResult {
  point: Vector3;
  barycentric: Barycentric;
  distanceSquared: number;
}

function resultFor(point: Vector3, query: Vector3, barycentric: Barycentric): ClosestPointOnTriangleResult {
  return { point, barycentric, distanceSquared: point.distanceToSquared(query) };
}

export function closestPointOnTriangle(point: Vector3, a: Vector3, b: Vector3, c: Vector3): ClosestPointOnTriangleResult {
  const ab = b.clone().sub(a);
  const ac = c.clone().sub(a);
  const ap = point.clone().sub(a);
  const d1 = ab.dot(ap);
  const d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) return resultFor(a.clone(), point, [1, 0, 0]);

  const bp = point.clone().sub(b);
  const d3 = ab.dot(bp);
  const d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) return resultFor(b.clone(), point, [0, 1, 0]);

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return resultFor(a.clone().addScaledVector(ab, v), point, [1 - v, v, 0]);
  }

  const cp = point.clone().sub(c);
  const d5 = ab.dot(cp);
  const d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) return resultFor(c.clone(), point, [0, 0, 1]);

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return resultFor(a.clone().addScaledVector(ac, w), point, [1 - w, 0, w]);
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return resultFor(b.clone().addScaledVector(c.clone().sub(b), w), point, [0, 1 - w, w]);
  }

  const barycentric = barycentricForPoint(point, a, b, c);
  const closest = pointFromBarycentric(a, b, c, barycentric);
  if (!Number.isFinite(closest.x) || !Number.isFinite(closest.y) || !Number.isFinite(closest.z)) {
    const candidates = [
      resultFor(a.clone(), point, [1, 0, 0]),
      resultFor(b.clone(), point, [0, 1, 0]),
      resultFor(c.clone(), point, [0, 0, 1]),
    ];
    candidates.sort((left, right) => left.distanceSquared - right.distanceSquared);
    return candidates[0]!;
  }
  return resultFor(closest, point, barycentric);
}
