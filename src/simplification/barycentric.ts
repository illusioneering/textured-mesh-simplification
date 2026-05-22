import { Vector2, Vector3 } from 'three';
import type { Barycentric } from './types';

const EPSILON = 1e-12;

export function pointFromBarycentric(a: Vector3, b: Vector3, c: Vector3, bary: Barycentric): Vector3 {
  return new Vector3()
    .addScaledVector(a, bary[0])
    .addScaledVector(b, bary[1])
    .addScaledVector(c, bary[2]);
}

export function interpolateVector2(a: Vector2, b: Vector2, c: Vector2, bary: Barycentric): Vector2 {
  return new Vector2(
    a.x * bary[0] + b.x * bary[1] + c.x * bary[2],
    a.y * bary[0] + b.y * bary[1] + c.y * bary[2],
  );
}

export function interpolateVector3(a: Vector3, b: Vector3, c: Vector3, bary: Barycentric): Vector3 {
  return new Vector3(
    a.x * bary[0] + b.x * bary[1] + c.x * bary[2],
    a.y * bary[0] + b.y * bary[1] + c.y * bary[2],
    a.z * bary[0] + b.z * bary[1] + c.z * bary[2],
  );
}

export function barycentricForPoint(point: Vector3, a: Vector3, b: Vector3, c: Vector3): Barycentric {
  const v0 = b.clone().sub(a);
  const v1 = c.clone().sub(a);
  const v2 = point.clone().sub(a);
  const d00 = v0.dot(v0);
  const d01 = v0.dot(v1);
  const d11 = v1.dot(v1);
  const d20 = v2.dot(v0);
  const d21 = v2.dot(v1);
  const denom = d00 * d11 - d01 * d01;
  if (Math.abs(denom) <= EPSILON) {
    const da = point.distanceToSquared(a);
    const db = point.distanceToSquared(b);
    const dc = point.distanceToSquared(c);
    if (da <= db && da <= dc) return [1, 0, 0];
    if (db <= dc) return [0, 1, 0];
    return [0, 0, 1];
  }
  const v = (d11 * d20 - d01 * d21) / denom;
  const w = (d00 * d21 - d01 * d20) / denom;
  const u = 1 - v - w;
  return [u, v, w];
}

export function clampBarycentricToTriangle(bary: Barycentric): Barycentric {
  const clamped: Barycentric = [Math.max(0, bary[0]), Math.max(0, bary[1]), Math.max(0, bary[2])];
  const sum = clamped[0] + clamped[1] + clamped[2];
  if (sum <= EPSILON) return [1, 0, 0];
  return [clamped[0] / sum, clamped[1] / sum, clamped[2] / sum];
}
