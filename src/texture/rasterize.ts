import type { Barycentric } from '../simplification/types';

export interface RasterSample {
  x: number;
  y: number;
  faceIndex: number;
  barycentric: Barycentric;
}

function barycentric2D(
  px: number,
  py: number,
  a: [number, number],
  b: [number, number],
  c: [number, number],
): Barycentric | null {
  const v0x = b[0] - a[0];
  const v0y = b[1] - a[1];
  const v1x = c[0] - a[0];
  const v1y = c[1] - a[1];
  const v2x = px - a[0];
  const v2y = py - a[1];
  const denom = v0x * v1y - v1x * v0y;
  if (Math.abs(denom) <= 1e-12) return null;
  const v = (v2x * v1y - v1x * v2y) / denom;
  const w = (v0x * v2y - v2x * v0y) / denom;
  return [1 - v - w, v, w];
}

export function rasterizeAtlasTriangle(
  faceIndex: number,
  pixelTriangle: [[number, number], [number, number], [number, number]],
  textureSize: number,
): RasterSample[] {
  const [a, b, c] = pixelTriangle;
  const minX = Math.max(0, Math.floor(Math.min(a[0], b[0], c[0])));
  const maxX = Math.min(textureSize - 1, Math.ceil(Math.max(a[0], b[0], c[0])));
  const minY = Math.max(0, Math.floor(Math.min(a[1], b[1], c[1])));
  const maxY = Math.min(textureSize - 1, Math.ceil(Math.max(a[1], b[1], c[1])));
  const samples: RasterSample[] = [];
  const epsilon = 1e-8;
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const barycentric = barycentric2D(x + 0.5, y + 0.5, a, b, c);
      if (!barycentric) continue;
      if (barycentric[0] >= -epsilon && barycentric[1] >= -epsilon && barycentric[2] >= -epsilon) {
        const sum = barycentric[0] + barycentric[1] + barycentric[2];
        samples.push({
          x,
          y,
          faceIndex,
          barycentric: [barycentric[0] / sum, barycentric[1] / sum, barycentric[2] / sum],
        });
      }
    }
  }
  return samples;
}
