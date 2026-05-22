import { describe, expect, it } from 'vitest';
import { Matrix4, Vector2, Vector3, Vector4 } from 'three';
import {
  computeAuthoredTangentFrame,
  computeFaceTangentFrame,
  computeSerializedTangents,
  normalRgbToVector,
  tangentNormalToWorld,
  transformAuthoredTangent,
  transformTangentSpaceNormal,
  vectorToNormalRgb,
} from '../../src/texture/tangentSpace';

function decode(color: [number, number, number, number]): Vector3 {
  return normalRgbToVector(color);
}

describe('tangent-space normal helpers', () => {
  it('round-trips neutral and axis-tilted normal-map colors', () => {
    expect(vectorToNormalRgb(new Vector3(0, 0, 1), 255)).toEqual([128, 128, 255, 255]);
    const decoded = decode(vectorToNormalRgb(new Vector3(1, 0, 1).normalize(), 200));
    expect(decoded.x).toBeCloseTo(Math.SQRT1_2, 2);
    expect(decoded.y).toBeCloseTo(0, 2);
    expect(decoded.z).toBeCloseTo(Math.SQRT1_2, 2);
  });

  it('builds a right-handed tangent frame from triangle UVs', () => {
    const frame = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)],
      normal: new Vector3(0, 0, 1),
    });

    expect(frame).not.toBeNull();
    expect(frame!.tangent.x).toBeCloseTo(1);
    expect(frame!.bitangent.y).toBeCloseTo(1);
    expect(frame!.normal.z).toBeCloseTo(1);
    expect(frame!.handedness).toBe(1);

    const bitangentNormal = tangentNormalToWorld(new Vector3(0, 1, 0), frame!);
    expect(bitangentNormal.x).toBeCloseTo(frame!.bitangent.x);
    expect(bitangentNormal.y).toBeCloseTo(frame!.bitangent.y);
    expect(bitangentNormal.z).toBeCloseTo(frame!.bitangent.z);
  });

  it('returns null for degenerate UV triangles', () => {
    expect(computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(0, 0), new Vector2(0, 0)],
      normal: new Vector3(0, 0, 1),
    })).toBeNull();
  });

  it('transforms authored tangents as directions instead of normals', () => {
    const transform = new Matrix4().makeScale(2, 1, 0.5);
    const tangent = new Vector4(1, 0, 1, -1);

    const transformed = transformAuthoredTangent(tangent, transform);

    const expected = new Vector3(1, 0, 1).transformDirection(transform);
    expect(transformed.x).toBeCloseTo(expected.x, 6);
    expect(transformed.y).toBeCloseTo(expected.y, 6);
    expect(transformed.z).toBeCloseTo(expected.z, 6);
    expect(transformed.w).toBe(-1);
  });

  it('builds authored tangent frames by interpolating corner bitangents', () => {
    const normals = [
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
      new Vector3(0, 0, 1),
    ] as [Vector3, Vector3, Vector3];
    const tangents = [
      new Vector4(1, 0, 0, -1),
      new Vector4(1, 0, 0, -1),
      new Vector4(0, 1, 0, 1),
    ] as [Vector4, Vector4, Vector4];

    const frame = computeAuthoredTangentFrame({
      normals,
      tangents,
      barycentric: [0.45, 0.45, 0.1],
    });

    expect(frame).not.toBeNull();
    expect(frame!.normal.z).toBeCloseTo(1, 6);
    expect(frame!.bitangent.y).toBeLessThan(0);
    expect(frame!.tangent.length()).toBeCloseTo(1, 6);
    expect(frame!.bitangent.length()).toBeCloseTo(1, 6);
  });

  it('converts a sampled source tangent normal into a rotated output tangent frame', () => {
    const source = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)],
      normal: new Vector3(0, 0, 1),
    })!;
    const output = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(0, 1), new Vector2(-1, 0)],
      normal: new Vector3(0, 0, 1),
    })!;

    const transformed = transformTangentSpaceNormal([255, 128, 128, 255], source, output);
    const decoded = decode(transformed);
    expect(decoded.x).toBeCloseTo(0, 2);
    expect(decoded.y).toBeCloseTo(1, 2);
    expect(decoded.z).toBeCloseTo(0, 2);
  });

  it('normalizes same-frame normal-map colors when transforming between frames', () => {
    const frame = computeFaceTangentFrame({
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)],
      normal: new Vector3(0, 0, 1),
    })!;

    const transformed = transformTangentSpaceNormal([200, 140, 80, 255], frame, frame);
    const decoded = decode(transformed);
    expect(Math.hypot(decoded.x, decoded.y, decoded.z)).toBeCloseTo(1, 2);
    expect(transformed).not.toEqual([200, 140, 80, 255]);
  });

  it('computes one glTF tangent vec4 per serialized vertex', () => {
    const tangents = computeSerializedTangents({
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      normals: [
        new Vector3(0, 0, 1),
        new Vector3(0, 0, 1),
        new Vector3(0, 0, 1),
      ],
      uvs: [
        new Vector2(0, 0),
        new Vector2(1, 0),
        new Vector2(0, 1),
      ],
      indices: [0, 1, 2],
    });

    expect(Array.from(tangents)).toEqual([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
  });
});
