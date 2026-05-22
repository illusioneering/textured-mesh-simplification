import { describe, expect, it } from 'vitest';
import { BufferAttribute, BufferGeometry, Vector3 } from 'three';
import { geometryToRawMesh } from '../../src/simplification/geometryConversion';
import { simplifyGeometry, simplifyRawMesh } from '../../src/simplification/simplify';

function squareGeometry(): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
  ]), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  return geometry;
}

describe('simplifyGeometry', () => {
  it('simplifies a square to the requested face target or lower', () => {
    const result = simplifyGeometry(squareGeometry(), { targetFaceCount: 1, virtualEdges: { mode: 'manual-global-radius', radius: 0 } });
    expect(result.stats.inputFaces).toBe(2);
    expect(result.stats.outputFaces).toBeLessThanOrEqual(1);
    const raw = geometryToRawMesh(result.geometry);
    for (const position of raw.positions) {
      expect(Number.isFinite(position.x)).toBe(true);
      expect(Number.isFinite(position.y)).toBe(true);
      expect(Number.isFinite(position.z)).toBe(true);
    }
  });

  it('uses targetRatio when targetFaceCount is omitted', () => {
    const result = simplifyGeometry(squareGeometry(), { targetRatio: 0.5, virtualEdges: { mode: 'manual-global-radius', radius: 0 } });
    expect(result.stats.outputFaces).toBeLessThanOrEqual(1);
  });

  it('creates virtual edges for close disconnected components', () => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0,
      0, 0, 0.05, 1, 0, 0.05, 0, 1, 0.05,
    ]), 3));
    geometry.setIndex([0, 1, 2, 3, 4, 5]);
    const result = simplifyGeometry(geometry, { targetFaceCount: 1, virtualEdges: { mode: 'manual-global-radius', radius: 0.1 } });
    expect(result.stats.virtualEdges).toBeGreaterThan(0);
  });

  it('uses auto-local-radius virtual edges by default', () => {
    const raw = geometryToRawMesh(squareGeometry());
    raw.positions.push(
      raw.positions[0]!.clone().add(new Vector3(0, 0, 0.01)),
      raw.positions[1]!.clone().add(new Vector3(0, 0, 0.01)),
      raw.positions[2]!.clone().add(new Vector3(0, 0, 0.01)),
    );
    raw.faces.push([4, 5, 6]);

    const result = simplifyRawMesh(raw, { targetFaceCount: 1 });

    expect(result.stats.virtualEdges).toBeGreaterThan(0);
    expect(result.stats.virtualEdgeDiagnostics).toMatchObject({
      mode: 'auto-local-radius',
    });
  });

  it('rejects invalid simplification options in the reusable API', () => {
    expect(() => simplifyGeometry(squareGeometry(), { targetRatio: Number.NaN })).toThrow(/targetRatio/);
    expect(() => simplifyGeometry(squareGeometry(), { targetRatio: -0.1 })).toThrow(/targetRatio/);
    expect(() => simplifyGeometry(squareGeometry(), { targetFaceCount: -1 })).toThrow(/targetFaceCount/);
    expect(() => simplifyGeometry(squareGeometry(), { maxIterations: 0 })).toThrow(/maxIterations/);
    expect(() => simplifyGeometry(squareGeometry(), { virtualEdges: { mode: 'manual-global-radius', radius: -1 } })).toThrow(/radius/);
    expect(() => simplifyGeometry(squareGeometry(), { virtualEdges: { mode: 'legacy' } as never })).toThrow(/virtual edge mode/i);
  });
});
