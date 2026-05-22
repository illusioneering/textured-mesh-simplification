import { describe, expect, it } from 'vitest';
import { BufferAttribute, BufferGeometry } from 'three';
import { geometryToRawMesh, rawMeshToGeometry } from '../../src/simplification/geometryConversion';

describe('geometry conversion', () => {
  it('converts indexed BufferGeometry to a raw triangle mesh', () => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]), 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);

    const raw = geometryToRawMesh(geometry);

    expect(raw.positions).toHaveLength(4);
    expect(raw.faces).toEqual([[0, 1, 2], [0, 2, 3]]);
  });

  it('converts non-indexed BufferGeometry to sequential faces', () => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]), 3));

    const raw = geometryToRawMesh(geometry);

    expect(raw.positions).toHaveLength(3);
    expect(raw.faces).toEqual([[0, 1, 2]]);
  });

  it('converts a raw triangle mesh back to BufferGeometry', () => {
    const raw = geometryToRawMesh(rawMeshToGeometry({
      positions: [
        { x: 0, y: 0, z: 0, clone() { return this; } } as never,
      ],
      faces: [],
    }));
    expect(raw.positions).toHaveLength(1);
    expect(raw.faces).toHaveLength(0);
  });
});
