import type { BufferGeometry } from 'three';
import type { RawMesh, RawSimplificationResult, SimplificationResult, SimplifyOptions } from './types';
import { geometryToRawMesh, rawMeshToGeometry } from './geometryConversion';
import { simplifyRawMeshCore } from './simplifier';

export function simplifyRawMesh(rawMesh: RawMesh, options: SimplifyOptions = {}): RawSimplificationResult {
  return simplifyRawMeshCore(rawMesh, options);
}

export function simplifyGeometry(input: BufferGeometry, options: SimplifyOptions = {}): SimplificationResult {
  const rawMesh = geometryToRawMesh(input);
  const result = simplifyRawMesh(rawMesh, options);
  const geometry = rawMeshToGeometry(result.rawMesh);
  return {
    geometry,
    stats: result.stats,
  };
}
