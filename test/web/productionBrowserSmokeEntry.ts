import { Vector3 } from 'three';
import { createInjectiveAtlas } from '../../src/texture/atlas';
import type { RawMesh } from '../../src/simplification/types';

type ProductionBrowserSmokeResult =
  | {
      ok: true;
      faceUvs: number;
      facePixelTriangles: number;
      islandCount?: number;
      textureSize: number;
    }
  | {
      ok: false;
      error: string;
      stack?: string;
    };

const smokeGlobal = globalThis as typeof globalThis & {
  __MESH_SIMPLIFICATION_SMOKE__?: ProductionBrowserSmokeResult;
};

function squareMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(1, 1, 0),
      new Vector3(0, 1, 0),
    ],
    faces: [
      [0, 1, 2],
      [0, 2, 3],
    ],
  };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function run(): Promise<void> {
  const mesh = squareMesh();
  const atlas = await createInjectiveAtlas(mesh, { textureSize: 64, padding: 2 });

  assert(atlas.textureSize === 64, `Expected textureSize 64, got ${atlas.textureSize}.`);
  assert(atlas.padding === 2, `Expected padding 2, got ${atlas.padding}.`);
  assert(atlas.faceUvs.length === mesh.faces.length, `Expected ${mesh.faces.length} face UV sets, got ${atlas.faceUvs.length}.`);
  assert(
    atlas.facePixelTriangles.length === mesh.faces.length,
    `Expected ${mesh.faces.length} face pixel triangles, got ${atlas.facePixelTriangles.length}.`,
  );
  assert(atlas.faceUvs.every((uvs) => uvs.length === 3), 'Each atlas face must contain three UV corners.');

  smokeGlobal.__MESH_SIMPLIFICATION_SMOKE__ = {
    ok: true,
    faceUvs: atlas.faceUvs.length,
    facePixelTriangles: atlas.facePixelTriangles.length,
    ...(atlas.islandCount !== undefined ? { islandCount: atlas.islandCount } : {}),
    textureSize: atlas.textureSize,
  };
}

void run().catch((error: unknown) => {
  smokeGlobal.__MESH_SIMPLIFICATION_SMOKE__ = {
    ok: false,
    error: error instanceof Error ? error.message : String(error),
    ...(error instanceof Error && error.stack ? { stack: error.stack } : {}),
  };
});
