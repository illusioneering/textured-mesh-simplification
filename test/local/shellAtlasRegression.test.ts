import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { readGlbScenePrimitives, groupScenePrimitives } from '../../src/local/scenePrimitiveGltfIo';
import { hasBakeableTextureData } from '../../src/pipeline/process';
import { processPrimitiveGeometries, type ProcessablePrimitiveEntry } from '../../src/pipeline/sceneProcessing';
import { createInjectiveAtlas } from '../../src/texture/atlas';
import { hasMaterialTextures } from '../../src/texture/types';
import { writePublicAtlasStressGlb } from '../fixtures/publicGltfFixtures';

let tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.map((dir) => rm(dir, { recursive: true, force: true })));
  tempDirs = [];
});

describe('synthetic atlas regression', () => {
  it('creates an atlas for a generated textured grid simplified to ratio 0.5', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-atlas-stress-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const inputFaces = await writePublicAtlasStressGlb(input, 12);

    const source = await readGlbScenePrimitives(input);
    const groups = groupScenePrimitives(source.entries, 'material-parent');
    const processable = groups.map<ProcessablePrimitiveEntry>((group) => ({
      id: group.id,
      rawMesh: group.rawMesh,
      texturedRawMesh: {
        rawMesh: group.rawMesh,
        faceAttributes: group.faceAttributes,
        materials: [group.sourceMaterial],
      },
      bakeable: hasBakeableTextureData({
        rawMesh: group.rawMesh,
        faceAttributes: group.faceAttributes,
        materials: [group.sourceMaterial],
      }),
      hasTexturedMaterial: hasMaterialTextures(group.sourceMaterial),
    }));

    expect(source.warnings).toEqual([]);
    expect(processable).toHaveLength(1);
    expect(processable[0]!.bakeable).toBe(true);

    const geometryResult = processPrimitiveGeometries(processable, {
      target: { kind: 'ratio', ratio: 0.5 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'auto-local-radius' },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 512,
      texturePadding: 4,
      textureFilter: 'linear',
    });

    expect(inputFaces).toBe(288);
    expect(geometryResult.stats.inputFaces).toBe(inputFaces);
    expect(geometryResult.stats.outputFaces).toBeGreaterThan(0);
    expect(geometryResult.stats.outputFaces).toBeLessThan(inputFaces);
    const outputRawMesh = geometryResult.entries[0]!.raw.rawMesh;
    const atlas = await createInjectiveAtlas(outputRawMesh, { textureSize: 512, padding: 4 });

    expect(atlas.faceUvs).toHaveLength(outputRawMesh.faces.length);
    expect(atlas.facePixelTriangles).toHaveLength(outputRawMesh.faces.length);
    expect(atlas.islandCount).toBeGreaterThan(0);
    for (const faceUvs of atlas.faceUvs) {
      for (const uv of faceUvs) {
        expect(Number.isFinite(uv.x)).toBe(true);
        expect(Number.isFinite(uv.y)).toBe(true);
        expect(uv.x).toBeGreaterThanOrEqual(0);
        expect(uv.x).toBeLessThanOrEqual(1);
        expect(uv.y).toBeGreaterThanOrEqual(0);
        expect(uv.y).toBeLessThanOrEqual(1);
      }
    }
  }, 15000);
});
