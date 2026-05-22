import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { NodeIO } from '@gltf-transform/core';
import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { writePublicNormalMapGlb } from '../fixtures/publicGltfFixtures';
import {
  groupScenePrimitives,
  readGlbScenePrimitives,
  replaceScenePrimitiveGroupGeometry,
  replaceScenePrimitiveGroupTextured,
  writeScenePrimitiveDocument,
} from '../../src/local/scenePrimitiveGltfIo';
import { hasBakeableTextureData } from '../../src/pipeline/process';
import {
  bakePrimitiveTextures,
  processPrimitiveGeometries,
  type ProcessablePrimitiveEntry,
} from '../../src/pipeline/sceneProcessing';
import type { RawMesh } from '../../src/simplification/types';
import type { AtlasLayout } from '../../src/texture/types';
import { hasMaterialTextures } from '../../src/texture/types';

interface BakedPrimitiveExpectation {
  id: string;
  positionCount: number;
  indexCount: number;
}

interface TangentHandednessCounts {
  positive: number;
  negative: number;
  zero: number;
}

function countSerializedTexturedVertices(outputRawMesh: RawMesh, atlas: AtlasLayout): number {
  const vertexKeys = new Set<string>();
  const uvKey = (value: number): string => String(Math.round(value * 1e9));
  for (let faceIndex = 0; faceIndex < outputRawMesh.faces.length; faceIndex += 1) {
    const face = outputRawMesh.faces[faceIndex]!;
    const faceUvs = atlas.faceUvs[faceIndex]!;
    for (let corner = 0; corner < 3; corner += 1) {
      const sourceVertexId = face[corner]!;
      const uv = faceUvs[corner]!;
      vertexKeys.add(`${sourceVertexId}/${uvKey(uv.x)}/${uvKey(uv.y)}`);
    }
  }
  return vertexKeys.size;
}

async function bakeFixture(
  inputPath: string,
  outputPath: string,
  ratio: number,
): Promise<BakedPrimitiveExpectation[]> {
  const source = await readGlbScenePrimitives(inputPath);
  expect(source.warnings).toEqual([]);
  const groups = groupScenePrimitives(source.entries, 'material-parent');
  const groupsById = new Map(groups.map((group) => [group.id, group]));
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
    requiresAttributeTransfer: true,
  }));
  expect(processable.some((entry) => entry.bakeable)).toBe(true);

  const options = {
    target: { kind: 'ratio' as const, ratio },
    primitiveGrouping: 'material-parent' as const,
    virtualEdges: { mode: 'auto-local-radius' as const },
    weldVertices: true,
    recomputeNormals: true,
    transferTextures: true,
    textureSize: 256,
    texturePadding: 4,
    textureFilter: 'linear' as const,
  };
  const geometryResult = processPrimitiveGeometries(processable, options);
  const texturedResult = await bakePrimitiveTextures(geometryResult, options);
  expect(texturedResult.entries.map((entry) => entry.id).sort()).toEqual(
    processable.map((entry) => entry.id).sort(),
  );

  const bakedExpectations: BakedPrimitiveExpectation[] = [];
  for (const entry of texturedResult.entries) {
    if (!entry.source.bakeable) continue;
    const group = groupsById.get(entry.id);
    expect(group).toBeDefined();
    expect(entry.baked).toBeDefined();
    const baked = entry.baked!;
    replaceScenePrimitiveGroupTextured(group!, {
      outputRawMesh: baked.raw.rawMesh,
      atlas: baked.baked.atlas,
      image: baked.baked.image,
      additionalTextures: baked.baked.additionalTextures,
    });
    bakedExpectations.push({
      id: entry.id,
      positionCount: countSerializedTexturedVertices(baked.raw.rawMesh, baked.baked.atlas),
      indexCount: baked.raw.rawMesh.faces.length * 3,
    });
  }
  expect(bakedExpectations.length).toBeGreaterThan(0);
  await writeScenePrimitiveDocument(source.document, outputPath);
  return bakedExpectations;
}

async function preserveMaterialFixture(
  inputPath: string,
  outputPath: string,
  ratio: number,
): Promise<void> {
  const source = await readGlbScenePrimitives(inputPath, { mode: 'geometry-with-texture-metadata' });
  expect(source.warnings).toEqual([]);
  const groups = groupScenePrimitives(source.entries, 'material-parent');
  const groupsById = new Map(groups.map((group) => [group.id, group]));
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
    requiresAttributeTransfer: true,
  }));

  const geometryResult = processPrimitiveGeometries(processable, {
    target: { kind: 'ratio', ratio },
    primitiveGrouping: 'material-parent',
    virtualEdges: { mode: 'manual-global-radius', radius: 0 },
    weldVertices: true,
    recomputeNormals: true,
    transferTextures: false,
    textureSize: 32,
    texturePadding: 1,
    textureFilter: 'linear',
  });

  for (const entry of geometryResult.entries) {
    const group = groupsById.get(entry.id);
    expect(group).toBeDefined();
    replaceScenePrimitiveGroupGeometry(group!, entry.geometry.raw.rawMesh, entry.transferredAttributes);
  }
  await writeScenePrimitiveDocument(source.document, outputPath);
}

function tangentHandednessCounts(tangent: { getCount(): number; getElement(index: number, target: number[]): void }): TangentHandednessCounts {
  const counts: TangentHandednessCounts = { positive: 0, negative: 0, zero: 0 };
  const element = [0, 0, 0, 1];
  for (let index = 0; index < tangent.getCount(); index += 1) {
    tangent.getElement(index, element);
    if (element[3]! > 0) counts.positive += 1;
    else if (element[3]! < 0) counts.negative += 1;
    else counts.zero += 1;
  }
  return counts;
}

async function normalMappedPrimitive(path: string) {
  const document = await new NodeIO().read(path);
  return document.getRoot().listMeshes()
    .flatMap((mesh) => mesh.listPrimitives())
    .find((primitive) => primitive.getMaterial()?.getNormalTexture() !== null);
}

async function expectBakedNormalTangentOutput(
  path: string,
  bakedExpectations: readonly BakedPrimitiveExpectation[],
): Promise<void> {
  const document = await new NodeIO().read(path);
  const primitives = document.getRoot().listMeshes()
    .flatMap((mesh) => mesh.listPrimitives())
    .filter((candidate) => candidate.getMaterial()?.getNormalTexture() !== null);
  const primitive = primitives.find((candidate) => {
    const position = candidate.getAttribute('POSITION');
    const indices = candidate.getIndices();
    return bakedExpectations.some((expected) => (
      position?.getCount() === expected.positionCount
      && indices?.getCount() === expected.indexCount
    ));
  });
  expect(primitive).toBeDefined();
  const position = primitive!.getAttribute('POSITION');
  const normal = primitive!.getAttribute('NORMAL');
  const tangent = primitive!.getAttribute('TANGENT');
  const uv = primitive!.getAttribute('TEXCOORD_0');
  const indices = primitive!.getIndices();
  expect(position).not.toBeNull();
  expect(normal).not.toBeNull();
  expect(tangent).not.toBeNull();
  expect(uv).not.toBeNull();
  expect(indices).not.toBeNull();
  expect(position!.getCount()).toBeGreaterThan(0);
  expect(indices!.getCount()).toBeGreaterThan(0);
  expect(position!.getType()).toBe('VEC3');
  expect(normal!.getType()).toBe('VEC3');
  expect(tangent!.getType()).toBe('VEC4');
  expect(uv!.getType()).toBe('VEC2');
  expect(indices!.getType()).toBe('SCALAR');
  expect(normal!.getCount()).toBe(position!.getCount());
  expect(tangent!.getCount()).toBe(position!.getCount());
  expect(uv!.getCount()).toBe(position!.getCount());
  const matchingExpectation = bakedExpectations.find((expected) => (
    expected.positionCount === position!.getCount()
    && expected.indexCount === indices!.getCount()
  ));
  expect(matchingExpectation).toBeDefined();
  expect(primitive!.getMaterial()?.getNormalTexture()?.getImage()).toBeTruthy();
}

describe('normal tangent fixture baking regressions', () => {
  it('bakes generated tangentless normal-map fixture with generated output tangents', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-normal-tangent-test-'));
    try {
      const input = join(dir, 'normal-map-input.glb');
      const output = join(dir, 'normal-tangent-test.glb');
      await writePublicNormalMapGlb(input);
      const bakedExpectations = await bakeFixture(input, output, 1);
      await expectBakedNormalTangentOutput(output, bakedExpectations);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60000);

  it('bakes generated authored-tangent normal-map fixture with generated output tangents', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-normal-tangent-mirror-test-'));
    try {
      const input = join(dir, 'authored-tangent-input.glb');
      const output = join(dir, 'normal-tangent-mirror-test.glb');
      await writePublicNormalMapGlb(input, { authoredTangents: true, negativeHandedness: true });
      const bakedExpectations = await bakeFixture(input, output, 0.5);
      await expectBakedNormalTangentOutput(output, bakedExpectations);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60000);

  it('preserves generated authored tangent handedness in preserve-material output', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-normal-tangent-mirror-preserve-'));
    try {
      const input = join(dir, 'authored-tangent-input.glb');
      const output = join(dir, 'normal-tangent-mirror-preserve.glb');
      await writePublicNormalMapGlb(input, { authoredTangents: true, negativeHandedness: true });
      await preserveMaterialFixture(input, output, 1);

      const sourcePrimitive = await normalMappedPrimitive(input);
      const outputPrimitive = await normalMappedPrimitive(output);
      const sourceTangents = sourcePrimitive?.getAttribute('TANGENT');
      const outputTangents = outputPrimitive?.getAttribute('TANGENT');
      const outputPositions = outputPrimitive?.getAttribute('POSITION');

      expect(sourcePrimitive).toBeDefined();
      expect(outputPrimitive).toBeDefined();
      expect(sourceTangents).not.toBeNull();
      expect(sourceTangents).toBeDefined();
      expect(outputTangents).not.toBeNull();
      expect(outputTangents).toBeDefined();
      expect(outputTangents?.getCount()).toBe(outputPositions?.getCount());
      expect(tangentHandednessCounts(outputTangents!)).toEqual(tangentHandednessCounts(sourceTangents!));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60000);

  it('emits positive preserve-material tangents for generated tangentless normal maps', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-normal-tangent-preserve-'));
    try {
      const input = join(dir, 'normal-map-input.glb');
      const output = join(dir, 'normal-tangent-preserve.glb');
      await writePublicNormalMapGlb(input);
      await preserveMaterialFixture(input, output, 1);

      const outputPrimitive = await normalMappedPrimitive(output);
      const outputTangents = outputPrimitive?.getAttribute('TANGENT');
      const outputPositions = outputPrimitive?.getAttribute('POSITION');

      expect(outputPrimitive).toBeDefined();
      expect(outputTangents).not.toBeNull();
      expect(outputTangents).toBeDefined();
      expect(outputTangents?.getCount()).toBe(outputPositions?.getCount());
      expect(tangentHandednessCounts(outputTangents!)).toEqual({
        positive: outputTangents!.getCount(),
        negative: 0,
        zero: 0,
      });
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60000);
});
