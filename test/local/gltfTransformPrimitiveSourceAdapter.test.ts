import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { Accessor, Document, NodeIO, Primitive } from '@gltf-transform/core';
import { afterEach, describe, expect, it } from 'vitest';
import { GltfTransformPrimitiveSourceAdapter } from '../../src/local/gltfTransformPrimitiveSourceAdapter';
import type { ProcessedPrimitiveEntry } from '../../src/pipeline/sceneProcessing';
import type { RawMesh, RawSimplificationResult } from '../../src/simplification/types';
import { writePublicTexturedMultiPrimitiveGlb } from '../fixtures/publicGltfFixtures';
import { Vector3 } from 'three';

let tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.map((dir) => rm(dir, { recursive: true, force: true })));
  tempDirs = [];
});

function makeTriangleAccessors(doc: Document, buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer()) {
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2]))
    .setBuffer(buffer);
  return { positions, indices };
}

async function writeSharedMaterialParentGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const material = doc.createMaterial('shared')
    .setBaseColorFactor([0.25, 0.5, 0.75, 1])
    .setMetallicFactor(0)
    .setRoughnessFactor(0.5);
  const parent = doc.createNode('parent');
  const scene = doc.createScene('Scene').addChild(parent);

  for (let i = 0; i < 2; i += 1) {
    const accessors = makeTriangleAccessors(doc, buffer);
    const primitive = doc.createPrimitive()
      .setMode(Primitive.Mode.TRIANGLES!)
      .setAttribute('POSITION', accessors.positions)
      .setIndices(accessors.indices)
      .setMaterial(material);
    const mesh = doc.createMesh(`mesh-${i}`).addPrimitive(primitive);
    parent.addChild(doc.createNode(`node-${i}`).setTranslation([i, 0, 0]).setMesh(mesh));
  }

  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeAdapterFixture(path: string): Promise<string> {
  await writePublicTexturedMultiPrimitiveGlb(path);
  return path;
}

function replacementRawMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(2, 0, 0),
      new Vector3(0, 2, 0),
    ],
    faces: [[0, 1, 2]],
  };
}

function rawResult(rawMesh: RawMesh): RawSimplificationResult {
  return {
    rawMesh,
    outputFaceIds: rawMesh.faces.map((_, index) => index),
    history: [],
    stats: {
      inputVertices: rawMesh.positions.length,
      inputFaces: rawMesh.faces.length,
      outputVertices: rawMesh.positions.length,
      outputFaces: rawMesh.faces.length,
      physicalEdges: 3,
      virtualEdges: 0,
      collapses: 0,
      stoppedReason: 'target-reached',
    },
  };
}

describe('GltfTransformPrimitiveSourceAdapter', () => {
  it('defers reading the input path until extraction or summarization', async () => {
    const adapter = await GltfTransformPrimitiveSourceAdapter.read('does-not-exist.glb');

    await expect(adapter.extractGroups({
      groupingMode: 'material-parent',
      mode: 'geometry-with-texture-metadata',
    })).rejects.toThrow();
  });

  it('summarizes and extracts one selected grouping mode', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-adapter-public-'));
    tempDirs.push(dir);
    const input = await writeAdapterFixture(join(dir, 'input.glb'));
    const adapter = await GltfTransformPrimitiveSourceAdapter.read(input);
    const summary = await adapter.summarize();
    expect(summary.inputFaces).toBeGreaterThan(0);
    const materialParent = await adapter.extractGroups({
      groupingMode: 'material-parent',
      mode: 'geometry-with-texture-metadata',
    });
    expect(materialParent.entries.length).toBeGreaterThan(0);
    expect(materialParent.applyMetadataByEntryId.size).toBe(materialParent.entries.length);
    materialParent.releaseProcessingData();
    expect(materialParent.entries).toEqual([]);
    expect(materialParent.applyMetadataByEntryId.size).toBeGreaterThan(0);
    materialParent.dispose();
    expect(materialParent.applyMetadataByEntryId.size).toBe(0);
    expect(materialParent.extractionApplyState.sourceByEntryId.size).toBe(0);
  });

  it('keeps stable entry ids across geometry and bake extraction', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-adapter-public-'));
    tempDirs.push(dir);
    const input = await writeAdapterFixture(join(dir, 'input.glb'));
    const adapter = await GltfTransformPrimitiveSourceAdapter.read(input);
    const geometry = await adapter.extractGroups({
      groupingMode: 'material-parent',
      mode: 'geometry-with-texture-metadata',
    });
    const bake = await adapter.extractGroups({
      groupingMode: 'material-parent',
      mode: 'bake',
    });
    expect(bake.entries.map((entry) => entry.id)).toEqual(geometry.entries.map((entry) => entry.id));
  });

  it('summarizes bakeable entries from texture metadata without decoded image payloads', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-adapter-public-'));
    tempDirs.push(dir);
    const input = await writeAdapterFixture(join(dir, 'input.glb'));
    const adapter = await GltfTransformPrimitiveSourceAdapter.read(input);

    const summary = await adapter.summarize();
    const bake = await adapter.extractGroups({
      groupingMode: 'material-parent',
      mode: 'bake',
    });

    expect(summary.bakeableEntryCount).toBeGreaterThan(0);
    expect(summary.bakeableEntryCount).toBe(bake.entries.filter((entry) => entry.bakeable).length);
  });

  it('summarizes factor-only material-parent entries as preservable but not bakeable', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-adapter-summary-group-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeSharedMaterialParentGlb(input);
    const adapter = await GltfTransformPrimitiveSourceAdapter.read(input);

    const summary = await adapter.summarize();
    const bake = await adapter.extractGroups({
      groupingMode: 'material-parent',
      mode: 'bake',
    });

    expect(summary.hasPreservableMaterialData).toBe(true);
    expect(summary.hasImageBackedTextureBakeData).toBe(false);
    expect(summary.bakeableEntryCount).toBe(0);
    expect(summary.bakeableEntryCount).toBe(bake.entries.filter((entry) => entry.bakeable).length);
  });

  it('applies geometry results to the retained document', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-adapter-public-'));
    tempDirs.push(dir);
    const input = await writeAdapterFixture(join(dir, 'input.glb'));
    const adapter = await GltfTransformPrimitiveSourceAdapter.read(input);
    const extraction = await adapter.extractGroups({
      groupingMode: 'none',
      mode: 'geometry-with-texture-metadata',
    });
    const source = extraction.entries[0]!;
    const raw = rawResult(replacementRawMesh());
    const processed: ProcessedPrimitiveEntry = {
      id: source.id,
      source,
      geometry: { raw, elapsedSeconds: 0 },
      raw,
    };

    const document = await adapter.applyResults(extraction, [processed]);

    expect(document).toBe(extraction.extractionApplyState.document);
    const metadata = extraction.applyMetadataByEntryId.get(source.id)!;
    expect(metadata.entries[0]!.primitive.getAttribute('POSITION')?.getCount()).toBe(3);
    expect(metadata.entries[0]!.primitive.getAttribute('NORMAL')?.getCount()).toBe(3);
  });
});
