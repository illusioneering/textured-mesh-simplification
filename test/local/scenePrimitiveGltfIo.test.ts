import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { afterEach, describe, expect, it } from 'vitest';
import { Accessor, Document, NodeIO, Primitive, TextureInfo, type Material, type Texture } from '@gltf-transform/core';
import { Matrix3, Matrix4, Vector2, Vector3 } from 'three';
import { encodePng } from '../../src/local/imageCodecs';
import { toProcessablePrimitiveEntry } from '../../src/pipeline/primitiveEntryMetadata';
import {
  groupScenePrimitives,
  readGlbScenePrimitives,
  replaceScenePrimitiveGroupGeometry,
  replaceScenePrimitiveGroupTextured,
  replaceScenePrimitiveGeometry,
  replaceScenePrimitiveTextured,
  writeScenePrimitiveDocument,
} from '../../src/local/scenePrimitiveGltfIo';
import {
  bakePrimitiveTextures,
  processPrimitiveGeometries,
  type ProcessablePrimitiveEntry,
} from '../../src/pipeline/sceneProcessing';
import type { TransferredMeshAttributes } from '../../src/simplification/attributes';
import { transferVertexAttributesToSimplifiedMesh } from '../../src/simplification/attributeTransfer';
import { SimplificationMesh } from '../../src/simplification/mesh';
import { computeConnectedComponents } from '../../src/simplification/virtualEdges';
import type { RawMesh } from '../../src/simplification/types';
import type { AtlasLayout } from '../../src/texture/types';
import { faceUvSet } from '../../src/texture/types';
import {
  writePublicNormalMapGlb,
  writePublicTexturedVertexColorGlb,
  writePublicVertexColorGlb,
} from '../fixtures/publicGltfFixtures';

let tempDirs: string[] = [];

const AUTHORED_NORMALS = [
  new Vector3(1, 2, 3).normalize(),
  new Vector3(-2, 1, 0.5).normalize(),
  new Vector3(0.25, -1, 2).normalize(),
] as const;

afterEach(async () => {
  await Promise.all(tempDirs.map((dir) => rm(dir, { recursive: true, force: true })));
  tempDirs = [];
});

function makeTriangleAccessors(doc: Document, buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer()) {
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    .setBuffer(buffer);
  const normals = doc.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]))
    .setBuffer(buffer);
  const uvs = doc.createAccessor('TEXCOORD_0')
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array([0, 0, 1, 0, 0, 1]))
    .setBuffer(buffer);
  const uv1s = doc.createAccessor('TEXCOORD_1')
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array([0.5, 0.5, 0.75, 0.5, 0.5, 0.75]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2]))
    .setBuffer(buffer);
  return { positions, normals, uvs, uv1s, indices };
}

function addMaterialTexture(doc: Document, name: string, rgba: readonly number[]) {
  return doc.createTexture(name)
    .setMimeType('image/png')
    .setImage(encodePng({ width: 1, height: 1, data: new Uint8Array(rgba) }));
}

function expectVectorClose(actual: Vector3 | undefined, expected: Vector3): void {
  expect(actual).toBeDefined();
  expect(actual!.x).toBeCloseTo(expected.x, 6);
  expect(actual!.y).toBeCloseTo(expected.y, 6);
  expect(actual!.z).toBeCloseTo(expected.z, 6);
}

async function writeTwoNodeGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const texturedAccessors = makeTriangleAccessors(doc, buffer);
  const plainAccessors = makeTriangleAccessors(doc, buffer);
  const linePositions = doc.createAccessor('line-position')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 0, 1, 0, 0]))
    .setBuffer(buffer);

  const png = encodePng({ width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) });
  const texture = doc.createTexture('red').setMimeType('image/png').setImage(png);
  const texturedMaterial = doc.createMaterial('textured')
    .setBaseColorTexture(texture)
    .setBaseColorFactor([0.7, 0.8, 0.9, 1])
    .setMetallicFactor(0)
    .setRoughnessFactor(0.42)
    .setDoubleSided(true);
  texturedMaterial.getBaseColorTextureInfo()!
    .setTexCoord(0)
    .setWrapS(TextureInfo.WrapMode.CLAMP_TO_EDGE!)
    .setWrapT(TextureInfo.WrapMode.CLAMP_TO_EDGE!)
    .setMagFilter(TextureInfo.MagFilter.NEAREST!);
  const plainMaterial = doc.createMaterial('plain')
    .setBaseColorFactor([0.1, 0.2, 0.3, 1])
    .setMetallicFactor(0.2)
    .setRoughnessFactor(0.6);

  const texturedPrimitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', texturedAccessors.positions)
    .setAttribute('NORMAL', texturedAccessors.normals)
    .setAttribute('TEXCOORD_0', texturedAccessors.uvs)
    .setIndices(texturedAccessors.indices)
    .setMaterial(texturedMaterial);
  const plainPrimitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', plainAccessors.positions)
    .setAttribute('NORMAL', plainAccessors.normals)
    .setIndices(plainAccessors.indices)
    .setMaterial(plainMaterial);
  const unsupportedPrimitive = doc.createPrimitive()
    .setMode(Primitive.Mode.LINES!)
    .setAttribute('POSITION', linePositions);

  const texturedMesh = doc.createMesh('textured-mesh').addPrimitive(texturedPrimitive);
  const mixedMesh = doc.createMesh('mixed-mesh')
    .addPrimitive(plainPrimitive)
    .addPrimitive(unsupportedPrimitive);
  const texturedNode = doc.createNode('textured-node')
    .setTranslation([5, 0, 0])
    .setMesh(texturedMesh);
  const mixedNode = doc.createNode('mixed-node')
    .setTranslation([0, 7, 0])
    .setMesh(mixedMesh);
  const scene = doc.createScene('Scene').addChild(texturedNode).addChild(mixedNode);
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeDuplicatePositionSeamGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const material = doc.createMaterial('seam-material')
    .setMetallicFactor(0)
    .setRoughnessFactor(0.5);
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 0,
    ]))
    .setBuffer(buffer);
  const normals = doc.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 1, 0,
    ]))
    .setBuffer(buffer);
  const uvs = doc.createAccessor('TEXCOORD_0')
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array([
      0, 0,
      1, 0,
      0, 1,
      0.5, 0.5,
    ]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2, 3, 2, 1]))
    .setBuffer(buffer);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', positions)
    .setAttribute('NORMAL', normals)
    .setAttribute('TEXCOORD_0', uvs)
    .setIndices(indices)
    .setMaterial(material);
  doc.createScene('scene').addChild(
    doc.createNode('seam-node').setMesh(doc.createMesh('seam-mesh').addPrimitive(primitive)),
  );
  await new NodeIO().write(path, doc);
}

async function writeTransformedAuthoredNormalsGlb(path: string, transform: Matrix4): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    .setBuffer(buffer);
  const normals = doc.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array(AUTHORED_NORMALS.flatMap((normal) => [normal.x, normal.y, normal.z])))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2]))
    .setBuffer(buffer);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', positions)
    .setAttribute('NORMAL', normals)
    .setIndices(indices);
  const mesh = doc.createMesh('normal-mesh').addPrimitive(primitive);
  const node = doc.createNode('normal-node')
    .setMatrix(transform.toArray())
    .setMesh(mesh);
  const scene = doc.createScene('Scene').addChild(node);
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeMismatchedNormalCountSceneGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    .setBuffer(buffer);
  const normals = doc.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 1, 0, 0, 1]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2]))
    .setBuffer(buffer);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', positions)
    .setAttribute('NORMAL', normals)
    .setIndices(indices);
  const mesh = doc.createMesh('mesh').addPrimitive(primitive);
  const scene = doc.createScene('Scene').addChild(doc.createNode('node').setMesh(mesh));
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeMismatchedColorCountSceneGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    .setBuffer(buffer);
  const colors = doc.createAccessor('COLOR_0')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([1, 0, 0, 0, 1, 0]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2]))
    .setBuffer(buffer);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', positions)
    .setAttribute('COLOR_0', colors)
    .setIndices(indices);
  const mesh = doc.createMesh('mesh').addPrimitive(primitive);
  const scene = doc.createScene('Scene').addChild(doc.createNode('node').setMesh(mesh));
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeSplitMaterialParentGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const material = doc.createMaterial('shared')
    .setBaseColorFactor([0.2, 0.3, 0.4, 1])
    .setMetallicFactor(0)
    .setRoughnessFactor(0.7);
  const parent = doc.createNode('shared-parent');
  const scene = doc.createScene('Scene').addChild(parent);

  for (let i = 0; i < 2; i += 1) {
    const accessors = makeTriangleAccessors(doc, buffer);
    const primitive = doc.createPrimitive()
      .setMode(Primitive.Mode.TRIANGLES!)
      .setAttribute('POSITION', accessors.positions)
      .setIndices(accessors.indices)
      .setMaterial(material);
    const mesh = doc.createMesh(`split-mesh-${i}`).addPrimitive(primitive);
    const node = doc.createNode(`split-node-${i}`)
      .setTranslation([i, 0, 0])
      .setMesh(mesh);
    parent.addChild(node);
  }

  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeMixedColorMaterialParentGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const material = doc.createMaterial('shared')
    .setBaseColorFactor([0.2, 0.3, 0.4, 1])
    .setMetallicFactor(0)
    .setRoughnessFactor(0.7);
  const parent = doc.createNode('shared-parent');
  const scene = doc.createScene('Scene').addChild(parent);

  const coloredAccessors = makeTriangleAccessors(doc, buffer);
  const colors = doc.createAccessor('COLOR_0')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]))
    .setBuffer(buffer);
  const coloredPrimitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', coloredAccessors.positions)
    .setAttribute('COLOR_0', colors)
    .setIndices(coloredAccessors.indices)
    .setMaterial(material);
  parent.addChild(doc.createNode('colored-node').setMesh(doc.createMesh('colored-mesh').addPrimitive(coloredPrimitive)));

  const plainAccessors = makeTriangleAccessors(doc, buffer);
  const plainPrimitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', plainAccessors.positions)
    .setIndices(plainAccessors.indices)
    .setMaterial(material);
  parent.addChild(doc.createNode('plain-node').setMesh(doc.createMesh('plain-mesh').addPrimitive(plainPrimitive)));

  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeSplitSharedBoundaryGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const material = doc.createMaterial('shared');
  const parent = doc.createNode('shared-parent');
  const scene = doc.createScene('Scene').addChild(parent);
  const triangles = [
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 1, 0],
  ];

  for (let i = 0; i < triangles.length; i += 1) {
    const positions = doc.createAccessor(`POSITION-${i}`)
      .setType(Accessor.Type.VEC3!)
      .setArray(new Float32Array(triangles[i]!))
      .setBuffer(buffer);
    const indices = doc.createAccessor(`indices-${i}`)
      .setType(Accessor.Type.SCALAR!)
      .setArray(new Uint16Array([0, 1, 2]))
      .setBuffer(buffer);
    const primitive = doc.createPrimitive()
      .setMode(Primitive.Mode.TRIANGLES!)
      .setAttribute('POSITION', positions)
      .setIndices(indices)
      .setMaterial(material);
    parent.addChild(doc.createNode(`split-node-${i}`).setMesh(doc.createMesh(`split-mesh-${i}`).addPrimitive(primitive)));
  }

  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeFullMaterialGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const accessors = makeTriangleAccessors(doc, buffer);
  const base = addMaterialTexture(doc, 'base', [255, 0, 0, 255]);
  const normal = addMaterialTexture(doc, 'normal', [128, 128, 255, 255]);
  const mr = addMaterialTexture(doc, 'metallic-roughness', [0, 127, 255, 255]);
  const ao = addMaterialTexture(doc, 'occlusion', [200, 0, 0, 255]);
  const emissive = addMaterialTexture(doc, 'emissive', [0, 0, 255, 255]);
  const material = doc.createMaterial('full-material')
    .setBaseColorTexture(base)
    .setNormalTexture(normal)
    .setNormalScale(0.75)
    .setMetallicRoughnessTexture(mr)
    .setMetallicFactor(0.25)
    .setRoughnessFactor(0.5)
    .setOcclusionTexture(ao)
    .setOcclusionStrength(0.6)
    .setEmissiveTexture(emissive)
    .setEmissiveFactor([0.1, 0.2, 0.3]);
  material.getBaseColorTextureInfo()!.setTexCoord(0);
  material.getNormalTextureInfo()!.setTexCoord(0);
  material.getMetallicRoughnessTextureInfo()!.setTexCoord(0);
  material.getOcclusionTextureInfo()!.setTexCoord(1);
  material.getEmissiveTextureInfo()!.setTexCoord(0);

  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', accessors.positions)
    .setAttribute('NORMAL', accessors.normals)
    .setAttribute('TEXCOORD_0', accessors.uvs)
    .setAttribute('TEXCOORD_1', accessors.uv1s)
    .setIndices(accessors.indices)
    .setMaterial(material);
  const mesh = doc.createMesh('full-material-mesh').addPrimitive(primitive);
  const node = doc.createNode('full-material-node').setMesh(mesh);
  const scene = doc.createScene('Scene').addChild(node);
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeFullMaterialSquareGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]))
    .setBuffer(buffer);
  const normals = doc.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]))
    .setBuffer(buffer);
  const uv0 = doc.createAccessor('TEXCOORD_0')
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array([
      0, 0,
      1, 0,
      1, 1,
      0, 1,
    ]))
    .setBuffer(buffer);
  const uv1 = doc.createAccessor('TEXCOORD_1')
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array([
      0.25, 0.25,
      0.75, 0.25,
      0.75, 0.75,
      0.25, 0.75,
    ]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2, 0, 2, 3]))
    .setBuffer(buffer);
  const base = addMaterialTexture(doc, 'base', [255, 0, 0, 255]);
  const normal = addMaterialTexture(doc, 'normal', [128, 128, 255, 255]);
  const material = doc.createMaterial('full-material-square')
    .setBaseColorTexture(base)
    .setNormalTexture(normal)
    .setNormalScale(0.75)
    .setMetallicFactor(0.25)
    .setRoughnessFactor(0.5);
  material.getBaseColorTextureInfo()!.setTexCoord(0);
  material.getNormalTextureInfo()!.setTexCoord(1);

  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', positions)
    .setAttribute('NORMAL', normals)
    .setAttribute('TEXCOORD_0', uv0)
    .setAttribute('TEXCOORD_1', uv1)
    .setIndices(indices)
    .setMaterial(material);
  const mesh = doc.createMesh('full-material-square-mesh').addPrimitive(primitive);
  const node = doc.createNode('full-material-square-node').setMesh(mesh);
  const scene = doc.createScene('Scene').addChild(node);
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeNormalMapUv1OnlySquareGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]))
    .setBuffer(buffer);
  const normals = doc.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]))
    .setBuffer(buffer);
  const uv1 = doc.createAccessor('TEXCOORD_1')
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array([
      0, 0,
      1, 0,
      1, 1,
      0, 1,
    ]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2, 0, 2, 3]))
    .setBuffer(buffer);
  const normal = addMaterialTexture(doc, 'normal', [128, 128, 255, 255]);
  const material = doc.createMaterial('uv1-normal-map-square')
    .setNormalTexture(normal);
  material.getNormalTextureInfo()!.setTexCoord(1);

  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', positions)
    .setAttribute('NORMAL', normals)
    .setAttribute('TEXCOORD_1', uv1)
    .setIndices(indices)
    .setMaterial(material);
  const mesh = doc.createMesh('uv1-normal-map-square-mesh').addPrimitive(primitive);
  const node = doc.createNode('uv1-normal-map-square-node').setMesh(mesh);
  const scene = doc.createScene('Scene').addChild(node);
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

async function writeMateriallessUv0SquareGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions = doc.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]))
    .setBuffer(buffer);
  const normals = doc.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]))
    .setBuffer(buffer);
  const uv0 = doc.createAccessor('TEXCOORD_0')
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array([
      0, 0,
      1, 0,
      1, 1,
      0, 1,
    ]))
    .setBuffer(buffer);
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2, 0, 2, 3]))
    .setBuffer(buffer);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', positions)
    .setAttribute('NORMAL', normals)
    .setAttribute('TEXCOORD_0', uv0)
    .setIndices(indices);
  const mesh = doc.createMesh('materialless-uv0-square-mesh').addPrimitive(primitive);
  const node = doc.createNode('materialless-uv0-square-node').setMesh(mesh);
  const scene = doc.createScene('Scene').addChild(node);
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

function replacementRawMesh(offset = 0): RawMesh {
  return {
    positions: [
      new Vector3(offset, 0, 0),
      new Vector3(offset + 2, 0, 0),
      new Vector3(offset, 2, 0),
    ],
    faces: [[0, 1, 2]],
  };
}

function seamRawMesh(): RawMesh {
  return {
    positions: [
      new Vector3(0, 0, 0),
      new Vector3(1, 0, 0),
      new Vector3(1, 1, 0),
      new Vector3(0, 1, 0),
    ],
    faces: [[0, 1, 2], [0, 2, 3]],
  };
}

function weldedTransferredAttributes(): TransferredMeshAttributes {
  return {
    vertices: [
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(0, 0) },
          { texCoord: 1, uv: new Vector2(0.1, 0.1) },
        ],
        normal: new Vector3(1, 0, 0),
      },
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(1, 0) },
          { texCoord: 1, uv: new Vector2(0.9, 0.1) },
        ],
        normal: new Vector3(0, 1, 0),
      },
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(1, 1) },
          { texCoord: 1, uv: new Vector2(0.9, 0.9) },
        ],
        normal: new Vector3(0, 0, 1),
      },
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(0, 1) },
          { texCoord: 1, uv: new Vector2(0.1, 0.9) },
        ],
        normal: new Vector3(0, -1, 0),
      },
    ],
  };
}

function transferredAttributesFromFaceAttributes(
  rawMesh: RawMesh,
  faceAttributes: readonly { uvSets: { texCoord: number; uvs: [Vector2, Vector2, Vector2] }[] }[],
): TransferredMeshAttributes {
  const vertices: TransferredMeshAttributes['vertices'] = rawMesh.positions.map(() => ({ uvSets: [] }));
  for (let faceIndex = 0; faceIndex < rawMesh.faces.length; faceIndex += 1) {
    const face = rawMesh.faces[faceIndex]!;
    const attributes = faceAttributes[faceIndex]!;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = vertices[face[corner]!]!;
      if (vertex.uvSets.length === 0) {
        vertex.uvSets = attributes.uvSets.map((uvSet) => ({
          texCoord: uvSet.texCoord,
          uv: uvSet.uvs[corner]!.clone(),
        }));
      }
    }
  }
  return { vertices };
}

function firstTrianglePrimitive(doc: Document): Primitive {
  const primitive = doc.getRoot()
    .listMeshes()
    .flatMap((mesh) => mesh.listPrimitives())
    .find((candidate) => candidate.getMode() === Primitive.Mode.TRIANGLES && candidate.getAttribute('POSITION'));
  if (!primitive) throw new Error('Expected a triangle primitive with POSITION.');
  return primitive;
}

function replacementAtlas(): AtlasLayout {
  return {
    textureSize: 4,
    padding: 1,
    faceUvs: [[new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)]],
    facePixelTriangles: [[[0, 0], [3, 0], [0, 3]]],
  };
}

function replacementAtlasForRawMesh(rawMesh: RawMesh): AtlasLayout {
  return {
    textureSize: 4,
    padding: 1,
    faceUvs: rawMesh.faces.map(() => [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)]),
    facePixelTriangles: rawMesh.faces.map(() => [[0, 0], [3, 0], [0, 3]]),
  };
}

function usedMaterials(doc: Document): Set<Material> {
  return new Set(doc.getRoot()
    .listMeshes()
    .flatMap((mesh) => mesh.listPrimitives())
    .map((primitive) => primitive.getMaterial())
    .filter((material): material is Material => material !== null));
}

function usedTextures(doc: Document): Set<Texture> {
  const textures = new Set<Texture>();
  for (const material of usedMaterials(doc)) {
    for (const texture of [
      material.getBaseColorTexture(),
      material.getNormalTexture(),
      material.getMetallicRoughnessTexture(),
      material.getOcclusionTexture(),
      material.getEmissiveTexture(),
    ]) {
      if (texture) textures.add(texture);
    }
  }
  return textures;
}

describe('scene-preserving primitive glTF I/O', () => {
  it('extracts supported primitives in primitive-local coordinates with decoded source materials and warnings', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-read-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeTwoNodeGlb(input);

    const read = await readGlbScenePrimitives(input);

    expect(read.entries).toHaveLength(2);
    expect(read.warnings).toHaveLength(1);
    expect(read.warnings[0]!.reason).toContain('mode');
    expect(read.entries.map((entry) => entry.nodeName)).toEqual(['textured-node', 'mixed-node']);
    expect(read.entries[0]!.rawMesh.positions[1]!.x).toBeCloseTo(1);
    expect(read.entries[0]!.rawMesh.positions[1]!.y).toBeCloseTo(0);
    expect(read.entries[0]!.sourceMaterial.baseColorTexture?.image?.width).toBe(1);
    expect(read.entries[0]!.sourceMaterial.metallicFactor).toBe(0);
    expect(read.entries[0]!.sourceMaterial.roughnessFactor).toBeCloseTo(0.42);
    expect(faceUvSet(read.entries[0]!.faceAttributes[0]!, 0)?.uvs[1].x).toBeCloseTo(1);
    expect(read.entries[0]!.faceAttributes[0]!.normalCorners?.[0].z).toBeCloseTo(1);
    expect(read.entries[0]!.faceAttributes[0]!.normalCorners?.[1].z).toBeCloseTo(1);
    expect(read.entries[0]!.faceAttributes[0]!.normalCorners?.[2].z).toBeCloseTo(1);
    expect(read.entries[1]!.sourceMaterial.baseColorTexture).toBeUndefined();
  });

  it('reads supplied tangents from a generated public fixture', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-public-tangent-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writePublicNormalMapGlb(input, { authoredTangents: true, negativeHandedness: true });

    const read = await readGlbScenePrimitives(input);
    const entry = read.entries[0]!;

    expect(entry.sourceMaterial.textureSlots.some((slot) => slot.slot === 'normal' && slot.hasImage)).toBe(true);
    expect(entry.faceAttributes.some((attributes) => attributes.normalMapYScale !== undefined)).toBe(false);
    expect(entry.faceAttributes.every((attributes) => attributes.tangentCorners?.length === 3)).toBe(true);
    expect(entry.faceAttributes.some((attributes) => attributes.tangentCorners?.some((tangent) => tangent.w < 0))).toBe(true);
  });

  it('extracts COLOR_0 face corners from local glTF primitives', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-public-color-read-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writePublicVertexColorGlb(input);

    const read = await readGlbScenePrimitives(input, { mode: 'geometry-with-texture-metadata' });
    const entry = read.entries.find((candidate) => candidate.faceAttributes.some((attributes) => attributes.colorCorners));

    expect(entry).toBeDefined();
    const coloredFace = entry!.faceAttributes.find((attributes) => attributes.colorCorners)!;
    expect(coloredFace.colorItemSize === 3 || coloredFace.colorItemSize === 4).toBe(true);
    expect(coloredFace.colorCorners).toHaveLength(3);
    expect(coloredFace.colorCorners!.some((color) => color.x > 0 || color.y > 0 || color.z > 0)).toBe(true);
    expect(coloredFace.colorCorners!.every((color) => color.w === 1)).toBe(true);

    const group = groupScenePrimitives([entry!], 'none')[0]!;
    expect(group.faceAttributes[0]!.colorItemSize).toBe(coloredFace.colorItemSize);
    expect(group.faceAttributes[0]!.colorCorners?.[0]).not.toBe(coloredFace.colorCorners![0]);
    expect(group.faceAttributes[0]!.colorCorners?.[0].toArray()).toEqual(coloredFace.colorCorners![0].toArray());
  });

  it('rejects primitives with a NORMAL accessor count that differs from POSITION count', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-bad-normal-count-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeMismatchedNormalCountSceneGlb(input);

    await expect(readGlbScenePrimitives(input)).rejects.toThrow(/NORMAL accessor count 2 does not match POSITION accessor count 3/);
  });

  it('rejects primitives with a COLOR_0 accessor count that differs from POSITION count', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-bad-color-count-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeMismatchedColorCountSceneGlb(input);

    await expect(readGlbScenePrimitives(input)).rejects.toThrow(/COLOR_0 accessor count 2 does not match POSITION accessor count 3/);
  });

  it('preserves COLOR_0 on geometry output after local attribute transfer', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-color-transfer-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writePublicVertexColorGlb(input);

    const read = await readGlbScenePrimitives(input, { mode: 'geometry-with-texture-metadata' });
    const group = groupScenePrimitives(read.entries, 'none').find((candidate) => (
      candidate.faceAttributes.some((attributes) => attributes.colorCorners)
    ))!;
    const processable = toProcessablePrimitiveEntry({
      id: group.id,
      rawMesh: group.rawMesh,
      texturedRawMesh: {
        rawMesh: group.rawMesh,
        faceAttributes: group.faceAttributes,
        materials: [group.sourceMaterial],
      },
    });
    const geometryResult = processPrimitiveGeometries([processable], {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'none',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    const processed = geometryResult.entries[0]!;
    const transferred = transferVertexAttributesToSimplifiedMesh({
      sourceFaceAttributes: group.faceAttributes,
      raw: processed.geometry.raw,
    });

    expect(transferred.vertices.every((vertex) => vertex.color)).toBe(true);
    replaceScenePrimitiveGroupGeometry(group, processed.geometry.raw.rawMesh, transferred);
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const primitive = firstTrianglePrimitive(doc);
    const positions = primitive.getAttribute('POSITION')!;
    const colors = primitive.getAttribute('COLOR_0')!;
    const colorArray = colors.getArray()!;

    expect(colors).toBeDefined();
    expect(colors.getCount()).toBe(positions.getCount());
    expect(colors.getType()).toBe(transferred.colorItemSize === 4 ? Accessor.Type.VEC4 : Accessor.Type.VEC3);
    expect(Array.from(colorArray).some((channel) => channel > 0)).toBe(true);
  });

  it('does not emit COLOR_0 on baked textured primitive replacement', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-color-baked-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writePublicVertexColorGlb(input);

    const read = await readGlbScenePrimitives(input, { mode: 'geometry-with-texture-metadata' });
    const group = groupScenePrimitives(read.entries, 'none').find((candidate) => (
      candidate.faceAttributes.some((attributes) => attributes.colorCorners)
    ))!;
    replaceScenePrimitiveGroupTextured(group, {
      outputRawMesh: group.rawMesh,
      atlas: replacementAtlasForRawMesh(group.rawMesh),
      image: { width: 4, height: 4, data: new Uint8Array(4 * 4 * 4).fill(255) },
    });
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const primitive = firstTrianglePrimitive(doc);
    const positionCount = primitive.getAttribute('POSITION')!.getCount();

    expect(primitive.getAttribute('COLOR_0')).toBeNull();
    expect(primitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(positionCount);
    expect(primitive.getAttribute('NORMAL')?.getCount()).toBe(positionCount);
  });

  it('inherits COLOR_0 on baked atlas output for textured colored primitives', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-baked-color0-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writePublicTexturedVertexColorGlb(input);
    const read = await readGlbScenePrimitives(input, { mode: 'bake' });
    const groups = groupScenePrimitives(read.entries, 'none');
    const group = groups.find((candidate) => (
      candidate.sourceMaterial.baseColorTexture
      && candidate.faceAttributes.some((attributes) => attributes.colorCorners)
    ))!;
    const processable = toProcessablePrimitiveEntry({
      id: group.id,
      rawMesh: group.rawMesh,
      texturedRawMesh: {
        rawMesh: group.rawMesh,
        faceAttributes: group.faceAttributes,
        materials: [group.sourceMaterial],
      },
    });

    const geometryResult = processPrimitiveGeometries([processable], {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'none',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 1024,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    const textured = await bakePrimitiveTextures(geometryResult, {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'none',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: true,
      textureSize: 1024,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    const processed = textured.entries[0]!;
    const transferredAttributes = processed.transferredAttributes;

    expect(processed.baked).toBeDefined();
    expect(transferredAttributes?.colorItemSize).toBe(4);
    replaceScenePrimitiveGroupTextured(group, {
      outputRawMesh: processed.baked!.raw.rawMesh,
      atlas: processed.baked!.baked.atlas,
      image: processed.baked!.baked.image,
      additionalTextures: processed.baked!.baked.additionalTextures,
      ...(transferredAttributes ? { transferredAttributes } : {}),
    });
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const primitive = firstTrianglePrimitive(doc);
    const position = primitive.getAttribute('POSITION')!;
    const color = primitive.getAttribute('COLOR_0')!;
    const sample: number[] = [];
    color.getElement(0, sample);

    expect(primitive.getMaterial()?.getBaseColorTexture()?.getImage()).toBeTruthy();
    expect(primitive.getAttribute('NORMAL')?.getCount()).toBe(position.getCount());
    expect(primitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(position.getCount());
    expect(color.getType()).toBe(Accessor.Type.VEC4);
    expect(color.getCount()).toBe(position.getCount());
    expect(sample[0]).toBeCloseTo(1);
    expect(sample[1]).toBeCloseTo(0);
    expect(sample[2]).toBeCloseTo(0);
    expect(sample[3]).toBeCloseTo(1);
  }, 15_000);

  it('preserves material texture slots, scalars, and required UV sets for geometry output', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-full-material-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeFullMaterialGlb(input);

    const read = await readGlbScenePrimitives(input);
    const entry = read.entries[0]!;

    expect(entry.sourceMaterial.textureSlots.map((slot) => slot.slot).sort()).toEqual([
      'baseColor',
      'emissive',
      'metallicRoughness',
      'normal',
      'occlusion',
    ]);
    const normalSlot = entry.sourceMaterial.textureSlots.find((slot) => slot.slot === 'normal');
    const metallicRoughnessSlot = entry.sourceMaterial.textureSlots.find((slot) => slot.slot === 'metallicRoughness');
    const occlusionSlot = entry.sourceMaterial.textureSlots.find((slot) => slot.slot === 'occlusion');
    const emissiveSlot = entry.sourceMaterial.textureSlots.find((slot) => slot.slot === 'emissive');
    const baseColorSlot = entry.sourceMaterial.textureSlots.find((slot) => slot.slot === 'baseColor');
    expect(baseColorSlot?.hasImage).toBe(true);
    expect(baseColorSlot?.image).toBeUndefined();
    expect(normalSlot?.image?.width).toBe(1);
    expect(Array.from(normalSlot!.image!.data.slice(0, 4))).toEqual([128, 128, 255, 255]);
    expect(metallicRoughnessSlot?.image?.width).toBe(1);
    expect(Array.from(metallicRoughnessSlot!.image!.data.slice(0, 4))).toEqual([0, 127, 255, 255]);
    expect(occlusionSlot?.image?.width).toBe(1);
    expect(Array.from(occlusionSlot!.image!.data.slice(0, 4))).toEqual([200, 0, 0, 255]);
    expect(emissiveSlot?.image?.width).toBe(1);
    expect(Array.from(emissiveSlot!.image!.data.slice(0, 4))).toEqual([0, 0, 255, 255]);
    expect(entry.sourceMaterial.normalScale).toBeCloseTo(0.75);
    expect(entry.sourceMaterial.occlusionStrength).toBeCloseTo(0.6);
    expect(entry.sourceMaterial.emissiveFactor).toEqual([0.1, 0.2, 0.3]);
    expect(entry.faceAttributes[0]!.uvSets.map((set) => set.texCoord)).toEqual([0, 1]);
    expect(entry.faceAttributes[0]!.normalMapYScale).toBe(-1);

    const replacement = replacementRawMesh();
    replaceScenePrimitiveGeometry(entry, replacement, transferredAttributesFromFaceAttributes(replacement, entry.faceAttributes));
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = doc.getRoot().listMeshes()[0]!.listPrimitives()[0]!;
    const outMaterial = outPrimitive.getMaterial();
    expect(outMaterial?.getBaseColorTexture()).toBeTruthy();
    expect(outMaterial?.getNormalTexture()).toBeTruthy();
    expect(outMaterial?.getMetallicRoughnessTexture()).toBeTruthy();
    expect(outMaterial?.getOcclusionTexture()).toBeTruthy();
    expect(outMaterial?.getEmissiveTexture()).toBeTruthy();
    expect(outMaterial?.getNormalScale()).toBeCloseTo(0.75);
    expect(outMaterial?.getOcclusionStrength()).toBeCloseTo(0.6);
    expect(outMaterial?.getEmissiveFactor()).toEqual([0.1, 0.2, 0.3]);
    expect(outPrimitive.getAttribute('TEXCOORD_0')).not.toBeNull();
    expect(outPrimitive.getAttribute('TEXCOORD_1')).not.toBeNull();
  });

  it('can read material texture metadata without retaining decoded image payloads', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-texture-metadata-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeFullMaterialGlb(input);

    const read = await readGlbScenePrimitives(input, { mode: 'geometry-with-texture-metadata' });
    const material = read.entries[0]!.sourceMaterial;
    const normalSlot = material.textureSlots.find((slot) => slot.slot === 'normal');

    expect(material.baseColorTexture?.image).toBeUndefined();
    expect(material.baseColorTexture?.texCoord).toBe(0);
    expect(material.textureSlots.every((slot) => slot.hasImage)).toBe(true);
    expect(material.textureSlots.every((slot) => slot.image === undefined)).toBe(true);
    expect(normalSlot?.hasImage).toBe(true);
    expect(normalSlot?.image).toBeUndefined();
    expect(material.normalScale).toBeCloseTo(0.75);
  });

  it('transfers UVs through grouped CLI-style geometry processing before preserving material textures', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-cli-transfer-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeFullMaterialSquareGlb(input);

    const read = await readGlbScenePrimitives(input);
    const groups = groupScenePrimitives(read.entries, 'material-parent');
    const group = groups[0]!;
    const processable: ProcessablePrimitiveEntry = {
      id: group.id,
      rawMesh: group.rawMesh,
      texturedRawMesh: {
        rawMesh: group.rawMesh,
        faceAttributes: group.faceAttributes,
        materials: [group.sourceMaterial],
      },
      bakeable: false,
      hasTexturedMaterial: true,
      requiresAttributeTransfer: true,
    };
    const geometryResult = processPrimitiveGeometries([processable], {
      target: { kind: 'faces', targetFaceCount: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    const processed = geometryResult.entries[0]!;

    expect(processed.geometry.raw.stats.collapses).toBeGreaterThan(0);
    expect(processed.geometry.raw.history).toHaveLength(processed.geometry.raw.stats.collapses);

    const transferred = transferVertexAttributesToSimplifiedMesh({
      sourceFaceAttributes: processable.texturedRawMesh!.faceAttributes,
      raw: processed.geometry.raw,
    });
    expect(transferred.normalMapYScale).toBe(-1);
    replaceScenePrimitiveGroupGeometry(group, processed.geometry.raw.rawMesh, transferred);
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = firstTrianglePrimitive(doc);
    const positionCount = outPrimitive.getAttribute('POSITION')!.getCount();
    expect(outPrimitive.getMaterial()?.getBaseColorTexture()).toBeTruthy();
    expect(outPrimitive.getMaterial()?.getNormalTexture()).toBeTruthy();
    expect(outPrimitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(positionCount);
    expect(outPrimitive.getAttribute('TEXCOORD_1')?.getCount()).toBe(positionCount);
  });

  it('serializes transferred geometry output with one vertex per welded raw vertex', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-uv-tuple-seam-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeFullMaterialSquareGlb(input);

    const read = await readGlbScenePrimitives(input);
    replaceScenePrimitiveGeometry(read.entries[0]!, seamRawMesh(), weldedTransferredAttributes());
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = doc.getRoot().listMeshes()[0]!.listPrimitives()[0]!;

    expect(outPrimitive.getAttribute('POSITION')?.getCount()).toBe(4);
    expect(outPrimitive.getAttribute('NORMAL')?.getCount()).toBe(4);
    expect(outPrimitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(4);
    expect(outPrimitive.getAttribute('TEXCOORD_1')?.getCount()).toBe(4);
    expect(outPrimitive.getIndices()?.getCount()).toBe(6);
  });

  it('derives preserved-material tangents when the output material references a normal map', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-normal-map-tangents-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeFullMaterialSquareGlb(input);

    const read = await readGlbScenePrimitives(input);
    replaceScenePrimitiveGeometry(read.entries[0]!, seamRawMesh(), weldedTransferredAttributes());
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = firstTrianglePrimitive(doc);
    const positionCount = outPrimitive.getAttribute('POSITION')!.getCount();
    const tangents = outPrimitive.getAttribute('TANGENT');
    expect(outPrimitive.getMaterial()?.getNormalTexture()).toBeTruthy();
    expect(tangents?.getCount()).toBe(positionCount);
    expect(tangents?.getType()).toBe(Accessor.Type.VEC4);
  });

  it('derives preserved-material tangents for normal texture references without decoded image payloads', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-normal-map-metadata-tangents-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeFullMaterialSquareGlb(input);

    const read = await readGlbScenePrimitives(input, { mode: 'geometry-with-texture-metadata' });
    const normalSlot = read.entries[0]!.sourceMaterial.textureSlots.find((slot) => slot.slot === 'normal');
    expect(normalSlot?.hasImage).toBe(true);
    expect(normalSlot?.image).toBeUndefined();
    replaceScenePrimitiveGeometry(read.entries[0]!, seamRawMesh(), weldedTransferredAttributes());
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = firstTrianglePrimitive(doc);
    const positionCount = outPrimitive.getAttribute('POSITION')!.getCount();
    const tangents = outPrimitive.getAttribute('TANGENT');
    expect(outPrimitive.getMaterial()?.getNormalTexture()).toBeTruthy();
    expect(tangents?.getCount()).toBe(positionCount);
    expect(tangents?.getType()).toBe(Accessor.Type.VEC4);
  });

  it('does not derive preserved-material tangents for material-less output without source tangent provenance', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-no-material-no-tangents-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeMateriallessUv0SquareGlb(input);

    const read = await readGlbScenePrimitives(input);
    replaceScenePrimitiveGeometry(read.entries[0]!, seamRawMesh(), {
      vertices: weldedTransferredAttributes().vertices.map((vertex) => ({
        uvSets: vertex.uvSets.filter((uvSet) => uvSet.texCoord === 0),
        ...(vertex.normal ? { normal: vertex.normal } : {}),
      })),
    });
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = firstTrianglePrimitive(doc);
    expect(outPrimitive.getMaterial()).toBeNull();
    expect(outPrimitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(4);
    expect(outPrimitive.getAttribute('TANGENT')).toBeNull();
  });

  it('derives preserved-material tangents from source tangent provenance without a normal map', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-source-tangent-provenance-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    const read = await readGlbScenePrimitives(input);
    const attributes = weldedTransferredAttributes();
    attributes.hasSourceTangents = true;
    replaceScenePrimitiveGeometry(read.entries[1]!, seamRawMesh(), attributes);
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = doc.getRoot()
      .listMeshes()
      .flatMap((mesh) => mesh.listPrimitives())
      .find((primitive) => primitive.getMaterial()?.getName() === 'plain')!;
    const positionCount = outPrimitive.getAttribute('POSITION')!.getCount();
    const tangents = outPrimitive.getAttribute('TANGENT');
    expect(outPrimitive.getMaterial()?.getNormalTexture()).toBeNull();
    expect(tangents?.getCount()).toBe(positionCount);
    expect(tangents?.getType()).toBe(Accessor.Type.VEC4);
  });

  it('omits preserved-material tangents when TEXCOORD_0 is unavailable', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-uv1-only-no-tangents-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeNormalMapUv1OnlySquareGlb(input);

    const read = await readGlbScenePrimitives(input);
    const attributes: TransferredMeshAttributes = {
      vertices: weldedTransferredAttributes().vertices.map((vertex) => ({
        uvSets: vertex.uvSets.filter((uvSet) => uvSet.texCoord === 1),
        ...(vertex.normal ? { normal: vertex.normal } : {}),
      })),
      hasSourceTangents: true,
    };
    replaceScenePrimitiveGeometry(read.entries[0]!, seamRawMesh(), attributes);
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const outPrimitive = firstTrianglePrimitive(doc);
    expect(outPrimitive.getAttribute('TEXCOORD_1')?.getCount()).toBe(4);
    expect(outPrimitive.getAttribute('TEXCOORD_0')).toBeNull();
    expect(outPrimitive.getAttribute('TANGENT')).toBeNull();
  });

  it('throws instead of silently dropping material UV sets missing from transferred faces', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-required-uv-missing-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeFullMaterialSquareGlb(input);

    const read = await readGlbScenePrimitives(input);
    const partialAttributes = weldedTransferredAttributes();
    partialAttributes.vertices[3] = {
      uvSets: [{ texCoord: 0, uv: new Vector2(0, 1) }],
      normal: new Vector3(0, -1, 0),
    };

    expect(() => replaceScenePrimitiveGeometry(read.entries[0]!, seamRawMesh(), partialAttributes))
      .toThrow('Missing transferred TEXCOORD_1 coordinates for output vertex 3.');
  });

  it('throws when a material-required UV set is missing from every transferred face', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-required-uv-all-missing-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeFullMaterialSquareGlb(input);

    const read = await readGlbScenePrimitives(input);
    const partialAttributes: TransferredMeshAttributes = {
      vertices: weldedTransferredAttributes().vertices.map((vertex) => ({
        uvSets: vertex.uvSets.filter((uvSet) => uvSet.texCoord === 0),
        ...(vertex.normal ? { normal: vertex.normal } : {}),
      })),
    };

    expect(() => replaceScenePrimitiveGeometry(read.entries[0]!, seamRawMesh(), partialAttributes))
      .toThrow('Missing transferred TEXCOORD_1 coordinates for output vertex 0.');
  });

  it('replaces each processed primitive without collapsing scene hierarchy or unsupported primitives', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-write-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    const read = await readGlbScenePrimitives(input);
    replaceScenePrimitiveGeometry(read.entries[1]!, replacementRawMesh(10));
    replaceScenePrimitiveTextured(read.entries[0]!, {
      outputRawMesh: replacementRawMesh(20),
      atlas: replacementAtlas(),
      image: { width: 4, height: 4, data: new Uint8Array(4 * 4 * 4).fill(255) },
    });
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const sceneChildren = doc.getRoot().getDefaultScene()!.listChildren();
    expect(sceneChildren.map((node) => node.getName())).toEqual(['textured-node', 'mixed-node']);
    expect(sceneChildren[0]!.getTranslation()).toEqual([5, 0, 0]);
    expect(sceneChildren[1]!.getTranslation()).toEqual([0, 7, 0]);

    const meshes = doc.getRoot().listMeshes();
    expect(meshes.map((mesh) => mesh.getName())).toEqual(['textured-mesh', 'mixed-mesh']);
    expect(meshes[0]!.listPrimitives()).toHaveLength(1);
    expect(meshes[1]!.listPrimitives()).toHaveLength(2);

    const bakedPrimitive = meshes[0]!.listPrimitives()[0]!;
    expect(bakedPrimitive.getAttribute('POSITION')?.getCount()).toBe(3);
    expect(bakedPrimitive.getAttribute('NORMAL')?.getCount()).toBe(3);
    expect(bakedPrimitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(3);
    expect(bakedPrimitive.getAttribute('TANGENT')).toBeNull();
    expect(bakedPrimitive.getMaterial()?.getBaseColorTexture()?.getImage()).toBeTruthy();
    expect(bakedPrimitive.getMaterial()?.getMetallicFactor()).toBe(0);
    expect(bakedPrimitive.getMaterial()?.getRoughnessFactor()).toBeCloseTo(0.42);

    const geometryPrimitive = meshes[1]!.listPrimitives()[0]!;
    expect(geometryPrimitive.getAttribute('POSITION')?.getCount()).toBe(3);
    expect(geometryPrimitive.getAttribute('NORMAL')?.getCount()).toBe(3);
    expect(geometryPrimitive.getAttribute('TEXCOORD_0')).toBeNull();
    expect(geometryPrimitive.getMaterial()?.getName()).toBe('plain');

    const unchangedLine = meshes[1]!.listPrimitives()[1]!;
    expect(unchangedLine.getMode()).toBe(Primitive.Mode.LINES);
    expect(unchangedLine.getAttribute('POSITION')?.getCount()).toBe(2);
  });

  it('prunes detached source meshes, materials, and textures after grouped replacement', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-prune-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    const read = await readGlbScenePrimitives(input);
    const group = groupScenePrimitives(read.entries, 'material-parent')
      .find((candidate) => candidate.sourceMaterial.name === 'textured');
    expect(group).toBeDefined();

    replaceScenePrimitiveGroupTextured(group!, {
      outputRawMesh: replacementRawMesh(20),
      atlas: replacementAtlas(),
      image: { width: 4, height: 4, data: new Uint8Array(4 * 4 * 4).fill(255) },
    });
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    expect(doc.getRoot().listMeshes().map((mesh) => mesh.listPrimitives().length)).not.toContain(0);
    expect(doc.getRoot().listMaterials()).toEqual(Array.from(usedMaterials(doc)));
    expect(doc.getRoot().listTextures()).toEqual(Array.from(usedTextures(doc)));
  });

  it('writes baked standard material textures on textured primitive replacement', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-baked-extra-textures-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeFullMaterialGlb(input);

    const read = await readGlbScenePrimitives(input);
    replaceScenePrimitiveTextured(read.entries[0]!, {
      outputRawMesh: read.entries[0]!.rawMesh,
      atlas: replacementAtlas(),
      image: { width: 4, height: 4, data: new Uint8Array(4 * 4 * 4).fill(255) },
      additionalTextures: [
        { slot: 'normal', image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) } },
        { slot: 'metallicRoughness', image: { width: 1, height: 1, data: new Uint8Array([0, 127, 255, 255]) } },
        { slot: 'occlusion', image: { width: 1, height: 1, data: new Uint8Array([200, 0, 0, 255]) } },
        { slot: 'emissive', image: { width: 1, height: 1, data: new Uint8Array([0, 0, 255, 255]) } },
      ],
    });
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const primitive = firstTrianglePrimitive(doc);
    const material = primitive.getMaterial()!;
    const positionCount = primitive.getAttribute('POSITION')!.getCount();
    expect(material.getBaseColorTexture()?.getImage()).toBeTruthy();
    expect(material.getNormalTexture()?.getImage()).toBeTruthy();
    expect(material.getMetallicRoughnessTexture()?.getImage()).toBeTruthy();
    expect(material.getOcclusionTexture()?.getImage()).toBeTruthy();
    expect(material.getEmissiveTexture()?.getImage()).toBeTruthy();
    expect(material.getNormalScale()).toBeCloseTo(0.75);
    expect(material.getOcclusionStrength()).toBeCloseTo(0.6);
    expect(material.getEmissiveFactor()).toEqual([0.1, 0.2, 0.3]);
    expect(material.getMetallicFactor()).toBeCloseTo(0.25);
    expect(material.getRoughnessFactor()).toBeCloseTo(0.5);
    for (const info of [
      material.getNormalTextureInfo(),
      material.getMetallicRoughnessTextureInfo(),
      material.getOcclusionTextureInfo(),
      material.getEmissiveTextureInfo(),
    ]) {
      expect(info?.getTexCoord()).toBe(0);
      expect(info?.getWrapS()).toBe(TextureInfo.WrapMode.CLAMP_TO_EDGE);
      expect(info?.getWrapT()).toBe(TextureInfo.WrapMode.CLAMP_TO_EDGE);
      expect(info?.getMagFilter()).toBe(TextureInfo.MagFilter.LINEAR);
      expect(info?.getMinFilter()).toBe(TextureInfo.MinFilter.LINEAR);
    }
    expect(primitive.getAttribute('TANGENT')?.getCount()).toBe(positionCount);
  });

  it('replaces original primitive entries directly while preserving node transforms', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-direct-entry-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    const read = await readGlbScenePrimitives(input);
    for (const entry of read.entries) {
      replaceScenePrimitiveGeometry(entry, entry.rawMesh, transferredAttributesFromFaceAttributes(entry.rawMesh, entry.faceAttributes));
    }
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const sceneChildren = doc.getRoot().getDefaultScene()!.listChildren();
    expect(sceneChildren.map((node) => node.getName())).toEqual(['textured-node', 'mixed-node']);
    expect(sceneChildren[0]!.getTranslation()).toEqual([5, 0, 0]);
    expect(sceneChildren[1]!.getTranslation()).toEqual([0, 7, 0]);

    const meshes = doc.getRoot().listMeshes();
    expect(meshes.map((mesh) => mesh.getName())).toEqual(['textured-mesh', 'mixed-mesh']);
    expect(meshes[0]!.listPrimitives()).toHaveLength(1);
    expect(meshes[1]!.listPrimitives()).toHaveLength(2);
    expect(meshes[0]!.listPrimitives()[0]!.getAttribute('TEXCOORD_0')?.getCount()).toBe(3);
    expect(meshes[1]!.listPrimitives()[0]!.getAttribute('POSITION')?.getCount()).toBe(3);
    expect(meshes[1]!.listPrimitives()[1]!.getMode()).toBe(Primitive.Mode.LINES);
  });

  it('groups same-material sibling primitives under their shared parent and replaces them once', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-group-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeSplitMaterialParentGlb(input);

    const read = await readGlbScenePrimitives(input);
    const groups = groupScenePrimitives(read.entries, 'material-parent');

    expect(groups).toHaveLength(1);
    expect(groups[0]!.entries).toHaveLength(2);
    expect(groups[0]!.rawMesh.faces).toHaveLength(2);
    expect(groups[0]!.rawMesh.positions).toHaveLength(5);
    expect(groups[0]!.rawMesh.positions.some((position) => Math.abs(position.x - 2) < 1e-6)).toBe(true);

    replaceScenePrimitiveGroupGeometry(groups[0]!, groups[0]!.rawMesh);
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const parent = doc.getRoot().listNodes().find((node) => node.getName() === 'shared-parent')!;
    const children = parent.listChildren();
    expect(children).toHaveLength(1);
    expect(children[0]!.getMesh()?.listPrimitives()).toHaveLength(1);

    const primitive = children[0]!.getMesh()!.listPrimitives()[0]!;
    expect(primitive.getAttribute('POSITION')?.getCount()).toBe(5);
    expect(primitive.getIndices()?.getArray()).toBeInstanceOf(Uint16Array);
    expect(primitive.getMaterial()?.getName()).toBe('shared');
  });

  it('separates same-material sibling primitives by COLOR_0 schema while grouping by material parent', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-mixed-color-group-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeMixedColorMaterialParentGlb(input);

    const read = await readGlbScenePrimitives(input, { mode: 'geometry-with-texture-metadata' });
    const groups = groupScenePrimitives(read.entries, 'material-parent');
    const coloredGroup = groups.find((group) => group.faceAttributes.some((attributes) => attributes.colorCorners));
    const plainGroup = groups.find((group) => group.faceAttributes.every((attributes) => !attributes.colorCorners));

    expect(groups).toHaveLength(2);
    expect(coloredGroup).toBeDefined();
    expect(plainGroup).toBeDefined();
    expect(coloredGroup!.entries.map((entry) => entry.nodeName)).toEqual(['colored-node']);
    expect(plainGroup!.entries.map((entry) => entry.nodeName)).toEqual(['plain-node']);

    const processable = toProcessablePrimitiveEntry({
      id: coloredGroup!.id,
      rawMesh: coloredGroup!.rawMesh,
      texturedRawMesh: {
        rawMesh: coloredGroup!.rawMesh,
        faceAttributes: coloredGroup!.faceAttributes,
        materials: [coloredGroup!.sourceMaterial],
      },
    });
    const geometryResult = processPrimitiveGeometries([processable], {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'material-parent',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: true,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    const processed = geometryResult.entries[0]!;

    replaceScenePrimitiveGroupGeometry(coloredGroup!, processed.geometry.raw.rawMesh, processed.transferredAttributes);
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const coloredPrimitive = doc.getRoot().listMeshes()
      .flatMap((mesh) => mesh.listPrimitives())
      .find((primitive) => primitive.getAttribute('COLOR_0'));
    const colorAccessor = coloredPrimitive?.getAttribute('COLOR_0');

    expect(colorAccessor).toBeDefined();
    expect(colorAccessor?.getType()).toBe(Accessor.Type.VEC3);
    expect(colorAccessor?.getCount()).toBe(coloredPrimitive?.getAttribute('POSITION')?.getCount());
    expect(Array.from(colorAccessor!.getArray()!).some((channel) => channel > 0)).toBe(true);
  });

  it('welds duplicate boundary vertices while grouping same-material sibling primitives', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-weld-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeSplitSharedBoundaryGlb(input);

    const read = await readGlbScenePrimitives(input);
    const groups = groupScenePrimitives(read.entries, 'material-parent');
    const mesh = SimplificationMesh.fromRawMesh(groups[0]!.rawMesh);
    const components = computeConnectedComponents(mesh);

    expect(groups).toHaveLength(1);
    expect(groups[0]!.rawMesh.faces).toHaveLength(2);
    expect(groups[0]!.rawMesh.positions).toHaveLength(4);
    expect(components.count).toBe(1);
  });

  it.each(['material-parent', 'material', 'none'] as const)(
    'uses weldVertices to control raw mesh construction for %s grouping',
    async (groupingMode) => {
      const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-seam-weld-'));
      tempDirs.push(dir);
      const input = join(dir, 'input.glb');
      await writeDuplicatePositionSeamGlb(input);

      const read = await readGlbScenePrimitives(input);
      const sourceVertexCount = read.entries[0]!.rawMesh.positions.length;
      const welded = groupScenePrimitives(read.entries, groupingMode, { weldVertices: true })[0]!;
      const unwelded = groupScenePrimitives(read.entries, groupingMode, { weldVertices: false })[0]!;

      expect(sourceVertexCount).toBe(4);
      expect(welded.rawMesh.positions.length).toBeLessThan(unwelded.rawMesh.positions.length);
      expect(unwelded.rawMesh.positions).toHaveLength(sourceVertexCount);
      expect(welded.faceAttributes).toHaveLength(read.entries[0]!.faceAttributes.length);
      expect(unwelded.faceAttributes).toHaveLength(read.entries[0]!.faceAttributes.length);
    },
  );

  it('keeps none-grouped source primitives separate without welding or parent transforms', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-none-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writeSplitMaterialParentGlb(input);

    const read = await readGlbScenePrimitives(input);
    const groups = groupScenePrimitives(read.entries, 'none');

    expect(groups).toHaveLength(2);
    for (const [index, group] of groups.entries()) {
      expect(group.entries).toEqual([read.entries[index]]);
      expect(group.parentNode?.getName()).toBe('shared-parent');
      expect(group.rawMesh.faces).toEqual([[0, 1, 2]]);
      expect(group.rawMesh.positions).toHaveLength(3);
      expect(group.rawMesh.positions.map((position) => position.x)).toEqual([0, 1, 0]);
      expect(group.faceAttributes).toHaveLength(1);
      expect(group.faceAttributes[0]).not.toBe(read.entries[index]!.faceAttributes[0]);
    }
  });

  it('transforms grouped authored normals with the node normal matrix while none grouping stays primitive-local', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-normal-transform-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const transform = new Matrix4()
      .makeRotationZ(Math.PI / 5)
      .scale(new Vector3(2, 3, 0.5));
    await writeTransformedAuthoredNormalsGlb(input, transform);

    const read = await readGlbScenePrimitives(input);
    const materialGroups = groupScenePrimitives(read.entries, 'material');
    const noneGroups = groupScenePrimitives(read.entries, 'none');
    const normalMatrix = new Matrix3().getNormalMatrix(transform);

    expect(materialGroups).toHaveLength(1);
    expect(noneGroups).toHaveLength(1);
    for (let corner = 0; corner < 3; corner += 1) {
      const expectedGrouped = AUTHORED_NORMALS[corner]!.clone().applyNormalMatrix(normalMatrix).normalize();
      expectVectorClose(materialGroups[0]!.faceAttributes[0]!.normalCorners?.[corner], expectedGrouped);
      expectVectorClose(noneGroups[0]!.faceAttributes[0]!.normalCorners?.[corner], AUTHORED_NORMALS[corner]!);
    }
  });

  it('preserves authored normals through none-grouped geometry output attribute transfer', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-normal-transfer-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTransformedAuthoredNormalsGlb(input, new Matrix4());

    const read = await readGlbScenePrimitives(input, { mode: 'geometry-with-texture-metadata' });
    const group = groupScenePrimitives(read.entries, 'none')[0]!;
    const processable = toProcessablePrimitiveEntry({
      id: group.id,
      rawMesh: group.rawMesh,
      texturedRawMesh: {
        rawMesh: group.rawMesh,
        faceAttributes: group.faceAttributes,
        materials: [group.sourceMaterial],
      },
    });
    const geometryResult = processPrimitiveGeometries([processable], {
      target: { kind: 'ratio', ratio: 1 },
      primitiveGrouping: 'none',
      virtualEdges: { mode: 'manual-global-radius', radius: 0 },
      weldVertices: true,
      recomputeNormals: false,
      transferTextures: false,
      textureSize: 32,
      texturePadding: 1,
      textureFilter: 'linear',
    });
    const processed = geometryResult.entries[0]!;

    expect(processed.transferredAttributes?.vertices).toHaveLength(3);
    replaceScenePrimitiveGroupGeometry(group, processed.geometry.raw.rawMesh, processed.transferredAttributes);
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const normals = firstTrianglePrimitive(doc).getAttribute('NORMAL')?.getArray();
    expect(normals).toBeInstanceOf(Float32Array);
    for (let index = 0; index < AUTHORED_NORMALS.length; index += 1) {
      const base = index * 3;
      expectVectorClose(new Vector3(
        (normals as Float32Array)[base]!,
        (normals as Float32Array)[base + 1]!,
        (normals as Float32Array)[base + 2]!,
      ), AUTHORED_NORMALS[index]!);
    }
  });

  it('transforms grouped authored tangents as directions under non-uniform scale', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-public-tangent-transform-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    await writePublicNormalMapGlb(input, { authoredTangents: true, negativeHandedness: true });

    const read = await readGlbScenePrimitives(input);
    const entry = read.entries[0]!;
    const original = entry.faceAttributes.find((attributes) => attributes.tangentCorners?.[0])!.tangentCorners![0]!;

    entry.node.setMatrix(new Matrix4().makeScale(2, 1, 0.5).toArray());
    const groups = groupScenePrimitives([entry], 'material-parent');
    const expected = new Vector3(original.x, original.y, original.z).transformDirection(new Matrix4().makeScale(2, 1, 0.5));

    expect(groups[0]!.faceAttributes[0]!.tangentCorners?.[0].x).toBeCloseTo(expected.x, 6);
    expect(groups[0]!.faceAttributes[0]!.tangentCorners?.[0].y).toBeCloseTo(expected.y, 6);
    expect(groups[0]!.faceAttributes[0]!.tangentCorners?.[0].z).toBeCloseTo(expected.z, 6);
    expect(groups[0]!.faceAttributes[0]!.tangentCorners?.[0].w).toBeCloseTo(original.w);
  });

  it('none-grouped group geometry replacement preserves original node transforms', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-none-group-replace-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    const read = await readGlbScenePrimitives(input);
    const groups = groupScenePrimitives(read.entries, 'none');
    for (const group of groups) {
      replaceScenePrimitiveGroupGeometry(group, group.rawMesh, transferredAttributesFromFaceAttributes(group.rawMesh, group.faceAttributes));
    }
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const sceneChildren = doc.getRoot().getDefaultScene()!.listChildren();
    expect(sceneChildren.map((node) => node.getName())).toEqual(['textured-node', 'mixed-node']);
    expect(sceneChildren[0]!.getTranslation()).toEqual([5, 0, 0]);
    expect(sceneChildren[1]!.getTranslation()).toEqual([0, 7, 0]);

    const meshes = doc.getRoot().listMeshes();
    expect(meshes.map((mesh) => mesh.getName())).toEqual(['textured-mesh', 'mixed-mesh']);
    expect(meshes[0]!.listPrimitives()).toHaveLength(1);
    expect(meshes[1]!.listPrimitives()).toHaveLength(2);
    expect(meshes[1]!.listPrimitives()[1]!.getMode()).toBe(Primitive.Mode.LINES);
  });

  it('none-grouped group textured replacement preserves original node transforms', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-none-group-textured-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    const read = await readGlbScenePrimitives(input);
    const groups = groupScenePrimitives(read.entries, 'none');
    replaceScenePrimitiveGroupTextured(groups[0]!, {
      outputRawMesh: groups[0]!.rawMesh,
      atlas: replacementAtlas(),
      image: { width: 4, height: 4, data: new Uint8Array(4 * 4 * 4).fill(255) },
    });
    await writeScenePrimitiveDocument(read.document, output);

    const doc = await new NodeIO().read(output);
    const sceneChildren = doc.getRoot().getDefaultScene()!.listChildren();
    expect(sceneChildren.map((node) => node.getName())).toEqual(['textured-node', 'mixed-node']);
    expect(sceneChildren[0]!.getTranslation()).toEqual([5, 0, 0]);
    expect(sceneChildren[1]!.getTranslation()).toEqual([0, 7, 0]);

    const meshes = doc.getRoot().listMeshes();
    expect(meshes.map((mesh) => mesh.getName())).toEqual(['textured-mesh', 'mixed-mesh']);
    expect(meshes[0]!.listPrimitives()[0]!.getMaterial()?.getBaseColorTexture()?.getImage()).toBeTruthy();
    expect(meshes[0]!.listPrimitives()[0]!.getAttribute('TEXCOORD_0')?.getCount()).toBe(3);
    expect(meshes[1]!.listPrimitives()).toHaveLength(2);
  });

  it('CLI none grouping replaces transformed primitives in place', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-cli-none-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    execFileSync(process.execPath, [
      '--import',
      'tsx',
      'src/local/main.ts',
      '--input',
      input,
      '--output',
      output,
      '--target-faces',
      '2',
      '--primitive-grouping',
      'none',
    ], { encoding: 'utf8' });

    const doc = await new NodeIO().read(output);
    const sceneChildren = doc.getRoot().getDefaultScene()!.listChildren();
    expect(sceneChildren.map((node) => node.getName())).toEqual(['textured-node', 'mixed-node']);
    expect(sceneChildren[0]!.getTranslation()).toEqual([5, 0, 0]);
    expect(sceneChildren[1]!.getTranslation()).toEqual([0, 7, 0]);

    const meshes = doc.getRoot().listMeshes();
    expect(meshes.map((mesh) => mesh.getName())).toEqual(['textured-mesh', 'mixed-mesh']);
    expect(meshes[0]!.listPrimitives()).toHaveLength(1);
    expect(meshes[1]!.listPrimitives()).toHaveLength(2);
    expect(meshes[0]!.listPrimitives()[0]!.getAttribute('POSITION')?.getCount()).toBe(3);
    expect(meshes[1]!.listPrimitives()[0]!.getAttribute('POSITION')?.getCount()).toBe(3);
    expect(meshes[1]!.listPrimitives()[1]!.getMode()).toBe(Primitive.Mode.LINES);
  });

  it('CLI none grouping preserves material textures with attribute transfer', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-cli-none-attribute-transfer-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeFullMaterialSquareGlb(input);

    execFileSync(process.execPath, [
      '--import',
      'tsx',
      'src/local/main.ts',
      '--input',
      input,
      '--output',
      output,
      '--target-faces',
      '1',
      '--primitive-grouping',
      'none',
    ], { encoding: 'utf8' });

    const doc = await new NodeIO().read(output);
    const outPrimitive = firstTrianglePrimitive(doc);
    const positionCount = outPrimitive.getAttribute('POSITION')!.getCount();
    expect(outPrimitive.getMaterial()?.getBaseColorTexture()).toBeTruthy();
    expect(outPrimitive.getMaterial()?.getNormalTexture()).toBeTruthy();
    expect(outPrimitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(positionCount);
    expect(outPrimitive.getAttribute('TEXCOORD_1')?.getCount()).toBe(positionCount);
    expect(outPrimitive.getIndices()?.getCount()).toBe(3);
  });

  it('CLI none grouping bakes textured primitives in place', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mesh-scene-cli-none-bake-'));
    tempDirs.push(dir);
    const input = join(dir, 'input.glb');
    const output = join(dir, 'output.glb');
    await writeTwoNodeGlb(input);

    execFileSync(process.execPath, [
      '--import',
      'tsx',
      'src/local/main.ts',
      '--input',
      input,
      '--output',
      output,
      '--target-faces',
      '2',
      '--primitive-grouping',
      'none',
      '--transfer-textures',
      '--texture-size',
      '16',
      '--texture-padding',
      '1',
    ], { encoding: 'utf8' });

    const doc = await new NodeIO().read(output);
    const sceneChildren = doc.getRoot().getDefaultScene()!.listChildren();
    expect(sceneChildren.map((node) => node.getName())).toEqual(['textured-node', 'mixed-node']);
    expect(sceneChildren[0]!.getTranslation()).toEqual([5, 0, 0]);
    expect(sceneChildren[1]!.getTranslation()).toEqual([0, 7, 0]);

    const meshes = doc.getRoot().listMeshes();
    expect(meshes.map((mesh) => mesh.getName())).toEqual(['textured-mesh', 'mixed-mesh']);
    expect(meshes[0]!.listPrimitives()).toHaveLength(1);
    expect(meshes[1]!.listPrimitives()).toHaveLength(2);

    const bakedPrimitive = meshes[0]!.listPrimitives()[0]!;
    expect(bakedPrimitive.getMaterial()?.getBaseColorTexture()?.getImage()).toBeTruthy();
    expect(bakedPrimitive.getAttribute('TEXCOORD_0')?.getCount()).toBe(3);
    expect(bakedPrimitive.getAttribute('NORMAL')?.getCount()).toBe(3);

    const factorOnlyPrimitive = meshes[1]!.listPrimitives()[0]!;
    expect(factorOnlyPrimitive.getMaterial()?.getName()).toBe('plain');
    expect(factorOnlyPrimitive.getMaterial()?.getBaseColorTexture()).toBeNull();
    expect(factorOnlyPrimitive.getMaterial()?.getBaseColorFactor()).toEqual([0.1, 0.2, 0.3, 1]);
    expect(factorOnlyPrimitive.getMaterial()?.getMetallicFactor()).toBeCloseTo(0.2);
    expect(factorOnlyPrimitive.getMaterial()?.getRoughnessFactor()).toBeCloseTo(0.6);
    expect(factorOnlyPrimitive.getAttribute('POSITION')?.getCount()).toBe(3);
    expect(factorOnlyPrimitive.getAttribute('NORMAL')?.getCount()).toBe(3);
    expect(meshes[1]!.listPrimitives()[1]!.getMode()).toBe(Primitive.Mode.LINES);
  });
});
