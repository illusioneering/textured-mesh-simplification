import { Accessor, Document, NodeIO, Primitive, TextureInfo } from '@gltf-transform/core';
import { encodePng } from '../../src/local/imageCodecs';

type VertexColorItemSize = 3 | 4;

interface NormalMapFixtureOptions {
  authoredTangents?: boolean;
  negativeHandedness?: boolean;
}

function png(rgba: readonly number[]): Uint8Array {
  return encodePng({ width: 1, height: 1, data: new Uint8Array(rgba) });
}

function addTexture(doc: Document, name: string, rgba: readonly number[]) {
  return doc.createTexture(name)
    .setMimeType('image/png')
    .setImage(png(rgba));
}

function createSquareAccessors(
  doc: Document,
  buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer(),
  flipUvHandedness = false,
) {
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
    .setArray(new Float32Array(flipUvHandedness
      ? [
          0, 1,
          1, 1,
          1, 0,
          0, 0,
        ]
      : [
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
  return { positions, normals, uv0, uv1, indices };
}

function createTriangleAccessors(doc: Document, buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer()) {
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
  const indices = doc.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(new Uint16Array([0, 1, 2]))
    .setBuffer(buffer);
  return { positions, normals, uvs, indices };
}

export async function writePublicTexturedMultiPrimitiveGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const base = addTexture(doc, 'public-base', [255, 0, 0, 255]);
  const normal = addTexture(doc, 'public-normal', [128, 128, 255, 255]);
  const mr = addTexture(doc, 'public-metallic-roughness', [0, 127, 255, 255]);
  const ao = addTexture(doc, 'public-occlusion', [200, 0, 0, 255]);
  const emissive = addTexture(doc, 'public-emissive', [0, 0, 255, 255]);
  const textured = doc.createMaterial('public-textured')
    .setBaseColorTexture(base)
    .setNormalTexture(normal)
    .setNormalScale(0.75)
    .setMetallicRoughnessTexture(mr)
    .setMetallicFactor(0.25)
    .setRoughnessFactor(0.5)
    .setOcclusionTexture(ao)
    .setOcclusionStrength(0.6)
    .setEmissiveTexture(emissive)
    .setEmissiveFactor([0.1, 0.2, 0.3])
    .setDoubleSided(true);
  textured.getBaseColorTextureInfo()!.setTexCoord(0).setMagFilter(TextureInfo.MagFilter.NEAREST!);
  textured.getNormalTextureInfo()!.setTexCoord(0);
  textured.getMetallicRoughnessTextureInfo()!.setTexCoord(0);
  textured.getOcclusionTextureInfo()!.setTexCoord(1);
  textured.getEmissiveTextureInfo()!.setTexCoord(0);

  const plain = doc.createMaterial('public-plain')
    .setBaseColorFactor([0.1, 0.2, 0.3, 1])
    .setMetallicFactor(0.2)
    .setRoughnessFactor(0.6);

  const sharedParent = doc.createNode('public-parent');
  const scene = doc.createScene('Scene').addChild(sharedParent);
  const square = createSquareAccessors(doc, buffer);
  const texturedPrimitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', square.positions)
    .setAttribute('NORMAL', square.normals)
    .setAttribute('TEXCOORD_0', square.uv0)
    .setAttribute('TEXCOORD_1', square.uv1)
    .setIndices(square.indices)
    .setMaterial(textured);
  sharedParent.addChild(doc.createNode('public-textured-node')
    .setMesh(doc.createMesh('public-textured-mesh').addPrimitive(texturedPrimitive)));

  const triangle = createTriangleAccessors(doc, buffer);
  const plainPrimitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', triangle.positions)
    .setAttribute('NORMAL', triangle.normals)
    .setIndices(triangle.indices)
    .setMaterial(plain);
  sharedParent.addChild(doc.createNode('public-plain-node')
    .setTranslation([2, 0, 0])
    .setMesh(doc.createMesh('public-plain-mesh').addPrimitive(plainPrimitive)));

  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

export async function writePublicNormalMapGlb(path: string, options: NormalMapFixtureOptions = {}): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const accessors = createSquareAccessors(doc, buffer, !options.authoredTangents);
  const normal = addTexture(doc, 'public-normal-map', [128, 128, 255, 255]);
  const material = doc.createMaterial(options.authoredTangents ? 'public-authored-tangent-normal-map' : 'public-tangentless-normal-map')
    .setNormalTexture(normal)
    .setNormalScale(0.7)
    .setMetallicFactor(0)
    .setRoughnessFactor(0.5);
  material.getNormalTextureInfo()!.setTexCoord(0);

  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', accessors.positions)
    .setAttribute('NORMAL', accessors.normals)
    .setAttribute('TEXCOORD_0', accessors.uv0)
    .setIndices(accessors.indices)
    .setMaterial(material);

  if (options.authoredTangents) {
    const handedness = options.negativeHandedness ? -1 : 1;
    primitive.setAttribute('TANGENT', doc.createAccessor('TANGENT')
      .setType(Accessor.Type.VEC4!)
      .setArray(new Float32Array([
        1, 0, 0, handedness,
        1, 0, 0, handedness,
        1, 0, 0, handedness,
        1, 0, 0, handedness,
      ]))
      .setBuffer(buffer));
  }

  const scene = doc.createScene('Scene').addChild(
    doc.createNode('public-normal-node').setMesh(doc.createMesh('public-normal-mesh').addPrimitive(primitive)),
  );
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

export async function writePublicVertexColorGlb(path: string, itemSize: VertexColorItemSize = 3): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const accessors = createTriangleAccessors(doc, buffer);
  const colorArray = itemSize === 4
    ? new Float32Array([1, 0, 0, 1, 0, 1, 0, 0.75, 0, 0, 1, 0.5])
    : new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const colors = doc.createAccessor('COLOR_0')
    .setType(itemSize === 4 ? Accessor.Type.VEC4! : Accessor.Type.VEC3!)
    .setArray(colorArray)
    .setBuffer(buffer);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', accessors.positions)
    .setAttribute('NORMAL', accessors.normals)
    .setAttribute('TEXCOORD_0', accessors.uvs)
    .setAttribute('COLOR_0', colors)
    .setIndices(accessors.indices)
    .setMaterial(doc.createMaterial('public-vertex-color').setMetallicFactor(0).setRoughnessFactor(0.5));
  const scene = doc.createScene('Scene').addChild(
    doc.createNode('public-color-node').setMesh(doc.createMesh('public-color-mesh').addPrimitive(primitive)),
  );
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

export async function writePublicTexturedVertexColorGlb(path: string): Promise<void> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const accessors = createTriangleAccessors(doc, buffer);
  const base = addTexture(doc, 'public-color-base', [255, 0, 0, 255]);
  const colors = doc.createAccessor('COLOR_0')
    .setType(Accessor.Type.VEC4!)
    .setArray(new Float32Array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]))
    .setBuffer(buffer);
  const material = doc.createMaterial('public-textured-color')
    .setBaseColorTexture(base)
    .setMetallicFactor(0)
    .setRoughnessFactor(0.5);
  material.getBaseColorTextureInfo()!.setTexCoord(0);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', accessors.positions)
    .setAttribute('NORMAL', accessors.normals)
    .setAttribute('TEXCOORD_0', accessors.uvs)
    .setAttribute('COLOR_0', colors)
    .setIndices(accessors.indices)
    .setMaterial(material);
  const scene = doc.createScene('Scene').addChild(
    doc.createNode('public-textured-color-node').setMesh(doc.createMesh('public-textured-color-mesh').addPrimitive(primitive)),
  );
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
}

export async function writePublicAtlasStressGlb(path: string, gridSize = 12): Promise<number> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  for (let y = 0; y <= gridSize; y += 1) {
    for (let x = 0; x <= gridSize; x += 1) {
      const z = ((x + y) % 2) * 0.02;
      positions.push(x / gridSize, y / gridSize, z);
      normals.push(0, 0, 1);
      uvs.push(x / gridSize, y / gridSize);
    }
  }
  const stride = gridSize + 1;
  for (let y = 0; y < gridSize; y += 1) {
    for (let x = 0; x < gridSize; x += 1) {
      const a = y * stride + x;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      indices.push(a, b, d, a, d, c);
    }
  }
  const material = doc.createMaterial('public-atlas-stress')
    .setBaseColorTexture(addTexture(doc, 'public-atlas-base', [255, 255, 255, 255]))
    .setMetallicFactor(0)
    .setRoughnessFactor(0.5);
  const primitive = doc.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', doc.createAccessor('POSITION').setType(Accessor.Type.VEC3!).setArray(new Float32Array(positions)).setBuffer(buffer))
    .setAttribute('NORMAL', doc.createAccessor('NORMAL').setType(Accessor.Type.VEC3!).setArray(new Float32Array(normals)).setBuffer(buffer))
    .setAttribute('TEXCOORD_0', doc.createAccessor('TEXCOORD_0').setType(Accessor.Type.VEC2!).setArray(new Float32Array(uvs)).setBuffer(buffer))
    .setIndices(doc.createAccessor('indices').setType(Accessor.Type.SCALAR!).setArray(new Uint32Array(indices)).setBuffer(buffer))
    .setMaterial(material);
  const scene = doc.createScene('Scene').addChild(
    doc.createNode('public-atlas-node').setMesh(doc.createMesh('public-atlas-mesh').addPrimitive(primitive)),
  );
  doc.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, doc);
  return indices.length / 3;
}
