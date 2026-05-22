import { describe, expect, it } from 'vitest';
import { BufferAttribute, BufferGeometry, ClampToEdgeWrapping, DataTexture, Float32BufferAttribute, Group, InterleavedBuffer, InterleavedBufferAttribute, LinearFilter, Material, Matrix3, Matrix4, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, MeshStandardMaterial, Object3D, RGBAFormat, SRGBColorSpace, Texture, Vector2, Vector3, Vector4 } from 'three';
import { decodeImage, encodePng } from '../../src/local/imageCodecs';
import type { TransferredMeshAttributes } from '../../src/simplification/attributes';
import { createInjectiveAtlas } from '../../src/texture/atlas';
import type { AtlasLayout } from '../../src/texture/types';
import { faceUvSet } from '../../src/texture/types';
import {
  createGeometryOutputScene,
  createPrimitiveOutputScene,
  createTexturedOutputScene,
  exportSceneToGlb,
  extractBrowserModelFromObject,
  extractBrowserPrimitiveGroupsFromObject,
  matchExternalGltfResource,
  parseGlbArrayBuffer,
  summarizeBrowserObject,
} from '../../src/web/browserGltfIo';
import { normalizeBrowserSceneMaterialsToCorePbr } from '../../src/web/corePbrMaterialNormalization';
import { cloneSceneForViewport } from '../../src/web/previewScene';
import type { TexturedOutputSceneOptions } from '../../src/web/browserGltfIo';
import { identitySceneTransformContext } from '../../src/web/sceneContext';

const AUTHORED_NORMALS = [
  new Vector3(1, 2, 3).normalize(),
  new Vector3(-2, 1, 0.5).normalize(),
  new Vector3(0.25, -1, 2).normalize(),
] as const;

function expectVectorClose(actual: Vector3 | undefined, expected: Vector3): void {
  expect(actual).toBeDefined();
  expect(actual!.x).toBeCloseTo(expected.x, 6);
  expect(actual!.y).toBeCloseTo(expected.y, 6);
  expect(actual!.z).toBeCloseTo(expected.z, 6);
}

function squareGeometry(): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
  ]), 3));
  geometry.setAttribute('normal', new BufferAttribute(new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]), 3));
  geometry.setAttribute('uv', new BufferAttribute(new Float32Array([
    0, 0,
    1, 0,
    1, 1,
    0, 1,
  ]), 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  geometry.addGroup(0, 3, 0);
  geometry.addGroup(3, 3, 1);
  return geometry;
}

function triangleGeometryWithAuthoredNormals(): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]), 3));
  geometry.setAttribute('normal', new BufferAttribute(new Float32Array(
    AUTHORED_NORMALS.flatMap((normal) => [normal.x, normal.y, normal.z]),
  ), 3));
  geometry.setIndex([0, 1, 2]);
  return geometry;
}

function squareGeometryWithUv1(): BufferGeometry {
  const geometry = squareGeometry();
  geometry.setAttribute('uv1', new BufferAttribute(new Float32Array([
    0.25, 0.25,
    0.75, 0.25,
    0.75, 0.75,
    0.25, 0.75,
  ]), 2));
  return geometry;
}

function duplicatePositionSeamRoot(): { root: Object3D; sourceVertexCount: number } {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 0, 0,
  ]), 3));
  geometry.setAttribute('normal', new BufferAttribute(new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 1, 0,
  ]), 3));
  geometry.setAttribute('uv', new BufferAttribute(new Float32Array([
    0, 0,
    1, 0,
    0, 1,
    0.5, 0.5,
  ]), 2));
  geometry.setIndex([0, 1, 2, 3, 2, 1]);
  const root = new Group();
  root.add(new Mesh(geometry, new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 })));
  return { root, sourceVertexCount: 4 };
}

function weldedTransferredAttributes(includeUv1 = true): TransferredMeshAttributes {
  return {
    vertices: [
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(0, 0) },
          ...(includeUv1 ? [{ texCoord: 1, uv: new Vector2(0.25, 0.25) }] : []),
        ],
        normal: new Vector3(1, 0, 0),
        color: new Vector4(1, 0, 0, 1),
      },
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(1, 0) },
          ...(includeUv1 ? [{ texCoord: 1, uv: new Vector2(0.75, 0.25) }] : []),
        ],
        normal: new Vector3(0, 1, 0),
        color: new Vector4(0, 1, 0, 0.75),
      },
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(1, 1) },
          ...(includeUv1 ? [{ texCoord: 1, uv: new Vector2(0.75, 0.75) }] : []),
        ],
        normal: new Vector3(0, 0, 1),
        color: new Vector4(0, 0, 1, 0.5),
      },
      {
        uvSets: [
          { texCoord: 0, uv: new Vector2(0, 1) },
          ...(includeUv1 ? [{ texCoord: 1, uv: new Vector2(0.25, 0.75) }] : []),
        ],
        normal: new Vector3(0, -1, 0),
        color: new Vector4(1, 1, 1, 0.25),
      },
    ],
    colorItemSize: 4,
  };
}

function weldedTransferredUv1OnlyAttributes(): TransferredMeshAttributes {
  return {
    hasSourceTangents: true,
    vertices: [
      { uvSets: [{ texCoord: 1, uv: new Vector2(0.25, 0.25) }] },
      { uvSets: [{ texCoord: 1, uv: new Vector2(0.75, 0.25) }] },
      { uvSets: [{ texCoord: 1, uv: new Vector2(0.75, 0.75) }] },
      { uvSets: [{ texCoord: 1, uv: new Vector2(0.25, 0.75) }] },
    ],
  };
}

function materialWithStandardTextureSlots(): MeshStandardMaterial {
  const base = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
  const normal = new DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1, RGBAFormat);
  const mr = new DataTexture(new Uint8Array([0, 128, 255, 255]), 1, 1, RGBAFormat);
  const ao = new DataTexture(new Uint8Array([200, 0, 0, 255]), 1, 1, RGBAFormat);
  const emissive = new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1, RGBAFormat);
  for (const texture of [base, normal, mr, ao, emissive]) texture.needsUpdate = true;
  normal.channel = 1;
  ao.channel = 1;
  const material = new MeshStandardMaterial({
    color: 0xffffff,
    map: base,
    normalMap: normal,
    metalnessMap: mr,
    roughnessMap: mr,
    aoMap: ao,
    emissiveMap: emissive,
    emissive: 0x112233,
    metalness: 0.25,
    roughness: 0.5,
  });
  material.normalScale.setScalar(0.75);
  material.aoMapIntensity = 0.6;
  return material;
}

function triangleGeometry(values: readonly number[]): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(values), 3));
  geometry.setIndex([0, 1, 2]);
  return geometry;
}

function vertexColorTriangleGeometry(itemSize: 3 | 4 = 3): BufferGeometry {
  const geometry = triangleGeometry([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  geometry.setAttribute('normal', new BufferAttribute(new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]), 3));
  geometry.setAttribute('uv', new BufferAttribute(new Float32Array([
    0, 0,
    1, 0,
    0, 1,
  ]), 2));
  geometry.setAttribute('color', new BufferAttribute(new Float32Array(itemSize === 4 ? [
    1, 0, 0, 0.25,
    0, 1, 0, 0.5,
    0, 0, 1, 0.75,
  ] : [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]), itemSize));
  return geometry;
}

function firstMesh(scene: Group | Object3D): Mesh {
  let found: Mesh | null = null;
  scene.traverse((object) => {
    if (!found && (object as Mesh).isMesh) found = object as Mesh;
  });
  if (!found) throw new Error('missing mesh');
  return found;
}

function allMeshes(scene: Group | Object3D): Mesh[] {
  const meshes: Mesh[] = [];
  scene.traverse((object) => {
    if ((object as Mesh).isMesh) meshes.push(object as Mesh);
  });
  return meshes;
}

function meshMaterialAt(mesh: Mesh, index: number): Material {
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  const material = materials[index];
  if (!material) throw new Error(`missing material ${index}`);
  return material;
}

function expectAllTangentHandedness(geometry: BufferGeometry, expected: 1 | -1): void {
  const tangent = geometry.getAttribute('tangent');
  expect(tangent).toBeDefined();
  expect(tangent!.itemSize).toBe(4);
  expect(tangent!.count).toBe(geometry.getAttribute('position').count);
  for (let index = 0; index < tangent!.count; index += 1) {
    expect(tangent!.getW(index)).toBe(expected);
  }
}

function minimalTexturedGltf(): ArrayBuffer {
  const binary = new Uint8Array(68);
  const view = new DataView(binary.buffer);
  [
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
  ].forEach((value, index) => view.setFloat32(index * 4, value, true));
  [0, 1, 2].forEach((value, index) => view.setUint16(36 + index * 2, value, true));
  [
    0, 0,
    1, 0,
    1, 1,
  ].forEach((value, index) => view.setFloat32(44 + index * 4, value, true));
  const gltf = {
    asset: { version: '2.0' },
    buffers: [{
      uri: `data:application/octet-stream;base64,${Buffer.from(binary).toString('base64')}`,
      byteLength: binary.byteLength,
    }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 6, target: 34963 },
      { buffer: 0, byteOffset: 44, byteLength: 24, target: 34962 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', max: [1, 1, 0], min: [0, 0, 0] },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
    ],
    images: [{ uri: 'textures/diffuse.png' }],
    textures: [{ source: 0 }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 0.5 } }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 2 }, indices: 1, material: 0 }] }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  return new TextEncoder().encode(JSON.stringify(gltf)).buffer;
}

function minimalNormalOnlyGltf(): ArrayBuffer {
  const binary = new Uint8Array(80);
  const view = new DataView(binary.buffer);
  [
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ].forEach((value, index) => view.setFloat32(index * 4, value, true));
  AUTHORED_NORMALS
    .flatMap((normal) => [normal.x, normal.y, normal.z])
    .forEach((value, index) => view.setFloat32(36 + index * 4, value, true));
  [0, 1, 2].forEach((value, index) => view.setUint16(72 + index * 2, value, true));
  const gltf = {
    asset: { version: '2.0' },
    buffers: [{
      uri: `data:application/octet-stream;base64,${Buffer.from(binary).toString('base64')}`,
      byteLength: binary.byteLength,
    }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 72, byteLength: 6, target: 34963 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', max: [1, 1, 0], min: [0, 0, 0] },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    materials: [{ name: 'normal-only' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2, material: 0 }] }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  return new TextEncoder().encode(JSON.stringify(gltf)).buffer;
}

function minimalClearcoatGltf(): ArrayBuffer {
  const binary = new Uint8Array(44);
  const view = new DataView(binary.buffer);
  [
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ].forEach((value, index) => view.setFloat32(index * 4, value, true));
  [0, 1, 2].forEach((value, index) => view.setUint16(36 + index * 2, value, true));
  const gltf = {
    asset: { version: '2.0' },
    extensionsUsed: ['KHR_materials_clearcoat'],
    buffers: [{
      uri: `data:application/octet-stream;base64,${Buffer.from(binary).toString('base64')}`,
      byteLength: binary.byteLength,
    }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 6, target: 34963 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', max: [1, 1, 0], min: [0, 0, 0] },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    materials: [{
      name: 'extended-clearcoat',
      pbrMetallicRoughness: {
        baseColorFactor: [0.2, 0.4, 0.6, 0.7],
        metallicFactor: 0.25,
        roughnessFactor: 0.5,
      },
      emissiveFactor: [0.1, 0.2, 0.3],
      alphaMode: 'BLEND',
      extensions: { KHR_materials_clearcoat: { clearcoatFactor: 1 } },
    }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  return new TextEncoder().encode(JSON.stringify(gltf)).buffer;
}

function minimalVertexColorClearcoatGltf(): ArrayBuffer {
  const binary = new Uint8Array(78);
  const view = new DataView(binary.buffer);
  [
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ].forEach((value, index) => view.setFloat32(index * 4, value, true));
  [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ].forEach((value, index) => view.setFloat32(36 + index * 4, value, true));
  [0, 1, 2].forEach((value, index) => view.setUint16(72 + index * 2, value, true));
  const gltf = {
    asset: { version: '2.0' },
    extensionsUsed: ['KHR_materials_clearcoat'],
    buffers: [{
      uri: `data:application/octet-stream;base64,${Buffer.from(binary).toString('base64')}`,
      byteLength: binary.byteLength,
    }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 72, byteLength: 6, target: 34963 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', max: [1, 1, 0], min: [0, 0, 0] },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3', max: [1, 1, 1], min: [0, 0, 0] },
      { bufferView: 2, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    materials: [{
      name: 'vertex-color-clearcoat',
      extensions: { KHR_materials_clearcoat: { clearcoatFactor: 1 } },
    }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 }, indices: 2, material: 0 }] }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  return new TextEncoder().encode(JSON.stringify(gltf)).buffer;
}

function minimalExternalBinGltf(): { gltf: ArrayBuffer; bin: ArrayBuffer } {
  const binary = new Uint8Array(44);
  const view = new DataView(binary.buffer);
  [
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ].forEach((value, index) => view.setFloat32(index * 4, value, true));
  [0, 1, 2].forEach((value, index) => view.setUint16(36 + index * 2, value, true));
  const gltf = {
    asset: { version: '2.0' },
    buffers: [{ uri: 'buffers/triangle.bin', byteLength: binary.byteLength }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 6, target: 34963 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', max: [1, 1, 0], min: [0, 0, 0] },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  return {
    gltf: new TextEncoder().encode(JSON.stringify(gltf)).buffer,
    bin: binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength) as ArrayBuffer,
  };
}

async function withBrowserImageLoading<T>(run: (pngBuffer: ArrayBuffer) => Promise<T>): Promise<T> {
  const testGlobal = globalThis as typeof globalThis & { self?: Window & typeof globalThis };
  const previousSelf = testGlobal.self;
  const previousCreateImageBitmap = globalThis.createImageBitmap;
  const previousProgressEvent = globalThis.ProgressEvent;
  testGlobal.self = globalThis as Window & typeof globalThis;
  globalThis.ProgressEvent = class TestProgressEvent extends Event {
    lengthComputable: boolean;
    loaded: number;
    total: number;

    constructor(type: string, eventInitDict: ProgressEventInit = {}) {
      super(type);
      this.lengthComputable = eventInitDict.lengthComputable ?? false;
      this.loaded = eventInitDict.loaded ?? 0;
      this.total = eventInitDict.total ?? 0;
    }
  } as typeof ProgressEvent;
  globalThis.createImageBitmap = (async (blob: Blob) => {
    const decoded = decodeImage(new Uint8Array(await blob.arrayBuffer()), blob.type || 'image/png');
    return { width: decoded.width, height: decoded.height, data: decoded.data } as unknown as ImageBitmap;
  }) as typeof createImageBitmap;
  const png = encodePng({ width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) });
  const pngBuffer = png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength) as ArrayBuffer;
  try {
    return await run(pngBuffer);
  } finally {
    if (previousSelf === undefined) {
      delete testGlobal.self;
    } else {
      testGlobal.self = previousSelf;
    }
    globalThis.createImageBitmap = previousCreateImageBitmap;
    globalThis.ProgressEvent = previousProgressEvent;
  }
}

async function withTestFileReader<T>(run: () => Promise<T>): Promise<T> {
  const testGlobal = globalThis as typeof globalThis & { FileReader?: typeof FileReader };
  const previousFileReader = testGlobal.FileReader;
  testGlobal.FileReader = class TestFileReader extends EventTarget {
    result: string | ArrayBuffer | null = null;
    error: DOMException | null = null;
    onloadend: ((this: FileReader, ev: ProgressEvent<FileReader>) => unknown) | null = null;

    private emitLoadEnd(): void {
      this.onloadend?.call(this as unknown as FileReader, {} as ProgressEvent<FileReader>);
    }

    readAsArrayBuffer(blob: Blob): void {
      void blob.arrayBuffer().then((buffer) => {
        this.result = buffer;
        this.emitLoadEnd();
      }).catch((error: unknown) => {
        this.error = error instanceof DOMException ? error : new DOMException(String(error));
        this.emitLoadEnd();
      });
    }
  } as unknown as typeof FileReader;
  try {
    return await run();
  } finally {
    if (previousFileReader === undefined) {
      delete testGlobal.FileReader;
    } else {
      testGlobal.FileReader = previousFileReader;
    }
  }
}

interface ExportedGlbJson {
  images?: unknown[];
  materials?: Array<{
    extensions?: Record<string, unknown>;
  }>;
  textures?: unknown[];
}

function decodeGlbJson(buffer: ArrayBuffer): ExportedGlbJson {
  const view = new DataView(buffer);
  expect(view.getUint32(0, true)).toBe(0x46546c67);
  const jsonChunkLength = view.getUint32(12, true);
  const jsonChunkType = view.getUint32(16, true);
  expect(jsonChunkType).toBe(0x4e4f534a);
  const bytes = new Uint8Array(buffer, 20, jsonChunkLength);
  return JSON.parse(new TextDecoder().decode(bytes).trim()) as ExportedGlbJson;
}

function twoIslandAtlas(): AtlasLayout {
  return {
    textureSize: 64,
    padding: 2,
    faceUvs: [
      [new Vector2(0.1, 0.1), new Vector2(0.4, 0.1), new Vector2(0.1, 0.4)],
      [new Vector2(0.6, 0.6), new Vector2(0.9, 0.6), new Vector2(0.6, 0.9)],
    ],
    facePixelTriangles: [
      [[6, 6], [26, 6], [6, 26]],
      [[38, 38], [58, 38], [38, 58]],
    ],
    islandCount: 2,
  };
}

function rawResultFor(rawMesh: { positions: Vector3[]; faces: [number, number, number][] }) {
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
      stoppedReason: 'target-reached' as const,
    },
  };
}

describe('browser glTF I/O helpers', () => {
  it('re-exports browser object summary scanning', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    const scene = new Group();
    scene.add(new Mesh(squareGeometry(), [
      new MeshStandardMaterial({ map: texture, metalness: 0, roughness: 0.5 }),
      new MeshStandardMaterial({ color: 0x00ff00, metalness: 0, roughness: 0.75 }),
    ]));

    const summary = await summarizeBrowserObject(scene);

    expect(summary.inputFaces).toBe(2);
    expect(summary.materials).toBe(2);
    expect(summary.materialsWithBaseColorImages).toBe(1);
  });

  it('matches uploaded external glTF resources by requested URI basename', () => {
    const resources = [
      { fileName: 'ignored.png', objectUrl: 'blob:ignored', kind: 'texture-image' as const },
      { fileName: 'diffuse.png', objectUrl: 'blob:diffuse', kind: 'texture-image' as const },
      { fileName: 'triangle.bin', objectUrl: 'blob:triangle', kind: 'binary-buffer' as const },
    ];

    expect(matchExternalGltfResource('textures/diffuse.png', resources)?.objectUrl).toBe('blob:diffuse');
    expect(matchExternalGltfResource('textures/diffuse.png?cache=1#main', resources)?.objectUrl).toBe('blob:diffuse');
    expect(matchExternalGltfResource('diffuse%20map.png', [{ fileName: 'diffuse map.png', objectUrl: 'blob:decoded', kind: 'texture-image' as const }])?.objectUrl).toBe('blob:decoded');
    expect(matchExternalGltfResource('buffers/triangle.bin', resources)?.objectUrl).toBe('blob:triangle');
    expect(matchExternalGltfResource('data:image/png;base64,aaaa', resources)).toBeUndefined();
    expect(matchExternalGltfResource('textures/missing.png', resources)).toBeUndefined();
  });

  it('parses external uploaded binary buffers through GLTFLoader URL resolution', async () => {
    const { gltf, bin } = minimalExternalBinGltf();
    const previousProgressEvent = globalThis.ProgressEvent;
    globalThis.ProgressEvent = class TestProgressEvent extends Event {
      lengthComputable: boolean;
      loaded: number;
      total: number;

      constructor(type: string, eventInitDict: ProgressEventInit = {}) {
        super(type);
        this.lengthComputable = eventInitDict.lengthComputable ?? false;
        this.loaded = eventInitDict.loaded ?? 0;
        this.total = eventInitDict.total ?? 0;
      }
    } as typeof ProgressEvent;
    try {
      const asset = await parseGlbArrayBuffer(gltf, {
        externalResourceFiles: [
          new File([bin], 'triangle.bin', { type: 'application/octet-stream' }),
        ],
      });
      const extraction = await asset.extractGroups({ groupingMode: 'material-parent', mode: 'geometry' });

      expect(asset.summary.inputFaces).toBe(1);
      expect(extraction.entries[0]!.rawMesh.positions).toHaveLength(3);
      expect(extraction.entries[0]!.rawMesh.faces).toEqual([[0, 1, 2]]);
      expect(asset.matchedExternalBinaryBufferFiles).toEqual(['triangle.bin']);
      expect(asset.matchedExternalTextureFiles).toEqual([]);
    } finally {
      globalThis.ProgressEvent = previousProgressEvent;
    }
  });

  it('loads browser glTF assets lazily without eager extraction state', async () => {
    const { gltf, bin } = minimalExternalBinGltf();
    const previousProgressEvent = globalThis.ProgressEvent;
    globalThis.ProgressEvent = class TestProgressEvent extends Event {
      lengthComputable: boolean;
      loaded: number;
      total: number;

      constructor(type: string, eventInitDict: ProgressEventInit = {}) {
        super(type);
        this.lengthComputable = eventInitDict.lengthComputable ?? false;
        this.loaded = eventInitDict.loaded ?? 0;
        this.total = eventInitDict.total ?? 0;
      }
    } as typeof ProgressEvent;
    try {
      const asset = await parseGlbArrayBuffer(gltf, {
        externalResourceFiles: [
          new File([bin], 'triangle.bin', { type: 'application/octet-stream' }),
        ],
      });

      expect('rawMesh' in asset).toBe(false);
      expect('textured' in asset).toBe(false);
      expect('primitiveEntries' in asset).toBe(false);
      expect('primitiveEntryGroups' in asset).toBe(false);
      expect(asset.summary.inputFaces).toBe(1);
    } finally {
      globalThis.ProgressEvent = previousProgressEvent;
    }
  });

  it('normalizes parsed browser materials before retaining the loaded scene', async () => {
    const previousProgressEvent = globalThis.ProgressEvent;
    globalThis.ProgressEvent = class TestProgressEvent extends Event {
      lengthComputable: boolean;
      loaded: number;
      total: number;

      constructor(type: string, eventInitDict: ProgressEventInit = {}) {
        super(type);
        this.lengthComputable = eventInitDict.lengthComputable ?? false;
        this.loaded = eventInitDict.loaded ?? 0;
        this.total = eventInitDict.total ?? 0;
      }
    } as typeof ProgressEvent;
    try {
      const asset = await parseGlbArrayBuffer(minimalClearcoatGltf());

      const material = firstMesh(asset.scene).material as MeshStandardMaterial;
      expect(material).toBeInstanceOf(MeshStandardMaterial);
      expect(material).not.toBeInstanceOf(MeshPhysicalMaterial);
      expect(material.name).toBe('extended-clearcoat');
      expect(material.color.r).toBeCloseTo(0.2);
      expect(material.color.g).toBeCloseTo(0.4);
      expect(material.color.b).toBeCloseTo(0.6);
      expect(material.opacity).toBeCloseTo(0.7);
      expect(material.transparent).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.metalness).toBeCloseTo(0.25);
      expect(material.roughness).toBeCloseTo(0.5);
      expect(material.emissive.r).toBeCloseTo(0.1);
      expect(material.emissive.g).toBeCloseTo(0.2);
      expect(material.emissive.b).toBeCloseTo(0.3);
      expect((material as MeshPhysicalMaterial).clearcoat).toBeUndefined();
      expect('clearcoatMap' in material).toBe(false);
      expect(asset.summary.materialsWithBaseColorImages).toBe(0);

      const preview = cloneSceneForViewport(asset.scene);
      const previewMaterial = firstMesh(preview).material as MeshStandardMaterial;
      expect('clearcoat' in previewMaterial).toBe(false);
      expect('clearcoatMap' in previewMaterial).toBe(false);

      const extraction = await asset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });
      const entry = extraction.entries[0];
      if (!entry?.texturedRawMesh) throw new Error('missing textured extraction entry');
      const materialMetadata = entry.texturedRawMesh.materials[0];
      if (!materialMetadata) throw new Error('missing material metadata');
      expect(materialMetadata.name).toBe('extended-clearcoat');
      expect(materialMetadata.metallicFactor).toBeCloseTo(0.25);
      expect(materialMetadata.roughnessFactor).toBeCloseTo(0.5);
      expect(materialMetadata.alphaMode).toBe('BLEND');
      expect(materialMetadata.emissiveFactor).toEqual([
        expect.closeTo(material.emissive.r),
        expect.closeTo(material.emissive.g),
        expect.closeTo(material.emissive.b),
      ]);
    } finally {
      if (previousProgressEvent === undefined) {
        delete (globalThis as typeof globalThis & { ProgressEvent?: typeof ProgressEvent }).ProgressEvent;
      } else {
        globalThis.ProgressEvent = previousProgressEvent;
      }
    }
  });

  it('preserves parsed core vertex colors when normalizing extension materials', async () => {
    const previousProgressEvent = globalThis.ProgressEvent;
    globalThis.ProgressEvent = class TestProgressEvent extends Event {
      lengthComputable: boolean;
      loaded: number;
      total: number;

      constructor(type: string, eventInitDict: ProgressEventInit = {}) {
        super(type);
        this.lengthComputable = eventInitDict.lengthComputable ?? false;
        this.loaded = eventInitDict.loaded ?? 0;
        this.total = eventInitDict.total ?? 0;
      }
    } as typeof ProgressEvent;
    try {
      const asset = await parseGlbArrayBuffer(minimalVertexColorClearcoatGltf());

      const material = firstMesh(asset.scene).material as MeshStandardMaterial;
      expect(material).toBeInstanceOf(MeshStandardMaterial);
      expect(material).not.toBeInstanceOf(MeshPhysicalMaterial);
      expect(material.vertexColors).toBe(true);
      expect('clearcoat' in material).toBe(false);
    } finally {
      if (previousProgressEvent === undefined) {
        delete (globalThis as typeof globalThis & { ProgressEvent?: typeof ProgressEvent }).ProgressEvent;
      } else {
        globalThis.ProgressEvent = previousProgressEvent;
      }
    }
  });

  it('extracts COLOR_0 face corners and item size from parsed vertex-color glTF geometry', async () => {
    const previousProgressEvent = globalThis.ProgressEvent;
    globalThis.ProgressEvent = class TestProgressEvent extends Event {
      lengthComputable: boolean;
      loaded: number;
      total: number;

      constructor(type: string, eventInitDict: ProgressEventInit = {}) {
        super(type);
        this.lengthComputable = eventInitDict.lengthComputable ?? false;
        this.loaded = eventInitDict.loaded ?? 0;
        this.total = eventInitDict.total ?? 0;
      }
    } as typeof ProgressEvent;
    try {
      const asset = await parseGlbArrayBuffer(minimalVertexColorClearcoatGltf());
      const extraction = await asset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });

      const attributes = extraction.entries[0]?.texturedRawMesh?.faceAttributes[0];
      expect(attributes?.colorItemSize).toBe(3);
      expect(attributes?.colorCorners).toHaveLength(3);
      expect(attributes?.colorCorners?.[0]).toEqual(expect.objectContaining({ x: 1, y: 0, z: 0, w: 1 }));
      expect(attributes?.colorCorners?.[1]).toEqual(expect.objectContaining({ x: 0, y: 1, z: 0, w: 1 }));
      expect(attributes?.colorCorners?.[2]).toEqual(expect.objectContaining({ x: 0, y: 0, z: 1, w: 1 }));
    } finally {
      if (previousProgressEvent === undefined) {
        delete (globalThis as typeof globalThis & { ProgressEvent?: typeof ProgressEvent }).ProgressEvent;
      } else {
        globalThis.ProgressEvent = previousProgressEvent;
      }
    }
  });

  it('keeps selected grouping extraction data scoped to that request', async () => {
    await withTestFileReader(async () => {
      const shared = new MeshStandardMaterial({ color: 0xff0000, metalness: 0, roughness: 0.5 });
      shared.name = 'shared';
      const accent = new MeshStandardMaterial({ color: 0x00ff00, metalness: 0, roughness: 0.25 });
      accent.name = 'accent';
      const parentA = new Group();
      parentA.name = 'Parent A';
      const parentB = new Group();
      parentB.name = 'Parent B';
      const splitMesh = new Mesh(squareGeometry(), [shared, accent]);
      splitMesh.name = 'Split material mesh';
      const sharedMesh = new Mesh(squareGeometry(), shared);
      sharedMesh.name = 'Shared material mesh';
      sharedMesh.position.set(2, 0, 0);
      parentA.add(splitMesh);
      parentB.add(sharedMesh);
      const scene = new Group();
      scene.name = 'Scene';
      scene.add(parentA, parentB);
      const asset = await parseGlbArrayBuffer(await exportSceneToGlb(scene));

      const materialParent = await asset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });
      const material = await asset.extractGroups({
        groupingMode: 'material',
        mode: 'geometry-with-texture-metadata',
      });
      const none = await asset.extractGroups({
        groupingMode: 'none',
        mode: 'geometry-with-texture-metadata',
      });

      expect(materialParent.extractionApplyState.groupingMode).toBe('material-parent');
      expect(material.extractionApplyState.groupingMode).toBe('material');
      expect(none.extractionApplyState.groupingMode).toBe('none');
      expect(materialParent.entries).toHaveLength(3);
      expect(material.entries).toHaveLength(2);
      expect(none.entries).toHaveLength(3);
      expect(materialParent.entries.map((entry) => entry.id)).not.toEqual(none.entries.map((entry) => entry.id));
      expect('primitiveEntryGroups' in asset).toBe(false);
      expect('primitiveEntries' in asset).toBe(false);
      materialParent.releaseProcessingData();
      expect(materialParent.entries).toEqual([]);
      expect(materialParent.applyMetadataByEntryId.size).toBe(3);
      expect(material.entries).toHaveLength(2);
      expect(none.entries).toHaveLength(3);
      materialParent.dispose();
      expect(materialParent.applyMetadataByEntryId.size).toBe(0);
      expect(material.applyMetadataByEntryId.size).toBe(2);
      expect(none.applyMetadataByEntryId.size).toBe(3);
    });
  });

  it('parses external uploaded texture images through GLTFLoader URL resolution', async () => {
    const testGlobal = globalThis as typeof globalThis & { self?: Window & typeof globalThis };
    const previousSelf = testGlobal.self;
    const previousCreateImageBitmap = globalThis.createImageBitmap;
    const previousProgressEvent = globalThis.ProgressEvent;
    testGlobal.self = globalThis as Window & typeof globalThis;
    globalThis.ProgressEvent = class TestProgressEvent extends Event {
      lengthComputable: boolean;
      loaded: number;
      total: number;

      constructor(type: string, eventInitDict: ProgressEventInit = {}) {
        super(type);
        this.lengthComputable = eventInitDict.lengthComputable ?? false;
        this.loaded = eventInitDict.loaded ?? 0;
        this.total = eventInitDict.total ?? 0;
      }
    } as typeof ProgressEvent;
    globalThis.createImageBitmap = (async (blob: Blob) => {
      const decoded = decodeImage(new Uint8Array(await blob.arrayBuffer()), blob.type || 'image/png');
      return { width: decoded.width, height: decoded.height, data: decoded.data } as unknown as ImageBitmap;
    }) as typeof createImageBitmap;
    const png = encodePng({ width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) });
    const pngBuffer = png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength) as ArrayBuffer;
    try {
      const asset = await parseGlbArrayBuffer(minimalTexturedGltf(), {
        externalResourceFiles: [
          new File([pngBuffer], 'diffuse.png', { type: 'image/png' }),
          new File([pngBuffer], 'unused.png', { type: 'image/png' }),
        ],
      });
      const extraction = await asset.extractGroups({ groupingMode: 'material-parent', mode: 'bake' });
      const textured = extraction.entries[0]!.texturedRawMesh!;

      expect(extraction.entries[0]!.rawMesh.faces).toEqual([[0, 1, 2]]);
      expect(asset.matchedExternalTextureFiles).toEqual(['diffuse.png']);
      expect(textured.materials[0]!.baseColorTexture?.image?.width).toBe(1);
      expect(Array.from(textured.materials[0]!.baseColorTexture!.image!.data.slice(0, 4))).toEqual([255, 0, 0, 255]);
    } finally {
      if (previousSelf === undefined) {
        delete testGlobal.self;
      } else {
        testGlobal.self = previousSelf;
      }
      globalThis.createImageBitmap = previousCreateImageBitmap;
      globalThis.ProgressEvent = previousProgressEvent;
    }
  });

  it('extracts selected browser primitive groups with texture metadata but without image payloads', async () => {
    await withBrowserImageLoading(async (pngBuffer) => {
      const asset = await parseGlbArrayBuffer(minimalTexturedGltf(), {
        externalResourceFiles: [
          new File([pngBuffer], 'diffuse.png', { type: 'image/png' }),
        ],
      });

      const extraction = await asset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });

      expect(extraction.entries).toHaveLength(1);
      expect(extraction.applyMetadataByEntryId.size).toBe(1);
      expect(extraction.entries[0]!.texturedRawMesh).toBeDefined();
      expect(extraction.entries[0]!.texturedRawMesh?.materials[0]?.baseColorTexture?.image).toBeUndefined();
      expect(extraction.entries[0]!.texturedRawMesh?.materials[0]?.textureSlots.every((slot) => slot.image === undefined)).toBe(true);
      expect(extraction.summary.bakeableEntryCount).toBe(1);
      expect(extraction.summary.hasTransferableTextureData).toBe(true);
      expect(extraction.summary.hasImageBackedTextureTransferData).toBe(true);
      expect(extraction.summary.hasTransferableVertexAttributes).toBe(true);
      extraction.releaseProcessingData();
      expect(extraction.entries).toEqual([]);
      expect(extraction.applyMetadataByEntryId.size).toBe(1);
    });
  });

  it('separates normal-only vertex attribute transfer from image-backed texture transfer in browser extraction', async () => {
    await withBrowserImageLoading(async () => {
      const asset = await parseGlbArrayBuffer(minimalNormalOnlyGltf());
      const extraction = await asset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });

      expect(extraction.entries[0]!.requiresAttributeTransfer).toBe(true);
      expect(extraction.summary.hasTransferableVertexAttributes).toBe(true);
      expect(extraction.summary.hasImageBackedTextureTransferData).toBe(false);
      expect(extraction.summary.hasTransferableTextureData).toBe(false);
      expect(extraction.summary.hasImageBackedTextureBakeData).toBe(false);
      expect(extraction.summary.bakeableEntryCount).toBe(0);
    });
  });

  it('extracts bake browser primitive groups with stable ids and image payloads', async () => {
    await withBrowserImageLoading(async (pngBuffer) => {
      const asset = await parseGlbArrayBuffer(minimalTexturedGltf(), {
        externalResourceFiles: [
          new File([pngBuffer], 'diffuse.png', { type: 'image/png' }),
        ],
      });

      const geometry = await asset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });
      const bake = await asset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'bake',
      });

      expect(bake.entries.map((entry) => entry.id)).toEqual(geometry.entries.map((entry) => entry.id));
      expect(bake.entries[0]!.texturedRawMesh?.materials[0]?.baseColorTexture?.image).toBeDefined();
    });
  });

  it('applies geometry-mode adapter results without preserving unusable texture bindings', async () => {
    await withBrowserImageLoading(async (pngBuffer) => {
      const asset = await parseGlbArrayBuffer(minimalTexturedGltf(), {
        externalResourceFiles: [
          new File([pngBuffer], 'diffuse.png', { type: 'image/png' }),
        ],
      });
      const extraction = await asset.extractGroups({ groupingMode: 'material-parent', mode: 'geometry' });
      const source = extraction.entries[0]!;
      const raw = rawResultFor(source.rawMesh);

      const output = await asset.applyResults(extraction, [{
        id: source.id,
        source,
        geometry: { raw, elapsedSeconds: 0 },
        raw,
      }]);

      const material = firstMesh(output).material as MeshStandardMaterial;
      const geometry = firstMesh(output).geometry as BufferGeometry;
      expect(material.map).toBeNull();
      expect(geometry.getAttribute('uv')).toBeUndefined();
    });
  });

  it('builds output scenes and applies baked adapter results from lazy extraction metadata', async () => {
    await withBrowserImageLoading(async (pngBuffer) => {
      const asset = await parseGlbArrayBuffer(minimalTexturedGltf(), {
        externalResourceFiles: [
          new File([pngBuffer], 'diffuse.png', { type: 'image/png' }),
        ],
      });
      const extraction = await asset.extractGroups({ groupingMode: 'material-parent', mode: 'bake' });
      const source = extraction.entries[0]!;
      const raw = rawResultFor(source.rawMesh);
      const atlas = await createInjectiveAtlas(source.rawMesh, { textureSize: 16, padding: 1 });
      const image = { width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) };
      const materialSettings = {
        alphaMode: 'OPAQUE' as const,
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 0,
        roughnessFactor: 0.5,
        emissiveFactor: [0, 0, 0] as [number, number, number],
        normalScale: 1,
        occlusionStrength: 1,
      };

      const applied = await asset.applyResults(extraction, [{
        id: source.id,
        source,
        geometry: { raw, elapsedSeconds: 0 },
        raw,
        baked: {
          raw,
          elapsedSeconds: 0,
          baked: {
            image,
            atlas,
            additionalTextures: [],
            stats: { filledPixels: 1, mappedPixels: 1, unmappedPixels: 0 },
          },
          materialSettings,
        },
      }]);
      const built = await asset.buildOutputScene(extraction, [{
        id: source.id,
        meshOrdinal: 0,
        sourceMeshOrdinals: [0],
        parentObjectOrdinal: 0,
        sourceMaterial: null,
        rawMesh: source.rawMesh,
        materialMode: 'baked',
        atlas,
        image,
        additionalTextures: [],
        materialSettings,
      }]);

      expect((firstMesh(applied).material as MeshStandardMaterial).map).toBeInstanceOf(DataTexture);
      expect((firstMesh(applied).geometry as BufferGeometry).getAttribute('uv').count).toBe(3);
      expect((firstMesh(built).material as MeshStandardMaterial).map).toBeInstanceOf(DataTexture);
    });
  });

  it('extracts world-space raw mesh, UVs, materials, groups, and data textures from a Three.js scene', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    texture.needsUpdate = true;
    const materials = [
      new MeshStandardMaterial({ color: 0xffffff, map: texture, metalness: 0, roughness: 0.5 }),
      new MeshStandardMaterial({ color: 0x00ff00, transparent: true, opacity: 0.25 }),
    ];
    const mesh = new Mesh(squareGeometry(), materials);
    mesh.position.set(1, 0, 0);
    const scene = new Group();
    scene.name = 'Scene';
    scene.add(mesh);

    const model = await extractBrowserModelFromObject(scene);
    expect(model.rawMesh.positions).toHaveLength(4);
    expect(model.rawMesh.positions[0]!.x).toBeCloseTo(1);
    expect(model.rawMesh.faces).toEqual([[0, 1, 2], [0, 2, 3]]);
    expect(model.textured.faceAttributes.map((attrs) => attrs.materialId)).toEqual([0, 1]);
    expect(faceUvSet(model.textured.faceAttributes[0]!, 0)?.uvs[2].x).toBeCloseTo(1);
    expect(model.textured.faceAttributes[0]!.normalCorners).toHaveLength(3);
    expect(model.textured.faceAttributes[0]!.normalCorners?.[0].z).toBeCloseTo(1);
    expect(model.textured.materials).toHaveLength(2);
    expect(model.textured.materials[0]!.baseColorTexture?.image?.width).toBe(1);
    expect(model.textured.materials[0]!.metallicFactor).toBe(0);
    expect(model.textured.materials[1]!.alphaMode).toBe('BLEND');
    expect(model.sceneContext.supported).toBe(true);
  });

  it('transforms grouped authored normals with normal matrices and preserves none-grouped local normals', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const mesh = new Mesh(triangleGeometryWithAuthoredNormals(), material);
    mesh.rotation.z = Math.PI / 5;
    mesh.scale.set(2, 3, 0.5);
    const scene = new Group();
    scene.name = 'Scene';
    scene.add(mesh);

    const model = await extractBrowserModelFromObject(scene);
    const normalMatrix = new Matrix3().getNormalMatrix(mesh.matrix);
    const materialParentEntry = model.primitiveEntryGroups['material-parent'][0]!;
    const noneEntry = model.primitiveEntryGroups.none[0]!;

    for (let corner = 0; corner < 3; corner += 1) {
      const expectedGrouped = AUTHORED_NORMALS[corner]!.clone().applyNormalMatrix(normalMatrix).normalize();
      expectVectorClose(model.textured.faceAttributes[0]!.normalCorners?.[corner], expectedGrouped);
      expectVectorClose(materialParentEntry.texturedRawMesh.faceAttributes[0]!.normalCorners?.[corner], expectedGrouped);
      expectVectorClose(noneEntry.texturedRawMesh.faceAttributes[0]!.normalCorners?.[corner], AUTHORED_NORMALS[corner]!);
    }
  });

  it('extracts base-color texture channel metadata and matching UV sets', async () => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]), 3));
    geometry.setAttribute('uv', new BufferAttribute(new Float32Array([
      0, 0,
      1, 0,
      0, 1,
    ]), 2));
    geometry.setAttribute('uv1', new BufferAttribute(new Float32Array([
      0.25, 0.25,
      0.75, 0.25,
      0.25, 0.75,
    ]), 2));
    geometry.setIndex([0, 1, 2]);
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    texture.channel = 1;
    texture.needsUpdate = true;
    const scene = new Group();
    scene.add(new Mesh(geometry, new MeshStandardMaterial({ map: texture })));

    const model = await extractBrowserModelFromObject(scene);

    expect(model.textured.materials[0]!.baseColorTexture?.texCoord).toBe(1);
    expect(model.textured.materials[0]!.textureSlots[0]!.texCoord).toBe(1);
    expect(faceUvSet(model.textured.faceAttributes[0]!, 0)?.uvs[1].x).toBeCloseTo(1);
    expect(faceUvSet(model.textured.faceAttributes[0]!, 1)?.uvs[1].x).toBeCloseTo(0.75);
    expect(model.primitiveEntries[0]!.bakeable).toBe(true);
  });

  it('extracts interleaved UV attributes from strided geometry', async () => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]), 3));
    geometry.setAttribute('uv1', new InterleavedBufferAttribute(
      new InterleavedBuffer(new Float32Array([
        99, 0.25, 0.25,
        99, 0.75, 0.25,
        99, 0.25, 0.75,
      ]), 3),
      2,
      1,
    ));
    geometry.setIndex([0, 1, 2]);
    const texture = new DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1, RGBAFormat);
    texture.channel = 1;
    texture.needsUpdate = true;
    const scene = new Group();
    scene.add(new Mesh(geometry, new MeshStandardMaterial({ normalMap: texture })));

    const model = await extractBrowserModelFromObject(scene);

    expect(faceUvSet(model.textured.faceAttributes[0]!, 1)?.uvs[1].x).toBeCloseTo(0.75);
    expect(model.primitiveEntries[0]!.hasTexturedMaterial).toBe(true);
    expect(model.primitiveEntries[0]!.requiresAttributeTransfer).toBe(true);
  });

  it('does not mark image-backed browser textures bakeable when their requested UV channel is missing', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    texture.channel = 1;
    texture.needsUpdate = true;
    const scene = new Group();
    scene.add(new Mesh(squareGeometry(), new MeshStandardMaterial({ map: texture })));

    const model = await extractBrowserModelFromObject(scene);

    expect(faceUvSet(model.textured.faceAttributes[0]!, 1)).toBeUndefined();
    expect(model.primitiveEntries[0]!.hasTexturedMaterial).toBe(true);
    expect(model.primitiveEntries[0]!.bakeable).toBe(false);
    expect(model.primitiveEntries[0]!.requiresAttributeTransfer).toBe(true);
  });

  it('normalizes in-memory browser materials to shared core PBR materials', () => {
    const clearcoat = new DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1, RGBAFormat);
    clearcoat.needsUpdate = true;
    const sourceMaterial = materialWithStandardTextureSlots() as MeshStandardMaterial & {
      clearcoat?: number;
      clearcoatMap?: Texture | null;
    };
    sourceMaterial.name = 'extended-clearcoat';
    sourceMaterial.clearcoat = 1;
    sourceMaterial.clearcoatMap = clearcoat;
    sourceMaterial.transparent = true;
    sourceMaterial.opacity = 0.7;
    const accentMaterial = new MeshPhysicalMaterial({
      color: 0x669933,
      clearcoat: 0.5,
      metalness: 0,
      roughness: 0.75,
    });
    const root = new Group();
    root.add(
      new Mesh(squareGeometry(), sourceMaterial),
      new Mesh(squareGeometry(), [sourceMaterial, accentMaterial]),
    );

    normalizeBrowserSceneMaterialsToCorePbr(root);

    const meshes = allMeshes(root);
    const normalizedMaterial = meshMaterialAt(meshes[0]!, 0) as MeshStandardMaterial;
    const normalizedAccentMaterial = meshMaterialAt(meshes[1]!, 1) as MeshStandardMaterial;
    expect(normalizedMaterial).toBeInstanceOf(MeshStandardMaterial);
    expect(normalizedMaterial).not.toBeInstanceOf(MeshPhysicalMaterial);
    expect(meshMaterialAt(meshes[1]!, 0)).toBe(normalizedMaterial);
    expect(normalizedMaterial).not.toBe(sourceMaterial);
    expect(normalizedMaterial.name).toBe('extended-clearcoat');
    expect(normalizedMaterial.color.getHex()).toBe(sourceMaterial.color.getHex());
    expect(normalizedMaterial.metalness).toBeCloseTo(0.25);
    expect(normalizedMaterial.roughness).toBeCloseTo(0.5);
    expect(normalizedMaterial.transparent).toBe(true);
    expect(normalizedMaterial.opacity).toBeCloseTo(0.7);
    expect(normalizedMaterial.map).toBe(sourceMaterial.map);
    expect(normalizedMaterial.normalMap).toBe(sourceMaterial.normalMap);
    expect(normalizedMaterial.metalnessMap).toBe(sourceMaterial.metalnessMap);
    expect(normalizedMaterial.roughnessMap).toBe(sourceMaterial.roughnessMap);
    expect(normalizedMaterial.aoMap).toBe(sourceMaterial.aoMap);
    expect(normalizedMaterial.emissiveMap).toBe(sourceMaterial.emissiveMap);
    expect(normalizedMaterial.normalScale.x).toBeCloseTo(sourceMaterial.normalScale.x);
    expect(normalizedMaterial.aoMapIntensity).toBeCloseTo(sourceMaterial.aoMapIntensity);
    expect(normalizedMaterial.emissive.r).toBeCloseTo(sourceMaterial.emissive.r);
    expect(normalizedMaterial.emissive.g).toBeCloseTo(sourceMaterial.emissive.g);
    expect(normalizedMaterial.emissive.b).toBeCloseTo(sourceMaterial.emissive.b);
    expect('clearcoat' in normalizedMaterial).toBe(false);
    expect('clearcoatMap' in normalizedMaterial).toBe(false);
    expect(normalizedAccentMaterial).toBeInstanceOf(MeshStandardMaterial);
    expect(normalizedAccentMaterial).not.toBeInstanceOf(MeshPhysicalMaterial);
    expect(normalizedAccentMaterial).not.toBe(accentMaterial);
    expect(normalizedAccentMaterial.color.getHex()).toBe(accentMaterial.color.getHex());
    expect(normalizedAccentMaterial.roughness).toBeCloseTo(0.75);

    const preview = cloneSceneForViewport(root);
    const previewMaterial = meshMaterialAt(firstMesh(preview), 0) as MeshStandardMaterial;
    expect(previewMaterial).toBeInstanceOf(MeshStandardMaterial);
    expect(previewMaterial).not.toBeInstanceOf(MeshPhysicalMaterial);
    expect(previewMaterial.name).toBe('extended-clearcoat');
    expect(previewMaterial.map).toBe(sourceMaterial.map);
    expect('clearcoat' in previewMaterial).toBe(false);
    expect('clearcoatMap' in previewMaterial).toBe(false);
  });

  it('extracts standard browser material texture slots and all matching UV sets', async () => {
    const material = materialWithStandardTextureSlots();
    const scene = new Group();
    scene.add(new Mesh(squareGeometryWithUv1(), material));

    const model = await extractBrowserModelFromObject(scene);

    expect(model.textured.materials[0]!.textureSlots.map((slot) => slot.slot).sort()).toEqual([
      'baseColor',
      'emissive',
      'metallicRoughness',
      'normal',
      'occlusion',
    ]);
    const normalSlot = model.textured.materials[0]!.textureSlots.find((slot) => slot.slot === 'normal');
    const occlusionSlot = model.textured.materials[0]!.textureSlots.find((slot) => slot.slot === 'occlusion');
    const baseColorSlot = model.textured.materials[0]!.textureSlots.find((slot) => slot.slot === 'baseColor');
    expect(baseColorSlot?.hasImage).toBe(true);
    expect(baseColorSlot?.image).toBeUndefined();
    expect(normalSlot?.image?.width).toBe(1);
    expect(Array.from(normalSlot!.image!.data.slice(0, 4))).toEqual([128, 128, 255, 255]);
    expect(occlusionSlot?.image?.width).toBe(1);
    expect(Array.from(occlusionSlot!.image!.data.slice(0, 4))).toEqual([200, 0, 0, 255]);
    expect(model.textured.materials[0]!.emissiveFactor).toEqual([
      expect.closeTo(material.emissive.r),
      expect.closeTo(material.emissive.g),
      expect.closeTo(material.emissive.b),
    ]);
    expect(model.textured.materials[0]!.normalScale).toBeCloseTo(0.75);
    expect(model.textured.materials[0]!.occlusionStrength).toBeCloseTo(0.6);
    expect(model.textured.faceAttributes[0]!.uvSets.map((set) => set.texCoord)).toEqual([0, 1]);
    expect(model.primitiveEntries[0]!.hasTexturedMaterial).toBe(true);
    expect(model.primitiveEntries[0]!.requiresAttributeTransfer).toBe(true);
  });

  it('marks standard normal-only material textures as bakeable', async () => {
    const normal = new DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1, RGBAFormat);
    normal.needsUpdate = true;
    const scene = new Group();
    scene.add(new Mesh(squareGeometry(), new MeshStandardMaterial({ normalMap: normal })));

    const model = await extractBrowserModelFromObject(scene);

    expect(model.primitiveEntries[0]!.hasTexturedMaterial).toBe(true);
    expect(model.primitiveEntries[0]!.requiresAttributeTransfer).toBe(true);
    expect(model.primitiveEntries[0]!.bakeable).toBe(true);
  });

  it('preserves browser source normal-map Y scale for tangentless geometry', async () => {
    const normal = new DataTexture(new Uint8Array([128, 255, 255, 255]), 1, 1, RGBAFormat);
    normal.needsUpdate = true;
    const material = new MeshStandardMaterial({ normalMap: normal });
    material.normalScale.set(0.75, -0.75);
    const scene = new Group();
    scene.add(new Mesh(squareGeometry(), material));

    const model = await extractBrowserModelFromObject(scene);

    expect(model.textured.materials[0]!.normalScale).toBeCloseTo(0.75);
    expect((model.textured.faceAttributes[0]! as typeof model.textured.faceAttributes[0] & { normalMapYScale?: number }).normalMapYScale).toBe(-1);
  });

  it('preserves browser source tangent corners as transformed directions', async () => {
    const normal = new DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1, RGBAFormat);
    normal.needsUpdate = true;
    const geometry = squareGeometry();
    geometry.setAttribute('tangent', new Float32BufferAttribute([
      1, 0, 1, -1,
      1, 0, 1, -1,
      1, 0, 1, -1,
      1, 0, 1, -1,
    ], 4));
    const mesh = new Mesh(geometry, new MeshStandardMaterial({ normalMap: normal }));
    mesh.scale.set(2, 1, 0.5);
    const scene = new Group();
    scene.add(mesh);

    const model = await extractBrowserModelFromObject(scene);
    const expected = new Vector3(1, 0, 1).transformDirection(mesh.matrixWorld);

    expect(model.textured.faceAttributes[0]!.normalMapYScale).toBeUndefined();
    expect(model.textured.faceAttributes[0]!.tangentCorners).toHaveLength(3);
    expect(model.textured.faceAttributes[0]!.tangentCorners?.[0].x).toBeCloseTo(expected.x, 6);
    expect(model.textured.faceAttributes[0]!.tangentCorners?.[0].y).toBeCloseTo(expected.y, 6);
    expect(model.textured.faceAttributes[0]!.tangentCorners?.[0].z).toBeCloseTo(expected.z, 6);
    expect(model.textured.faceAttributes[0]!.tangentCorners?.[0].w).toBeCloseTo(-1);
  });

  it('ignores extension texture properties during direct browser extraction compatibility', async () => {
    const bump = new DataTexture(new Uint8Array([127, 127, 127, 255]), 1, 1, RGBAFormat);
    bump.channel = 1;
    bump.needsUpdate = true;
    const material = new MeshStandardMaterial() as MeshStandardMaterial & { bumpMap?: Texture | null };
    material.bumpMap = bump;
    const scene = new Group();
    scene.add(new Mesh(squareGeometryWithUv1(), material));

    const model = await extractBrowserModelFromObject(scene);

    expect(model.textured.materials[0]!.textureSlots).toEqual([]);
    expect(model.textured.faceAttributes[0]!.uvSets.map((set) => set.texCoord)).toEqual([0, 1]);
    expect(model.primitiveEntries[0]!.hasTexturedMaterial).toBe(false);
    expect(model.primitiveEntries[0]!.requiresAttributeTransfer).toBe(true);
    expect(model.primitiveEntries[0]!.bakeable).toBe(false);
  });

  it('does not count empty Texture objects as image-backed material textures', async () => {
    const material = new MeshStandardMaterial({ map: new Texture() });
    const scene = new Group();
    scene.add(new Mesh(squareGeometry(), material));

    const model = await extractBrowserModelFromObject(scene);

    expect(model.textured.materials[0]!.baseColorTexture?.image).toBeUndefined();
    expect(model.textured.materials[0]!.textureSlots[0]).toMatchObject({
      slot: 'baseColor',
      hasImage: false,
    });
    expect(model.primitiveEntries[0]!.hasTexturedMaterial).toBe(false);
    expect(model.primitiveEntries[0]!.requiresAttributeTransfer).toBe(true);
    expect(model.primitiveEntries[0]!.bakeable).toBe(false);
  });

  it('keeps non-default base-color factors preservable but not bakeable when an empty base-color texture has no UVs', async () => {
    const geometry = triangleGeometry([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    const material = new MeshStandardMaterial({ color: 0x00ff00, map: new Texture() });
    const scene = new Group();
    scene.add(new Mesh(geometry, material));

    const model = await extractBrowserModelFromObject(scene);

    expect(model.textured.faceAttributes[0]!.uvSets).toEqual([]);
    expect(model.primitiveEntries[0]!.hasTexturedMaterial).toBe(false);
    expect(model.primitiveEntries[0]!.hasPreservableMaterialData).toBe(true);
    expect(model.primitiveEntries[0]!.requiresAttributeTransfer).toBeUndefined();
    expect(model.primitiveEntries[0]!.bakeable).toBe(false);
  });

  it('exports preserved factor-only output from a normalized source scene', async () => {
    const testGlobal = globalThis as typeof globalThis & { FileReader?: typeof FileReader };
    const previousFileReader = testGlobal.FileReader;
    testGlobal.FileReader = class TestFileReader extends EventTarget {
      result: string | ArrayBuffer | null = null;
      error: DOMException | null = null;
      onloadend: ((this: FileReader, ev: ProgressEvent<FileReader>) => unknown) | null = null;

      private emitLoadEnd(): void {
        this.onloadend?.call(this as unknown as FileReader, {} as ProgressEvent<FileReader>);
      }

      readAsArrayBuffer(blob: Blob): void {
        void blob.arrayBuffer().then((buffer) => {
          this.result = buffer;
          this.emitLoadEnd();
        }).catch((error: unknown) => {
          this.error = error instanceof DOMException ? error : new DOMException(String(error));
          this.emitLoadEnd();
        });
      }
    } as unknown as typeof FileReader;
    try {
      const geometry = triangleGeometry([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ]);
      const sourceMaterial = new MeshStandardMaterial({ color: 0x00ff00, map: new Texture() });
      const root = new Group();
      root.add(new Mesh(geometry, sourceMaterial));
      normalizeBrowserSceneMaterialsToCorePbr(root);
      const normalizedMaterial = firstMesh(root).material as MeshStandardMaterial;
      const output = createPrimitiveOutputScene(root, [{
        id: 'mesh-0',
        meshOrdinal: 0,
        rawMesh: {
          positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
          faces: [[0, 1, 2] as [number, number, number]],
        },
        materialMode: 'preserve',
        sourceMaterial: normalizedMaterial,
      }]);

      const mesh = firstMesh(output);
      expect((mesh.material as MeshStandardMaterial).map).toBeNull();
      await expect(exportSceneToGlb(output)).resolves.toBeInstanceOf(ArrayBuffer);
    } finally {
      if (previousFileReader === undefined) {
        delete testGlobal.FileReader;
      } else {
        testGlobal.FileReader = previousFileReader;
      }
    }
  });

  it('extracts default material-parent processing entries with material metadata', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    texture.needsUpdate = true;
    const textured = new Mesh(squareGeometry(), new MeshStandardMaterial({ color: 0xffffff, map: texture, metalness: 0, roughness: 0.5 }));
    textured.name = 'Chair seat';
    const plain = new Mesh(squareGeometry(), new MeshStandardMaterial({ color: 0x00ff00, metalness: 0, roughness: 0.75 }));
    plain.name = 'Chair legs';
    plain.position.set(2, 0, 0);
    const scene = new Group();
    scene.name = 'Scene';
    scene.add(textured, plain);

    const model = await extractBrowserModelFromObject(scene);

    expect(model.primitiveEntries).toHaveLength(2);
    expect(model.primitiveEntries.map((entry) => entry.meshOrdinal)).toEqual([0, 1]);
    expect(model.primitiveEntries.map((entry) => entry.sourceMeshOrdinals)).toEqual([[0], [1]]);
    expect(model.primitiveEntries[0]!.rawMesh.faces).toHaveLength(2);
    expect(model.primitiveEntries[0]!.texturedRawMesh.materials[0]!.baseColorTexture?.image?.width).toBe(1);
    expect(model.primitiveEntries[0]!.bakeable).toBe(true);
    expect(model.primitiveEntries[1]!.hasPreservableMaterialData).toBe(true);
    expect(model.primitiveEntries[1]!.bakeable).toBe(false);
    expect(model.primitiveEntries[1]!.rawMesh.positions[0]!.x).toBeCloseTo(2);
    expect(model.rawMesh.faces).toHaveLength(4);
  });

  it('warns and ignores source colors when COLOR_0 count does not match POSITION count', async () => {
    const geometry = triangleGeometry([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    geometry.setAttribute('color', new BufferAttribute(new Float32Array([
      1, 0, 0,
      0, 1, 0,
    ]), 3));
    const root = new Group();
    root.add(new Mesh(geometry, new MeshStandardMaterial({ vertexColors: true })));

    const model = await extractBrowserModelFromObject(root);

    expect(model.warnings).toContain('Mesh <unnamed> has COLOR_0 count 2 but POSITION count 3; source colors were ignored.');
    expect(model.textured.faceAttributes[0]?.colorCorners).toBeUndefined();
    expect(model.primitiveEntryGroups['material-parent'][0]?.texturedRawMesh.faceAttributes[0]?.colorCorners).toBeUndefined();
  });

  it('separates same-material browser groups by COLOR_0 presence and item size', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, vertexColors: true, metalness: 0, roughness: 0.5 });
    const root = new Group();
    root.add(
      new Mesh(triangleGeometry([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ]), material),
      new Mesh(vertexColorTriangleGeometry(3), material),
      new Mesh(vertexColorTriangleGeometry(4), material),
    );

    const materialParent = await extractBrowserPrimitiveGroupsFromObject(root, {
      groupingMode: 'material-parent',
      mode: 'geometry-with-texture-metadata',
    });
    const materialOnly = await extractBrowserPrimitiveGroupsFromObject(root, {
      groupingMode: 'material',
      mode: 'geometry-with-texture-metadata',
    });

    for (const extraction of [materialParent, materialOnly]) {
      const colorItemSizes = extraction.entries
        .map((entry) => entry.texturedRawMesh?.faceAttributes[0]?.colorItemSize ?? 0)
        .sort((a, b) => a - b);
      expect(colorItemSizes).toEqual([0, 3, 4]);
    }
  });

  it('groups processing entries by material and direct parent by default', async () => {
    const sharedMaterial = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const otherMaterial = new MeshStandardMaterial({ color: 0x00ff00, metalness: 0, roughness: 0.75 });
    const parentA = new Group();
    parentA.name = 'Parent A';
    parentA.position.set(10, 0, 0);
    const parentB = new Group();
    parentB.name = 'Parent B';
    parentB.position.set(20, 0, 0);

    const sharedA1 = new Mesh(squareGeometry(), sharedMaterial);
    sharedA1.position.set(1, 0, 0);
    const sharedA2 = new Mesh(squareGeometry(), sharedMaterial);
    sharedA2.position.set(2, 0, 0);
    const sharedB = new Mesh(squareGeometry(), sharedMaterial);
    sharedB.position.set(3, 0, 0);
    const otherA = new Mesh(squareGeometry(), otherMaterial);
    otherA.position.set(4, 0, 0);
    parentA.add(sharedA1, sharedA2, otherA);
    parentB.add(sharedB);
    const scene = new Group();
    scene.name = 'Scene';
    scene.add(parentA, parentB);

    const model = await extractBrowserModelFromObject(scene);

    expect(model.primitiveEntries).toBe(model.primitiveEntryGroups['material-parent']);
    expect(model.primitiveEntryGroups['material-parent']).toHaveLength(3);
    const parentShared = model.primitiveEntryGroups['material-parent'].find((entry) => entry.sourceMeshOrdinals.join(',') === '0,1');
    expect(parentShared?.rawMesh.faces).toHaveLength(4);
    expect(parentShared?.parentObjectOrdinal).toBeGreaterThan(0);
    expect(parentShared?.rawMesh.positions[0]!.x).toBeCloseTo(1);
    expect(parentShared?.rawMesh.positions.some((position) => Math.abs(position.x - 2) < 1e-6)).toBe(true);

    expect(model.primitiveEntryGroups.material).toHaveLength(2);
    const materialShared = model.primitiveEntryGroups.material.find((entry) => entry.sourceMeshOrdinals.join(',') === '0,1,3');
    expect(materialShared?.rawMesh.faces).toHaveLength(6);
    expect(materialShared?.parentObjectOrdinal).toBe(0);
    expect(materialShared?.rawMesh.positions[0]!.x).toBeCloseTo(11);
    expect(materialShared?.rawMesh.positions.some((position) => Math.abs(position.x - 23) < 1e-6)).toBe(true);
  });

  it('welds duplicate boundary vertices when building grouped processing entries', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const parent = new Group();
    parent.add(
      new Mesh(triangleGeometry([0, 0, 0, 1, 0, 0, 0, 1, 0]), material),
      new Mesh(triangleGeometry([1, 0, 0, 1, 1, 0, 0, 1, 0]), material),
    );
    const root = new Group();
    root.add(parent);

    const model = await extractBrowserModelFromObject(root);
    const group = model.primitiveEntryGroups['material-parent'][0]!;

    expect(group.rawMesh.faces).toHaveLength(2);
    expect(group.rawMesh.positions).toHaveLength(4);
    expect(group.rawMesh.faces).toEqual([[0, 1, 2], [1, 3, 2]]);
  });

  it.each(['material-parent', 'material', 'none'] as const)(
    'uses weldVertices to control browser raw mesh construction for %s grouping',
    async (groupingMode) => {
      const { root, sourceVertexCount } = duplicatePositionSeamRoot();

      const welded = await extractBrowserPrimitiveGroupsFromObject(root, {
        groupingMode,
        mode: 'geometry-with-texture-metadata',
        weldVertices: true,
      });
      const unwelded = await extractBrowserPrimitiveGroupsFromObject(root, {
        groupingMode,
        mode: 'geometry-with-texture-metadata',
        weldVertices: false,
      });

      expect(welded.entries[0]!.rawMesh.positions.length).toBeLessThan(unwelded.entries[0]!.rawMesh.positions.length);
      expect(unwelded.entries[0]!.rawMesh.positions).toHaveLength(sourceVertexCount);
      expect(welded.entries[0]!.texturedRawMesh?.faceAttributes).toHaveLength(2);
      expect(unwelded.entries[0]!.texturedRawMesh?.faceAttributes).toHaveLength(2);
    },
  );

  it('builds none-grouped entries without coordinate welding when disabled', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]), 3));
    geometry.setIndex([0, 1, 2, 3, 4, 5]);
    const mesh = new Mesh(geometry, material);
    mesh.name = 'Split triangles';
    mesh.position.set(5, 0, 0);
    const parent = new Group();
    parent.name = 'Parent';
    const root = new Group();
    parent.add(mesh);
    root.add(parent);

    const extraction = await extractBrowserPrimitiveGroupsFromObject(root, {
      groupingMode: 'none',
      mode: 'geometry-with-texture-metadata',
      weldVertices: false,
    });
    const none = extraction.entries;

    expect(none).toHaveLength(1);
    expect(none[0]!.rawMesh.positions).toHaveLength(6);
    expect(none[0]!.rawMesh.positions[0]!.x).toBeCloseTo(0);
    expect(none[0]!.rawMesh.faces).toEqual([[0, 1, 2], [3, 4, 5]]);
  });

  it('creates geometry output scenes with position, normal, and index attributes', () => {
    const rawMesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2] as [number, number, number]],
    };
    const scene = createGeometryOutputScene(rawMesh, identitySceneTransformContext('Scene'));
    const geometry = firstMesh(scene).geometry as BufferGeometry;
    expect(geometry.getAttribute('position').count).toBe(3);
    expect(geometry.getAttribute('normal').count).toBe(3);
    expect(geometry.getIndex()?.count).toBe(3);
  });

  it('creates textured output scenes with shared planar chart vertices and material scalars', async () => {
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };
    const scene = createTexturedOutputScene({
      rawMesh,
      atlas: await createInjectiveAtlas(rawMesh, { textureSize: 64, padding: 2 }),
      image: { width: 64, height: 64, data: new Uint8Array(64 * 64 * 4).fill(255) },
      materialSettings: {
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: true,
        metallicFactor: 0,
        roughnessFactor: 0.42,
        emissiveFactor: [0, 0, 0],
        normalScale: 1,
        occlusionStrength: 1,
      },
      sceneContext: identitySceneTransformContext('Scene'),
    });
    const mesh = firstMesh(scene);
    const geometry = mesh.geometry as BufferGeometry;
    const material = mesh.material as MeshStandardMaterial;
    expect(geometry.getAttribute('position').count).toBe(4);
    expect(geometry.getAttribute('uv').count).toBe(4);
    expect(geometry.getAttribute('normal').count).toBe(4);
    expect(geometry.getAttribute('tangent')).toBeUndefined();
    expect(geometry.getIndex()?.count).toBe(6);
    expect(material.map).toBeTruthy();
    expect(material.metalness).toBe(0);
    expect(material.roughness).toBeCloseTo(0.42);
  });

  it('creates output DataTextures with a zero-copy view of RGBA image data', async () => {
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };
    const backing = new Uint8Array([9, 9, 255, 0, 0, 255, 9, 9]);
    const imageData = new Uint8ClampedArray(backing.buffer, 2, 4);
    const scene = createTexturedOutputScene({
      rawMesh,
      atlas: await createInjectiveAtlas(rawMesh, { textureSize: 16, padding: 1 }),
      image: { width: 1, height: 1, data: imageData },
      materialSettings: {
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 1,
        roughnessFactor: 1,
        emissiveFactor: [0, 0, 0],
        normalScale: 1,
        occlusionStrength: 1,
      },
      sceneContext: identitySceneTransformContext('Scene'),
    });

    const material = firstMesh(scene).material as MeshStandardMaterial;
    const textureData = (material.map as DataTexture).image.data as Uint8Array;
    expect(textureData).toBeInstanceOf(Uint8Array);
    expect(textureData.buffer).toBe(imageData.buffer);
    expect(textureData.byteOffset).toBe(imageData.byteOffset);
    expect(textureData.byteLength).toBe(imageData.byteLength);
    imageData[0] = 128;
    expect(textureData[0]).toBe(128);
  });

  it('creates baked output materials with standard additional texture maps', async () => {
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };
    const scene = createTexturedOutputScene({
      rawMesh,
      atlas: await createInjectiveAtlas(rawMesh, { textureSize: 16, padding: 1 }),
      image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) },
      additionalTextures: [
        { slot: 'normal', image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) } },
        { slot: 'metallicRoughness', image: { width: 1, height: 1, data: new Uint8Array([0, 127, 255, 255]) } },
        { slot: 'occlusion', image: { width: 1, height: 1, data: new Uint8Array([200, 0, 0, 255]) } },
        { slot: 'emissive', image: { width: 1, height: 1, data: new Uint8Array([0, 0, 255, 255]) } },
      ],
      materialSettings: {
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 0.25,
        roughnessFactor: 0.5,
        normalScale: 0.75,
        occlusionStrength: 0.6,
        emissiveFactor: [0.1, 0.2, 0.3],
      },
      sceneContext: identitySceneTransformContext('Scene'),
    });

    const material = firstMesh(scene).material as MeshStandardMaterial;
    const geometry = firstMesh(scene).geometry as BufferGeometry;
    const tangent = geometry.getAttribute('tangent');
    expect(tangent?.itemSize).toBe(4);
    expect(tangent?.count).toBe(geometry.getAttribute('position').count);
    expect(material.map).toBeTruthy();
    expect(material.normalMap).toBeTruthy();
    expect(material.metalnessMap).toBeTruthy();
    expect(material.roughnessMap).toBe(material.metalnessMap);
    expect(material.aoMap).toBeTruthy();
    expect(material.emissiveMap).toBeTruthy();
    expect(material.map!.colorSpace).toBe(SRGBColorSpace);
    expect(material.emissiveMap!.colorSpace).toBe(SRGBColorSpace);
    expect(material.normalMap!.colorSpace).not.toBe(SRGBColorSpace);
    expect(material.metalnessMap!.colorSpace).not.toBe(SRGBColorSpace);
    expect(material.aoMap!.colorSpace).not.toBe(SRGBColorSpace);
    for (const texture of [material.map, material.normalMap, material.metalnessMap, material.aoMap, material.emissiveMap]) {
      expect(texture?.wrapS).toBe(ClampToEdgeWrapping);
      expect(texture?.wrapT).toBe(ClampToEdgeWrapping);
      expect(texture?.magFilter).toBe(LinearFilter);
      expect(texture?.minFilter).toBe(LinearFilter);
    }
    expect(material.normalScale.x).toBeCloseTo(0.75);
    expect(material.normalScale.y).toBeCloseTo(0.75);
    expect(material.aoMapIntensity).toBeCloseTo(0.6);
    expect(material.emissive.r).toBeCloseTo(0.1);
    expect(material.emissive.g).toBeCloseTo(0.2);
    expect(material.emissive.b).toBeCloseTo(0.3);
  });

  it('recomputes baked tangents after mirrored scene transform changes handedness', async () => {
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };
    const atlas = await createInjectiveAtlas(rawMesh, { textureSize: 16, padding: 1 });
    const baseOptions: Omit<TexturedOutputSceneOptions, 'sceneContext'> = {
      rawMesh,
      atlas,
      image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) },
      additionalTextures: [
        { slot: 'normal' as const, image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) } },
      ],
      materialSettings: {
        alphaMode: 'OPAQUE' as const,
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 1,
        roughnessFactor: 1,
        emissiveFactor: [0, 0, 0],
        normalScale: 1,
        occlusionStrength: 1,
      },
    };
    const identityScene = createTexturedOutputScene({
      ...baseOptions,
      sceneContext: identitySceneTransformContext('Scene'),
    });
    const mirroredScene = createTexturedOutputScene({
      ...baseOptions,
      sceneContext: {
        sceneName: 'Scene',
        nodes: [],
        attachmentWorldMatrix: new Matrix4().makeScale(-1, 1, 1).toArray(),
        supported: true,
      },
    });

    const identityTangent = (firstMesh(identityScene).geometry as BufferGeometry).getAttribute('tangent');
    const mirroredTangent = (firstMesh(mirroredScene).geometry as BufferGeometry).getAttribute('tangent');
    expect(mirroredTangent.getW(0)).toBe(-identityTangent.getW(0));
  });

  it('shares prepared baked output normals across UV seam duplicates in textured output scenes', async () => {
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(0, 0, 1),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const scene = createTexturedOutputScene({
      rawMesh,
      atlas: twoIslandAtlas(),
      image: { width: 64, height: 64, data: new Uint8Array(64 * 64 * 4).fill(255) },
      materialSettings: {
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 1,
        roughnessFactor: 1,
        emissiveFactor: [0, 0, 0],
        normalScale: 1,
        occlusionStrength: 1,
      },
      sceneContext: identitySceneTransformContext('Scene'),
    });

    const geometry = firstMesh(scene).geometry as BufferGeometry;
    const position = geometry.getAttribute('position');
    const normal = geometry.getAttribute('normal');
    const originNormals: Vector3[] = [];
    for (let index = 0; index < position.count; index += 1) {
      if (position.getX(index) === 0 && position.getY(index) === 0 && position.getZ(index) === 0) {
        originNormals.push(new Vector3(normal.getX(index), normal.getY(index), normal.getZ(index)));
      }
    }

    expect(position.count).toBeGreaterThan(rawMesh.positions.length);
    expect(originNormals.length).toBeGreaterThan(1);
    const expectedOriginNormal = new Vector3(1, 0, 1).normalize();
    expect(originNormals.every((normal) => normal.distanceTo(expectedOriginNormal) < 1e-5)).toBe(true);
  });

  it('creates primitive output scenes by replacing meshes without flattening the scene', () => {
    const textured = new Mesh(squareGeometry(), new MeshStandardMaterial({ color: 0xffffff, map: new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat) }));
    const plainMaterial = new MeshStandardMaterial({ color: 0x00ff00, metalness: 0, roughness: 0.75 });
    const plain = new Mesh(squareGeometry(), plainMaterial);
    plain.position.set(2, 0, 0);
    const root = new Group();
    root.name = 'Scene';
    root.add(textured, plain);
    const rawMesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [
      {
        id: 'mesh-0',
        meshOrdinal: 0,
        rawMesh,
        materialMode: 'neutral',
      },
      {
        id: 'mesh-1',
        meshOrdinal: 1,
        rawMesh,
        materialMode: 'preserve',
      },
    ]);

    const meshes = allMeshes(output);
    expect(meshes).toHaveLength(2);
    expect((meshes[0]!.geometry as BufferGeometry).getIndex()?.count).toBe(3);
    expect((meshes[1]!.geometry as BufferGeometry).getIndex()?.count).toBe(3);
    expect(meshes[0]!.material).toBeInstanceOf(MeshStandardMaterial);
    expect((meshes[0]!.material as MeshStandardMaterial).map).toBeNull();
    expect(meshes[1]!.material).not.toBe(plainMaterial);
    expect((meshes[1]!.material as MeshStandardMaterial).color.getHex()).toBe(plainMaterial.color.getHex());
    expect(meshes[1]!.position.x).toBeCloseTo(2);
  });

  it('preserves shared cloned materials for unreplaced primitive output meshes', () => {
    const sharedMaterial = new MeshStandardMaterial({ color: 0x336699, metalness: 0, roughness: 0.5 });
    const first = new Mesh(squareGeometry(), sharedMaterial);
    const second = new Mesh(squareGeometry(), sharedMaterial);
    const replaced = new Mesh(squareGeometry(), new MeshStandardMaterial({ color: 0xff0000 }));
    const root = new Group();
    root.add(first, second, replaced);
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-2',
      meshOrdinal: 2,
      rawMesh,
      materialMode: 'neutral',
    }]);

    const meshes = allMeshes(output);
    const firstOutputMaterial = meshMaterialAt(meshes[0]!, 0) as MeshStandardMaterial;
    const secondOutputMaterial = meshMaterialAt(meshes[1]!, 0) as MeshStandardMaterial;
    expect(firstOutputMaterial).toBe(secondOutputMaterial);
    expect(firstOutputMaterial).not.toBe(sharedMaterial);
    expect(firstOutputMaterial.color.getHex()).toBe(sharedMaterial.color.getHex());
    expect(firstOutputMaterial.metalness).toBeCloseTo(sharedMaterial.metalness);
    expect(firstOutputMaterial.roughness).toBeCloseTo(sharedMaterial.roughness);
  });

  it('preserves shared cloned materials for preserved primitive replacements', () => {
    const sharedMaterial = new MeshStandardMaterial({ color: 0x225533, metalness: 0.25, roughness: 0.4 });
    const root = new Group();
    root.add(
      new Mesh(squareGeometry(), sharedMaterial),
      new Mesh(squareGeometry(), sharedMaterial),
    );
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [
      {
        id: 'mesh-0',
        meshOrdinal: 0,
        sourceMaterial: sharedMaterial,
        rawMesh,
        materialMode: 'preserve',
      },
      {
        id: 'mesh-1',
        meshOrdinal: 1,
        sourceMaterial: sharedMaterial,
        rawMesh,
        materialMode: 'preserve',
      },
    ]);

    const meshes = allMeshes(output);
    const firstOutputMaterial = meshMaterialAt(meshes[0]!, 0) as MeshStandardMaterial;
    const secondOutputMaterial = meshMaterialAt(meshes[1]!, 0) as MeshStandardMaterial;
    expect(firstOutputMaterial).toBe(secondOutputMaterial);
    expect(firstOutputMaterial).not.toBe(sharedMaterial);
    expect(firstOutputMaterial.color.getHex()).toBe(sharedMaterial.color.getHex());
    expect(firstOutputMaterial.metalness).toBeCloseTo(sharedMaterial.metalness);
    expect(firstOutputMaterial.roughness).toBeCloseTo(sharedMaterial.roughness);
  });

  it('preserves cloned source material texture maps and transferred UV attributes in primitive output scenes', () => {
    const sourceMaterial = materialWithStandardTextureSlots();
    const root = new Group();
    root.add(new Mesh(squareGeometryWithUv1(), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };
    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      sourceMaterial,
      rawMesh,
      materialMode: 'preserve',
      transferredAttributes: { vertices: weldedTransferredAttributes().vertices.slice(0, 3) },
    }]);

    const mesh = firstMesh(output);
    const geometry = mesh.geometry as BufferGeometry;
    const material = mesh.material as MeshStandardMaterial;
    expect(material).not.toBe(sourceMaterial);
    expect(material.map).toBe(sourceMaterial.map);
    expect(material.normalMap).toBe(sourceMaterial.normalMap);
    expect(material.metalnessMap).toBe(sourceMaterial.metalnessMap);
    expect(material.roughnessMap).toBe(sourceMaterial.roughnessMap);
    expect(material.aoMap).toBe(sourceMaterial.aoMap);
    expect(material.emissiveMap).toBe(sourceMaterial.emissiveMap);
    expect(geometry.getAttribute('position').count).toBe(rawMesh.positions.length);
    expect(geometry.getAttribute('uv').count).toBe(3);
    expect(geometry.getAttribute('uv1').count).toBe(3);
  });

  it('derives preserved browser tangents when the preserved material references a normal map', () => {
    const sourceMaterial = materialWithStandardTextureSlots();
    const root = new Group();
    root.add(new Mesh(squareGeometryWithUv1(), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const transferredAttributes = weldedTransferredAttributes();
    for (const vertex of transferredAttributes.vertices) vertex.normal = new Vector3(0, 0, 1);

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      sourceMaterial,
      rawMesh,
      materialMode: 'preserve',
      transferredAttributes,
    }]);

    const geometry = firstMesh(output).geometry as BufferGeometry;
    expectAllTangentHandedness(geometry, -1);
  });

  it('derives preserved browser tangents from source tangent provenance without a normal map', () => {
    const sourceMaterial = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const root = new Group();
    root.add(new Mesh(squareGeometry(), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };
    const transferredAttributes = weldedTransferredAttributes(false);
    transferredAttributes.hasSourceTangents = true;
    for (const vertex of transferredAttributes.vertices) vertex.normal = new Vector3(0, 0, 1);

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      sourceMaterial,
      rawMesh,
      materialMode: 'preserve',
      transferredAttributes,
    }]);

    const geometry = firstMesh(output).geometry as BufferGeometry;
    expectAllTangentHandedness(geometry, -1);
  });

  it('omits preserved browser tangents when TEXCOORD_0 is unavailable', () => {
    const normalMap = new DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1, RGBAFormat);
    normalMap.channel = 1;
    normalMap.needsUpdate = true;
    const sourceMaterial = new MeshStandardMaterial({ normalMap, metalness: 0, roughness: 0.5 });
    const root = new Group();
    root.add(new Mesh(squareGeometryWithUv1(), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      sourceMaterial,
      rawMesh,
      materialMode: 'preserve',
      transferredAttributes: weldedTransferredUv1OnlyAttributes(),
    }]);

    const geometry = firstMesh(output).geometry as BufferGeometry;
    expect(geometry.getAttribute('uv1').count).toBe(4);
    expect(geometry.getAttribute('uv')).toBeUndefined();
    expect(geometry.getAttribute('tangent')).toBeUndefined();
  });

  it('does not request preserved browser tangents for missing source materials', () => {
    const root = new Group();
    const sourceMesh = new Mesh<BufferGeometry, Material>(squareGeometry(), new MeshStandardMaterial());
    sourceMesh.material = null as unknown as Material;
    root.add(sourceMesh);
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      sourceMaterial: null,
      rawMesh,
      materialMode: 'preserve',
      transferredAttributes: weldedTransferredAttributes(false),
    }]);

    expect((firstMesh(output).geometry as BufferGeometry).getAttribute('tangent')).toBeUndefined();
  });

  it('emits transferred COLOR_0 and enables preserved vertex color materials in primitive output scenes', () => {
    const sourceMaterial = new MeshStandardMaterial({ color: 0xffffff, vertexColors: true, metalness: 0, roughness: 0.5 });
    const root = new Group();
    root.add(new Mesh(vertexColorTriangleGeometry(4), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      sourceMaterial,
      rawMesh,
      materialMode: 'preserve',
      transferredAttributes: {
        colorItemSize: 4,
        vertices: [
          { uvSets: [], normal: new Vector3(0, 0, 1), color: new Vector4(1, 0, 0, 0.25) },
          { uvSets: [], normal: new Vector3(0, 0, 1), color: new Vector4(0, 1, 0, 0.5) },
          { uvSets: [], normal: new Vector3(0, 0, 1), color: new Vector4(0, 0, 1, 0.75) },
        ],
      },
    }]);

    const mesh = firstMesh(output);
    const color = (mesh.geometry as BufferGeometry).getAttribute('color');
    expect(color.itemSize).toBe(4);
    expect(color.count).toBe(3);
    expect(color.getX(0)).toBeCloseTo(1);
    expect(color.getY(1)).toBeCloseTo(1);
    expect(color.getZ(2)).toBeCloseTo(1);
    expect(color.getW(2)).toBeCloseTo(0.75);
    expect((mesh.material as MeshStandardMaterial).vertexColors).toBe(true);
  });

  it('keeps preserved material clones separate for colored and uncolored replacement geometry', () => {
    const sharedMaterial = new MeshStandardMaterial({ color: 0xffffff, vertexColors: true, metalness: 0.1, roughness: 0.6 });
    const root = new Group();
    root.add(
      new Mesh(squareGeometry(), sharedMaterial),
      new Mesh(squareGeometry(), sharedMaterial),
    );
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [
      {
        id: 'colored',
        meshOrdinal: 0,
        sourceMaterial: sharedMaterial,
        rawMesh,
        materialMode: 'preserve',
        transferredAttributes: {
          colorItemSize: 3,
          vertices: [
            { uvSets: [], normal: new Vector3(0, 0, 1), color: new Vector4(1, 0, 0, 1) },
            { uvSets: [], normal: new Vector3(0, 0, 1), color: new Vector4(0, 1, 0, 1) },
            { uvSets: [], normal: new Vector3(0, 0, 1), color: new Vector4(0, 0, 1, 1) },
          ],
        },
      },
      {
        id: 'uncolored',
        meshOrdinal: 1,
        sourceMaterial: sharedMaterial,
        rawMesh,
        materialMode: 'preserve',
      },
    ]);

    const meshes = allMeshes(output);
    const coloredGeometry = meshes[0]!.geometry as BufferGeometry;
    const uncoloredGeometry = meshes[1]!.geometry as BufferGeometry;
    const coloredMaterial = meshMaterialAt(meshes[0]!, 0) as MeshStandardMaterial;
    const uncoloredMaterial = meshMaterialAt(meshes[1]!, 0) as MeshStandardMaterial;

    expect(coloredGeometry.getAttribute('color')?.itemSize).toBe(3);
    expect(coloredMaterial.vertexColors).toBe(true);
    expect(uncoloredGeometry.getAttribute('color')).toBeUndefined();
    expect(uncoloredMaterial.vertexColors).toBe(false);
    expect(coloredMaterial).not.toBe(uncoloredMaterial);
  });

  it('keeps vertex colors enabled on retained cloned meshes with COLOR_0 geometry', () => {
    const sharedMaterial = new MeshStandardMaterial({ color: 0xffffff, vertexColors: true, metalness: 0.1, roughness: 0.6 });
    const root = new Group();
    root.add(
      new Mesh(vertexColorTriangleGeometry(3), sharedMaterial),
      new Mesh(squareGeometry(), sharedMaterial),
    );
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'replaced',
      meshOrdinal: 1,
      sourceMaterial: sharedMaterial,
      rawMesh,
      materialMode: 'preserve',
    }]);

    const meshes = allMeshes(output);
    const retainedGeometry = meshes[0]!.geometry as BufferGeometry;
    const retainedMaterial = meshMaterialAt(meshes[0]!, 0) as MeshStandardMaterial;
    const replacedMaterial = meshMaterialAt(meshes[1]!, 0) as MeshStandardMaterial;

    expect(retainedGeometry.getAttribute('color')?.count).toBe(3);
    expect(retainedMaterial.vertexColors).toBe(true);
    expect(replacedMaterial.vertexColors).toBe(false);
    expect(retainedMaterial).not.toBe(replacedMaterial);
  });

  it('adds tangents to baked primitive replacement geometry when a normal map is baked', async () => {
    const root = new Group();
    root.add(new Mesh(squareGeometry(), new MeshStandardMaterial()));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };
    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      rawMesh,
      materialMode: 'baked',
      atlas: await createInjectiveAtlas(rawMesh, { textureSize: 16, padding: 1 }),
      image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) },
      additionalTextures: [
        { slot: 'normal', image: { width: 1, height: 1, data: new Uint8Array([128, 128, 255, 255]) } },
      ],
      materialSettings: {
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 1,
        roughnessFactor: 1,
        normalScale: 1,
        occlusionStrength: 1,
        emissiveFactor: [0, 0, 0],
      },
    }]);

    const geometry = firstMesh(output).geometry as BufferGeometry;
    const tangent = geometry.getAttribute('tangent');
    expect(tangent?.itemSize).toBe(4);
    expect(tangent?.count).toBe(geometry.getAttribute('position').count);
  });

  it('does not emit COLOR_0 for baked primitive replacement geometry without atlas colors', async () => {
    const sourceMaterial = new MeshStandardMaterial({ color: 0xffffff, vertexColors: true, metalness: 0, roughness: 0.5 });
    const root = new Group();
    root.add(new Mesh(vertexColorTriangleGeometry(3), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      rawMesh,
      materialMode: 'baked',
      atlas: await createInjectiveAtlas(rawMesh, { textureSize: 16, padding: 1 }),
      image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) },
      materialSettings: {
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 1,
        roughnessFactor: 1,
        normalScale: 1,
        occlusionStrength: 1,
        emissiveFactor: [0, 0, 0],
      },
    }]);

    const geometry = firstMesh(output).geometry as BufferGeometry;
    expect(geometry.getAttribute('color')).toBeUndefined();
    expect(geometry.getAttribute('normal').count).toBe(3);
    expect(geometry.getAttribute('uv').count).toBe(3);
  });

  it('emits inherited COLOR_0 and enables vertex colors for baked primitive replacement geometry', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const root = new Group();
    root.add(new Mesh(squareGeometry(), material));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      rawMesh,
      transferredAttributes: weldedTransferredAttributes(),
      materialMode: 'baked',
      atlas: twoIslandAtlas(),
      image: { width: 64, height: 64, data: new Uint8Array(64 * 64 * 4).fill(255) },
      materialSettings: {
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: false,
        metallicFactor: 1,
        roughnessFactor: 1,
        normalScale: 1,
        occlusionStrength: 1,
        emissiveFactor: [0, 0, 0],
      },
    }]);

    const mesh = firstMesh(output);
    const geometry = mesh.geometry as BufferGeometry;
    const color = geometry.getAttribute('color');

    expect(color?.itemSize).toBe(4);
    expect(color?.count).toBe(geometry.getAttribute('position').count);
    expect(color?.getX(0)).toBeCloseTo(1);
    expect((mesh.material as MeshStandardMaterial).vertexColors).toBe(true);
  });

  it('drops optional partial UV sets from primitive output scenes', () => {
    const sourceMaterial = new MeshStandardMaterial({ color: 0x00ff00, roughness: 0.5, metalness: 0 });
    const root = new Group();
    root.add(new Mesh(squareGeometryWithUv1(), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const output = createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      rawMesh,
      materialMode: 'preserve',
      sourceMaterial,
      transferredAttributes: {
        vertices: weldedTransferredAttributes().vertices.map((vertex, index) => ({
          uvSets: index === 3 ? vertex.uvSets.filter((uvSet) => uvSet.texCoord === 0) : vertex.uvSets,
          ...(vertex.normal ? { normal: vertex.normal } : {}),
        })),
      },
    }]);

    const geometry = firstMesh(output).geometry as BufferGeometry;
    expect(geometry.getAttribute('position').count).toBe(rawMesh.positions.length);
    expect(geometry.getAttribute('uv').count).toBe(4);
    expect(geometry.getAttribute('uv1')).toBeUndefined();
  });

  it('throws when transferred face attributes omit a material-required UV set', () => {
    const sourceMaterial = materialWithStandardTextureSlots();
    const root = new Group();
    root.add(new Mesh(squareGeometryWithUv1(), sourceMaterial));
    const rawMesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    expect(() => createPrimitiveOutputScene(root, [{
      id: 'mesh-0',
      meshOrdinal: 0,
      rawMesh,
      materialMode: 'preserve',
      sourceMaterial,
      transferredAttributes: {
        vertices: weldedTransferredAttributes().vertices.map((vertex, index) => ({
          uvSets: index === 3 ? vertex.uvSets.filter((uvSet) => uvSet.texCoord === 0) : vertex.uvSets,
          ...(vertex.normal ? { normal: vertex.normal } : {}),
        })),
      },
    }])).toThrow('Missing transferred TEXCOORD_1 coordinates for output vertex 3.');
  });

  it('creates grouped output scenes by removing source meshes once and inserting replacements under their group parent', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const parent = new Group();
    parent.name = 'Parent';
    parent.position.set(10, 0, 0);
    const meshA = new Mesh(squareGeometry(), material);
    meshA.position.set(1, 0, 0);
    const meshB = new Mesh(squareGeometry(), material);
    meshB.position.set(2, 0, 0);
    parent.add(meshA, meshB);
    const root = new Group();
    root.name = 'Scene';
    root.add(parent);
    const model = await extractBrowserModelFromObject(root);
    const entry = model.primitiveEntryGroups['material-parent'][0]!;

    const output = createPrimitiveOutputScene(root, [{
      id: entry.id,
      meshOrdinal: entry.meshOrdinal,
      sourceMeshOrdinals: entry.sourceMeshOrdinals,
      parentObjectOrdinal: entry.parentObjectOrdinal,
      rawMesh: {
        positions: [new Vector3(1, 0, 0), new Vector3(2, 0, 0), new Vector3(1, 1, 0)],
        faces: [[0, 1, 2] as [number, number, number]],
      },
      materialMode: 'preserve',
      sourceMaterial: entry.sourceMaterial,
    }]);

    const outputParent = output.children[0] as Group;
    const meshes = allMeshes(outputParent);
    expect(meshes).toHaveLength(1);
    expect(outputParent.children).toHaveLength(1);
    expect(meshes[0]!.position.x).toBeCloseTo(0);
    expect((meshes[0]!.geometry as BufferGeometry).getAttribute('position').getX(0)).toBeCloseTo(1);
    expect(meshes[0]!.material).not.toBe(material);
    expect((meshes[0]!.material as MeshStandardMaterial).roughness).toBeCloseTo(material.roughness);
  });

  it('does not copy source mesh transforms for material-parent replacements with misleading ids', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const parent = new Group();
    parent.name = 'Parent';
    parent.position.set(10, 0, 0);
    const mesh = new Mesh(squareGeometry(), material);
    mesh.position.set(2, 0, 0);
    parent.add(mesh);
    const root = new Group();
    root.name = 'Scene';
    root.add(parent);
    const model = await extractBrowserModelFromObject(root);
    const entry = model.primitiveEntryGroups['material-parent'][0]!;

    const output = createPrimitiveOutputScene(root, [{
      id: 'none-misleading-material-parent',
      meshOrdinal: entry.meshOrdinal,
      sourceMeshOrdinals: entry.sourceMeshOrdinals,
      parentObjectOrdinal: entry.parentObjectOrdinal,
      rawMesh: {
        positions: [new Vector3(2, 0, 0), new Vector3(3, 0, 0), new Vector3(2, 1, 0)],
        faces: [[0, 1, 2] as [number, number, number]],
      },
      materialMode: 'preserve',
      sourceMaterial: entry.sourceMaterial,
    }]);

    const outputParent = output.children[0] as Group;
    const meshes = allMeshes(outputParent);
    expect(meshes).toHaveLength(1);
    expect(meshes[0]!.position.x).toBeCloseTo(0);
    expect((meshes[0]!.geometry as BufferGeometry).getAttribute('position').getX(0)).toBeCloseTo(2);
  });

  it('creates material-grouped output scenes under the root', async () => {
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const parentA = new Group();
    parentA.position.set(10, 0, 0);
    const parentB = new Group();
    parentB.position.set(20, 0, 0);
    parentA.add(new Mesh(squareGeometry(), material));
    parentB.add(new Mesh(squareGeometry(), material));
    const root = new Group();
    root.name = 'Scene';
    root.add(parentA, parentB);
    const model = await extractBrowserModelFromObject(root);
    const entry = model.primitiveEntryGroups.material[0]!;

    const output = createPrimitiveOutputScene(root, [{
      id: entry.id,
      meshOrdinal: entry.meshOrdinal,
      sourceMeshOrdinals: entry.sourceMeshOrdinals,
      parentObjectOrdinal: entry.parentObjectOrdinal,
      rawMesh: {
        positions: [new Vector3(10, 0, 0), new Vector3(20, 0, 0), new Vector3(10, 1, 0)],
        faces: [[0, 1, 2] as [number, number, number]],
      },
      materialMode: 'preserve',
      sourceMaterial: entry.sourceMaterial,
    }]);

    expect(allMeshes(output)).toHaveLength(1);
    expect(output.children.some((child) => (child as Mesh).isMesh)).toBe(true);
    expect((firstMesh(output).geometry as BufferGeometry).getAttribute('position').getX(0)).toBeCloseTo(10);
  });

  it('creates multiple none-grouped output replacements from one transformed source mesh', async () => {
    const red = new MeshStandardMaterial({ color: 0xff0000, metalness: 0, roughness: 0.5 });
    const green = new MeshStandardMaterial({ color: 0x00ff00, metalness: 0, roughness: 0.25 });
    const mesh = new Mesh(squareGeometry(), [red, green]);
    mesh.name = 'Panel';
    mesh.position.set(2, 3, 4);
    mesh.rotation.z = Math.PI / 5;
    mesh.scale.set(1.5, 2, 0.75);
    const root = new Group();
    root.add(mesh);
    const model = await extractBrowserModelFromObject(root);
    const entries = model.primitiveEntryGroups.none;

    const output = createPrimitiveOutputScene(root, entries.map((entry) => ({
      id: entry.id,
      meshOrdinal: entry.meshOrdinal,
      sourceMeshOrdinals: entry.sourceMeshOrdinals,
      parentObjectOrdinal: entry.parentObjectOrdinal,
      ...(entry.preserveSourceMeshTransform === true ? { preserveSourceMeshTransform: true } : {}),
      rawMesh: entry.rawMesh,
      materialMode: 'preserve' as const,
      sourceMaterial: entry.sourceMaterial,
    })));

    const meshes = allMeshes(output);
    expect(entries).toHaveLength(2);
    expect(meshes).toHaveLength(2);
    for (const outputMesh of meshes) {
      expect(outputMesh.position.x).toBeCloseTo(mesh.position.x);
      expect(outputMesh.position.y).toBeCloseTo(mesh.position.y);
      expect(outputMesh.position.z).toBeCloseTo(mesh.position.z);
      expect(outputMesh.quaternion.x).toBeCloseTo(mesh.quaternion.x);
      expect(outputMesh.quaternion.y).toBeCloseTo(mesh.quaternion.y);
      expect(outputMesh.quaternion.z).toBeCloseTo(mesh.quaternion.z);
      expect(outputMesh.quaternion.w).toBeCloseTo(mesh.quaternion.w);
      expect(outputMesh.scale.x).toBeCloseTo(mesh.scale.x);
      expect(outputMesh.scale.y).toBeCloseTo(mesh.scale.y);
      expect(outputMesh.scale.z).toBeCloseTo(mesh.scale.z);
    }
    expect((meshes[0]!.material as MeshStandardMaterial).color.getHex()).toBe(red.color.getHex());
    expect((meshes[1]!.material as MeshStandardMaterial).color.getHex()).toBe(green.color.getHex());
    expect((meshes[0]!.geometry as BufferGeometry).getAttribute('position').getX(0)).toBeCloseTo(0);
  });
});
