import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import { decodeImage } from '../src/local/imageCodecs';
import { GltfTransformPrimitiveSourceAdapter } from '../src/local/gltfTransformPrimitiveSourceAdapter';
import type { PrimitiveGroupingMode } from '../src/pipeline/options';
import type { ProcessablePrimitiveEntry } from '../src/pipeline/sceneProcessing';
import type { SourceFaceAttributes, SourceMaterialTextureInfo, SourceTexture, TextureSampler } from '../src/texture/types';
import { parseGlbArrayBuffer } from '../src/web/browserGltfIo';
import {
  writePublicNormalMapGlb,
  writePublicTexturedMultiPrimitiveGlb,
  writePublicVertexColorGlb,
} from './fixtures/publicGltfFixtures';

type NormalizedVector2 = [number, number];
type NormalizedVector3 = [number, number, number];
type NormalizedVector4 = [number, number, number, number];

interface NormalizedFaceUvSet {
  texCoord: number;
  uvs: [NormalizedVector2, NormalizedVector2, NormalizedVector2];
}

interface NormalizedFaceAttributes {
  materialId: number;
  uvSets: NormalizedFaceUvSet[];
  normalCorners?: [NormalizedVector3, NormalizedVector3, NormalizedVector3];
  tangentCorners?: [NormalizedVector4, NormalizedVector4, NormalizedVector4];
  colorCorners?: [NormalizedVector4, NormalizedVector4, NormalizedVector4];
  colorItemSize?: number;
  normalMapYScale?: number;
}

interface NormalizedEntry {
  faceCount: number;
  vertexCount: number;
  positions: NormalizedVector3[];
  vertexSignatures: string[];
  faces: [number, number, number][];
  faceAttributes: NormalizedFaceAttributes[];
  materialName: string;
  baseColorFactor: [number, number, number, number];
  baseColorTexture?: string;
  alphaMode: string;
  alphaCutoff: number;
  doubleSided: boolean;
  metallicFactor: number;
  roughnessFactor: number;
  normalScale: number;
  occlusionStrength: number;
  emissiveFactor: [number, number, number];
  textureSlots: string[];
  textureSlotMetadata: string[];
  uvSets: number[];
  bakeable: boolean;
  requiresAttributeTransfer: boolean;
}

function roundCoordinate(value: number): number {
  const rounded = Number(value.toFixed(6));
  return Object.is(rounded, -0) ? 0 : rounded;
}

function compareVector3(a: NormalizedVector3, b: NormalizedVector3): number {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

function positionKey(position: NormalizedVector3): string {
  return position.join(',');
}

function vertexKey(position: NormalizedVector3, incidents: readonly string[]): string {
  return `${positionKey(position)}|${incidents.join('|')}`;
}

function normalizeVector3(vector: { x: number; y: number; z: number }): NormalizedVector3 {
  return [
    roundCoordinate(vector.x),
    roundCoordinate(vector.y),
    roundCoordinate(vector.z),
  ];
}

function normalizeVector4(vector: { x: number; y: number; z: number; w: number }): NormalizedVector4 {
  return [
    roundCoordinate(vector.x),
    roundCoordinate(vector.y),
    roundCoordinate(vector.z),
    roundCoordinate(vector.w),
  ];
}

function normalizeSampler(sampler: TextureSampler): string {
  return `${sampler.wrapS}:${sampler.wrapT}:${sampler.filter}`;
}

function normalizeTexture(texture: SourceTexture): string {
  const name = texture.name ?? '';
  const mimeType = texture.mimeType ?? '';
  return `${texture.texCoord}:${normalizeSampler(texture.sampler)}:${texture.image !== undefined}:${name}:${mimeType}`;
}

function normalizeTextureSlot(slot: SourceMaterialTextureInfo): string {
  const name = slot.name ?? '';
  const mimeType = slot.mimeType ?? '';
  return `${slot.slot}:${slot.texCoord}:${slot.hasImage}:${normalizeSampler(slot.sampler)}:${slot.image !== undefined}:${name}:${mimeType}`;
}

function normalizeOptionalVector3(vector: NormalizedVector3 | undefined): string {
  return vector ? vector.join(',') : '';
}

function normalizeOptionalVector4(vector: NormalizedVector4 | undefined): string {
  return vector ? vector.join(',') : '';
}

function normalizeFaceAttributes(attributes: SourceFaceAttributes): NormalizedFaceAttributes {
  return {
    materialId: attributes.materialId,
    uvSets: attributes.uvSets
      .map((uvSet) => ({
        texCoord: uvSet.texCoord,
        uvs: uvSet.uvs.map((uv) => [
          roundCoordinate(uv.x),
          roundCoordinate(uv.y),
        ]) as [NormalizedVector2, NormalizedVector2, NormalizedVector2],
      }))
      .sort((a, b) => a.texCoord - b.texCoord),
    ...(attributes.normalCorners
      ? { normalCorners: attributes.normalCorners.map(normalizeVector3) as [NormalizedVector3, NormalizedVector3, NormalizedVector3] }
      : {}),
    ...(attributes.tangentCorners
      ? { tangentCorners: attributes.tangentCorners.map(normalizeVector4) as [NormalizedVector4, NormalizedVector4, NormalizedVector4] }
      : {}),
    ...(attributes.colorCorners && attributes.colorItemSize
      ? {
        colorCorners: attributes.colorCorners.map(normalizeVector4) as [NormalizedVector4, NormalizedVector4, NormalizedVector4],
        colorItemSize: attributes.colorItemSize,
      }
      : {}),
    ...(attributes.normalMapYScale !== undefined ? { normalMapYScale: roundCoordinate(attributes.normalMapYScale) } : {}),
  };
}

function cornerSignature(
  attributes: NormalizedFaceAttributes,
  faceIndex: number,
  cornerIndex: number,
): string {
  const uvSets = attributes.uvSets
    .map((uvSet) => `${uvSet.texCoord}:${uvSet.uvs[cornerIndex]!.join(',')}`)
    .join(';');
  return [
    faceIndex,
    cornerIndex,
    attributes.materialId,
    uvSets,
    normalizeOptionalVector3(attributes.normalCorners?.[cornerIndex]),
    normalizeOptionalVector4(attributes.tangentCorners?.[cornerIndex]),
    normalizeOptionalVector4(attributes.colorCorners?.[cornerIndex]),
    attributes.colorItemSize ?? '',
    attributes.normalMapYScale ?? '',
  ].join('/');
}

function vertexIncidentSignatures(
  faces: readonly [number, number, number][],
  faceAttributes: readonly NormalizedFaceAttributes[],
  vertexCount: number,
): string[][] {
  const incidents = Array.from({ length: vertexCount }, () => [] as string[]);
  for (let faceIndex = 0; faceIndex < faces.length; faceIndex += 1) {
    const face = faces[faceIndex]!;
    const attributes = faceAttributes[faceIndex];
    if (!attributes) throw new Error(`Face ${faceIndex} is missing attributes.`);
    for (let cornerIndex = 0; cornerIndex < face.length; cornerIndex += 1) {
      const vertexId = face[cornerIndex]!;
      const vertexIncidents = incidents[vertexId];
      if (!vertexIncidents) throw new Error(`Face ${faceIndex} references missing vertex ${vertexId}.`);
      vertexIncidents.push(cornerSignature(attributes, faceIndex, cornerIndex));
    }
  }
  for (const vertexIncidents of incidents) vertexIncidents.sort();
  return incidents;
}

function normalizeEntry(entry: ProcessablePrimitiveEntry): NormalizedEntry {
  const textured = entry.texturedRawMesh;
  if (!textured) throw new Error(`Entry ${entry.id} is missing textured metadata.`);
  const material = textured.materials[0]!;
  const roundedPositions = entry.rawMesh.positions.map((position) => [
    roundCoordinate(position.x),
    roundCoordinate(position.y),
    roundCoordinate(position.z),
  ] as NormalizedVector3);
  const faceAttributes = textured.faceAttributes.map(normalizeFaceAttributes);
  const vertexIncidents = vertexIncidentSignatures(entry.rawMesh.faces, faceAttributes, roundedPositions.length);
  const vertexSignatures = roundedPositions.map((position, vertexId) => vertexKey(position, vertexIncidents[vertexId]!));
  const canonicalVertexIds = new Map<string, number>();
  for (const signature of vertexSignatures.slice().sort()) {
    if (!canonicalVertexIds.has(signature)) canonicalVertexIds.set(signature, canonicalVertexIds.size);
  }
  return {
    faceCount: entry.rawMesh.faces.length,
    vertexCount: entry.rawMesh.positions.length,
    positions: roundedPositions.slice().sort(compareVector3),
    vertexSignatures: vertexSignatures.slice().sort(),
    faces: entry.rawMesh.faces.map((face) => face.map((vertexId) => {
      const signature = vertexSignatures[vertexId];
      if (!signature) throw new Error(`Entry ${entry.id} face references missing vertex ${vertexId}.`);
      const canonicalId = canonicalVertexIds.get(signature);
      if (canonicalId === undefined) throw new Error(`Entry ${entry.id} has an unregistered vertex position.`);
      return canonicalId;
    }) as [number, number, number]),
    faceAttributes,
    materialName: material.name === 'material' ? 'default-material' : material.name,
    baseColorFactor: material.baseColorFactor.map(roundCoordinate) as [number, number, number, number],
    ...(material.baseColorTexture ? { baseColorTexture: normalizeTexture(material.baseColorTexture) } : {}),
    alphaMode: material.alphaMode,
    alphaCutoff: roundCoordinate(material.alphaCutoff),
    doubleSided: material.doubleSided,
    metallicFactor: roundCoordinate(material.metallicFactor),
    roughnessFactor: roundCoordinate(material.roughnessFactor),
    normalScale: roundCoordinate(material.normalScale),
    occlusionStrength: roundCoordinate(material.occlusionStrength),
    emissiveFactor: material.emissiveFactor.map(roundCoordinate) as [number, number, number],
    textureSlots: material.textureSlots
      .map((slot) => `${slot.slot}:${slot.texCoord}:${slot.hasImage}`)
      .sort(),
    textureSlotMetadata: material.textureSlots.map(normalizeTextureSlot).sort(),
    uvSets: Array.from(new Set(
      textured.faceAttributes.flatMap((attributes) => attributes.uvSets.map((set) => set.texCoord)),
    )).sort((a, b) => a - b),
    bakeable: entry.bakeable,
    requiresAttributeTransfer: entry.requiresAttributeTransfer === true,
  };
}

function sortNormalized(entries: NormalizedEntry[]): NormalizedEntry[] {
  return entries.slice().sort((a: NormalizedEntry, b: NormalizedEntry) => {
    const material = a.materialName.localeCompare(b.materialName);
    if (material !== 0) return material;
    if (a.faceCount !== b.faceCount) return a.faceCount - b.faceCount;
    if (a.vertexCount !== b.vertexCount) return a.vertexCount - b.vertexCount;
    const textureSlots = a.textureSlots.join('|').localeCompare(b.textureSlots.join('|'));
    if (textureSlots !== 0) return textureSlots;
    return JSON.stringify(a).localeCompare(JSON.stringify(b));
  });
}

async function withBrowserImageLoading<T>(run: () => Promise<T>): Promise<T> {
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
  try {
    return await run();
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

const GROUPING_MODES = ['material-parent', 'material', 'none'] as const satisfies readonly PrimitiveGroupingMode[];
const FIXTURE_KINDS = ['textured', 'normal-map', 'authored-tangent', 'vertex-color'] as const;
type FixtureKind = typeof FIXTURE_KINDS[number];

const EQUIVALENCE_CASES = FIXTURE_KINDS.flatMap((fixtureKind) => (
  GROUPING_MODES.map((groupingMode) => ({ fixtureKind, groupingMode }))
));

let fixtureDir: string | undefined;
let fixturePaths: Record<FixtureKind, string>;

beforeAll(async () => {
  fixtureDir = await mkdtemp(join(tmpdir(), 'mesh-adapter-equivalence-fixtures-'));
  fixturePaths = {
    textured: join(fixtureDir, 'textured.glb'),
    'normal-map': join(fixtureDir, 'normal-map.glb'),
    'authored-tangent': join(fixtureDir, 'authored-tangent.glb'),
    'vertex-color': join(fixtureDir, 'vertex-color.glb'),
  };
  await writePublicTexturedMultiPrimitiveGlb(fixturePaths.textured);
  await writePublicNormalMapGlb(fixturePaths['normal-map']);
  await writePublicNormalMapGlb(fixturePaths['authored-tangent'], { authoredTangents: true, negativeHandedness: true });
  await writePublicVertexColorGlb(fixturePaths['vertex-color']);
});

afterAll(async () => {
  if (fixtureDir) await rm(fixtureDir, { recursive: true, force: true });
});

async function readFixtureArrayBuffer(fixturePath: string): Promise<ArrayBuffer> {
  const bytes = await readFile(fixturePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

function seamEntry(id: string, faces: [number, number, number][]): ProcessablePrimitiveEntry {
  return {
    id,
    rawMesh: {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
      ],
      faces,
    },
    texturedRawMesh: {
      rawMesh: {
        positions: [
          new Vector3(0, 0, 0),
          new Vector3(0, 0, 0),
          new Vector3(1, 0, 0),
          new Vector3(0, 1, 0),
        ],
        faces,
      },
      faceAttributes: [
        {
          materialId: 0,
          uvSets: [{ texCoord: 0, uvs: [new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1)] }],
        },
        {
          materialId: 0,
          uvSets: [{ texCoord: 0, uvs: [new Vector2(0.5, 0), new Vector2(0, 1), new Vector2(1, 0)] }],
        },
      ],
      materials: [{
        name: 'seam',
        baseColorFactor: [1, 1, 1, 1],
        textureSlots: [],
        alphaMode: 'OPAQUE',
        alphaCutoff: 0.5,
        doubleSided: false,
        emissiveFactor: [0, 0, 0],
        metallicFactor: 1,
        roughnessFactor: 1,
        normalScale: 1,
        occlusionStrength: 1,
      }],
    },
    bakeable: false,
  };
}

describe('primitive adapter equivalence', () => {
  it('preserves duplicate-position seam topology in normalized entries', () => {
    const splitSeam = normalizeEntry(seamEntry('split', [[0, 2, 3], [1, 3, 2]]));
    const collapsedSeam = normalizeEntry(seamEntry('collapsed', [[0, 2, 3], [0, 3, 2]]));

    expect(splitSeam.faces).not.toEqual(collapsedSeam.faces);
  });

  it('covers normal-map scale, authored tangent, and vertex color fixture fields', async () => {
    await withBrowserImageLoading(async () => {
      const normalMapScaleAsset = await parseGlbArrayBuffer(await readFixtureArrayBuffer(fixturePaths['normal-map']));
      const normalMapScaleExtraction = await normalMapScaleAsset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });
      const tangentAsset = await parseGlbArrayBuffer(await readFixtureArrayBuffer(fixturePaths['authored-tangent']));
      const tangentExtraction = await tangentAsset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });

      expect(normalMapScaleExtraction.entries.some((entry) => (
        normalizeEntry(entry).faceAttributes.some((attributes) => attributes.normalMapYScale !== undefined)
      ))).toBe(true);
      expect(tangentExtraction.entries.some((entry) => (
        normalizeEntry(entry).faceAttributes.some((attributes) => attributes.tangentCorners !== undefined)
      ))).toBe(true);
      const vertexColorAsset = await parseGlbArrayBuffer(await readFixtureArrayBuffer(fixturePaths['vertex-color']));
      const vertexColorExtraction = await vertexColorAsset.extractGroups({
        groupingMode: 'material-parent',
        mode: 'geometry-with-texture-metadata',
      });
      expect(vertexColorExtraction.entries.some((entry) => (
        normalizeEntry(entry).faceAttributes.some((attributes) => attributes.colorCorners !== undefined)
      ))).toBe(true);
    });
  });

  it.each(EQUIVALENCE_CASES)(
    'extracts equivalent $groupingMode entries for public $fixtureKind fixture through browser and glTF-Transform adapters',
    async ({ fixtureKind, groupingMode }) => {
      const fixturePath = fixturePaths[fixtureKind];
      await withBrowserImageLoading(async () => {
        const browserAsset = await parseGlbArrayBuffer(await readFixtureArrayBuffer(fixturePath));
        const browserExtraction = await browserAsset.extractGroups({
          groupingMode,
          mode: 'geometry-with-texture-metadata',
        });
        const cliAdapter = await GltfTransformPrimitiveSourceAdapter.read(fixturePath);
        const cliExtraction = await cliAdapter.extractGroups({
          groupingMode,
          mode: 'geometry-with-texture-metadata',
        });

        const browserEntries = sortNormalized(browserExtraction.entries.map(normalizeEntry));
        const cliEntries = sortNormalized(cliExtraction.entries.map(normalizeEntry));

        expect(browserEntries).toEqual(cliEntries);
      });
    },
  );
});
