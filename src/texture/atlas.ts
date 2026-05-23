import * as watlas from 'watlas';
import { Vector2 } from 'three';
import type { RawMesh } from '../simplification/types';
import type { AtlasLayout } from './types';

export interface WatlasMeshBuffers {
  positions: Float32Array;
  indices: Uint32Array;
}

export interface AtlasOptions {
  textureSize: number;
  padding: number;
}

export const WATLAS_MIN_REASONABLE_EXTENT = 1;
export const WATLAS_MAX_REASONABLE_EXTENT = 10_000;
export const WATLAS_TARGET_EXTENT = 10;

export function watlasPositionScaleForExtent(maxExtent: number): number {
  if (!Number.isFinite(maxExtent) || maxExtent <= 0) return 1;
  if (maxExtent >= WATLAS_MIN_REASONABLE_EXTENT && maxExtent <= WATLAS_MAX_REASONABLE_EXTENT) return 1;
  return WATLAS_TARGET_EXTENT / maxExtent;
}

let watlasInitialization: Promise<void> | null = null;

async function ensureWatlasInitialized(): Promise<void> {
  watlasInitialization ??= watlas.Initialize();
  await watlasInitialization;
}

function validateOptions(faceCount: number, options: AtlasOptions): void {
  if (!Number.isInteger(faceCount) || faceCount < 0) throw new Error('faceCount must be a non-negative integer.');
  if (!Number.isInteger(options.textureSize) || options.textureSize <= 0) throw new Error('textureSize must be a positive integer.');
  if (!Number.isInteger(options.padding) || options.padding < 0) throw new Error('padding must be a non-negative integer.');
}

export function watlasChartOptions(): watlas.ChartOptions {
  return {
    fixWinding: false,
    useInputMeshUvs: false,
  };
}

export function watlasPackOptions(options: AtlasOptions, texelsPerUnit?: number): watlas.PackOptions {
  return {
    maxChartSize: Math.max(1, options.textureSize - 2 * options.padding),
    resolution: options.textureSize,
    padding: options.padding,
    bilinear: false,
    blockAlign: true,
    rotateCharts: true,
    rotateChartsToAxis: true,
    ...(texelsPerUnit !== undefined ? { texelsPerUnit } : {}),
  };
}

export function meshToWatlasBuffers(mesh: RawMesh): WatlasMeshBuffers {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (let index = 0; index < mesh.positions.length; index += 1) {
    const position = mesh.positions[index]!;
    if (!Number.isFinite(position.x) || !Number.isFinite(position.y) || !Number.isFinite(position.z)) {
      throw new Error(`Output vertex ${index} has non-finite position coordinates.`);
    }
    minX = Math.min(minX, position.x);
    minY = Math.min(minY, position.y);
    minZ = Math.min(minZ, position.z);
    maxX = Math.max(maxX, position.x);
    maxY = Math.max(maxY, position.y);
    maxZ = Math.max(maxZ, position.z);
  }

  const hasPositions = mesh.positions.length > 0;
  const centerX = hasPositions ? (minX + maxX) / 2 : 0;
  const centerY = hasPositions ? (minY + maxY) / 2 : 0;
  const centerZ = hasPositions ? (minZ + maxZ) / 2 : 0;
  const maxExtent = hasPositions ? Math.max(maxX - minX, maxY - minY, maxZ - minZ) : 0;
  const positionScale = watlasPositionScaleForExtent(maxExtent);

  const positions = new Float32Array(mesh.positions.length * 3);
  mesh.positions.forEach((position, index) => {
    positions[index * 3] = (position.x - centerX) * positionScale;
    positions[index * 3 + 1] = (position.y - centerY) * positionScale;
    positions[index * 3 + 2] = (position.z - centerZ) * positionScale;
  });
  const indices = new Uint32Array(mesh.faces.length * 3);
  mesh.faces.forEach((face, faceIndex) => {
    for (let corner = 0; corner < 3; corner += 1) {
      const vertexId = face[corner]!;
      if (!Number.isInteger(vertexId) || vertexId < 0 || vertexId >= mesh.positions.length) {
        throw new Error(`Output face ${faceIndex} corner ${corner} references invalid vertex ${vertexId}.`);
      }
      indices[faceIndex * 3 + corner] = vertexId;
    }
  });

  return { positions, indices };
}

function assignCornerByXref(face: readonly [number, number, number], usedCorners: boolean[], xref: number): number {
  for (let corner = 0; corner < 3; corner += 1) {
    if (!usedCorners[corner] && face[corner] === xref) {
      usedCorners[corner] = true;
      return corner;
    }
  }
  throw new Error(`watlas returned vertex xref ${xref} that does not match the source face.`);
}

function extractAtlasLayout(mesh: RawMesh, atlas: watlas.Atlas, options: AtlasOptions): AtlasLayout {
  if (atlas.meshCount !== 1) throw new Error(`Expected watlas to return one mesh, got ${atlas.meshCount}.`);
  if (atlas.atlasCount !== 1) {
    throw new Error('watlas produced multiple atlases. Increase --texture-size or reduce --target-faces.');
  }
  if (atlas.width > options.textureSize || atlas.height > options.textureSize) {
    throw new Error(`watlas atlas ${atlas.width}x${atlas.height} exceeds requested --texture-size ${options.textureSize}.`);
  }

  const resultMesh = atlas.getMesh(0);
  if (resultMesh.indexCount !== mesh.faces.length * 3) {
    throw new Error(`watlas returned ${resultMesh.indexCount} indices for ${mesh.faces.length} triangles.`);
  }

  const indices = new Uint32Array(resultMesh.indexCount);
  if (!resultMesh.getIndexArray(indices)) throw new Error('watlas failed to copy output indices.');

  const faceUvs: AtlasLayout['faceUvs'] = [];
  const facePixelTriangles: AtlasLayout['facePixelTriangles'] = [];

  for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex += 1) {
    const face = mesh.faces[faceIndex]!;
    const usedCorners = [false, false, false];
    const uvs = [new Vector2(), new Vector2(), new Vector2()] as [Vector2, Vector2, Vector2];
    const pixels = [[0, 0], [0, 0], [0, 0]] as [[number, number], [number, number], [number, number]];

    for (let corner = 0; corner < 3; corner += 1) {
      const outputVertexId = indices[faceIndex * 3 + corner]!;
      const vertex = resultMesh.getVertex(outputVertexId);
      if (vertex.atlasIndex < 0) {
        throw new Error(`watlas left output vertex ${outputVertexId} unassigned to an atlas.`);
      }
      if (vertex.atlasIndex !== 0) {
        throw new Error('watlas returned a vertex outside the first atlas. Increase --texture-size or reduce --target-faces.');
      }
      if (!Number.isInteger(vertex.xref) || vertex.xref < 0 || vertex.xref >= mesh.positions.length) {
        throw new Error(`watlas returned vertex xref ${vertex.xref} that does not map to a source vertex.`);
      }
      const sourceCorner = assignCornerByXref(face, usedCorners, vertex.xref);
      const px = vertex.uv[0];
      const py = vertex.uv[1];
      if (!Number.isFinite(px) || !Number.isFinite(py) || px < 0 || py < 0 || px > options.textureSize || py > options.textureSize) {
        throw new Error(`watlas returned UV texel (${px}, ${py}) outside ${options.textureSize}x${options.textureSize}.`);
      }
      pixels[sourceCorner] = [px, py];
      uvs[sourceCorner] = new Vector2(px / options.textureSize, py / options.textureSize);
    }

    faceUvs.push(uvs);
    facePixelTriangles.push(pixels);
  }

  return {
    textureSize: options.textureSize,
    padding: options.padding,
    faceUvs,
    facePixelTriangles,
    islandCount: resultMesh.chartCount,
  };
}

function packAtlasCharts(atlas: watlas.Atlas, options: AtlasOptions): void {
  atlas.computeCharts(watlasChartOptions());
  atlas.packCharts(watlasPackOptions(options));
  for (let attempt = 0; attempt < 8; attempt += 1) {
    if (atlas.atlasCount === 1 && atlas.width <= options.textureSize && atlas.height <= options.textureSize) return;
    const scaleToFit = atlas.width > options.textureSize || atlas.height > options.textureSize
      ? Math.min(options.textureSize / atlas.width, options.textureSize / atlas.height) * 0.95
      : 0.8;
    atlas.packCharts(watlasPackOptions(options, atlas.texelsPerUnit * scaleToFit));
  }
}

export async function createInjectiveAtlas(mesh: RawMesh, options: AtlasOptions): Promise<AtlasLayout> {
  validateOptions(mesh.faces.length, options);
  if (mesh.faces.length === 0) {
    return { textureSize: options.textureSize, padding: options.padding, faceUvs: [], facePixelTriangles: [], islandCount: 0 };
  }

  await ensureWatlasInitialized();
  const buffers = meshToWatlasBuffers(mesh);
  const atlas = new watlas.Atlas();
  try {
    atlas.addMesh({
      vertexPositionData: buffers.positions,
      vertexCount: buffers.positions.length / 3,
      vertexPositionStride: 12,
      indexData: buffers.indices,
      indexCount: buffers.indices.length,
    });
    packAtlasCharts(atlas, options);
    return extractAtlasLayout(mesh, atlas, options);
  } finally {
    atlas.delete();
  }
}

export function countSerializedTexturedVertices(mesh: RawMesh, atlas: AtlasLayout): number {
  if (atlas.faceUvs.length !== mesh.faces.length) throw new Error('Atlas face UV count must match output mesh face count.');

  const vertexKeys = new Set<string>();
  const uvKey = (value: number): string => String(Math.round(value * 1e9));
  for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex += 1) {
    const face = mesh.faces[faceIndex]!;
    const faceUvs = atlas.faceUvs[faceIndex]!;
    for (let corner = 0; corner < 3; corner += 1) {
      const sourceVertexId = face[corner]!;
      const uv = faceUvs[corner]!;
      vertexKeys.add(`${sourceVertexId}/${uvKey(uv.x)}/${uvKey(uv.y)}`);
    }
  }
  return vertexKeys.size;
}
