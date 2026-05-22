import { Vector3, Vector4 } from 'three';
import type { Barycentric, CollapseHistoryRecord, RawMesh } from '../simplification/types';
import { computeAreaWeightedVertexNormals } from '../simplification/normals';
import { interpolateVector3 } from '../simplification/barycentric';
import { createHistoryTraceIndex, mapOutputSampleToInput } from '../simplification/successiveMapping';
import { rasterizeAtlasTriangle } from './rasterize';
import { sampleSourceBaseColor, sampleSourceMaterialTexture, sampleSourceMaterialTextureColor } from './sampling';
import {
  computeAuthoredTangentFrame,
  computeFaceTangentFrame,
  normalRgbToVector,
  tangentNormalToWorld,
  vectorToNormalRgb,
  worldNormalToTangent,
} from './tangentSpace';
import type {
  AtlasLayout,
  BakedMaterialTexture,
  BakedTextureResult,
  BakeTextureProgress,
  Rgba,
  RgbaImage,
  StandardBakedTextureSlot,
  TexturedRawMesh,
} from './types';
import { faceUvSet } from './types';

export const DEFAULT_TEXTURE_BAKE_BATCH_SAMPLES = 65_536;
const NORMAL_SCALE_EPSILON = 1e-8;

export interface TextureBakeBatch {
  id: number;
  startFaceIndex: number;
  endFaceIndex: number;
  sampleCount: number;
}

export interface TextureBakePlan {
  batches: TextureBakeBatch[];
  totalFaces: number;
  totalSamples: number;
}

export interface TextureBakeBatchRunInput {
  source: TexturedRawMesh;
  outputRawMesh: RawMesh;
  outputFaceIds: number[];
  history: CollapseHistoryRecord[];
  atlas: AtlasLayout;
  activeSlots: StandardBakedTextureSlot[];
  outputNormalScale?: number;
  outputVertexNormals?: Vector3[];
  batches: TextureBakeBatch[];
  totalFaces: number;
  totalSamples: number;
}

export interface TextureBakeBatchProgress {
  batchId: number;
  completedBatches: number;
  totalBatches: number;
  processedFaces: number;
  totalFaces: number;
  processedSamples: number;
  totalSamples: number;
  mappedPixels: number;
  unmappedPixels: number;
}

export interface TextureBakeBatchResult {
  batchId: number;
  processedFaces: number;
  sampleCount: number;
  mappedPixels: number;
  unmappedPixels: number;
  pixelIndices: Uint32Array;
  baseColor: Uint8Array;
  additionalTextures: Array<{ slot: StandardBakedTextureSlot; data: Uint8Array }>;
}

export type TextureBakeBatchResultHandler = (result: TextureBakeBatchResult) => void;

export type TextureBakeBatchRunner = (
  input: TextureBakeBatchRunInput,
  onBatchResult: TextureBakeBatchResultHandler,
  onProgress?: (progress: TextureBakeBatchProgress) => void,
) => Promise<void>;

export interface TextureBakeBatchGroupOptions {
  collectResults?: boolean;
}

export interface BakeImages {
  image: RgbaImage;
  additionalTextures: BakedMaterialTexture[];
  filled: Uint8Array;
}

interface NormalMapScale2D {
  x: number;
  y: number;
}

function createEmptyImage(textureSize: number): RgbaImage {
  return {
    width: textureSize,
    height: textureSize,
    data: new Uint8ClampedArray(textureSize * textureSize * 4),
  };
}

export function createEmptyBakeImages(textureSize: number, activeSlots: StandardBakedTextureSlot[]): BakeImages {
  return {
    image: createEmptyImage(textureSize),
    additionalTextures: activeSlots.map((slot) => ({ slot, image: createEmptyImage(textureSize) })),
    filled: new Uint8Array(textureSize * textureSize),
  };
}

export function writePixel(image: RgbaImage, x: number, y: number, color: Rgba): void {
  const offset = (y * image.width + x) * 4;
  image.data[offset] = color[0];
  image.data[offset + 1] = color[1];
  image.data[offset + 2] = color[2];
  image.data[offset + 3] = color[3];
}

export function readPixel(image: RgbaImage, x: number, y: number): Rgba {
  const offset = (y * image.width + x) * 4;
  return [image.data[offset] ?? 0, image.data[offset + 1] ?? 0, image.data[offset + 2] ?? 0, image.data[offset + 3] ?? 0];
}

function writePackedColor(data: Uint8Array, index: number, color: Rgba): void {
  const offset = index * 4;
  data[offset] = color[0];
  data[offset + 1] = color[1];
  data[offset + 2] = color[2];
  data[offset + 3] = color[3];
}

function writePixelIndex(image: RgbaImage, pixelIndex: number, color: Uint8Array, colorIndex: number): void {
  const sourceOffset = colorIndex * 4;
  const targetOffset = pixelIndex * 4;
  image.data[targetOffset] = color[sourceOffset] ?? 0;
  image.data[targetOffset + 1] = color[sourceOffset + 1] ?? 0;
  image.data[targetOffset + 2] = color[sourceOffset + 2] ?? 0;
  image.data[targetOffset + 3] = color[sourceOffset + 3] ?? 0;
}

// Normal-map rebaking interprets source tangent-space normals using original source provenance.
// Final baked output normals are recomputed later from the simplified mesh.
function sourceFaceNormalCorners(
  source: TexturedRawMesh,
  faceId: number,
  fallbackNormals: readonly Vector3[],
): [Vector3, Vector3, Vector3] {
  const attributes = source.faceAttributes[faceId];
  if (attributes?.normalCorners) return attributes.normalCorners;
  const face = source.rawMesh.faces[faceId];
  if (!face) throw new Error(`Missing source face ${faceId}.`);
  return [
    fallbackNormals[face[0]] ?? new Vector3(0, 0, 1),
    fallbackNormals[face[1]] ?? new Vector3(0, 0, 1),
    fallbackNormals[face[2]] ?? new Vector3(0, 0, 1),
  ];
}

function sourceFaceTangentCorners(
  source: TexturedRawMesh,
  faceId: number,
): [Vector4, Vector4, Vector4] | undefined {
  return source.faceAttributes[faceId]?.tangentCorners;
}

function normalMapScale2D(normalScale: number, yScale = 1): NormalMapScale2D {
  const scale = Number.isFinite(normalScale) ? normalScale : 1;
  const yMultiplier = Number.isFinite(yScale) ? yScale : 1;
  return { x: scale, y: scale * yMultiplier };
}

function applyMaterialNormalScale(tangentNormal: Vector3, normalScale: NormalMapScale2D): Vector3 {
  const scaled = new Vector3(tangentNormal.x * normalScale.x, tangentNormal.y * normalScale.y, tangentNormal.z);
  return scaled.lengthSq() <= NORMAL_SCALE_EPSILON ? new Vector3(0, 0, 1) : scaled.normalize();
}

function removePreservedNormalScale(tangentNormal: Vector3, normalScale: NormalMapScale2D): Vector3 {
  if (Math.abs(normalScale.x) <= NORMAL_SCALE_EPSILON && Math.abs(normalScale.y) <= NORMAL_SCALE_EPSILON) {
    return new Vector3(0, 0, tangentNormal.z < 0 ? -1 : 1);
  }
  return new Vector3(
    Math.abs(normalScale.x) <= NORMAL_SCALE_EPSILON ? 0 : tangentNormal.x / normalScale.x,
    Math.abs(normalScale.y) <= NORMAL_SCALE_EPSILON ? 0 : tangentNormal.y / normalScale.y,
    tangentNormal.z,
  );
}

function transformNormalSamplePreservingScale(
  color: Rgba,
  sourceFrame: NonNullable<ReturnType<typeof computeFaceTangentFrame>>,
  outputFrame: NonNullable<ReturnType<typeof computeFaceTangentFrame>>,
  sourceNormalScale: NormalMapScale2D,
  outputNormalScale: NormalMapScale2D,
): Rgba {
  const sourceTangentNormal = applyMaterialNormalScale(normalRgbToVector(color), sourceNormalScale);
  const worldNormal = tangentNormalToWorld(sourceTangentNormal, sourceFrame);
  const outputTangentNormal = worldNormalToTangent(worldNormal, outputFrame);
  return vectorToNormalRgb(removePreservedNormalScale(outputTangentNormal, outputNormalScale), color[3]);
}

function compensateNormalSampleScaleWithoutFrame(
  color: Rgba,
  sourceNormalScale: NormalMapScale2D,
  outputNormalScale: NormalMapScale2D,
): Rgba {
  const sourceTangentNormal = applyMaterialNormalScale(normalRgbToVector(color), sourceNormalScale);
  return vectorToNormalRgb(removePreservedNormalScale(sourceTangentNormal, outputNormalScale), color[3]);
}

function rebakeNormalSample(options: {
  source: TexturedRawMesh;
  outputRawMesh: RawMesh;
  atlas: AtlasLayout;
  outputVertexNormals: readonly Vector3[];
  sourceVertexNormals: readonly Vector3[];
  outputFaceIndex: number;
  outputBarycentric: Barycentric;
  outputNormalScale: number;
  mappedFaceId: number;
  mappedBarycentric: Barycentric;
}): Rgba {
  const sourceAttributes = options.source.faceAttributes[options.mappedFaceId];
  if (!sourceAttributes) throw new Error(`Missing source face attributes for source face ${options.mappedFaceId}.`);
  const material = options.source.materials[sourceAttributes.materialId];
  if (!material) throw new Error(`Missing material ${sourceAttributes.materialId} for source face ${options.mappedFaceId}.`);
  const normalSlot = material.textureSlots.find((slot) => slot.slot === 'normal' && slot.hasImage && slot.image);
  if (!normalSlot?.image) return [128, 128, 255, 255];
  const sourceNormalScale = normalMapScale2D(material.normalScale, sourceAttributes.normalMapYScale);
  const outputNormalScale = normalMapScale2D(options.outputNormalScale);

  const sourceUvSet = faceUvSet(sourceAttributes, normalSlot.texCoord);
  if (!sourceUvSet) {
    return sampleSourceMaterialTexture(options.source, options.mappedFaceId, options.mappedBarycentric, 'normal');
  }

  const sourceFace = options.source.rawMesh.faces[options.mappedFaceId];
  if (!sourceFace) throw new Error(`Missing source face ${options.mappedFaceId}.`);
  const outputFace = options.outputRawMesh.faces[options.outputFaceIndex];
  if (!outputFace) throw new Error(`Missing output face ${options.outputFaceIndex}.`);
  const outputUvs = options.atlas.faceUvs[options.outputFaceIndex];
  if (!outputUvs) throw new Error(`Missing atlas UVs for output face ${options.outputFaceIndex}.`);

  const sourceNormals = sourceFaceNormalCorners(options.source, options.mappedFaceId, options.sourceVertexNormals);
  const sourceNormal = interpolateVector3(sourceNormals[0], sourceNormals[1], sourceNormals[2], options.mappedBarycentric);
  const sourceTangents = sourceFaceTangentCorners(options.source, options.mappedFaceId);
  const sourceFrame = sourceTangents
    ? computeAuthoredTangentFrame({
      normals: sourceNormals,
      tangents: sourceTangents,
      barycentric: options.mappedBarycentric,
    })
    : computeFaceTangentFrame({
      positions: [
        options.source.rawMesh.positions[sourceFace[0]]!,
        options.source.rawMesh.positions[sourceFace[1]]!,
        options.source.rawMesh.positions[sourceFace[2]]!,
      ],
      uvs: sourceUvSet.uvs,
      normal: sourceNormal,
    });
  const outputFrame = computeFaceTangentFrame({
    positions: [
      options.outputRawMesh.positions[outputFace[0]]!,
      options.outputRawMesh.positions[outputFace[1]]!,
      options.outputRawMesh.positions[outputFace[2]]!,
    ],
    uvs: outputUvs,
    normal: interpolateVector3(
      options.outputVertexNormals[outputFace[0]] ?? new Vector3(0, 0, 1),
      options.outputVertexNormals[outputFace[1]] ?? new Vector3(0, 0, 1),
      options.outputVertexNormals[outputFace[2]] ?? new Vector3(0, 0, 1),
      options.outputBarycentric,
    ),
  });
  const color = sampleSourceMaterialTextureColor(options.source, options.mappedFaceId, options.mappedBarycentric, 'normal');
  if (!sourceFrame || !outputFrame) {
    return compensateNormalSampleScaleWithoutFrame(color, sourceNormalScale, outputNormalScale);
  }

  return transformNormalSamplePreservingScale(color, sourceFrame, outputFrame, sourceNormalScale, outputNormalScale);
}

export function planTextureBakeBatches(
  atlas: AtlasLayout,
  options: { targetSamplesPerBatch?: number } = {},
): TextureBakePlan {
  const targetSamplesPerBatch = Math.max(1, Math.floor(options.targetSamplesPerBatch ?? DEFAULT_TEXTURE_BAKE_BATCH_SAMPLES));
  const sampleCounts = atlas.facePixelTriangles.map((triangle, faceIndex) => (
    rasterizeAtlasTriangle(faceIndex, triangle, atlas.textureSize).length
  ));
  const totalFaces = sampleCounts.length;
  const totalSamples = sampleCounts.reduce((total, count) => total + count, 0);
  const batches: TextureBakeBatch[] = [];
  let startFaceIndex = 0;
  let sampleCount = 0;

  for (let faceIndex = 0; faceIndex < totalFaces; faceIndex += 1) {
    const faceSampleCount = sampleCounts[faceIndex] ?? 0;
    if (faceIndex > startFaceIndex && sampleCount + faceSampleCount > targetSamplesPerBatch) {
      batches.push({
        id: batches.length,
        startFaceIndex,
        endFaceIndex: faceIndex,
        sampleCount,
      });
      startFaceIndex = faceIndex;
      sampleCount = 0;
    }
    sampleCount += faceSampleCount;
  }

  if (startFaceIndex < totalFaces) {
    batches.push({
      id: batches.length,
      startFaceIndex,
      endFaceIndex: totalFaces,
      sampleCount,
    });
  }

  return { batches, totalFaces, totalSamples };
}

export const runTextureBakeBatchesSerial: TextureBakeBatchRunner = async (input, onBatchResult, onProgress) => (
  runTextureBakeBatchGroup(input, onBatchResult, onProgress, { collectResults: false }).then(() => undefined)
);

export async function runTextureBakeBatchGroup(
  input: TextureBakeBatchRunInput,
  onBatchResult?: TextureBakeBatchResultHandler,
  onProgress?: (progress: TextureBakeBatchProgress) => void,
  options: TextureBakeBatchGroupOptions = {},
): Promise<TextureBakeBatchResult[]> {
  const collectResults = options.collectResults ?? true;
  const historyIndex = createHistoryTraceIndex(input.history);
  const rebakeNormalMaps = input.activeSlots.includes('normal');
  const outputNormalScale = input.outputNormalScale ?? input.source.materials[0]?.normalScale ?? 1;
  const outputVertexNormals = rebakeNormalMaps ? input.outputVertexNormals ?? computeAreaWeightedVertexNormals(input.outputRawMesh) : [];
  const sourceVertexNormals = rebakeNormalMaps ? computeAreaWeightedVertexNormals(input.source.rawMesh) : [];
  const results: TextureBakeBatchResult[] = [];
  let processedFaces = 0;
  let processedSamples = 0;
  let mappedPixels = 0;
  let unmappedPixels = 0;
  let completedBatches = 0;

  for (const batch of input.batches) {
    const pixelIndices = new Uint32Array(batch.sampleCount);
    const baseColor = new Uint8Array(batch.sampleCount * 4);
    const additionalTextures = input.activeSlots.map((slot) => ({
      slot,
      data: new Uint8Array(batch.sampleCount * 4),
    }));
    let sampleIndex = 0;

    for (let faceIndex = batch.startFaceIndex; faceIndex < batch.endFaceIndex; faceIndex += 1) {
      const pixelTriangle = input.atlas.facePixelTriangles[faceIndex];
      if (!pixelTriangle) throw new Error(`Missing atlas face at index ${faceIndex}.`);
      const samples = rasterizeAtlasTriangle(faceIndex, pixelTriangle, input.atlas.textureSize);
      for (const sample of samples) {
        const mapped = mapOutputSampleToInput({
          outputRawMesh: input.outputRawMesh,
          outputFaceIds: input.outputFaceIds,
          outputFaceIndex: sample.faceIndex,
          outputBarycentric: sample.barycentric,
          history: input.history,
          historyIndex,
        });
        pixelIndices[sampleIndex] = sample.y * input.atlas.textureSize + sample.x;
        writePackedColor(baseColor, sampleIndex, sampleSourceBaseColor(input.source, mapped.faceId, mapped.barycentric));
        for (const texture of additionalTextures) {
          const color = texture.slot === 'normal'
            ? rebakeNormalSample({
              source: input.source,
              outputRawMesh: input.outputRawMesh,
              atlas: input.atlas,
              outputVertexNormals,
              sourceVertexNormals,
              outputFaceIndex: sample.faceIndex,
              outputBarycentric: sample.barycentric,
              outputNormalScale,
              mappedFaceId: mapped.faceId,
              mappedBarycentric: mapped.barycentric,
            })
            : sampleSourceMaterialTexture(input.source, mapped.faceId, mapped.barycentric, texture.slot);
          writePackedColor(texture.data, sampleIndex, color);
        }
        sampleIndex += 1;
      }
    }

    const actualPixelIndices = sampleIndex === pixelIndices.length ? pixelIndices : pixelIndices.slice(0, sampleIndex);
    const actualBaseColor = sampleIndex * 4 === baseColor.length ? baseColor : baseColor.slice(0, sampleIndex * 4);
    const actualAdditionalTextures = additionalTextures.map((texture) => ({
      slot: texture.slot,
      data: sampleIndex * 4 === texture.data.length ? texture.data : texture.data.slice(0, sampleIndex * 4),
    }));
    const result: TextureBakeBatchResult = {
      batchId: batch.id,
      processedFaces: batch.endFaceIndex - batch.startFaceIndex,
      sampleCount: sampleIndex,
      mappedPixels: sampleIndex,
      unmappedPixels: 0,
      pixelIndices: actualPixelIndices,
      baseColor: actualBaseColor,
      additionalTextures: actualAdditionalTextures,
    };
    if (collectResults) results.push(result);
    onBatchResult?.(result);
    processedFaces += result.processedFaces;
    processedSamples += result.sampleCount;
    mappedPixels += result.mappedPixels;
    unmappedPixels += result.unmappedPixels;
    completedBatches += 1;
    onProgress?.({
      batchId: batch.id,
      completedBatches,
      totalBatches: input.batches.length,
      processedFaces,
      totalFaces: input.totalFaces,
      processedSamples,
      totalSamples: input.totalSamples,
      mappedPixels,
      unmappedPixels,
    });
  }

  return results;
}

export function mergeTextureBakeBatchResult(
  images: BakeImages,
  result: TextureBakeBatchResult,
): BakedTextureResult['stats'] {
  let filledPixels = 0;
  const additionalImages = new Map(images.additionalTextures.map((texture) => [texture.slot, texture.image]));

  for (let colorIndex = 0; colorIndex < result.pixelIndices.length; colorIndex += 1) {
    const pixelIndex = result.pixelIndices[colorIndex];
    if (pixelIndex === undefined) continue;
    writePixelIndex(images.image, pixelIndex, result.baseColor, colorIndex);
    for (const texture of result.additionalTextures) {
      const image = additionalImages.get(texture.slot);
      if (image) writePixelIndex(image, pixelIndex, texture.data, colorIndex);
    }
    if (!images.filled[pixelIndex]) {
      images.filled[pixelIndex] = 1;
      filledPixels += 1;
    }
  }

  return { filledPixels, mappedPixels: result.mappedPixels, unmappedPixels: result.unmappedPixels };
}

export function mergeTextureBakeBatchResults(
  images: BakeImages,
  results: TextureBakeBatchResult[],
): BakedTextureResult['stats'] {
  let filledPixels = 0;
  let mappedPixels = 0;
  let unmappedPixels = 0;
  const sortedResults = [...results].sort((left, right) => left.batchId - right.batchId);

  for (const result of sortedResults) {
    const stats = mergeTextureBakeBatchResult(images, result);
    filledPixels += stats.filledPixels;
    mappedPixels += stats.mappedPixels;
    unmappedPixels += stats.unmappedPixels;
  }

  return { filledPixels, mappedPixels, unmappedPixels };
}

export function toBakeTextureProgress(progress: TextureBakeBatchProgress): BakeTextureProgress {
  return {
    stage: 'resampling',
    completedBatches: progress.completedBatches,
    totalBatches: progress.totalBatches,
    processedFaces: progress.processedFaces,
    totalFaces: progress.totalFaces,
    processedSamples: progress.processedSamples,
    totalSamples: progress.totalSamples,
    mappedPixels: progress.mappedPixels,
    unmappedPixels: progress.unmappedPixels,
  };
}
