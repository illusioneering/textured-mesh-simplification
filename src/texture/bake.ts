import { Vector3 } from 'three';
import type { CollapseHistoryRecord, RawMesh } from '../simplification/types';
import type { TransferredMeshAttributes } from '../simplification/attributes';
import { computeAreaWeightedVertexNormals } from '../simplification/normals';
import { createInjectiveAtlas, type AtlasOptions } from './atlas';
import {
  createEmptyBakeImages,
  mergeTextureBakeBatchResult,
  planTextureBakeBatches,
  readPixel,
  runTextureBakeBatchesSerial,
  toBakeTextureProgress,
  type TextureBakeBatchResult,
  type TextureBakeBatchRunner,
  writePixel,
} from './bakeBatch';
import type {
  BakedTextureResult,
  BakeTextureProgress,
  Rgba,
  StandardBakedTextureSlot,
  TexturedRawMesh,
} from './types';
import { deriveWatlasInputFaceUvs } from './watlasInputUvs';

export interface BakeTextureOptions extends AtlasOptions {
  gutterPasses?: number;
  outputNormalScale?: number;
  onProgress?: (progress: BakeTextureProgress) => void;
  batchRunner?: TextureBakeBatchRunner;
  maxOutputTextureBytes?: number;
}

const STANDARD_BAKED_TEXTURE_SLOTS = ['normal', 'metallicRoughness', 'occlusion', 'emissive'] as const satisfies readonly StandardBakedTextureSlot[];

export function estimateBakeOutputTextureBytes(
  textureSize: number,
  activeSlots: readonly StandardBakedTextureSlot[],
): number {
  return (textureSize * textureSize * 4 * (1 + activeSlots.length)) + (textureSize * textureSize);
}

function activeStandardTextureSlots(source: TexturedRawMesh): StandardBakedTextureSlot[] {
  return STANDARD_BAKED_TEXTURE_SLOTS.filter((slot) => source.materials.some((material) => (
    material.textureSlots.some((texture) => texture.slot === slot && texture.hasImage && texture.image)
  )));
}

function bytesToMiB(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

function dilateGutters(
  image: BakedTextureResult['image'],
  filled: Uint8Array,
  passes: number,
  onProgress?: (progress: BakeTextureProgress) => void,
): void {
  const { width, height } = image;
  const neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]] as const;
  for (let pass = 0; pass < passes; pass += 1) {
    onProgress?.({ stage: 'dilating', gutterPass: pass + 1, gutterPasses: passes });
    const nextFilled = new Uint8Array(filled);
    const writes: { x: number; y: number; color: Rgba }[] = [];
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        if (filled[index]) continue;
        let total: [number, number, number, number] = [0, 0, 0, 0];
        let count = 0;
        for (const [dx, dy] of neighbors) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          if (!filled[ny * width + nx]) continue;
          const color = readPixel(image, nx, ny);
          total = [total[0] + color[0], total[1] + color[1], total[2] + color[2], total[3] + color[3]];
          count += 1;
        }
        if (count > 0) {
          writes.push({
            x,
            y,
            color: [Math.round(total[0] / count), Math.round(total[1] / count), Math.round(total[2] / count), Math.round(total[3] / count)],
          });
          nextFilled[index] = 1;
        }
      }
    }
    for (const write of writes) writePixel(image, write.x, write.y, write.color);
    filled.set(nextFilled);
    if (writes.length === 0) break;
  }
}

function outputVertexNormalsFromTransferred(
  rawMesh: RawMesh,
  transferredAttributes: TransferredMeshAttributes | undefined,
): Vector3[] {
  const fallbackNormals = computeAreaWeightedVertexNormals(rawMesh);
  if (!transferredAttributes || transferredAttributes.vertices.length !== rawMesh.positions.length) {
    return fallbackNormals;
  }
  return rawMesh.positions.map((_, vertexIndex) => {
    const normal = transferredAttributes.vertices[vertexIndex]?.normal;
    if (normal && normal.lengthSq() > 0) return normal;
    return fallbackNormals[vertexIndex] ?? new Vector3(0, 0, 1);
  });
}

export async function bakeStandardMaterialTextures(options: {
  source: TexturedRawMesh;
  outputRawMesh: RawMesh;
  outputFaceIds: number[];
  history: CollapseHistoryRecord[];
  transferredAttributes?: TransferredMeshAttributes;
} & BakeTextureOptions): Promise<BakedTextureResult> {
  const inputFaceUvs = deriveWatlasInputFaceUvs({
    source: options.source,
    outputRawMesh: options.outputRawMesh,
    outputFaceIds: options.outputFaceIds,
    history: options.history,
    ...(options.transferredAttributes ? { transferredAttributes: options.transferredAttributes } : {}),
  });
  const atlas = await createInjectiveAtlas(options.outputRawMesh, {
    textureSize: options.textureSize,
    padding: options.padding,
    ...(inputFaceUvs ? { inputFaceUvs } : {}),
  });
  options.onProgress?.({
    stage: 'atlas-created',
    totalFaces: atlas.facePixelTriangles.length,
    ...(atlas.islandCount !== undefined ? { islandCount: atlas.islandCount } : {}),
  });
  const activeSlots = activeStandardTextureSlots(options.source);
  const estimatedOutputTextureBytes = estimateBakeOutputTextureBytes(options.textureSize, activeSlots);
  if (options.maxOutputTextureBytes !== undefined && estimatedOutputTextureBytes > options.maxOutputTextureBytes) {
    throw new Error(
      `Texture atlas output would allocate approximately ${bytesToMiB(estimatedOutputTextureBytes)} MiB, `
      + `exceeding the configured cap of ${bytesToMiB(options.maxOutputTextureBytes)} MiB. `
      + 'Use a lower texture size or fewer texture slots.',
    );
  }
  const images = createEmptyBakeImages(options.textureSize, activeSlots);
  const plan = planTextureBakeBatches(atlas);
  const batchInput = {
    source: options.source,
    outputRawMesh: options.outputRawMesh,
    outputFaceIds: options.outputFaceIds,
    history: options.history,
    atlas,
    activeSlots,
    outputNormalScale: options.outputNormalScale ?? options.source.materials[0]?.normalScale ?? 1,
    ...(activeSlots.includes('normal')
      ? { outputVertexNormals: outputVertexNormalsFromTransferred(options.outputRawMesh, options.transferredAttributes) }
      : {}),
    batches: plan.batches,
    totalFaces: plan.totalFaces,
    totalSamples: plan.totalSamples,
  };
  const runner = options.batchRunner ?? runTextureBakeBatchesSerial;
  const stats = { filledPixels: 0, mappedPixels: 0, unmappedPixels: 0 };
  const pendingResults = new Map<number, TextureBakeBatchResult>();
  const seenBatchIds = new Set<number>();
  let nextBatchIdToMerge = 0;
  const mergeResult = (result: TextureBakeBatchResult): void => {
    const update = mergeTextureBakeBatchResult(images, result);
    stats.filledPixels += update.filledPixels;
    stats.mappedPixels += update.mappedPixels;
    stats.unmappedPixels += update.unmappedPixels;
  };
  const flushReadyResults = (): void => {
    while (pendingResults.has(nextBatchIdToMerge)) {
      const result = pendingResults.get(nextBatchIdToMerge)!;
      pendingResults.delete(nextBatchIdToMerge);
      mergeResult(result);
      nextBatchIdToMerge += 1;
    }
  };
  await runner(batchInput, (result) => {
    if (result.batchId < 0 || result.batchId >= batchInput.batches.length) {
      throw new Error(`Texture bake runner emitted invalid batch ${result.batchId}.`);
    }
    if (seenBatchIds.has(result.batchId)) {
      throw new Error(`Texture bake runner emitted duplicate batch ${result.batchId}.`);
    }
    seenBatchIds.add(result.batchId);
    pendingResults.set(result.batchId, result);
    flushReadyResults();
  }, (progress) => {
    options.onProgress?.(toBakeTextureProgress(progress));
  });
  if (nextBatchIdToMerge !== batchInput.batches.length) {
    throw new Error(`Texture bake runner is missing expected batch ${nextBatchIdToMerge}.`);
  }
  const gutterPasses = options.gutterPasses ?? options.padding;
  dilateGutters(images.image, new Uint8Array(images.filled), gutterPasses, options.onProgress);
  for (const texture of images.additionalTextures) {
    dilateGutters(texture.image, new Uint8Array(images.filled), gutterPasses);
  }
  options.onProgress?.({ stage: 'complete', ...stats });
  return { image: images.image, additionalTextures: images.additionalTextures, atlas, stats };
}

export const bakeBaseColorTexture = bakeStandardMaterialTextures;
