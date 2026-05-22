#!/usr/bin/env node

import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { performance } from 'node:perf_hooks';
import { Command, InvalidArgumentError } from 'commander';
import type { PrimitiveGroupingMode, ProcessingOptions } from '../pipeline/options';
import { bakePrimitiveTextures } from '../pipeline/sceneProcessing';
import { countSerializedTexturedVertices } from '../texture/atlas';
import { GltfTransformPrimitiveSourceAdapter } from './gltfTransformPrimitiveSourceAdapter';
import { processPrimitiveGeometriesInNodeWorkers } from './primitiveWorkerPool';
import { createNodeTextureBakeBatchRunner } from './textureBakeWorkerPool';
import { writeScenePrimitiveDocument } from './scenePrimitiveGltfIo';

function parsePositiveNumber(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new InvalidArgumentError(`Expected a positive number, got "${value}".`);
  }
  return parsed;
}

function parseNonNegativeNumber(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new InvalidArgumentError(`Expected a non-negative number, got "${value}".`);
  }
  return parsed;
}

function parsePositiveInteger(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new InvalidArgumentError(`Expected a positive integer, got "${value}".`);
  }
  return parsed;
}

function parseNonNegativeInteger(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new InvalidArgumentError(`Expected a non-negative integer, got "${value}".`);
  }
  return parsed;
}

function parseTextureFilter(value: string): 'nearest' | 'linear' {
  if (value === 'nearest' || value === 'linear') return value;
  throw new InvalidArgumentError(`Expected "nearest" or "linear", got "${value}".`);
}

type CliVirtualEdgeMode = 'auto-local-radius' | 'auto-global-radius' | 'manual-global-radius';

function parseVirtualEdgeMode(value: string): CliVirtualEdgeMode {
  if (value === 'auto-local-radius' || value === 'auto-global-radius' || value === 'manual-global-radius') return value;
  throw new InvalidArgumentError(`Expected auto-local-radius, auto-global-radius, or manual-global-radius, got "${value}".`);
}

function parseVirtualEdgeCandidateCap(value: string): number | 'none' {
  if (value === 'none') return 'none';
  return parseNonNegativeInteger(value);
}

function parsePrimitiveGrouping(value: string): PrimitiveGroupingMode {
  if (value === 'material-parent' || value === 'material' || value === 'none') return value;
  throw new InvalidArgumentError(`Expected "material-parent", "material", or "none", got "${value}".`);
}

const program = new Command();
program
  .name('mesh-simplify')
  .description('Command-line simplification prototype for wild triangle meshes, with optional standard material texture transfer.')
  .requiredOption('-i, --input <path>', 'input .glb path')
  .requiredOption('-o, --output <path>', 'output .glb path')
  .option('-r, --ratio <number>', 'target output face ratio, e.g. 0.5', parsePositiveNumber)
  .option('-f, --target-faces <integer>', 'target output face count', parsePositiveInteger)
  .option('--virtual-edge-mode <auto-local-radius|auto-global-radius|manual-global-radius>', 'virtual edge radius mode', parseVirtualEdgeMode, 'auto-local-radius')
  .option('--virtual-radius <number>', 'manual global virtual edge radius r; required when --virtual-edge-mode manual-global-radius', parseNonNegativeNumber)
  .option('--virtual-edge-candidate-cap <integer|none>', 'diagnostic auto-local-radius cap per component pair; "none" disables the cap', parseVirtualEdgeCandidateCap)
  .option('--no-weld-vertices', 'skip duplicate-position vertex welding during input mesh extraction')
  .option('--no-recompute-normals', 'preserve transferred source normals when available')
  .option('--max-iterations <integer>', 'debug cap on collapse iterations', parsePositiveInteger)
  .option('--progress-interval <integer>', 'print progress every N collapses', parsePositiveInteger, 1000)
  .option('--primitive-grouping <material-parent|material|none>', 'group source primitives before simplification', parsePrimitiveGrouping, 'material-parent')
  .option('--transfer-textures', 'bake standard PBR material textures and base-color factors using a watlas-generated chart atlas')
  .option('--texture-size <integer>', 'output texture width/height in pixels', parsePositiveInteger, 1024)
  .option('--texture-padding <integer>', 'atlas chart padding/gutter size in pixels', parseNonNegativeInteger, 2)
  .option('--texture-filter <nearest|linear>', 'source texture sampling filter', parseTextureFilter, 'linear');

type CliOptions = {
  input: string;
  output: string;
  ratio?: number;
  targetFaces?: number;
  virtualEdgeMode: CliVirtualEdgeMode;
  virtualRadius?: number;
  virtualEdgeCandidateCap?: number | 'none';
  weldVertices: boolean;
  recomputeNormals: boolean;
  maxIterations?: number;
  progressInterval: number;
  primitiveGrouping: PrimitiveGroupingMode;
  transferTextures?: boolean;
  textureSize: number;
  texturePadding: number;
  textureFilter: 'nearest' | 'linear';
};

async function main(): Promise<void> {
  program.parse(process.argv);
  const opts = program.opts<CliOptions>();

  if (opts.ratio !== undefined && opts.targetFaces !== undefined) {
    throw new Error('Specify either --ratio or --target-faces, not both.');
  }
  if (opts.virtualEdgeMode !== 'manual-global-radius' && opts.virtualRadius !== undefined) {
    throw new Error('--virtual-radius only applies when --virtual-edge-mode is manual-global-radius.');
  }
  if (opts.virtualEdgeMode === 'manual-global-radius' && opts.virtualRadius === undefined) {
    throw new Error('--virtual-radius is required when --virtual-edge-mode is manual-global-radius.');
  }
  if (opts.virtualEdgeMode !== 'auto-local-radius' && opts.virtualEdgeCandidateCap !== undefined) {
    throw new Error('--virtual-edge-candidate-cap only applies when using auto-local-radius virtual edges.');
  }

  const started = performance.now();
  await mkdir(dirname(opts.output), { recursive: true });

  const progress = opts.progressInterval > 0
    ? (stats: Parameters<NonNullable<import('../simplification/types').SimplifyOptions['onProgress']>>[0]) => {
      if (stats.iteration % opts.progressInterval === 0) {
        console.log(
          `  ${stats.iteration} collapses: ${stats.activeFaces} faces, ${stats.activeVertices} vertices, last cost ${stats.lastCost.toExponential(3)}`,
        );
      }
    }
    : undefined;

  const processingOptions: ProcessingOptions = {
    target: opts.targetFaces !== undefined
      ? { kind: 'faces', targetFaceCount: opts.targetFaces }
      : { kind: 'ratio', ratio: opts.ratio ?? 0.5 },
    virtualEdges: opts.virtualEdgeMode === 'auto-local-radius'
      ? (opts.virtualEdgeCandidateCap === undefined
        ? { mode: 'auto-local-radius' }
        : { mode: 'auto-local-radius', maxPairsPerComponentPair: opts.virtualEdgeCandidateCap === 'none' ? null : opts.virtualEdgeCandidateCap })
      : opts.virtualEdgeMode === 'auto-global-radius'
        ? { mode: 'auto-global-radius' }
        : { mode: 'manual-global-radius', radius: opts.virtualRadius! },
    weldVertices: opts.weldVertices,
    recomputeNormals: opts.recomputeNormals,
    ...(opts.maxIterations !== undefined ? { maxIterations: opts.maxIterations } : {}),
    transferTextures: opts.transferTextures ?? false,
    primitiveGrouping: opts.primitiveGrouping,
    textureSize: opts.textureSize,
    texturePadding: opts.texturePadding,
    textureFilter: opts.textureFilter,
  };

  console.log(`Reading ${opts.input} ...`);
  const adapter = await GltfTransformPrimitiveSourceAdapter.read(opts.input);
  const extraction = await adapter.extractGroups({
    groupingMode: processingOptions.primitiveGrouping,
    mode: opts.transferTextures ? 'bake' : 'geometry-with-texture-metadata',
    weldVertices: processingOptions.weldVertices,
  });

  try {
    for (const warning of extraction.summary.warnings) {
      console.warn(`Warning: ${warning}`);
    }
    const processable = extraction.entries;
    if (processable.length === 0) throw new Error(`No supported triangle primitives found in ${opts.input}.`);
    if (opts.transferTextures && !processable.some((entry) => entry.bakeable)) {
      throw new Error('Input has no primitives with usable standard material texture data and required UVs to transfer.');
    }

    const simplifyLabel = extraction.summary.hasTransferableVertexAttributes === true
      ? 'Simplifying primitives with source attribute transfer ...'
      : 'Simplifying primitives ...';
    console.log(simplifyLabel);
    const geometryResult = await processPrimitiveGeometriesInNodeWorkers(
      processable,
      processingOptions,
      progress ? {
        onAggregateProgress: (stageProgress) => {
          if (stageProgress.stage !== 'simplification') return;
          progress({
            iteration: stageProgress.collapses,
            activeFaces: stageProgress.activeFaces,
            activeVertices: stageProgress.activeVertices,
            activeEdges: stageProgress.activeEdges,
            lastCost: 0,
          });
        },
      } : {},
    );
    if (geometryResult.stats.stoppedReason === 'queue-empty') {
      console.warn('Warning: simplification queue emptied before reaching the requested target for at least one primitive.');
    }

    let bakeStats: { filledPixels: number; mappedPixels: number; unmappedPixels: number; islandCount: number; outputVertices: number } | undefined;
    if (opts.transferTextures) {
      console.log('Baking standard material texture atlases ...');
      const texturedResult = await bakePrimitiveTextures(
        geometryResult,
        processingOptions,
        {},
        { batchRunner: createNodeTextureBakeBatchRunner() },
      );
      bakeStats = texturedResult.entries
        .filter((entry) => entry.baked)
        .reduce((sum, entry) => ({
          filledPixels: sum.filledPixels + (entry.baked?.baked.stats.filledPixels ?? 0),
          mappedPixels: sum.mappedPixels + (entry.baked?.baked.stats.mappedPixels ?? 0),
          unmappedPixels: sum.unmappedPixels + (entry.baked?.baked.stats.unmappedPixels ?? 0),
          islandCount: sum.islandCount + (entry.baked?.baked.atlas.islandCount ?? 0),
          outputVertices: sum.outputVertices + (entry.baked
            ? countSerializedTexturedVertices(entry.baked.raw.rawMesh, entry.baked.baked.atlas)
            : 0),
        }), { filledPixels: 0, mappedPixels: 0, unmappedPixels: 0, islandCount: 0, outputVertices: 0 });
      await adapter.applyResults(extraction, texturedResult.entries);
    } else {
      await adapter.applyResults(extraction, geometryResult.entries);
    }

    console.log(`Writing ${opts.output} ...`);
    await writeScenePrimitiveDocument(extraction.extractionApplyState.document, opts.output);
    const elapsedSeconds = (performance.now() - started) / 1000;

    console.log('Done.');
    console.log(JSON.stringify({
      ...geometryResult.stats,
      ...(bakeStats ? { bake: bakeStats } : {}),
      textureSize: opts.textureSize,
      texturePadding: opts.texturePadding,
      weldVertices: opts.weldVertices,
      recomputeNormals: opts.recomputeNormals,
      primitiveGrouping: opts.primitiveGrouping,
      elapsedSeconds,
    }, null, 2));
  } finally {
    extraction.dispose();
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
