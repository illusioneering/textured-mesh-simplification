import './styles.css';

import type { Material, Object3D } from 'three';
import { defaultProcessingOptions, type ProcessingOptions } from '../pipeline/options';
import type { PrimitiveExtractionOptions, PrimitiveExtractionResult } from '../pipeline/primitiveExtraction';
import { countSerializedTexturedVertices } from '../texture/atlas';
import { createPrimitiveOutputScene, exportSceneToGlb, parseGlbArrayBuffer, type BrowserLoadedAsset, type BrowserPrimitiveApplyMetadata, type PrimitiveOutputReplacement } from './browserGltfIo';
import { browserMaterialTextureSlots, browserTextureHasImageSource } from './browserMaterialTextures';
import { initializeControlTabs, renderControlPanel } from './controlPanel';
import { parseProcessingOptionsValues, parseTextureBakeOptionsValues, type ProcessingFormValues, type TextureBakeFormValues } from './formOptions';
import { pickModelUploadFiles } from './modelFileInput';
import {
  collectMaterialsForPbrControls,
  defaultPbrMaterialPropertyState,
  detectPbrMaterialPropertyAvailability,
  textureSizeForLargestBaseColorMap,
  type PbrMaterialPropertyId,
  type PbrMaterialPropertyState,
} from './pbrControls';
import {
  formatProcessingCompleteStatus,
  inputModelStatItems,
  processedOutputStatItems,
  summarizeProcessingResult,
  type ProcessingSummary,
  type ProcessingSummaryInput,
  type TexturedMeshStats,
} from './modelStats';
import { cloneSceneForViewport } from './previewScene';
import {
  collectTransferables,
  deserializeAtlas,
  deserializeBakedMaterialTexture,
  deserializeSimplifiedRawMesh,
  deserializeRgbaImage,
  deserializeTransferredMeshAttributes,
  serializePrimitiveEntries,
} from './serialization';
import { createActiveProcessedState, createSimplificationOptionsKey, isActiveProcessedBakeAvailableForAsset, isActiveProcessedOutputForAsset, isActiveProcessedStateForAsset, type ActiveProcessedState, type PendingSimplifyState } from './processingState';
import { createModelViewport, gridFrameForObjectBounds, type RenderingMode } from './viewport';
import type { WorkerRequestMessage, WorkerResponseMessage } from './workerProtocol';

const controls = requiredElement<HTMLDivElement>('#controls');
const stats = requiredElement<HTMLDivElement>('#stats');
const statusLog = requiredElement<HTMLDivElement>('#status-log');
const inputViewportElement = requiredElement<HTMLDivElement>('#input-viewport');
const outputViewportElement = requiredElement<HTMLDivElement>('#output-viewport');

controls.innerHTML = renderControlPanel(defaultProcessingOptions());
initializeControlTabs(controls);

const fileInput = requiredElement<HTMLInputElement>('#model-file');
const modelDropZone = requiredElement<HTMLDivElement>('#model-drop-zone');
const modelFileName = requiredElement<HTMLSpanElement>('#model-file-name');
const form = requiredElement<HTMLFormElement>('#processing-form');
const simplifyButton = requiredElement<HTMLButtonElement>('#simplify-button');
const bakeButton = requiredElement<HTMLButtonElement>('#bake-button');
const exportButton = requiredElement<HTMLButtonElement>('#export-button');
const renderingModeSelect = requiredElement<HTMLSelectElement>('#rendering-mode');
const targetModeSelect = requiredElement<HTMLSelectElement>('#target-mode');
const virtualEdgeModeSelect = requiredElement<HTMLSelectElement>('#virtual-edge-mode');
const textureSizeSelect = requiredElement<HTMLSelectElement>('#texture-size');
const pbrPropertyInputs = Array.from(document.querySelectorAll<HTMLElement>('[data-pbr-property]'))
  .flatMap((element): HTMLInputElement[] => {
    const input = element instanceof HTMLInputElement ? element : element.querySelector<HTMLInputElement>('input');
    const property = element.dataset.pbrProperty ?? input?.dataset.pbrProperty;
    if (!input || !property) return [];
    input.dataset.pbrProperty = property;
    return [input];
  });

const inputViewport = createModelViewport(inputViewportElement);
const outputViewport = createModelViewport(outputViewportElement);
const worker = new Worker(new URL('./processingWorker.ts', import.meta.url), { type: 'module' });

let loadedModel: BrowserLoadedAsset | null = null;
let outputScene: Object3D | null = null;
let outputSummary: ProcessingSummary | null = null;
let inputSummary: TexturedMeshStats | null = null;
let inputGridSpacing: number | undefined;
let currentRequestId: string | null = null;
let currentOperation: 'simplify' | 'bake' | null = null;
let busy = false;
let lastObjectUrl: string | null = null;
let hasSimplifiedGeometryForLoadedModel = false;
let currentInputFileName: string | null = null;
let currentMatchedExternalResourceCount = 0;
let activeProcessedState: ActiveProcessedState | null = null;
let pendingSimplifyState: PendingSimplifyState | null = null;
let pbrPropertyState: PbrMaterialPropertyState = defaultPbrMaterialPropertyState(false);

appendStatus('Ready. Load a GLB/GLTF to begin.');
renderStats();
updateModelFileName();
updateControlVisibility();
updateRenderingMode();
updateViewportPbrState();
updateButtons();

fileInput.addEventListener('change', () => {
  const files = fileInput.files;
  if (files && files.length > 0) void loadInputFiles(files);
});
modelDropZone.addEventListener('click', () => {
  if (busy) return;
  fileInput.click();
});
modelDropZone.addEventListener('keydown', (event: KeyboardEvent) => {
  if (busy || (event.key !== 'Enter' && event.key !== ' ')) return;
  event.preventDefault();
  fileInput.click();
});
modelDropZone.addEventListener('dragenter', handleModelDrag);
modelDropZone.addEventListener('dragover', handleModelDrag);
modelDropZone.addEventListener('dragleave', (event: DragEvent) => {
  if (event.relatedTarget instanceof Node && modelDropZone.contains(event.relatedTarget)) return;
  setModelDragOver(false);
});
modelDropZone.addEventListener('drop', (event: DragEvent) => {
  event.preventDefault();
  setModelDragOver(false);
  if (busy) return;
  const files = event.dataTransfer?.files;
  if (!files || files.length === 0) {
    appendStatus('Drop failed: no file was provided. Drop a .glb or .gltf model file, optionally with .bin buffers and texture images.', 'error');
    return;
  }
  void loadInputFiles(files);
});
form.addEventListener('submit', (event) => {
  event.preventDefault();
});
simplifyButton.addEventListener('click', () => {
  void simplifyLoadedModel();
});
bakeButton.addEventListener('click', () => {
  void bakeTextureAtlas();
});
exportButton.addEventListener('click', () => {
  void exportProcessedModel();
});
renderingModeSelect.addEventListener('change', updateRenderingMode);
targetModeSelect.addEventListener('change', updateControlVisibility);
virtualEdgeModeSelect.addEventListener('change', updateControlVisibility);
pbrPropertyInputs.forEach((input) => {
  input.addEventListener('change', () => {
    const property = input.dataset.pbrProperty as PbrMaterialPropertyId | undefined;
    if (!property || !(property in pbrPropertyState)) return;
    pbrPropertyState = { ...pbrPropertyState, [property]: input.checked };
    updateViewportPbrState();
  });
});
worker.addEventListener('message', (event: MessageEvent<WorkerResponseMessage>) => handleWorkerMessage(event.data));
worker.addEventListener('error', (event) => {
  currentRequestId = null;
  currentOperation = null;
  disposePendingSimplifyState();
  setBusy(false);
  appendStatus(`Worker error: ${event.message}`, 'error');
});

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Required element ${selector} is missing.`);
  return element;
}

function inputValue(id: string): string {
  return requiredElement<HTMLInputElement | HTMLSelectElement>(`#${id}`).value;
}

function inputChecked(id: string): boolean {
  return requiredElement<HTMLInputElement>(`#${id}`).checked;
}

function formValues(): ProcessingFormValues {
  return {
    targetMode: inputValue('target-mode') as ProcessingFormValues['targetMode'],
    targetRatio: inputValue('target-ratio'),
    targetFaceCount: inputValue('target-face-count'),
    primitiveGrouping: inputValue('primitive-grouping') as ProcessingFormValues['primitiveGrouping'],
    virtualEdgeMode: inputValue('virtual-edge-mode') as ProcessingFormValues['virtualEdgeMode'],
    virtualEdgeRadius: inputValue('virtual-edge-radius'),
    weldVertices: inputChecked('weld-vertices'),
    recomputeNormals: inputChecked('recompute-normals'),
    maxIterations: inputValue('max-iterations'),
    textureSize: inputValue('texture-size'),
    texturePadding: inputValue('texture-padding'),
    textureFilter: inputValue('texture-filter') as ProcessingFormValues['textureFilter'],
  };
}

function textureBakeFormValues(): TextureBakeFormValues {
  return {
    textureSize: inputValue('texture-size'),
    texturePadding: inputValue('texture-padding'),
    textureFilter: inputValue('texture-filter') as TextureBakeFormValues['textureFilter'],
  };
}

function updateControlVisibility(): void {
  const targetMode = targetModeSelect.value;
  document.querySelectorAll<HTMLElement>('[data-target-mode]').forEach((element) => {
    element.hidden = element.dataset.targetMode !== targetMode;
  });
  const virtualEdgeMode = virtualEdgeModeSelect.value;
  document.querySelectorAll<HTMLElement>('[data-virtual-edge-mode]').forEach((element) => {
    element.hidden = element.dataset.virtualEdgeMode !== virtualEdgeMode;
  });
}

function updateRenderingMode(): void {
  const mode = renderingModeSelect.value as RenderingMode;
  inputViewport.setRenderingMode(mode);
  outputViewport.setRenderingMode(mode);
}

function updateViewportPbrState(): void {
  inputViewport.setPbrPropertyState(pbrPropertyState);
  outputViewport.setPbrPropertyState(pbrPropertyState);
}

function updatePbrControlsForModel(model: BrowserLoadedAsset | null): void {
  const availability = model
    ? detectPbrMaterialPropertyAvailability(collectMaterialsForPbrControls(model.scene))
    : defaultPbrMaterialPropertyState(false);
  pbrPropertyState = { ...availability };
  for (const input of pbrPropertyInputs) {
    const property = input.dataset.pbrProperty as PbrMaterialPropertyId | undefined;
    if (!property || !(property in availability)) continue;
    input.disabled = !availability[property];
    input.checked = availability[property];
  }
  updateViewportPbrState();
}

function textureDimensionsFromSummary(values: readonly string[]): Array<{ width: number; height: number }> {
  return values.flatMap((value) => {
    const [widthText, heightText] = value.split(/[x×]/u);
    const width = Number(widthText);
    const height = Number(heightText);
    return Number.isFinite(width) && Number.isFinite(height) ? [{ width, height }] : [];
  });
}

function updateTextureSizeForModel(model: BrowserLoadedAsset): void {
  const dimensions = textureDimensionsFromSummary(model.summary.textureDimensions);
  if (dimensions.length === 0) return;
  textureSizeSelect.value = String(textureSizeForLargestBaseColorMap(dimensions, Number(textureSizeSelect.value)));
}

function updateButtons(): void {
  const activeOutputCurrent = loadedModel
    ? isActiveProcessedOutputForAsset(activeProcessedState, outputScene, loadedModel.assetRevision)
    : false;
  fileInput.disabled = busy;
  modelDropZone.classList.toggle('is-disabled', busy);
  modelDropZone.setAttribute('aria-disabled', String(busy));
  simplifyButton.disabled = busy || !loadedModel;
  bakeButton.disabled = busy
    || !loadedModel
    || !hasSimplifiedGeometryForLoadedModel
    || !isActiveProcessedBakeAvailableForAsset(activeProcessedState, loadedModel.assetRevision);
  exportButton.disabled = busy || !activeOutputCurrent;
}

function setBusy(value: boolean): void {
  busy = value;
  form.classList.toggle('is-busy', busy);
  updateButtons();
}

function disposePendingSimplifyState(): void {
  pendingSimplifyState?.extraction.dispose();
  pendingSimplifyState = null;
}

function appendStatus(message: string, kind: 'info' | 'error' | 'success' = 'info'): void {
  const entry = document.createElement('p');
  entry.className = `status-entry status-${kind}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  statusLog.prepend(entry);
  while (statusLog.childElementCount > 80) statusLog.lastElementChild?.remove();
}

function resetOutput(): void {
  outputScene = null;
  outputSummary = null;
  hasSimplifiedGeometryForLoadedModel = false;
  outputViewport.setModel(null, { fitCamera: false, updateGrid: false });
}

function updateModelFileName(): void {
  if (!currentInputFileName) {
    modelFileName.textContent = 'No model selected';
    return;
  }
  modelFileName.textContent = currentMatchedExternalResourceCount > 0
    ? `${currentInputFileName} + ${currentMatchedExternalResourceCount.toLocaleString()} external resource${currentMatchedExternalResourceCount === 1 ? '' : 's'}`
    : currentInputFileName;
}

function dragHasFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes('Files');
}

function setModelDragOver(active: boolean): void {
  modelDropZone.classList.toggle('is-drag-over', active && !busy);
}

function handleModelDrag(event: DragEvent): void {
  if (!dragHasFiles(event)) return;
  event.preventDefault();
  if (event.dataTransfer) event.dataTransfer.dropEffect = busy ? 'none' : 'copy';
  setModelDragOver(true);
}

async function loadInputFiles(files: ArrayLike<File>): Promise<void> {
  if (busy) return;
  const selection = pickModelUploadFiles(files);
  if (selection.skippedUnsupported > 0) {
    appendStatus(`Ignored ${selection.skippedUnsupported.toLocaleString()} unsupported uploaded file${selection.skippedUnsupported === 1 ? '' : 's'}.`);
  }
  if (selection.skippedModelFiles > 0 && selection.modelFile) {
    appendStatus(`Multiple model files uploaded; loading ${selection.modelFile.name} and ignoring ${selection.skippedModelFiles.toLocaleString()} additional model file${selection.skippedModelFiles === 1 ? '' : 's'}.`);
  }
  if (!selection.modelFile) {
    appendStatus('Load failed: no .glb or .gltf model file was provided.', 'error');
    fileInput.value = '';
    return;
  }
  await loadInputFile(selection.modelFile, selection.externalResourceFiles, selection.textureFiles, selection.binaryBufferFiles);
}

async function loadInputFile(
  file: File,
  externalResourceFiles: readonly File[],
  textureFiles: readonly File[],
  binaryBufferFiles: readonly File[],
): Promise<void> {
  if (busy) return;
  setBusy(true);
  currentInputFileName = file.name;
  currentMatchedExternalResourceCount = 0;
  updateModelFileName();
  try {
    appendStatus(`Reading ${file.name} (${file.size.toLocaleString()} bytes)…`);
    const buffer = await file.arrayBuffer();
    appendStatus('Parsing GLB/GLTF and extracting raw mesh data...');
    const model = await parseGlbArrayBuffer(buffer, { externalResourceFiles });
    const matchedExternalTextureFiles = model.matchedExternalTextureFiles ?? [];
    const matchedExternalBinaryBufferFiles = model.matchedExternalBinaryBufferFiles ?? [];
    currentMatchedExternalResourceCount = matchedExternalTextureFiles.length + matchedExternalBinaryBufferFiles.length;
    inputSummary = {
      vertices: model.summary.inputVertices,
      faces: model.summary.inputFaces,
      materials: model.summary.materials,
      materialsWithTextures: model.summary.materialsWithTextures,
      materialsWithBaseColorImages: model.summary.materialsWithBaseColorImages,
      facesWithUvs: model.summary.facesWithUvs,
      textureSlotKinds: model.summary.textureSlotKinds,
      textureDimensions: model.summary.textureDimensions,
    };
    loadedModel = model;
    activeProcessedState = null;
    disposePendingSimplifyState();
    updatePbrControlsForModel(model);
    updateTextureSizeForModel(model);
    resetOutput();
    const inputPreviewScene = cloneSceneForViewport(model.scene);
    inputGridSpacing = gridFrameForObjectBounds(inputPreviewScene).spacing;
    inputViewport.setModel(inputPreviewScene);
    outputViewport.setGridFromObject(inputPreviewScene);
    outputViewport.setCameraPose(inputViewport.getCameraPose());
    updateModelFileName();
    renderStats();
    appendStatus(`Loaded ${file.name}: ${model.summary.inputVertices.toLocaleString()} vertices, ${model.summary.inputFaces.toLocaleString()} faces.`, 'success');
    if (matchedExternalTextureFiles.length > 0) {
      appendStatus(`Matched ${matchedExternalTextureFiles.length.toLocaleString()} uploaded texture image${matchedExternalTextureFiles.length === 1 ? '' : 's'}: ${matchedExternalTextureFiles.join(', ')}.`, 'success');
    }
    if (matchedExternalBinaryBufferFiles.length > 0) {
      appendStatus(`Matched ${matchedExternalBinaryBufferFiles.length.toLocaleString()} uploaded binary buffer file${matchedExternalBinaryBufferFiles.length === 1 ? '' : 's'}: ${matchedExternalBinaryBufferFiles.join(', ')}.`, 'success');
    }
    const ignoredTextureCount = Math.max(0, textureFiles.length - matchedExternalTextureFiles.length);
    if (ignoredTextureCount > 0) {
      appendStatus(`Ignored ${ignoredTextureCount.toLocaleString()} uploaded texture image${ignoredTextureCount === 1 ? '' : 's'} not referenced by filename in the model.`);
    }
    const ignoredBufferCount = Math.max(0, binaryBufferFiles.length - matchedExternalBinaryBufferFiles.length);
    if (ignoredBufferCount > 0) {
      appendStatus(`Ignored ${ignoredBufferCount.toLocaleString()} uploaded binary buffer file${ignoredBufferCount === 1 ? '' : 's'} not referenced by filename in the model.`);
    }
    for (const warning of model.warnings) appendStatus(warning, 'error');
    if (!model.summary.hasImageBackedTextureBakeData) {
      appendStatus('Texture atlas baking disabled for the default material-parent grouping: no image-backed standard material texture data is available with all required UVs. Other groupings may still be bakeable after simplification.');
    }
  } catch (error) {
    loadedModel = null;
    activeProcessedState = null;
    disposePendingSimplifyState();
    inputSummary = null;
    inputGridSpacing = undefined;
    currentInputFileName = null;
    currentMatchedExternalResourceCount = 0;
    resetOutput();
    inputViewport.setModel(null);
    outputViewport.setModel(null);
    updatePbrControlsForModel(null);
    updateModelFileName();
    renderStats();
    appendStatus(`Load failed: ${error instanceof Error ? error.message : String(error)}`, 'error');
  } finally {
    fileInput.value = '';
    setBusy(false);
  }
}

function entriesWithMeshOrdinals(extraction: PrimitiveExtractionResult<BrowserPrimitiveApplyMetadata, PrimitiveExtractionOptions>) {
  return extraction.entries.map((entry) => {
    const metadata = extraction.applyMetadataByEntryId.get(entry.id);
    if (!metadata) throw new Error(`Missing browser apply metadata for primitive ${entry.id}.`);
    if (!entry.texturedRawMesh) throw new Error(`Primitive ${entry.id} is missing textured metadata for worker processing.`);
    return {
      id: entry.id,
      label: entry.label ?? entry.id,
      meshOrdinal: metadata.meshOrdinal,
      rawMesh: entry.rawMesh,
      texturedRawMesh: entry.texturedRawMesh,
      bakeable: entry.bakeable,
      hasTexturedMaterial: entry.hasTexturedMaterial ?? false,
      ...(entry.requiresAttributeTransfer === true ? { requiresAttributeTransfer: true } : {}),
    };
  });
}

function countImageBackedTextureBindings(material: Material | null | undefined): number {
  if (!material) return 0;
  return browserMaterialTextureSlots(material)
    .filter((slot) => browserTextureHasImageSource(slot.texture))
    .length;
}

function countTransferredMaterialTextureBindings(replacements: readonly PrimitiveOutputReplacement[]): number {
  return replacements.reduce((sum, replacement) => {
    if (replacement.materialMode !== 'preserve' || !replacement.transferredAttributes) return sum;
    return sum + countImageBackedTextureBindings(replacement.sourceMaterial);
  }, 0);
}

async function buildSimplifyWorkerRequest(options: ProcessingOptions): Promise<WorkerRequestMessage> {
  if (!loadedModel) throw new Error('Load a model before simplifying geometry.');
  disposePendingSimplifyState();
  const extraction = await loadedModel.extractGroups({
    groupingMode: options.primitiveGrouping,
    mode: 'geometry-with-texture-metadata',
    weldVertices: options.weldVertices,
  });
  pendingSimplifyState = {
    simplifyOptions: options,
    simplificationOptionsKey: createSimplificationOptionsKey({
      primitiveGrouping: options.primitiveGrouping,
      weldVertices: options.weldVertices,
      recomputeNormals: options.recomputeNormals,
    }),
    extraction,
  };
  return {
    type: 'simplify',
    id: crypto.randomUUID(),
    options,
    input: {
      kind: 'primitives',
      entries: serializePrimitiveEntries(entriesWithMeshOrdinals(extraction), { includeImages: false }),
    },
  };
}

async function buildBakeWorkerRequest(options: ProcessingOptions): Promise<WorkerRequestMessage> {
  if (!loadedModel) throw new Error('Load a model before baking a texture atlas.');
  if (!isActiveProcessedStateForAsset(activeProcessedState, loadedModel.assetRevision)) {
    throw new Error('Simplify geometry before baking a texture atlas.');
  }
  if (activeProcessedState.bakeableEntryCount === 0) {
    throw new Error('Texture atlas baking is unavailable for the latest simplified geometry.');
  }
  const simplifyOptions = activeProcessedState.simplifyOptions;
  const extraction = await loadedModel.extractGroups({
    groupingMode: simplifyOptions.primitiveGrouping,
    mode: 'bake',
    weldVertices: simplifyOptions.weldVertices,
  });
  try {
    if (!extraction.entries.some((entry) => entry.bakeable)) {
      throw new Error('Texture atlas baking is unavailable for the latest simplified geometry.');
    }
    return {
      type: 'bake',
      id: crypto.randomUUID(),
      options,
      source: { kind: 'primitives', entries: serializePrimitiveEntries(entriesWithMeshOrdinals(extraction)) },
    };
  } finally {
    extraction.dispose();
  }
}

async function simplifyLoadedModel(): Promise<void> {
  if (!loadedModel || busy) return;
  setBusy(true);
  resetOutput();
  renderStats();
  try {
    const options = parseProcessingOptionsValues(formValues(), false);
    const request = await buildSimplifyWorkerRequest(options);
    currentRequestId = request.id;
    currentOperation = 'simplify';
    appendStatus('Starting worker geometry simplification…');
    worker.postMessage(request, collectTransferables(request));
    pendingSimplifyState?.extraction.releaseProcessingData();
  } catch (error) {
    currentRequestId = null;
    currentOperation = null;
    disposePendingSimplifyState();
    setBusy(false);
    appendStatus(`Simplify failed: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
}

async function bakeTextureAtlas(): Promise<void> {
  if (!loadedModel || busy) return;
  setBusy(true);
  try {
    if (!activeProcessedState) throw new Error('Simplify geometry before baking a texture atlas.');
    const options = parseTextureBakeOptionsValues(textureBakeFormValues(), activeProcessedState.simplifyOptions);
    const request = await buildBakeWorkerRequest(options);
    currentRequestId = request.id;
    currentOperation = 'bake';
    appendStatus('Starting worker texture-atlas baking…');
    worker.postMessage(request, collectTransferables(request));
  } catch (error) {
    currentRequestId = null;
    currentOperation = null;
    setBusy(false);
    appendStatus(`Bake failed: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
}

function handleWorkerMessage(message: WorkerResponseMessage): void {
  if (message.id !== currentRequestId) return;
  if (message.type === 'progress') {
    appendStatus(message.message);
    return;
  }

  const operation = currentOperation;
  currentRequestId = null;
  currentOperation = null;
  setBusy(false);
  if (message.type === 'error') {
    if (operation === 'simplify') {
      disposePendingSimplifyState();
    }
    appendStatus(`${operation === 'bake' ? 'Bake' : 'Simplify'} failed: ${message.message}`, 'error');
    return;
  }

  if (!loadedModel) {
    appendStatus('Worker finished, but the input model was unloaded.', 'error');
    return;
  }

  if (message.result.kind === 'primitives') {
    const metadataByEntryId = operation === 'simplify'
      ? pendingSimplifyState?.extraction.applyMetadataByEntryId
      : activeProcessedState?.applyMetadataByEntryId;
    if (!metadataByEntryId) {
      appendStatus('Worker finished, but primitive apply metadata was unavailable.', 'error');
      return;
    }
    const replacements = message.result.entries.map<PrimitiveOutputReplacement>((entry) => {
      const rawMesh = deserializeSimplifiedRawMesh(entry.raw);
      const source = metadataByEntryId.get(entry.id);
      if (!source) throw new Error(`Missing browser apply metadata for primitive ${entry.id}.`);
      const replacementMetadata = {
        sourceMeshOrdinals: source.sourceMeshOrdinals,
        ...(source.parentObjectOrdinal !== undefined ? { parentObjectOrdinal: source.parentObjectOrdinal } : {}),
        ...(source.preserveSourceMeshTransform === true ? { preserveSourceMeshTransform: true } : {}),
        sourceMaterial: source.sourceMaterial,
      };
      if (entry.baked) {
        return {
          id: entry.id,
          meshOrdinal: entry.meshOrdinal,
          ...replacementMetadata,
          rawMesh,
          ...(entry.transferredAttributes
            ? { transferredAttributes: deserializeTransferredMeshAttributes(entry.transferredAttributes) }
            : {}),
          materialMode: 'baked' as const,
          atlas: deserializeAtlas(entry.baked.atlas),
          image: deserializeRgbaImage(entry.baked.image),
          additionalTextures: entry.baked.additionalTextures.map(deserializeBakedMaterialTexture),
          materialSettings: entry.baked.materialSettings,
        };
      }
      return {
        id: entry.id,
        meshOrdinal: entry.meshOrdinal,
        ...replacementMetadata,
        rawMesh,
        ...(entry.transferredAttributes
          ? { transferredAttributes: deserializeTransferredMeshAttributes(entry.transferredAttributes) }
          : {}),
        materialMode: 'preserve' as const,
      };
    });
    outputScene = createPrimitiveOutputScene(loadedModel.scene, replacements);
    const transferredMaterialTextureBindings = operation === 'simplify'
      ? countTransferredMaterialTextureBindings(replacements)
      : 0;
    const bake = message.result.entries.reduce((sum, entry) => {
      if (!entry.baked) return sum;
      return {
        filledPixels: sum.filledPixels + entry.baked.stats.filledPixels,
        mappedPixels: sum.mappedPixels + entry.baked.stats.mappedPixels,
        unmappedPixels: sum.unmappedPixels + entry.baked.stats.unmappedPixels,
        islandCount: sum.islandCount + (entry.baked.atlas.islandCount ?? 0),
        outputVertices: sum.outputVertices + countSerializedTexturedVertices(
          deserializeSimplifiedRawMesh(entry.raw),
          deserializeAtlas(entry.baked.atlas),
        ),
      };
    }, { filledPixels: 0, mappedPixels: 0, unmappedPixels: 0, islandCount: 0, outputVertices: 0 });
    outputSummary = summarizeProcessingResult({
      stats: message.result.stats,
      ...(bake.filledPixels > 0 || bake.mappedPixels > 0 || bake.islandCount > 0 ? { bake } : {}),
      elapsedSeconds: message.result.elapsedSeconds,
    });
    hasSimplifiedGeometryForLoadedModel = true;
    if (operation === 'simplify') {
      if (!pendingSimplifyState) {
        appendStatus('Worker finished, but pending simplify state was unavailable.', 'error');
        return;
      }
      activeProcessedState = createActiveProcessedState({
        assetRevision: loadedModel.assetRevision,
        pending: pendingSimplifyState,
        simplifyResult: message.result,
        outputScene,
      });
      disposePendingSimplifyState();
    } else if (activeProcessedState) {
      activeProcessedState = { ...activeProcessedState, outputScene };
    }
    outputViewport.setModel(cloneSceneForViewport(outputScene), { fitCamera: false, updateGrid: false });
    renderStats();
    updateButtons();
    appendStatus(formatProcessingCompleteStatus(operation === 'bake' ? 'bake' : 'simplify', message.result.stats, bake), 'success');
    if (transferredMaterialTextureBindings > 0) {
      appendStatus(`Transferred ${transferredMaterialTextureBindings.toLocaleString()} material texture binding${transferredMaterialTextureBindings === 1 ? '' : 's'} to the simplified mesh. Bake a new texture atlas to resample texture maps for the simplified geometry.`);
    }
    return;
  }

  appendStatus('Worker returned an unsupported non-primitive browser result.', 'error');
}

async function exportProcessedModel(): Promise<void> {
  if (
    !loadedModel
    || !outputScene
    || !isActiveProcessedOutputForAsset(activeProcessedState, outputScene, loadedModel.assetRevision)
    || !outputSummary
    || busy
  ) {
    return;
  }
  setBusy(true);
  try {
    appendStatus('Exporting processed scene to binary GLB…');
    const glb = await exportSceneToGlb(outputScene);
    const summaryInput: ProcessingSummaryInput = {
      stats: outputSummary,
      exportedBytes: glb.byteLength,
      ...(outputSummary.elapsedSeconds !== undefined ? { elapsedSeconds: outputSummary.elapsedSeconds } : {}),
    };
    if (outputSummary.bakeFilledPixels !== undefined) {
      summaryInput.bake = {
        filledPixels: outputSummary.bakeFilledPixels,
        mappedPixels: outputSummary.bakeMappedPixels ?? 0,
        unmappedPixels: outputSummary.bakeUnmappedPixels ?? 0,
        ...(outputSummary.bakeIslandCount !== undefined ? { islandCount: outputSummary.bakeIslandCount } : {}),
        ...(outputSummary.bakeOutputVertices !== undefined ? { outputVertices: outputSummary.bakeOutputVertices } : {}),
      };
    }
    outputSummary = summarizeProcessingResult(summaryInput);
    renderStats();
    if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
    const blob = new Blob([glb], { type: 'model/gltf-binary' });
    lastObjectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = lastObjectUrl;
    link.download = exportFilename();
    document.body.append(link);
    link.click();
    link.remove();
    appendStatus(`Exported ${link.download} (${glb.byteLength.toLocaleString()} bytes).`, 'success');
  } catch (error) {
    appendStatus(`Export failed: ${error instanceof Error ? error.message : String(error)}`, 'error');
  } finally {
    setBusy(false);
  }
}

function exportFilename(): string {
  const name = currentInputFileName ?? 'model.glb';
  return name.replace(/\.(glb|gltf)$/i, '') + '.simplified.glb';
}

function keyValueList(items: Array<[string, string | number | undefined]>): HTMLElement {
  const dl = document.createElement('dl');
  dl.className = 'key-value-list';
  for (const [key, value] of items) {
    if (value === undefined) continue;
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = typeof value === 'number' ? value.toLocaleString() : value;
    dl.append(dt, dd);
  }
  return dl;
}

function section(title: string, content: HTMLElement): HTMLElement {
  const wrapper = document.createElement('section');
  wrapper.className = 'stats-section';
  const heading = document.createElement('h3');
  heading.textContent = title;
  wrapper.append(heading, content);
  return wrapper;
}

function renderStats(): void {
  if (!loadedModel) {
    const placeholder = document.createElement('p');
    placeholder.className = 'placeholder';
    placeholder.textContent = 'No model loaded.';
    stats.replaceChildren(placeholder);
    return;
  }

  const input = inputSummary ?? {
    vertices: loadedModel.summary.inputVertices,
    faces: loadedModel.summary.inputFaces,
    materials: loadedModel.summary.materials,
    materialsWithTextures: loadedModel.summary.materialsWithTextures,
    materialsWithBaseColorImages: loadedModel.summary.materialsWithBaseColorImages,
    facesWithUvs: loadedModel.summary.facesWithUvs,
    textureSlotKinds: loadedModel.summary.textureSlotKinds,
    textureDimensions: loadedModel.summary.textureDimensions,
  };
  const transformPreservation = 'scene graph preserved';
  const inputStats = section('Input', keyValueList(inputModelStatItems(input, transformPreservation, inputGridSpacing)));

  if (!outputSummary) {
    const placeholder = document.createElement('p');
    placeholder.className = 'placeholder';
    placeholder.textContent = hasSimplifiedGeometryForLoadedModel
      ? 'Bake a texture atlas or export the geometry-only output.'
      : 'Simplify the loaded model to see output statistics.';
    stats.replaceChildren(inputStats, placeholder);
    return;
  }

  const outputStats = section('Processed output', keyValueList(processedOutputStatItems(outputSummary)));
  stats.replaceChildren(inputStats, outputStats);
}
