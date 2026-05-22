import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import {
  Accessor,
  Document,
  NodeIO,
  Primitive,
  type Property,
  TextureInfo,
  type Material,
  type Mesh,
  type Node as GltfNode,
  type Scene,
  type Texture,
} from '@gltf-transform/core';
import { Matrix3, Matrix4, Vector2, Vector3, Vector4 } from 'three';
import type { PrimitiveGroupingMode } from '../pipeline/options';
import type { TransferredMeshAttributes, VertexColorItemSize } from '../simplification/attributes';
import type { FaceIndices, RawMesh } from '../simplification/types';
import { appendTransformedRawMesh, createWeldedVertexAppendState } from '../simplification/weld';
import { transformAuthoredTangent } from '../texture/tangentSpace';
import type {
  AtlasLayout,
  BakedMaterialTexture,
  RgbaImage,
  SourceFaceAttributes,
  SourceMaterial,
  SourceMaterialTextureInfo,
  SourceMaterialTextureSlot,
  SourceTexture,
  TextureSampler,
  WrapMode,
} from '../texture/types';
import {
  attachAdditionalBakedTextures,
  configureBakedTextureInfo,
  createBakedTexture,
} from './bakedMaterialTextures';
import { decodeImage } from './imageCodecs';
import type { PrimitiveExtractionMode } from '../pipeline/primitiveExtraction';
import {
  buildAttributeTransferredPrimitiveGeometryData,
  buildAtlasPrimitiveGeometryData,
  buildIndexedPrimitiveGeometryData,
} from '../pipeline/primitiveOutputGeometry';

export interface ScenePrimitiveWarning {
  meshName: string;
  nodeName?: string;
  primitiveIndex: number;
  reason: string;
}

export interface ScenePrimitiveEntry {
  id: number;
  document: Document;
  mesh: Mesh;
  primitive: Primitive;
  primitiveIndex: number;
  scene: Scene;
  node: GltfNode;
  parentNode: GltfNode | null;
  nodeName?: string;
  rawMesh: RawMesh;
  faceAttributes: SourceFaceAttributes[];
  sourceMaterial: SourceMaterial;
  originalMaterial: Material | null;
}

export interface ScenePrimitiveGroup {
  id: string;
  mode: PrimitiveGroupingMode;
  document: Document;
  scene: Scene;
  parentNode: GltfNode | null;
  entries: ScenePrimitiveEntry[];
  rawMesh: RawMesh;
  faceAttributes: SourceFaceAttributes[];
  sourceMaterial: SourceMaterial;
  originalMaterial: Material | null;
}

export interface ScenePrimitiveReadResult {
  document: Document;
  entries: ScenePrimitiveEntry[];
  warnings: ScenePrimitiveWarning[];
}

export interface ReadGlbScenePrimitiveOptions {
  mode?: PrimitiveExtractionMode;
}

export interface GroupScenePrimitiveOptions {
  weldVertices?: boolean;
}

function mapWrapMode(mode: number | null | undefined): WrapMode {
  if (mode === TextureInfo.WrapMode.CLAMP_TO_EDGE) return 'clamp';
  if (mode === TextureInfo.WrapMode.MIRRORED_REPEAT) return 'mirrored-repeat';
  return 'repeat';
}

function mapFilter(info: TextureInfo | null): TextureSampler['filter'] {
  if (!info) return 'linear';
  const mag = info.getMagFilter();
  const min = info.getMinFilter();
  return mag === TextureInfo.MagFilter.NEAREST || min === TextureInfo.MinFilter.NEAREST ? 'nearest' : 'linear';
}

function materialName(material: Material | null): string {
  const name = material?.getName();
  return name && name.length > 0 ? name : 'material';
}

function cloneFactor(factor: readonly number[]): [number, number, number, number] {
  return [factor[0] ?? 1, factor[1] ?? 1, factor[2] ?? 1, factor[3] ?? 1];
}

function cloneVec3(factor: readonly number[]): [number, number, number] {
  return [factor[0] ?? 0, factor[1] ?? 0, factor[2] ?? 0];
}

function samplerFromTextureInfo(info: TextureInfo | null): TextureSampler {
  return {
    wrapS: mapWrapMode(info?.getWrapS()),
    wrapT: mapWrapMode(info?.getWrapT()),
    filter: mapFilter(info),
  };
}

function decodeTextureImage(texture: Texture, textureImages: Map<Texture, RgbaImage>): RgbaImage | undefined {
  const cached = textureImages.get(texture);
  if (cached) return cached;
  const image = texture.getImage();
  if (!image) return undefined;
  const decoded = decodeImage(image, texture.getMimeType());
  textureImages.set(texture, decoded);
  return decoded;
}

function textureInfoToSourceTexture(
  texture: Texture | null,
  info: TextureInfo | null,
  textureImages: Map<Texture, RgbaImage>,
  includeImagePayloads: boolean,
): SourceTexture | undefined {
  if (!texture || !info) return undefined;
  const image = includeImagePayloads ? decodeTextureImage(texture, textureImages) : undefined;
  const name = texture.getName();
  const mimeType = texture.getMimeType();
  return {
    ...(image ? { image } : {}),
    sampler: samplerFromTextureInfo(info),
    texCoord: info.getTexCoord(),
    ...(name ? { name } : {}),
    ...(mimeType ? { mimeType } : {}),
  };
}

function textureInfoToSlot(
  slot: SourceMaterialTextureSlot,
  texture: Texture | null,
  info: TextureInfo | null,
  textureImages: Map<Texture, RgbaImage>,
  includeImagePayloads: boolean,
): SourceMaterialTextureInfo | null {
  if (!texture || !info) return null;
  const hasImage = Boolean(texture.getImage());
  const image = includeImagePayloads ? decodeTextureImage(texture, textureImages) : undefined;
  const name = texture.getName();
  const mimeType = texture.getMimeType();
  return {
    slot,
    texCoord: info.getTexCoord(),
    sampler: samplerFromTextureInfo(info),
    hasImage,
    ...(image && slot !== 'baseColor' ? { image } : {}),
    ...(name ? { name } : {}),
    ...(mimeType ? { mimeType } : {}),
  };
}

function defaultSourceMaterial(): SourceMaterial {
  return {
    name: 'default-material',
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
  };
}

function decodeSourceMaterial(
  material: Material | null,
  textureImages: Map<Texture, RgbaImage>,
  mode: PrimitiveExtractionMode,
): SourceMaterial {
  if (!material) return defaultSourceMaterial();
  const includeImagePayloads = mode === 'bake';

  const baseColorTexture = textureInfoToSourceTexture(
    material.getBaseColorTexture(),
    material.getBaseColorTextureInfo(),
    textureImages,
    includeImagePayloads,
  );
  const textureSlots = [
    textureInfoToSlot('baseColor', material.getBaseColorTexture(), material.getBaseColorTextureInfo(), textureImages, includeImagePayloads),
    textureInfoToSlot('normal', material.getNormalTexture(), material.getNormalTextureInfo(), textureImages, includeImagePayloads),
    textureInfoToSlot('metallicRoughness', material.getMetallicRoughnessTexture(), material.getMetallicRoughnessTextureInfo(), textureImages, includeImagePayloads),
    textureInfoToSlot('occlusion', material.getOcclusionTexture(), material.getOcclusionTextureInfo(), textureImages, includeImagePayloads),
    textureInfoToSlot('emissive', material.getEmissiveTexture(), material.getEmissiveTextureInfo(), textureImages, includeImagePayloads),
  ].filter((slot): slot is SourceMaterialTextureInfo => slot !== null);

  const sourceMaterial: SourceMaterial = {
    name: materialName(material),
    baseColorFactor: cloneFactor(material.getBaseColorFactor()),
    ...(baseColorTexture ? { baseColorTexture } : {}),
    textureSlots,
    alphaMode: material.getAlphaMode() as SourceMaterial['alphaMode'],
    alphaCutoff: material.getAlphaCutoff(),
    doubleSided: material.getDoubleSided(),
    emissiveFactor: cloneVec3(material.getEmissiveFactor()),
    metallicFactor: material.getMetallicFactor(),
    roughnessFactor: material.getRoughnessFactor(),
    normalScale: material.getNormalScale(),
    occlusionStrength: material.getOcclusionStrength(),
  };
  return sourceMaterial;
}

function hasNormalTextureImage(material: SourceMaterial): boolean {
  return material.textureSlots.some((slot) => slot.slot === 'normal' && slot.hasImage);
}

function sourceNormalMapYScale(material: SourceMaterial, primitive: Primitive): number | undefined {
  return hasNormalTextureImage(material) && !primitive.getAttribute('TANGENT') ? -1 : undefined;
}

function meshHasSkinnedInstance(nodes: GltfNode[], mesh: Mesh): boolean {
  return nodes.some((node) => node.getMesh() === mesh && node.getSkin() !== null);
}

function unsupportedReason(primitive: Primitive, mesh: Mesh, nodes: GltfNode[]): string | null {
  if (primitive.getMode() !== Primitive.Mode.TRIANGLES) {
    return `Unsupported primitive mode ${primitive.getMode()}; only TRIANGLES are supported.`;
  }
  const positionAccessor = primitive.getAttribute('POSITION');
  if (!positionAccessor) return 'Primitive is missing POSITION.';
  if (positionAccessor.getType() !== Accessor.Type.VEC3) {
    return `Unsupported POSITION accessor type ${positionAccessor.getType()}; expected VEC3.`;
  }
  if (primitive.listTargets().length > 0) return 'Morph targets are not supported for primitive replacement.';
  const customSemantics = primitive
    .listSemantics()
    .filter((semantic) => (
      semantic !== 'POSITION'
      && semantic !== 'NORMAL'
      && semantic !== 'TANGENT'
      && semantic !== 'COLOR_0'
      && !/^TEXCOORD_\d+$/.test(semantic)
    ));
  if (customSemantics.length > 0) return `Custom vertex attributes are not supported: ${customSemantics.join(', ')}.`;
  if (meshHasSkinnedInstance(nodes, mesh)) return 'Skinned mesh instances are not supported for primitive replacement.';
  return null;
}

function listPrimitiveUvAccessors(primitive: Primitive): Array<{ texCoord: number; accessor: Accessor }> {
  const uvAccessors: Array<{ texCoord: number; accessor: Accessor }> = [];
  for (const semantic of primitive.listSemantics()) {
    const match = /^TEXCOORD_(\d+)$/.exec(semantic);
    if (!match) continue;
    const accessor = primitive.getAttribute(semantic);
    if (!accessor) continue;
    if (accessor.getType() !== Accessor.Type.VEC2) {
      throw new Error(`Unsupported ${semantic} accessor type ${accessor.getType()}; expected VEC2.`);
    }
    uvAccessors.push({ texCoord: Number(match[1]), accessor });
  }
  return uvAccessors.sort((a, b) => a.texCoord - b.texCoord);
}

function readLocalPrimitive(entry: {
  document: Document;
  mesh: Mesh;
  primitive: Primitive;
  primitiveIndex: number;
  scene: Scene;
  node: GltfNode;
  parentNode: GltfNode | null;
  nodeName?: string;
  id: number;
  textureImages: Map<Texture, RgbaImage>;
  mode: PrimitiveExtractionMode;
}): ScenePrimitiveEntry {
  const positionAccessor = entry.primitive.getAttribute('POSITION');
  if (!positionAccessor) throw new Error('Cannot read a supported primitive without POSITION.');

  const rawMesh: RawMesh = { positions: [], faces: [] };
  const positionElement: number[] = [0, 0, 0];
  for (let i = 0; i < positionAccessor.getCount(); i += 1) {
    positionAccessor.getElement(i, positionElement);
    rawMesh.positions.push(new Vector3(positionElement[0]!, positionElement[1]!, positionElement[2]!));
  }

  const localIndices: number[] = [];
  const indexAccessor = entry.primitive.getIndices();
  if (indexAccessor) {
    if (indexAccessor.getCount() % 3 !== 0) throw new Error(`Primitive index count ${indexAccessor.getCount()} is not divisible by 3.`);
    for (let i = 0; i < indexAccessor.getCount(); i += 1) localIndices.push(indexAccessor.getScalar(i));
  } else {
    if (positionAccessor.getCount() % 3 !== 0) throw new Error(`Non-indexed primitive vertex count ${positionAccessor.getCount()} is not divisible by 3.`);
    for (let i = 0; i < positionAccessor.getCount(); i += 1) localIndices.push(i);
  }

  const material = entry.primitive.getMaterial();
  const sourceMaterial = decodeSourceMaterial(material, entry.textureImages, entry.mode);
  const normalMapYScale = sourceNormalMapYScale(sourceMaterial, entry.primitive);
  const uvAccessors = listPrimitiveUvAccessors(entry.primitive);
  const normalAccessor = entry.primitive.getAttribute('NORMAL');
  if (normalAccessor && normalAccessor.getType() !== Accessor.Type.VEC3) {
    throw new Error(`Unsupported NORMAL accessor type ${normalAccessor.getType()}; expected VEC3.`);
  }
  if (normalAccessor && normalAccessor.getCount() !== positionAccessor.getCount()) {
    throw new Error(`NORMAL accessor count ${normalAccessor.getCount()} does not match POSITION accessor count ${positionAccessor.getCount()}.`);
  }
  const tangentAccessor = entry.primitive.getAttribute('TANGENT');
  if (tangentAccessor && tangentAccessor.getType() !== Accessor.Type.VEC4) {
    throw new Error(`Unsupported TANGENT accessor type ${tangentAccessor.getType()}; expected VEC4.`);
  }
  if (tangentAccessor && tangentAccessor.getCount() !== positionAccessor.getCount()) {
    throw new Error(`TANGENT accessor count ${tangentAccessor.getCount()} does not match POSITION accessor count ${positionAccessor.getCount()}.`);
  }
  const colorAccessor = entry.primitive.getAttribute('COLOR_0');
  if (
    colorAccessor
    && colorAccessor.getType() !== Accessor.Type.VEC3
    && colorAccessor.getType() !== Accessor.Type.VEC4
  ) {
    throw new Error(`Unsupported COLOR_0 accessor type ${colorAccessor.getType()}; expected VEC3 or VEC4.`);
  }
  if (colorAccessor && colorAccessor.getCount() !== positionAccessor.getCount()) {
    throw new Error(`COLOR_0 accessor count ${colorAccessor.getCount()} does not match POSITION accessor count ${positionAccessor.getCount()}.`);
  }

  const uvElement: number[] = [0, 0];
  const readUv = (accessor: Accessor, localVertexId: number): Vector2 => {
    accessor.getElement(localVertexId, uvElement);
    return new Vector2(uvElement[0]!, uvElement[1]!);
  };

  const meshNormals: Vector3[] = [];
  if (normalAccessor) {
    const normalElement: number[] = [0, 0, 0];
    for (let i = 0; i < normalAccessor.getCount(); i += 1) {
      normalAccessor.getElement(i, normalElement);
      meshNormals.push(new Vector3(normalElement[0]!, normalElement[1]!, normalElement[2]!).normalize());
    }
  }
  const meshTangents: Vector4[] = [];
  if (tangentAccessor) {
    const tangentElement: number[] = [0, 0, 0, 1];
    for (let i = 0; i < tangentAccessor.getCount(); i += 1) {
      tangentAccessor.getElement(i, tangentElement);
      meshTangents.push(transformAuthoredTangent(
        new Vector4(tangentElement[0]!, tangentElement[1]!, tangentElement[2]!, tangentElement[3]!),
        new Matrix4().identity(),
      ));
    }
  }
  const meshColors: Vector4[] = [];
  const colorItemSize: VertexColorItemSize | undefined = colorAccessor
    ? (colorAccessor.getType() === Accessor.Type.VEC4 ? 4 : 3)
    : undefined;
  if (colorAccessor) {
    const colorElement: number[] = [0, 0, 0, 1];
    for (let i = 0; i < colorAccessor.getCount(); i += 1) {
      colorAccessor.getElement(i, colorElement);
      meshColors.push(new Vector4(
        colorElement[0]!,
        colorElement[1]!,
        colorElement[2]!,
        colorItemSize === 4 ? colorElement[3]! : 1,
      ));
    }
  }

  const faceAttributes: SourceFaceAttributes[] = [];
  for (let i = 0; i < localIndices.length; i += 3) {
    const a = localIndices[i]!;
    const b = localIndices[i + 1]!;
    const c = localIndices[i + 2]!;
    const normalCorners = meshNormals.length === positionAccessor.getCount()
      ? [meshNormals[a]!.clone(), meshNormals[b]!.clone(), meshNormals[c]!.clone()] as [Vector3, Vector3, Vector3]
      : undefined;
    const tangentCorners = meshTangents.length === positionAccessor.getCount()
      ? [meshTangents[a]!.clone(), meshTangents[b]!.clone(), meshTangents[c]!.clone()] as [Vector4, Vector4, Vector4]
      : undefined;
    const colorCorners = meshColors.length === positionAccessor.getCount()
      ? [meshColors[a]!.clone(), meshColors[b]!.clone(), meshColors[c]!.clone()] as [Vector4, Vector4, Vector4]
      : undefined;
    rawMesh.faces.push([a, b, c] as FaceIndices);
    faceAttributes.push({
      materialId: 0,
      uvSets: uvAccessors.map(({ texCoord, accessor }) => ({
        texCoord,
        uvs: [readUv(accessor, a), readUv(accessor, b), readUv(accessor, c)] as [Vector2, Vector2, Vector2],
      })),
      ...(normalCorners ? { normalCorners } : {}),
      ...(tangentCorners ? { tangentCorners } : {}),
      ...(colorCorners && colorItemSize !== undefined ? { colorCorners, colorItemSize } : {}),
      ...(normalMapYScale !== undefined ? { normalMapYScale } : {}),
    });
  }

  const result: ScenePrimitiveEntry = {
    id: entry.id,
    document: entry.document,
    mesh: entry.mesh,
    primitive: entry.primitive,
    primitiveIndex: entry.primitiveIndex,
    scene: entry.scene,
    node: entry.node,
    parentNode: entry.parentNode,
    rawMesh,
    faceAttributes,
    sourceMaterial,
    originalMaterial: material,
  };
  if (entry.nodeName !== undefined) result.nodeName = entry.nodeName;
  return result;
}

interface NodeInstance {
  scene: Scene;
  node: GltfNode;
  parentNode: GltfNode | null;
}

function collectSceneNodeInstances(scenes: Scene[]): NodeInstance[] {
  const instances: NodeInstance[] = [];
  const visit = (scene: Scene, node: GltfNode, parentNode: GltfNode | null): void => {
    instances.push({ scene, node, parentNode });
    for (const child of node.listChildren()) visit(scene, child, node);
  };
  for (const scene of scenes) {
    for (const child of scene.listChildren()) visit(scene, child, null);
  }
  return instances;
}

export async function readGlbScenePrimitives(
  inputPath: string,
  options: ReadGlbScenePrimitiveOptions = {},
): Promise<ScenePrimitiveReadResult> {
  const document = await new NodeIO().read(inputPath);
  const mode = options.mode ?? 'bake';
  const root = document.getRoot();
  const nodes = root.listNodes();
  const nodeInstances = collectSceneNodeInstances(root.listScenes());
  const textureImages = new Map<Texture, RgbaImage>();
  const entries: ScenePrimitiveEntry[] = [];
  const warnings: ScenePrimitiveWarning[] = [];

  for (const { scene, node, parentNode } of nodeInstances) {
    const mesh = node.getMesh();
    if (!mesh) continue;
    const primitives = mesh.listPrimitives();
    for (let primitiveIndex = 0; primitiveIndex < primitives.length; primitiveIndex += 1) {
      const primitive = primitives[primitiveIndex]!;
      const reason = unsupportedReason(primitive, mesh, nodes);
      if (reason) {
        const warning: ScenePrimitiveWarning = {
          meshName: mesh.getName(),
          primitiveIndex,
          reason,
        };
        const nodeName = node?.getName();
        if (nodeName !== undefined) warning.nodeName = nodeName;
        warnings.push(warning);
        continue;
      }
      const readOptions: {
        document: Document;
        mesh: Mesh;
        primitive: Primitive;
        primitiveIndex: number;
        scene: Scene;
        node: GltfNode;
        parentNode: GltfNode | null;
        nodeName?: string;
        id: number;
        textureImages: Map<Texture, RgbaImage>;
        mode: PrimitiveExtractionMode;
      } = {
        document,
        mesh,
        primitive,
        primitiveIndex,
        scene,
        node,
        parentNode,
        id: entries.length,
        textureImages,
        mode,
      };
      const nodeName = node?.getName();
      if (nodeName !== undefined) readOptions.nodeName = nodeName;
      entries.push(readLocalPrimitive(readOptions));
    }
  }

  return { document, entries, warnings };
}

function objectIds<T extends object>(values: readonly T[]): Map<T, number> {
  const ids = new Map<T, number>();
  for (const value of values) {
    if (!ids.has(value)) ids.set(value, ids.size);
  }
  return ids;
}

function groupTransform(entry: ScenePrimitiveEntry, mode: PrimitiveGroupingMode): Matrix4 {
  return mode === 'material-parent'
    ? new Matrix4().fromArray(entry.node.getMatrix())
    : new Matrix4().fromArray(entry.node.getWorldMatrix());
}

function transformFaceNormals(
  normalCorners: [Vector3, Vector3, Vector3] | undefined,
  transform: Matrix4,
): [Vector3, Vector3, Vector3] | undefined {
  if (!normalCorners) return undefined;
  const normalMatrix = new Matrix3().getNormalMatrix(transform);
  return normalCorners.map((normal) => normal.clone().applyNormalMatrix(normalMatrix).normalize()) as [Vector3, Vector3, Vector3];
}

function cloneFaceNormals(
  normalCorners: [Vector3, Vector3, Vector3] | undefined,
): [Vector3, Vector3, Vector3] | undefined {
  if (!normalCorners) return undefined;
  return normalCorners.map((normal) => normal.clone()) as [Vector3, Vector3, Vector3];
}

function transformFaceTangents(
  tangentCorners: [Vector4, Vector4, Vector4] | undefined,
  transform: Matrix4,
): [Vector4, Vector4, Vector4] | undefined {
  if (!tangentCorners) return undefined;
  return tangentCorners.map((tangent) => transformAuthoredTangent(tangent, transform)) as [Vector4, Vector4, Vector4];
}

function cloneFaceTangents(
  tangentCorners: [Vector4, Vector4, Vector4] | undefined,
): [Vector4, Vector4, Vector4] | undefined {
  if (!tangentCorners) return undefined;
  return tangentCorners.map((tangent) => tangent.clone()) as [Vector4, Vector4, Vector4];
}

function cloneFaceColors(
  colorCorners: [Vector4, Vector4, Vector4] | undefined,
): [Vector4, Vector4, Vector4] | undefined {
  if (!colorCorners) return undefined;
  return colorCorners.map((color) => color.clone()) as [Vector4, Vector4, Vector4];
}

function cloneFaceAttributes(
  faceAttributes: readonly SourceFaceAttributes[],
  materialId?: number,
  transform?: Matrix4,
): SourceFaceAttributes[] {
  return faceAttributes.map((attributes) => {
    const normalCorners = transform
      ? transformFaceNormals(attributes.normalCorners, transform)
      : cloneFaceNormals(attributes.normalCorners);
    const tangentCorners = transform
      ? transformFaceTangents(attributes.tangentCorners, transform)
      : cloneFaceTangents(attributes.tangentCorners);
    const colorCorners = cloneFaceColors(attributes.colorCorners);
    return {
      materialId: materialId ?? attributes.materialId,
      uvSets: attributes.uvSets.map((uvSet) => ({
        texCoord: uvSet.texCoord,
        uvs: [uvSet.uvs[0].clone(), uvSet.uvs[1].clone(), uvSet.uvs[2].clone()] as [Vector2, Vector2, Vector2],
      })),
      ...(normalCorners ? { normalCorners } : {}),
      ...(tangentCorners ? { tangentCorners } : {}),
      ...(colorCorners && attributes.colorItemSize !== undefined ? { colorCorners, colorItemSize: attributes.colorItemSize } : {}),
      ...(attributes.normalMapYScale !== undefined ? { normalMapYScale: attributes.normalMapYScale } : {}),
    };
  });
}

function entryColorSchemaKey(entry: ScenePrimitiveEntry): string {
  const schemas = new Set(entry.faceAttributes.map((attributes) => (
    attributes.colorCorners ? `c${attributes.colorItemSize ?? 3}` : 'c0'
  )));
  return Array.from(schemas).sort().join('+');
}

export function groupScenePrimitives(
  entries: readonly ScenePrimitiveEntry[],
  mode: PrimitiveGroupingMode,
  options: GroupScenePrimitiveOptions = {},
): ScenePrimitiveGroup[] {
  const weldVertices = options.weldVertices ?? true;
  if (mode === 'none') {
    return entries.map((entry) => {
      const rawMesh: RawMesh = { positions: [], faces: [] };
      appendTransformedRawMesh(
        rawMesh,
        entry.rawMesh,
        new Matrix4().identity(),
        createWeldedVertexAppendState(),
        { weldVertices },
      );
      return {
        id: String(entry.id),
        mode,
        document: entry.document,
        scene: entry.scene,
        parentNode: entry.parentNode,
        entries: [entry],
        rawMesh,
        faceAttributes: cloneFaceAttributes(entry.faceAttributes),
        sourceMaterial: entry.sourceMaterial,
        originalMaterial: entry.originalMaterial,
      };
    });
  }

  const materialIds = objectIds(entries.map((entry) => entry.originalMaterial ?? entry.document));
  const parentIds = objectIds(entries.map((entry) => entry.parentNode ?? entry.scene));
  const groups = new Map<string, ScenePrimitiveGroup>();
  const appendStates = new Map<string, ReturnType<typeof createWeldedVertexAppendState>>();
  for (const entry of entries) {
    const materialId = materialIds.get(entry.originalMaterial ?? entry.document)!;
    const baseKey = mode === 'material-parent'
      ? `${materialId}/${parentIds.get(entry.parentNode ?? entry.scene)!}`
      : `${materialId}`;
    const colorSchemaKey = entryColorSchemaKey(entry);
    const key = colorSchemaKey === 'c0' ? baseKey : `${baseKey}/${colorSchemaKey}`;
    let group = groups.get(key);
    if (!group) {
      group = {
        id: key,
        mode,
        document: entry.document,
        scene: entry.scene,
        parentNode: mode === 'material-parent' ? entry.parentNode : null,
        entries: [],
        rawMesh: { positions: [], faces: [] },
        faceAttributes: [],
        sourceMaterial: entry.sourceMaterial,
        originalMaterial: entry.originalMaterial,
      };
      groups.set(key, group);
      appendStates.set(key, createWeldedVertexAppendState());
    }
    group.entries.push(entry);
    const transform = groupTransform(entry, mode);
    appendTransformedRawMesh(group.rawMesh, entry.rawMesh, transform, appendStates.get(key)!, { weldVertices });
    group.faceAttributes.push(...cloneFaceAttributes(entry.faceAttributes, 0, transform));
  }
  return [...groups.values()];
}

function outputBuffer(document: Document) {
  return document.getRoot().listBuffers()[0] ?? document.createBuffer();
}

function createPositionAccessor(document: Document, positions: Float32Array): Accessor {
  return document.createAccessor('POSITION')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array(positions))
    .setBuffer(outputBuffer(document));
}

function createNormalAccessor(document: Document, normals: Float32Array): Accessor {
  return document.createAccessor('NORMAL')
    .setType(Accessor.Type.VEC3!)
    .setArray(new Float32Array(normals))
    .setBuffer(outputBuffer(document));
}

function createTangentAccessor(document: Document, tangents: Float32Array): Accessor {
  return document.createAccessor('TANGENT')
    .setType(Accessor.Type.VEC4!)
    .setArray(new Float32Array(tangents))
    .setBuffer(outputBuffer(document));
}

function createColorAccessor(document: Document, colors: Float32Array, itemSize: VertexColorItemSize): Accessor {
  return document.createAccessor('COLOR_0')
    .setType(itemSize === 4 ? Accessor.Type.VEC4! : Accessor.Type.VEC3!)
    .setArray(new Float32Array(colors))
    .setBuffer(outputBuffer(document));
}

function createIndexAccessor(document: Document, indices: Uint16Array | Uint32Array): Accessor {
  const array = indices instanceof Uint32Array ? new Uint32Array(indices) : new Uint16Array(indices);
  return document.createAccessor('indices')
    .setType(Accessor.Type.SCALAR!)
    .setArray(array)
    .setBuffer(outputBuffer(document));
}

function createUvAccessor(document: Document, texCoord: number, uvs: Float32Array): Accessor {
  return document.createAccessor(`TEXCOORD_${texCoord}`)
    .setType(Accessor.Type.VEC2!)
    .setArray(new Float32Array(uvs))
    .setBuffer(outputBuffer(document));
}

function clearVertexAttributes(primitive: Primitive): void {
  for (const semantic of primitive.listSemantics()) primitive.setAttribute(semantic, null);
}

function requiredMaterialTexCoords(material: Material | null): number[] {
  if (!material) return [];
  return [
    material.getBaseColorTextureInfo(),
    material.getNormalTextureInfo(),
    material.getMetallicRoughnessTextureInfo(),
    material.getOcclusionTextureInfo(),
    material.getEmissiveTextureInfo(),
  ]
    .filter((info): info is TextureInfo => info !== null)
    .map((info) => info.getTexCoord());
}

function materialReferencesNormalMap(material: Material | null): boolean {
  return material !== null && material.getNormalTexture() !== null;
}

function createIndexedGeometryPrimitive(document: Document, outputRawMesh: RawMesh, material: Material | null): Primitive {
  const data = buildIndexedPrimitiveGeometryData(outputRawMesh);

  return document.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', createPositionAccessor(document, data.positions))
    .setAttribute('NORMAL', createNormalAccessor(document, data.normals))
    .setIndices(createIndexAccessor(document, data.indices))
    .setMaterial(material);
}

function createGeometryPrimitive(
  document: Document,
  outputRawMesh: RawMesh,
  material: Material | null,
  transferredAttributes?: TransferredMeshAttributes,
): Primitive {
  if (!transferredAttributes) return createIndexedGeometryPrimitive(document, outputRawMesh, material);
  const emitTangents = materialReferencesNormalMap(material) || transferredAttributes.hasSourceTangents === true;
  const data = buildAttributeTransferredPrimitiveGeometryData(outputRawMesh, transferredAttributes, {
    requiredTexCoords: requiredMaterialTexCoords(material),
    emitTangents,
    ...(emitTangents ? { tangentHandednessScale: -1 as const } : {}),
  });

  const primitive = document.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', createPositionAccessor(document, data.positions))
    .setAttribute('NORMAL', createNormalAccessor(document, data.normals))
    .setIndices(createIndexAccessor(document, data.indices))
    .setMaterial(material);
  for (const [texCoord, texCoords] of data.texCoordsBySet) {
    primitive.setAttribute(`TEXCOORD_${texCoord}`, createUvAccessor(document, texCoord, texCoords));
  }
  if (data.tangents) primitive.setAttribute('TANGENT', createTangentAccessor(document, data.tangents));
  if (data.colors && data.colorItemSize) primitive.setAttribute('COLOR_0', createColorAccessor(document, data.colors, data.colorItemSize));
  return primitive;
}

function applyPrimitiveReplacement(target: Primitive, replacement: Primitive): void {
  clearVertexAttributes(target);
  target.setMode(Primitive.Mode.TRIANGLES!);
  for (const semantic of replacement.listSemantics()) {
    target.setAttribute(semantic, replacement.getAttribute(semantic));
  }
  target
    .setIndices(replacement.getIndices())
    .setMaterial(replacement.getMaterial());
}

export function replaceScenePrimitiveGeometry(
  entry: ScenePrimitiveEntry,
  outputRawMesh: RawMesh,
  transferredAttributes?: TransferredMeshAttributes,
): void {
  const replacement = createGeometryPrimitive(
    entry.document,
    outputRawMesh,
    entry.originalMaterial,
    transferredAttributes,
  );
  applyPrimitiveReplacement(entry.primitive, replacement);
}

function createTexturedPrimitive(document: Document, sourceMaterial: SourceMaterial, options: {
  outputRawMesh: RawMesh;
  atlas: AtlasLayout;
  image: RgbaImage;
  additionalTextures?: readonly BakedMaterialTexture[];
  transferredAttributes?: TransferredMeshAttributes;
  materialName?: string;
}): Primitive {
  const texture = createBakedTexture(document, `${sourceMaterial.name}-baked-base-color`, options.image);
  const material = document.createMaterial(options.materialName ?? `${sourceMaterial.name}-baked`)
    .setBaseColorTexture(texture)
    .setBaseColorFactor([1, 1, 1, 1])
    .setAlphaMode(sourceMaterial.alphaMode)
    .setAlphaCutoff(sourceMaterial.alphaCutoff)
    .setDoubleSided(sourceMaterial.doubleSided)
    .setEmissiveFactor(sourceMaterial.emissiveFactor)
    .setMetallicFactor(sourceMaterial.metallicFactor)
    .setRoughnessFactor(sourceMaterial.roughnessFactor)
    .setNormalScale(sourceMaterial.normalScale)
    .setOcclusionStrength(sourceMaterial.occlusionStrength);
  configureBakedTextureInfo(material.getBaseColorTextureInfo());
  attachAdditionalBakedTextures(
    document,
    material,
    options.additionalTextures ?? [],
    sourceMaterial,
    `${sourceMaterial.name}-baked`,
  );
  const data = buildAtlasPrimitiveGeometryData(
    options.outputRawMesh,
    options.atlas,
    options.additionalTextures,
    options.transferredAttributes,
  );

  const primitive = document.createPrimitive()
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', createPositionAccessor(document, data.positions))
    .setAttribute('NORMAL', createNormalAccessor(document, data.normals))
    .setAttribute('TEXCOORD_0', createUvAccessor(document, 0, data.texCoordsBySet.get(0) ?? new Float32Array()))
    .setIndices(createIndexAccessor(document, data.indices))
    .setMaterial(material);
  if (data.tangents) primitive.setAttribute('TANGENT', createTangentAccessor(document, data.tangents));
  if (data.colors && data.colorItemSize) primitive.setAttribute('COLOR_0', createColorAccessor(document, data.colors, data.colorItemSize));
  return primitive;
}

export function replaceScenePrimitiveTextured(entry: ScenePrimitiveEntry, options: {
  outputRawMesh: RawMesh;
  atlas: AtlasLayout;
  image: RgbaImage;
  additionalTextures?: readonly BakedMaterialTexture[];
  transferredAttributes?: TransferredMeshAttributes;
  materialName?: string;
}): void {
  clearVertexAttributes(entry.primitive);
  const replacement = createTexturedPrimitive(entry.document, entry.sourceMaterial, options);
  entry.primitive
    .setMode(Primitive.Mode.TRIANGLES!)
    .setAttribute('POSITION', replacement.getAttribute('POSITION'))
    .setAttribute('NORMAL', replacement.getAttribute('NORMAL'))
    .setAttribute('TEXCOORD_0', replacement.getAttribute('TEXCOORD_0'))
    .setAttribute('TANGENT', replacement.getAttribute('TANGENT'))
    .setAttribute('COLOR_0', replacement.getAttribute('COLOR_0'))
    .setIndices(replacement.getIndices())
    .setMaterial(replacement.getMaterial());
}

function removeEmptySourceNodes(group: ScenePrimitiveGroup): void {
  const seen = new Set<GltfNode>();
  for (const entry of group.entries) {
    if (seen.has(entry.node)) continue;
    seen.add(entry.node);
    const mesh = entry.node.getMesh();
    if (!mesh || mesh.listPrimitives().length > 0) continue;
    if (entry.node.listChildren().length > 0) {
      entry.node.setMesh(null);
    } else if (entry.parentNode) {
      entry.parentNode.removeChild(entry.node);
    } else {
      entry.scene.removeChild(entry.node);
    }
  }
}

function removeProcessedPrimitives(group: ScenePrimitiveGroup): void {
  for (const entry of group.entries) {
    entry.mesh.removePrimitive(entry.primitive);
    entry.primitive.dispose();
  }
  removeEmptySourceNodes(group);
}

function addGroupReplacement(group: ScenePrimitiveGroup, primitive: Primitive, name: string): void {
  removeProcessedPrimitives(group);
  const mesh = group.document.createMesh(name).addPrimitive(primitive);
  const node = group.document.createNode(name).setMesh(mesh);
  if (group.parentNode) {
    group.parentNode.addChild(node);
  } else {
    group.scene.addChild(node);
  }
}

function hasNonRootParent(document: Document, property: Property): boolean {
  const root = document.getRoot();
  return property.listParents().some((parent) => parent !== root);
}

function reachableSceneNodes(document: Document): Set<GltfNode> {
  const reachable = new Set<GltfNode>();
  for (const scene of document.getRoot().listScenes()) {
    scene.traverse((node) => {
      reachable.add(node);
    });
  }
  return reachable;
}

export function pruneDetachedSceneResources(document: Document): void {
  const root = document.getRoot();
  const reachableNodes = reachableSceneNodes(document);
  for (const node of root.listNodes()) {
    if (!reachableNodes.has(node)) node.dispose();
  }

  let pruned = true;
  while (pruned) {
    pruned = false;
    const pruneIf = <T extends Property>(properties: readonly T[], predicate: (property: T) => boolean): void => {
      for (const property of properties) {
        if (!property.isDisposed() && predicate(property)) {
          property.dispose();
          pruned = true;
        }
      }
    };

    pruneIf(root.listMeshes(), (mesh) => mesh.listPrimitives().length === 0 || !hasNonRootParent(document, mesh));
    pruneIf(root.listMaterials(), (material) => !hasNonRootParent(document, material));
    pruneIf(root.listTextures(), (texture) => !hasNonRootParent(document, texture));
    pruneIf(root.listAccessors(), (accessor) => !hasNonRootParent(document, accessor));
    pruneIf(root.listBuffers(), (buffer) => !hasNonRootParent(document, buffer));
  }
}

export function replaceScenePrimitiveGroupGeometry(
  group: ScenePrimitiveGroup,
  outputRawMesh: RawMesh,
  transferredAttributes?: TransferredMeshAttributes,
): void {
  if (group.mode === 'none') {
    const entry = group.entries[0];
    if (!entry || group.entries.length !== 1) {
      throw new Error('None-grouped primitive replacement requires exactly one source entry.');
    }
    replaceScenePrimitiveGeometry(entry, outputRawMesh, transferredAttributes);
    return;
  }
  const primitive = createGeometryPrimitive(group.document, outputRawMesh, group.originalMaterial, transferredAttributes);
  addGroupReplacement(group, primitive, `simplified-${group.id}`);
}

export function replaceScenePrimitiveGroupTextured(group: ScenePrimitiveGroup, options: {
  outputRawMesh: RawMesh;
  atlas: AtlasLayout;
  image: RgbaImage;
  additionalTextures?: readonly BakedMaterialTexture[];
  transferredAttributes?: TransferredMeshAttributes;
  materialName?: string;
}): void {
  if (group.mode === 'none') {
    const entry = group.entries[0];
    if (!entry || group.entries.length !== 1) {
      throw new Error('None-grouped primitive replacement requires exactly one source entry.');
    }
    replaceScenePrimitiveTextured(entry, options);
    return;
  }
  const primitive = createTexturedPrimitive(group.document, group.sourceMaterial, options);
  addGroupReplacement(group, primitive, `simplified-${group.id}`);
}

export async function writeScenePrimitiveDocument(document: Document, outputPath: string): Promise<void> {
  await mkdir(dirname(outputPath), { recursive: true });
  pruneDetachedSceneResources(document);
  await new NodeIO().write(outputPath, document);
}
