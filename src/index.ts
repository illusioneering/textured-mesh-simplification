export type {
  Barycentric,
  CollapseHistoryRecord,
  FaceIndices,
  FaceSnapshot,
  RawMesh,
  RawSimplificationResult,
  SimplificationProgress,
  SimplificationResult,
  SimplifyOptions,
} from './simplification/types';
export { geometryToRawMesh, rawMeshToGeometry } from './simplification/geometryConversion';
export { computeAreaWeightedVertexNormals } from './simplification/normals';
export { simplifyGeometry, simplifyRawMesh } from './simplification/simplify';
export { barycentricForPoint, clampBarycentricToTriangle, interpolateVector2, pointFromBarycentric } from './simplification/barycentric';
export { closestPointOnTriangle, type ClosestPointOnTriangleResult } from './simplification/projection';
export {
  createHistoryTraceIndex,
  mapOutputSampleToInput,
  type HistoryTraceIndex,
  type MappedSample,
} from './simplification/successiveMapping';
export type {
  SourceFaceColorCorners,
  SourceFaceAttributes,
  SourceFaceUvSet,
  TransferredMeshAttributes,
  TransferredVertexAttributes,
  TransferredVertexUvSet,
  VertexColorItemSize,
} from './simplification/attributes';
export { faceUvSet, transferredVertexColor, transferredVertexUv } from './simplification/attributes';
export { transferVertexAttributesToSimplifiedMesh, type AttributeTransferProgress } from './simplification/attributeTransfer';

export type {
  AtlasLayout,
  BakedMaterialTexture,
  BakedTextureResult,
  Rgba,
  RgbaImage,
  SourceMaterial,
  SourceMaterialTextureInfo,
  SourceMaterialTextureSlot,
  SourceTexture,
  StandardBakedTextureSlot,
  TextureSampler,
  TexturedRawMesh,
  WrapMode,
} from './texture/types';
export { hasMaterialTextures } from './texture/types';
export { applyWrap, sampleImageBilinear, sampleImageNearest, sampleSourceBaseColor, sampleSourceMaterialTexture } from './texture/sampling';
export { countSerializedTexturedVertices, createInjectiveAtlas, type AtlasOptions } from './texture/atlas';
export { rasterizeAtlasTriangle, type RasterSample } from './texture/rasterize';
export { bakeBaseColorTexture, bakeStandardMaterialTextures, type BakeTextureOptions } from './texture/bake';
export {
  computeFaceTangentFrame,
  computeSerializedTangents,
  normalRgbToVector,
  tangentNormalToWorld,
  transformTangentSpaceNormal,
  vectorToNormalRgb,
  worldNormalToTangent,
  type FaceTangentFrameInput,
  type SerializedTangentsInput,
  type TangentFrame,
} from './texture/tangentSpace';
