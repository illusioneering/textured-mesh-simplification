import { classifyExternalGltfResourceFileName } from './externalGltfResources';
export {
  isSupportedBinaryBufferFileName,
  isSupportedExternalResourceFileName,
  isSupportedTextureFileName,
} from './externalGltfResources';

export interface ModelUploadFileSelection {
  modelFile: File | null;
  externalResourceFiles: File[];
  textureFiles: File[];
  binaryBufferFiles: File[];
  skippedUnsupported: number;
  skippedModelFiles: number;
}

export function isSupportedModelFileName(name: string): boolean {
  return /\.(glb|gltf)$/i.test(name);
}

export function pickModelUploadFiles(files: ArrayLike<File>): ModelUploadFileSelection {
  let modelFile: File | null = null;
  const externalResourceFiles: File[] = [];
  const textureFiles: File[] = [];
  const binaryBufferFiles: File[] = [];
  let skippedUnsupported = 0;
  let skippedModelFiles = 0;

  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    if (!file) continue;
    if (isSupportedModelFileName(file.name)) {
      if (modelFile) {
        skippedModelFiles += 1;
      } else {
        modelFile = file;
      }
      continue;
    }

    const resourceKind = classifyExternalGltfResourceFileName(file.name);
    if (resourceKind === 'texture-image') {
      externalResourceFiles.push(file);
      textureFiles.push(file);
    } else if (resourceKind === 'binary-buffer') {
      externalResourceFiles.push(file);
      binaryBufferFiles.push(file);
    } else {
      skippedUnsupported += 1;
    }
  }

  return { modelFile, externalResourceFiles, textureFiles, binaryBufferFiles, skippedUnsupported, skippedModelFiles };
}
