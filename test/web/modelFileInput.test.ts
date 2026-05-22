import { describe, expect, it } from 'vitest';
import {
  isSupportedBinaryBufferFileName,
  isSupportedExternalResourceFileName,
  isSupportedModelFileName,
  isSupportedTextureFileName,
  pickModelUploadFiles,
} from '../../src/web/modelFileInput';

function file(name: string): File {
  return new File(['model'], name);
}

describe('web model file input helpers', () => {
  it('accepts GLB and GLTF filenames case-insensitively', () => {
    expect(isSupportedModelFileName('mesh.glb')).toBe(true);
    expect(isSupportedModelFileName('scene.GLTF')).toBe(true);
    expect(isSupportedModelFileName('texture.png')).toBe(false);
  });

  it('accepts common web texture image filenames case-insensitively', () => {
    expect(isSupportedTextureFileName('baseColor.png')).toBe(true);
    expect(isSupportedTextureFileName('albedo.JPG')).toBe(true);
    expect(isSupportedTextureFileName('normal.jpeg')).toBe(true);
    expect(isSupportedTextureFileName('preview.webp')).toBe(true);
    expect(isSupportedTextureFileName('scene.glb')).toBe(false);
  });

  it('accepts external glTF binary buffer filenames case-insensitively', () => {
    expect(isSupportedBinaryBufferFileName('mesh.bin')).toBe(true);
    expect(isSupportedBinaryBufferFileName('MESH.BIN')).toBe(true);
    expect(isSupportedBinaryBufferFileName('mesh.glb')).toBe(false);
    expect(isSupportedExternalResourceFileName('mesh.bin')).toBe(true);
    expect(isSupportedExternalResourceFileName('baseColor.png')).toBe(true);
    expect(isSupportedExternalResourceFileName('scene.gltf')).toBe(false);
  });

  it('classifies a model file with external resource candidates', () => {
    const result = pickModelUploadFiles([
      file('readme.txt'),
      file('mesh.gltf'),
      file('mesh.bin'),
      file('baseColor.png'),
      file('roughness.JPG'),
      file('extra.glb'),
    ]);

    expect(result.modelFile?.name).toBe('mesh.gltf');
    expect(result.externalResourceFiles.map((candidate) => candidate.name)).toEqual(['mesh.bin', 'baseColor.png', 'roughness.JPG']);
    expect(result.binaryBufferFiles.map((candidate) => candidate.name)).toEqual(['mesh.bin']);
    expect(result.textureFiles.map((candidate) => candidate.name)).toEqual(['baseColor.png', 'roughness.JPG']);
    expect(result.skippedUnsupported).toBe(1);
    expect(result.skippedModelFiles).toBe(1);
  });

  it('keeps external resources even when no model file is selected', () => {
    const result = pickModelUploadFiles([file('baseColor.png'), file('mesh.bin')]);

    expect(result.modelFile).toBeNull();
    expect(result.externalResourceFiles.map((candidate) => candidate.name)).toEqual(['baseColor.png', 'mesh.bin']);
    expect(result.textureFiles.map((candidate) => candidate.name)).toEqual(['baseColor.png']);
    expect(result.binaryBufferFiles.map((candidate) => candidate.name)).toEqual(['mesh.bin']);
    expect(result.skippedUnsupported).toBe(0);
    expect(result.skippedModelFiles).toBe(0);
  });

  it('reports empty or unsupported uploads without selecting a model file', () => {
    expect(pickModelUploadFiles([])).toEqual({
      modelFile: null,
      externalResourceFiles: [],
      textureFiles: [],
      binaryBufferFiles: [],
      skippedUnsupported: 0,
      skippedModelFiles: 0,
    });
    expect(pickModelUploadFiles([file('notes.txt')])).toEqual({
      modelFile: null,
      externalResourceFiles: [],
      textureFiles: [],
      binaryBufferFiles: [],
      skippedUnsupported: 1,
      skippedModelFiles: 0,
    });
  });
});
