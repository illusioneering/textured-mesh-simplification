export type ExternalGltfResourceKind = 'binary-buffer' | 'texture-image';

export function isSupportedTextureFileName(name: string): boolean {
  return /\.(png|jpe?g|webp)$/i.test(name);
}

export function isSupportedBinaryBufferFileName(name: string): boolean {
  return /\.bin$/i.test(name);
}

export function classifyExternalGltfResourceFileName(name: string): ExternalGltfResourceKind | undefined {
  if (isSupportedBinaryBufferFileName(name)) return 'binary-buffer';
  if (isSupportedTextureFileName(name)) return 'texture-image';
  return undefined;
}

export function isSupportedExternalResourceFileName(name: string): boolean {
  return classifyExternalGltfResourceFileName(name) !== undefined;
}

export function requestedExternalResourceFileName(uri: string): string | undefined {
  if (/^(data|blob):/i.test(uri)) return undefined;
  const withoutFragment = uri.split('#')[0] ?? uri;
  const withoutQuery = withoutFragment.split('?')[0] ?? withoutFragment;
  const normalized = withoutQuery.replace(/\\/g, '/');
  const encodedName = normalized.split('/').filter(Boolean).pop();
  if (!encodedName) return undefined;
  try {
    return decodeURIComponent(encodedName);
  } catch {
    return encodedName;
  }
}
