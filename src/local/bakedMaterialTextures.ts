import { TextureInfo, type Document, type Material, type Texture } from '@gltf-transform/core';
import type { BakedMaterialTexture, RgbaImage } from '../texture/types';
import { encodePng } from './imageCodecs';

export interface BakedMaterialTextureScalars {
  normalScale: number;
  occlusionStrength: number;
  emissiveFactor: [number, number, number];
}

export function createBakedTexture(document: Document, name: string, image: RgbaImage): Texture {
  return document.createTexture(name)
    .setImage(encodePng(image))
    .setMimeType('image/png');
}

export function configureBakedTextureInfo(info: TextureInfo | null): void {
  info
    ?.setTexCoord(0)
    .setWrapS(TextureInfo.WrapMode.CLAMP_TO_EDGE!)
    .setWrapT(TextureInfo.WrapMode.CLAMP_TO_EDGE!)
    .setMagFilter(TextureInfo.MagFilter.LINEAR!)
    .setMinFilter(TextureInfo.MinFilter.LINEAR!);
}

export function attachAdditionalBakedTextures(
  document: Document,
  material: Material,
  textures: readonly BakedMaterialTexture[],
  scalars: BakedMaterialTextureScalars,
  textureNamePrefix: string,
): void {
  for (const texture of textures) {
    const bakedTexture = createBakedTexture(document, `${textureNamePrefix}-${texture.slot}`, texture.image);
    if (texture.slot === 'normal') {
      material.setNormalTexture(bakedTexture).setNormalScale(scalars.normalScale);
      configureBakedTextureInfo(material.getNormalTextureInfo());
    } else if (texture.slot === 'metallicRoughness') {
      material.setMetallicRoughnessTexture(bakedTexture);
      configureBakedTextureInfo(material.getMetallicRoughnessTextureInfo());
    } else if (texture.slot === 'occlusion') {
      material.setOcclusionTexture(bakedTexture).setOcclusionStrength(scalars.occlusionStrength);
      configureBakedTextureInfo(material.getOcclusionTextureInfo());
    } else if (texture.slot === 'emissive') {
      material.setEmissiveTexture(bakedTexture).setEmissiveFactor(scalars.emissiveFactor);
      configureBakedTextureInfo(material.getEmissiveTextureInfo());
    }
  }
}
