import { PNG } from 'pngjs';
import * as jpeg from 'jpeg-js';
import type { RgbaImage } from '../texture/types';

function cloneBytes(data: Uint8Array): Uint8Array {
  return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

export function decodeImage(bytes: Uint8Array, mimeType: string): RgbaImage {
  const normalized = mimeType.toLowerCase();
  if (normalized === 'image/png') {
    const decoded = PNG.sync.read(Buffer.from(bytes));
    return {
      width: decoded.width,
      height: decoded.height,
      data: cloneBytes(decoded.data),
    };
  }
  if (normalized === 'image/jpeg' || normalized === 'image/jpg') {
    const decoded = jpeg.decode(Buffer.from(bytes), { useTArray: true });
    return {
      width: decoded.width,
      height: decoded.height,
      data: cloneBytes(decoded.data),
    };
  }
  throw new Error(`Unsupported image MIME type "${mimeType}". Only PNG and JPEG are supported.`);
}

export function encodePng(image: RgbaImage): Uint8Array {
  if (!Number.isInteger(image.width) || image.width <= 0 || !Number.isInteger(image.height) || image.height <= 0) {
    throw new Error('PNG image dimensions must be positive integers.');
  }
  if (image.data.length !== image.width * image.height * 4) {
    throw new Error(`RGBA image data length ${image.data.length} does not match ${image.width}x${image.height}.`);
  }
  const png = new PNG({ width: image.width, height: image.height });
  png.data.set(image.data);
  return new Uint8Array(PNG.sync.write(png));
}
