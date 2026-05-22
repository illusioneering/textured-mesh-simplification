import { describe, expect, it } from 'vitest';
import { decodeImage, encodePng } from '../../src/local/imageCodecs';

const rgba2x2 = {
  width: 2,
  height: 2,
  data: new Uint8Array([
    255, 0, 0, 255, 0, 255, 0, 255,
    0, 0, 255, 255, 255, 255, 255, 128,
  ]),
};

describe('local image codecs', () => {
  it('round-trips RGBA PNG bytes', () => {
    const encoded = encodePng(rgba2x2);
    const decoded = decodeImage(encoded, 'image/png');
    expect(decoded.width).toBe(2);
    expect(decoded.height).toBe(2);
    expect(Array.from(decoded.data)).toEqual(Array.from(rgba2x2.data));
  });

  it('throws for unsupported image MIME types', () => {
    expect(() => decodeImage(new Uint8Array(), 'image/gif')).toThrow(/unsupported/i);
  });
});
