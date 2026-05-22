import { describe, expect, it } from 'vitest';
import {
  Color,
  BoxGeometry,
  DoubleSide,
  FrontSide,
  Group,
  Mesh,
  MeshStandardMaterial,
  Texture,
} from 'three';
import {
  allPbrMaterialProperties,
  cloneMaterialForPbrPreview,
  collectMaterialsForPbrControls,
  defaultPbrMaterialPropertyState,
  detectPbrMaterialPropertyAvailability,
  textureSizeForLargestBaseColorMap,
  textureSizeOptions,
  type TextureSizeOption,
} from '../../src/web/pbrControls';

function texture(width = 1, height = 1): Texture {
  const result = new Texture({ width, height });
  result.needsUpdate = true;
  return result;
}

describe('PBR controls helpers', () => {
  it('defines PBR material property metadata in UI order', () => {
    expect(textureSizeOptions).toEqual([1024, 2048, 4096, 8192]);
    expect(allPbrMaterialProperties).toEqual([
      { id: 'baseColor', label: 'Base color' },
      { id: 'metallic', label: 'Metallic' },
      { id: 'roughness', label: 'Roughness' },
      { id: 'normal', label: 'Normal' },
      { id: 'occlusion', label: 'Occlusion' },
      { id: 'emissive', label: 'Emissive' },
      { id: 'alpha', label: 'Alpha' },
      { id: 'doubleSided', label: 'Double-sided' },
    ]);
  });

  it('detects non-default PBR scalar and texture properties on standard materials', () => {
    const material = new MeshStandardMaterial({
      color: 0xff00ff,
      map: texture(),
      metalness: 0.25,
      metalnessMap: texture(),
      roughness: 0.5,
      roughnessMap: texture(),
      normalMap: texture(),
      aoMap: texture(),
      emissive: 0x220011,
      emissiveMap: texture(),
      transparent: true,
      opacity: 0.5,
      alphaTest: 0.2,
      side: DoubleSide,
    });

    expect(detectPbrMaterialPropertyAvailability([material])).toEqual({
      baseColor: true,
      metallic: true,
      roughness: true,
      normal: true,
      occlusion: true,
      emissive: true,
      alpha: true,
      doubleSided: true,
    });
  });

  it('does not detect explicit glTF metallic and roughness defaults as present', () => {
    const material = new MeshStandardMaterial({ metalness: 1, roughness: 1 });

    expect(detectPbrMaterialPropertyAvailability([material])).toEqual(
      defaultPbrMaterialPropertyState(false),
    );
  });

  it('clones materials for PBR preview and resets unchecked properties to glTF defaults', () => {
    const source = new MeshStandardMaterial({
      color: 0x336699,
      map: texture(),
      metalness: 0.25,
      metalnessMap: texture(),
      roughness: 0.5,
      roughnessMap: texture(),
      normalMap: texture(),
      aoMap: texture(),
      emissive: 0x220011,
      emissiveMap: texture(),
      transparent: true,
      opacity: 0.5,
      alphaTest: 0.2,
      side: DoubleSide,
    });

    const clone = cloneMaterialForPbrPreview(source, defaultPbrMaterialPropertyState(false));

    expect(clone).not.toBe(source);
    expect(clone).toBeInstanceOf(MeshStandardMaterial);
    const preview = clone as MeshStandardMaterial;
    expect(preview.color.getHex()).toBe(new Color(0xffffff).getHex());
    expect(preview.map).toBeNull();
    expect(preview.metalness).toBe(1);
    expect(preview.metalnessMap).toBeNull();
    expect(preview.roughness).toBe(1);
    expect(preview.roughnessMap).toBeNull();
    expect(preview.normalMap).toBeNull();
    expect(preview.aoMap).toBeNull();
    expect(preview.emissive.getHex()).toBe(new Color(0x000000).getHex());
    expect(preview.emissiveMap).toBeNull();
    expect(preview.transparent).toBe(false);
    expect(preview.opacity).toBe(1);
    expect(preview.alphaTest).toBe(0);
    expect(preview.side).toBe(FrontSide);
    expect(preview.version).toBeGreaterThan(0);

    expect(source.color.getHex()).toBe(new Color(0x336699).getHex());
    expect(source.map).toBeInstanceOf(Texture);
    expect(source.side).toBe(DoubleSide);
  });

  it('filters each material in an array without sharing source material objects', () => {
    const firstMap = texture();
    const firstNormalMap = texture();
    const secondMap = texture();
    const secondNormalMap = texture();
    const first = new MeshStandardMaterial({
      color: 0xff0000,
      map: firstMap,
      normalMap: firstNormalMap,
    });
    const second = new MeshStandardMaterial({
      color: 0x00ff00,
      map: secondMap,
      normalMap: secondNormalMap,
    });
    const state = defaultPbrMaterialPropertyState(true);
    state.baseColor = false;
    state.normal = false;

    const clones = [first, second].map((material) => cloneMaterialForPbrPreview(material, state));

    expect(clones).toHaveLength(2);
    expect(clones[0]).not.toBe(first);
    expect(clones[1]).not.toBe(second);
    expect(clones[0]).not.toBe(clones[1]);
    expect(first.map).toBe(firstMap);
    expect(first.normalMap).toBe(firstNormalMap);
    expect(second.map).toBe(secondMap);
    expect(second.normalMap).toBe(secondNormalMap);

    for (const clone of clones) {
      const preview = clone as MeshStandardMaterial;
      expect(preview.map).toBeNull();
      expect(preview.normalMap).toBeNull();
    }
  });

  it('rounds texture size up to supported options and clamps to the maximum', () => {
    const cases: Array<{
      dimensions: Array<{ width: number; height: number }>;
      currentValue: number;
      expected: TextureSizeOption;
    }> = [
      { dimensions: [{ width: 512, height: 1024 }], currentValue: 2048, expected: 1024 },
      { dimensions: [{ width: 1025, height: 700 }], currentValue: 1024, expected: 2048 },
      { dimensions: [{ width: 4096, height: 4097 }], currentValue: 1024, expected: 8192 },
      { dimensions: [{ width: 10000, height: 9000 }], currentValue: 1024, expected: 8192 },
      { dimensions: [{ width: 2048, height: 2048 }, { width: 3000, height: 1024 }], currentValue: 1024, expected: 4096 },
    ];

    for (const testCase of cases) {
      expect(textureSizeForLargestBaseColorMap(testCase.dimensions, testCase.currentValue)).toBe(testCase.expected);
    }
  });

  it('preserves the current supported texture size when dimensions are unavailable', () => {
    expect(textureSizeForLargestBaseColorMap([], 4096)).toBe(4096);
    expect(textureSizeForLargestBaseColorMap([], 1536)).toBe(1024);
  });

  it('collects single and array mesh materials from a scene', () => {
    const first = new MeshStandardMaterial();
    const second = new MeshStandardMaterial();
    const root = new Group();
    root.add(new Mesh(new BoxGeometry(), first));
    root.add(new Mesh(new BoxGeometry(), [first, second]));

    expect(collectMaterialsForPbrControls(root)).toEqual([first, first, second]);
  });
});
