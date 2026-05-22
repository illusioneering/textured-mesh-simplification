import { describe, expect, it } from 'vitest';
import {
  BufferAttribute,
  BufferGeometry,
  DataTexture,
  Group,
  Mesh,
  MeshStandardMaterial,
  RGBAFormat,
} from 'three';
import { summarizeBrowserObject } from '../../src/web/browserModelSummary';

function triangleGeometry(offsetX: number): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array([
    offsetX, 0, 0,
    offsetX + 1, 0, 0,
    offsetX, 1, 0,
  ]), 3));
  geometry.setAttribute('uv', new BufferAttribute(new Float32Array([
    0, 0,
    1, 0,
    0, 1,
  ]), 2));
  geometry.setIndex([0, 1, 2]);
  return geometry;
}

function triangleGeometryWithUv1(): BufferGeometry {
  const geometry = triangleGeometry(0);
  geometry.deleteAttribute('uv');
  geometry.setAttribute('uv1', new BufferAttribute(new Float32Array([
    0, 0,
    1, 0,
    0, 1,
  ]), 2));
  return geometry;
}

function triangleGeometryWithoutUvs(offsetX: number): BufferGeometry {
  const geometry = triangleGeometry(offsetX);
  geometry.deleteAttribute('uv');
  return geometry;
}

describe('browser model summary scanner', () => {
  it('summarizes mesh, material, UV, and image texture metadata without extraction internals', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    const scene = new Group();
    scene.add(
      new Mesh(triangleGeometry(0), new MeshStandardMaterial({ map: texture, metalness: 0, roughness: 0.5 })),
      new Mesh(triangleGeometry(2), new MeshStandardMaterial({ color: 0x336699, metalness: 0, roughness: 0.75 })),
    );

    const summary = await summarizeBrowserObject(scene);

    expect(summary.inputVertices).toBe(6);
    expect(summary.inputFaces).toBe(2);
    expect(summary.materials).toBe(2);
    expect(summary.materialsWithTextures).toBe(1);
    expect(summary.materialsWithBaseColorImages).toBe(1);
    expect(summary.facesWithUvs).toBe(2);
    expect(summary.hasPreservableMaterialData).toBe(true);
    expect(summary.hasImageBackedTextureTransferData).toBe(true);
    expect(summary.hasImageBackedTextureBakeData).toBe(true);
    expect(summary.bakeableEntryCount).toBe(1);
    expect(summary.hasTransferableTextureData).toBe(true);
    expect(summary.warnings).toEqual([]);
    expect(summary.textureSlotKinds).toContain('baseColor');
    expect(summary.textureDimensions).toEqual(['1x1']);
    expect('rawMesh' in summary).toBe(false);
    expect('textured' in summary).toBe(false);
    expect('primitiveEntries' in summary).toBe(false);
    expect('primitiveEntryGroups' in summary).toBe(false);
  });

  it('matches texture channel 1 to uv1 like browser extraction', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    texture.channel = 1;
    const scene = new Group();
    scene.add(new Mesh(triangleGeometryWithUv1(), new MeshStandardMaterial({ map: texture, metalness: 0, roughness: 0.5 })));

    const summary = await summarizeBrowserObject(scene);

    expect(summary.inputFaces).toBe(1);
    expect(summary.facesWithUvs).toBe(1);
    expect(summary.materialsWithBaseColorImages).toBe(1);
    expect(summary.bakeableEntryCount).toBe(1);
    expect(summary.hasTransferableTextureData).toBe(true);
  });

  it('counts each bakeable material entry once when it has texture and base-color factors', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    const scene = new Group();
    scene.add(new Mesh(triangleGeometry(0), new MeshStandardMaterial({
      color: 0x336699,
      map: texture,
      metalness: 0,
      roughness: 0.5,
    })));

    const summary = await summarizeBrowserObject(scene);

    expect(summary.materials).toBe(1);
    expect(summary.materialsWithTextures).toBe(1);
    expect(summary.materialsWithBaseColorImages).toBe(1);
    expect(summary.bakeableEntryCount).toBe(1);
    expect(summary.hasTransferableTextureData).toBe(true);
  });

  it('ignores unused mesh material array entries when geometry has no groups', async () => {
    const used = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 });
    const unusedTexture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    const unusedTextured = new MeshStandardMaterial({ map: unusedTexture, metalness: 0, roughness: 0.5 });
    const scene = new Group();
    scene.add(new Mesh(triangleGeometry(0), [used, unusedTextured]));

    const summary = await summarizeBrowserObject(scene);

    expect(summary.materials).toBe(1);
    expect(summary.materialsWithTextures).toBe(0);
    expect(summary.materialsWithBaseColorImages).toBe(0);
    expect(summary.textureSlotKinds).toEqual([]);
    expect(summary.textureDimensions).toEqual([]);
    expect(summary.bakeableEntryCount).toBe(0);
    expect(summary.hasTransferableTextureData).toBe(false);
  });

  it('does not report texture atlas availability for factor-only materials without texture maps', async () => {
    const scene = new Group();
    scene.add(new Mesh(triangleGeometry(0), new MeshStandardMaterial({
      color: 0x336699,
      metalness: 0,
      roughness: 0.5,
    })));

    const summary = await summarizeBrowserObject(scene);

    expect(summary.materials).toBe(1);
    expect(summary.materialsWithTextures).toBe(0);
    expect(summary.materialsWithBaseColorImages).toBe(0);
    expect(summary.hasPreservableMaterialData).toBe(true);
    expect(summary.hasImageBackedTextureTransferData).toBe(false);
    expect(summary.hasImageBackedTextureBakeData).toBe(false);
    expect(summary.bakeableEntryCount).toBe(0);
    expect(summary.hasTransferableTextureData).toBe(false);
  });

  it('counts shared material sibling meshes under one parent as one material-parent bakeable entry', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    const material = new MeshStandardMaterial({ map: texture, metalness: 0, roughness: 0.5 });
    const parent = new Group();
    parent.add(
      new Mesh(triangleGeometry(0), material),
      new Mesh(triangleGeometry(2), material),
    );
    const scene = new Group();
    scene.add(parent);

    const summary = await summarizeBrowserObject(scene);

    expect(summary.inputFaces).toBe(2);
    expect(summary.materials).toBe(1);
    expect(summary.bakeableEntryCount).toBe(1);
    expect(summary.hasTransferableTextureData).toBe(true);
  });

  it('keeps transfer availability when a less restrictive grouping can bake a shared-material primitive', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    const material = new MeshStandardMaterial({ map: texture, metalness: 0, roughness: 0.5 });
    const parent = new Group();
    parent.add(
      new Mesh(triangleGeometry(0), material),
      new Mesh(triangleGeometryWithoutUvs(2), material),
    );
    const scene = new Group();
    scene.add(parent);

    const summary = await summarizeBrowserObject(scene);

    expect(summary.inputFaces).toBe(2);
    expect(summary.materials).toBe(1);
    expect(summary.bakeableEntryCount).toBe(0);
    expect(summary.hasImageBackedTextureTransferData).toBe(true);
    expect(summary.hasImageBackedTextureBakeData).toBe(false);
    expect(summary.hasTransferableTextureData).toBe(true);
  });

  it('ignores extension texture properties because only core glTF material slots are supported', async () => {
    const texture = new DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1, RGBAFormat);
    const material = new MeshStandardMaterial({ color: 0xffffff, metalness: 0, roughness: 0.5 }) as MeshStandardMaterial & {
      clearcoatMap?: DataTexture;
    };
    material.clearcoatMap = texture;
    const scene = new Group();
    scene.add(new Mesh(triangleGeometry(0), material));

    const summary = await summarizeBrowserObject(scene);

    expect(summary.materials).toBe(1);
    expect(summary.materialsWithTextures).toBe(0);
    expect(summary.materialsWithBaseColorImages).toBe(0);
    expect(summary.textureSlotKinds).toEqual([]);
    expect(summary.textureDimensions).toEqual([]);
    expect(summary.hasImageBackedTextureTransferData).toBe(false);
    expect(summary.hasImageBackedTextureBakeData).toBe(false);
    expect(summary.bakeableEntryCount).toBe(0);
  });

  it('does not mark non-default base color bakeable when image-backed base color needs a missing UV channel', async () => {
    const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1, RGBAFormat);
    texture.channel = 1;
    const material = new MeshStandardMaterial({
      color: 0x336699,
      map: texture,
      metalness: 0,
      roughness: 0.5,
    });
    const scene = new Group();
    scene.add(new Mesh(triangleGeometry(0), material));

    const summary = await summarizeBrowserObject(scene);

    expect(summary.materials).toBe(1);
    expect(summary.materialsWithTextures).toBe(1);
    expect(summary.materialsWithBaseColorImages).toBe(1);
    expect(summary.textureSlotKinds).toContain('baseColor');
    expect(summary.bakeableEntryCount).toBe(0);
    expect(summary.hasImageBackedTextureTransferData).toBe(false);
    expect(summary.hasImageBackedTextureBakeData).toBe(false);
    expect(summary.hasTransferableTextureData).toBe(false);
  });
});
