import { describe, expect, it } from 'vitest';
import { Vector2, Vector3 } from 'three';
import {
  countSerializedTexturedVertices,
  createInjectiveAtlas,
  meshToWatlasBuffers,
  WATLAS_MAX_REASONABLE_EXTENT,
  WATLAS_MIN_REASONABLE_EXTENT,
  WATLAS_TARGET_EXTENT,
  watlasChartOptions,
  watlasPackOptions,
  watlasPositionScaleForExtent,
} from '../../src/texture/atlas';
import type { AtlasLayout } from '../../src/texture/types';

function meshWithFaces(faceCount: number) {
  return {
    positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
    faces: Array.from({ length: faceCount }, () => [0, 1, 2] as [number, number, number]),
  };
}

function pixelBounds(triangle: [[number, number], [number, number], [number, number]]) {
  const xs = triangle.map(([x]) => x);
  const ys = triangle.map(([, y]) => y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function wavyGridMesh(size: number, amplitude: number) {
  const positions: Vector3[] = [];
  const faces: [number, number, number][] = [];
  for (let y = 0; y <= size; y += 1) {
    for (let x = 0; x <= size; x += 1) {
      positions.push(new Vector3(x / size, y / size, Math.sin(x * 2.1) * Math.cos(y * 1.7) * amplitude));
    }
  }
  const vertexId = (x: number, y: number): number => y * (size + 1) + x;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      faces.push([vertexId(x, y), vertexId(x + 1, y), vertexId(x + 1, y + 1)]);
      faces.push([vertexId(x, y), vertexId(x + 1, y + 1), vertexId(x, y + 1)]);
    }
  }
  return { positions, faces };
}

function disconnectedTrianglesMesh(faceCount: number) {
  const positions: Vector3[] = [];
  const faces: [number, number, number][] = [];
  for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
    const base = positions.length;
    const offset = faceIndex * 2;
    positions.push(
      new Vector3(offset, 0, 0),
      new Vector3(offset + 1, 0, 0),
      new Vector3(offset, 1, 0),
    );
    faces.push([base, base + 1, base + 2]);
  }
  return { positions, faces };
}

function tinyShellPatchMesh() {
  const sourcePositions = new Map<number, Vector3>([
    [0, new Vector3(0.025370588526129723, 0.0031326571479439735, 0.035582851618528366)],
    [1, new Vector3(0.025380656123161316, 0.0035792244598269463, 0.035662151873111725)],
    [2, new Vector3(0.03467908129096031, 0.003431384451687336, 0.03088834509253502)],
    [3, new Vector3(0.008287638425827026, 0.002248571952804923, 0.037607092410326004)],
    [17, new Vector3(0.008932778611779213, 0.0021706526167690754, 0.03724917024374008)],
    [224, new Vector3(0.02567298524081707, 0.00294151296839118, 0.03571656346321106)],
    [225, new Vector3(0.02574838697910309, 0.0033005487639456987, 0.0358259491622448)],
    [226, new Vector3(0.025361929088830948, 0.003374907188117504, 0.03562413901090622)],
    [227, new Vector3(0.025139570236206055, 0.0033574337139725685, 0.03527887910604477)],
    [228, new Vector3(0.025162162259221077, 0.003015916794538498, 0.03528786450624466)],
    [229, new Vector3(0.025536032393574715, 0.0026922058314085007, 0.035292524844408035)],
    [230, new Vector3(0.025761188939213753, 0.0036049699410796165, 0.03577273339033127)],
    [231, new Vector3(0.02571711502969265, 0.003733583725988865, 0.035418782383203506)],
    [232, new Vector3(0.025295648723840714, 0.0036663159262388945, 0.035322174429893494)],
    [233, new Vector3(0.034464601427316666, 0.0035818987525999546, 0.031327053904533386)],
    [234, new Vector3(0.0337483249604702, 0.0034176462795585394, 0.031173720955848694)],
    [235, new Vector3(0.03425649181008339, 0.0031880002934485674, 0.030762888491153717)],
    [236, new Vector3(0.03558093309402466, 0.0032495709601789713, 0.029815616086125374)],
    [237, new Vector3(0.035789668560028076, 0.0037003590259701014, 0.030209384858608246)],
    [238, new Vector3(0.036085713654756546, 0.003915170207619667, 0.030770622193813324)],
    [239, new Vector3(0.008721536956727505, 0.0024761210661381483, 0.037477754056453705)],
    [240, new Vector3(0.008743616752326488, 0.001914038322865963, 0.037490539252758026)],
    [241, new Vector3(0.008736768737435341, 0.0018285836558789015, 0.037864286452531815)],
    [242, new Vector3(0.008320857770740986, 0.0022480268962681293, 0.03792329505085945)],
    [243, new Vector3(0.008719812147319317, 0.0026322973426431417, 0.03778912127017975)],
  ]);
  const sourceFaces = [
    [0, 224, 225], [0, 225, 226], [0, 226, 227], [0, 227, 228], [0, 228, 229], [0, 229, 224],
    [1, 230, 231], [1, 231, 232], [1, 232, 227], [1, 227, 226], [1, 226, 225], [1, 225, 230],
    [2, 233, 234], [2, 234, 235], [2, 235, 236], [2, 236, 237], [2, 237, 238], [2, 238, 233],
    [3, 239, 17], [3, 17, 240], [3, 240, 241], [3, 241, 242], [3, 242, 243], [3, 243, 239],
  ] as const;
  const vertexIds = Array.from(sourcePositions.keys());
  const remap = new Map(vertexIds.map((id, index) => [id, index]));
  return {
    positions: vertexIds.map((id) => sourcePositions.get(id)!.clone()),
    faces: sourceFaces.map((face) => face.map((id) => remap.get(id)!) as [number, number, number]),
  };
}

describe('createInjectiveAtlas', () => {
  it('uses geometry-only watlas charting and explicit non-bilinear block-aligned packing', () => {
    expect(watlasChartOptions()).toEqual({
      fixWinding: false,
      useInputMeshUvs: false,
    });

    expect(watlasPackOptions({ textureSize: 64, padding: 2 })).toEqual({
      maxChartSize: 60,
      resolution: 64,
      padding: 2,
      bilinear: false,
      blockAlign: true,
      rotateCharts: true,
      rotateChartsToAxis: true,
    });

    expect(watlasPackOptions({ textureSize: 4, padding: 4 })).toMatchObject({
      maxChartSize: 1,
      bilinear: false,
      blockAlign: true,
    });

    expect(watlasPackOptions({ textureSize: 64, padding: 2 }, 12.5)).toMatchObject({
      texelsPerUnit: 12.5,
    });
  });

  it('uses original watlas scale only inside the reasonable extent range', () => {
    expect(WATLAS_MIN_REASONABLE_EXTENT).toBe(1);
    expect(WATLAS_MAX_REASONABLE_EXTENT).toBe(10_000);
    expect(WATLAS_TARGET_EXTENT).toBe(10);

    expect(watlasPositionScaleForExtent(Number.NaN)).toBe(1);
    expect(watlasPositionScaleForExtent(0)).toBe(1);

    const shellExtent = 0.139482282102108;
    expect(watlasPositionScaleForExtent(shellExtent)).toBeCloseTo(WATLAS_TARGET_EXTENT / shellExtent, 12);

    expect(watlasPositionScaleForExtent(WATLAS_MIN_REASONABLE_EXTENT)).toBe(1);
    expect(watlasPositionScaleForExtent(8.279043674468994)).toBe(1);
    expect(watlasPositionScaleForExtent(WATLAS_MAX_REASONABLE_EXTENT)).toBe(1);

    const hugeExtent = WATLAS_MAX_REASONABLE_EXTENT + 0.1;
    expect(watlasPositionScaleForExtent(hugeExtent)).toBeCloseTo(WATLAS_TARGET_EXTENT / hugeExtent, 12);
  });

  it('builds watlas input vertices directly from output mesh positions and faces', () => {
    const mesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const buffers = meshToWatlasBuffers(mesh);

    expect(buffers.positions).toHaveLength(4 * 3);
    expect(buffers.positions).toEqual(new Float32Array([
      -0.5, -0.5, 0,
      0.5, -0.5, 0,
      0.5, 0.5, 0,
      -0.5, 0.5, 0,
    ]));
    expect(buffers.indices).toEqual(new Uint32Array([0, 1, 2, 0, 2, 3]));
    expect((buffers as { uvs?: unknown }).uvs).toBeUndefined();
    expect((buffers as { sourceVertexByXref?: unknown }).sourceVertexByXref).toBeUndefined();
  });

  it('uses watlas charts to share UVs across a planar mesh', async () => {
    const mesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };

    const atlas = await createInjectiveAtlas(mesh, { textureSize: 64, padding: 2 });
    const again = await createInjectiveAtlas(mesh, { textureSize: 64, padding: 2 });

    expect(atlas.facePixelTriangles).toEqual(again.facePixelTriangles);
    expect(atlas.faceUvs).toHaveLength(2);
    expect(atlas.facePixelTriangles).toHaveLength(2);
    expect(atlas.islandCount).toBeGreaterThanOrEqual(1);
    expect(atlas.islandCount).toBeLessThanOrEqual(2);
    expect(countSerializedTexturedVertices(mesh, atlas)).toBeLessThan(6);
    for (const faceUvs of atlas.faceUvs) {
      for (const uv of faceUvs) {
        expect(Number.isFinite(uv.x)).toBe(true);
        expect(Number.isFinite(uv.y)).toBe(true);
        expect(uv.x).toBeGreaterThanOrEqual(0);
        expect(uv.x).toBeLessThanOrEqual(1);
        expect(uv.y).toBeGreaterThanOrEqual(0);
        expect(uv.y).toBeLessThanOrEqual(1);
      }
    }
  });

  it('returns UV and pixel triangles in original face corner order', async () => {
    const mesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(2, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(2, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [1, 3, 2] as [number, number, number]],
    };

    const atlas = await createInjectiveAtlas(mesh, { textureSize: 128, padding: 2 });

    expect(atlas.faceUvs).toHaveLength(mesh.faces.length);
    const face0Uvs = atlas.faceUvs[0]!;
    const face1Uvs = atlas.faceUvs[1]!;
    expect(face0Uvs[1]!.x).toBeCloseTo(face1Uvs[0]!.x, 5);
    expect(face0Uvs[1]!.y).toBeCloseTo(face1Uvs[0]!.y, 5);
    expect(face0Uvs[2]!.x).toBeCloseTo(face1Uvs[2]!.x, 5);
    expect(face0Uvs[2]!.y).toBeCloseTo(face1Uvs[2]!.y, 5);

    for (let faceIndex = 0; faceIndex < mesh.faces.length; faceIndex += 1) {
      const faceUvs = atlas.faceUvs[faceIndex]!;
      const pixels = atlas.facePixelTriangles[faceIndex]!;
      const uvPixels = faceUvs.map((uv) => [uv.x * atlas.textureSize, uv.y * atlas.textureSize]);
      for (let corner = 0; corner < 3; corner += 1) {
        expect(pixels[corner]![0]).toBeCloseTo(uvPixels[corner]![0]!, 5);
        expect(pixels[corner]![1]).toBeCloseTo(uvPixels[corner]![1]!, 5);
      }
    }
  });

  it('assigns folded adjacent faces to separate atlas domains', async () => {
    const mesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(0, 0, 1),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 1, 3] as [number, number, number]],
    };

    const atlas = await createInjectiveAtlas(mesh, { textureSize: 64, padding: 2 });
    const first = pixelBounds(atlas.facePixelTriangles[0]!);
    const second = pixelBounds(atlas.facePixelTriangles[1]!);

    expect(atlas.islandCount).toBe(2);
    expect(first.maxX < second.minX || second.maxX < first.minX || first.maxY < second.minY || second.maxY < first.minY).toBe(true);
  });

  it('throws a helpful error when chart packing cannot fit the requested padding', async () => {
    await expect(createInjectiveAtlas(meshWithFaces(100), { textureSize: 16, padding: 2 })).rejects.toThrow(
      /texture-size|target-faces|texture-padding/i,
    );
  });

  it('rescales watlas packing when the initial chart scale exceeds the requested texture size', async () => {
    const atlas = await createInjectiveAtlas(wavyGridMesh(4, 0.5), { textureSize: 64, padding: 1 });

    expect(atlas.textureSize).toBe(64);
    for (const pixels of atlas.facePixelTriangles) {
      for (const [x, y] of pixels) {
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThanOrEqual(64);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThanOrEqual(64);
      }
    }
  });

  it('continues rescaling when watlas initially splits chart-heavy meshes into multiple atlases', async () => {
    const atlas = await createInjectiveAtlas(disconnectedTrianglesMesh(50), { textureSize: 128, padding: 2 });

    expect(atlas.textureSize).toBe(128);
    expect(atlas.faceUvs).toHaveLength(50);
    expect(atlas.facePixelTriangles).toHaveLength(50);
    for (const pixels of atlas.facePixelTriangles) {
      for (const [x, y] of pixels) {
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThanOrEqual(128);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThanOrEqual(128);
      }
    }
  });

  it('normalizes tiny shell-scale coordinates before watlas charting', async () => {
    const mesh = tinyShellPatchMesh();

    const atlas = await createInjectiveAtlas(mesh, { textureSize: 4096, padding: 4 });

    expect(atlas.faceUvs).toHaveLength(mesh.faces.length);
    expect(atlas.facePixelTriangles).toHaveLength(mesh.faces.length);
    expect(atlas.islandCount).toBeGreaterThan(0);
    for (const faceUvs of atlas.faceUvs) {
      for (const uv of faceUvs) {
        expect(Number.isFinite(uv.x)).toBe(true);
        expect(Number.isFinite(uv.y)).toBe(true);
        expect(uv.x).toBeGreaterThanOrEqual(0);
        expect(uv.x).toBeLessThanOrEqual(1);
        expect(uv.y).toBeGreaterThanOrEqual(0);
        expect(uv.y).toBeLessThanOrEqual(1);
      }
    }
  });

  it.each([
    ['fractional', 1.5],
    ['NaN', Number.NaN],
  ])('rejects %s face vertex indices before watlas conversion', async (_caseName, invalidVertexId) => {
    const mesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(1, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, invalidVertexId, 2] as [number, number, number]],
    };

    await expect(createInjectiveAtlas(mesh, { textureSize: 64, padding: 2 })).rejects.toThrow(
      /Output face 0 corner 1 references invalid vertex/,
    );
  });

  it('rejects non-finite position coordinates before watlas conversion', async () => {
    const mesh = {
      positions: [new Vector3(0, 0, 0), new Vector3(Number.NaN, 0, 0), new Vector3(0, 1, 0)],
      faces: [[0, 1, 2] as [number, number, number]],
    };

    await expect(createInjectiveAtlas(mesh, { textureSize: 64, padding: 2 })).rejects.toThrow(
      /Output vertex 1 has non-finite position coordinates/,
    );
  });

  it('counts serialized textured vertices using source vertex and rounded UV identity', () => {
    const mesh = {
      positions: [
        new Vector3(0, 0, 0),
        new Vector3(1, 0, 0),
        new Vector3(1, 1, 0),
        new Vector3(0, 1, 0),
      ],
      faces: [[0, 1, 2] as [number, number, number], [0, 2, 3] as [number, number, number]],
    };
    const injectiveAtlas: AtlasLayout = {
      textureSize: 64,
      padding: 2,
      islandCount: 2,
      facePixelTriangles: [
        [
          [2, 2],
          [30, 2],
          [2, 30],
        ],
        [
          [34, 34],
          [62, 34],
          [34, 62],
        ],
      ],
      faceUvs: [
        [new Vector2(2 / 64, 2 / 64), new Vector2(30 / 64, 2 / 64), new Vector2(2 / 64, 30 / 64)],
        [new Vector2(34 / 64, 34 / 64), new Vector2(62 / 64, 34 / 64), new Vector2(34 / 64, 62 / 64)],
      ],
    };
    const sharedUvAtlas: AtlasLayout = {
      ...injectiveAtlas,
      faceUvs: [
        injectiveAtlas.faceUvs[0]!,
        [
          injectiveAtlas.faceUvs[0]![0]!.clone(),
          injectiveAtlas.faceUvs[0]![2]!.clone(),
          injectiveAtlas.faceUvs[1]![2]!.clone(),
        ] as [Vector2, Vector2, Vector2],
      ],
    };

    expect(countSerializedTexturedVertices(mesh, injectiveAtlas)).toBe(6);
    expect(countSerializedTexturedVertices(mesh, sharedUvAtlas)).toBe(4);
  });
});
