import { BufferAttribute, BufferGeometry, Vector3 } from 'three';
import type { FaceIndices, RawMesh } from './types';

function assertFiniteVector(position: Vector3, index: number): void {
  if (!Number.isFinite(position.x) || !Number.isFinite(position.y) || !Number.isFinite(position.z)) {
    throw new Error(`Position ${index} contains a non-finite coordinate.`);
  }
}

export function geometryToRawMesh(geometry: BufferGeometry): RawMesh {
  const positionAttribute = geometry.getAttribute('position');
  if (!positionAttribute) {
    throw new Error('BufferGeometry is missing a position attribute.');
  }
  if (positionAttribute.itemSize !== 3) {
    throw new Error(`Expected position itemSize 3, got ${positionAttribute.itemSize}.`);
  }

  const positions: Vector3[] = [];
  for (let i = 0; i < positionAttribute.count; i += 1) {
    const position = new Vector3(
      positionAttribute.getX(i),
      positionAttribute.getY(i),
      positionAttribute.getZ(i),
    );
    assertFiniteVector(position, i);
    positions.push(position);
  }

  const faces: FaceIndices[] = [];
  const index = geometry.getIndex();
  if (index) {
    if (index.count % 3 !== 0) {
      throw new Error(`Indexed geometry has ${index.count} indices, which is not divisible by 3.`);
    }
    for (let i = 0; i < index.count; i += 3) {
      faces.push([index.getX(i), index.getX(i + 1), index.getX(i + 2)]);
    }
  } else {
    if (positions.length % 3 !== 0) {
      throw new Error(`Non-indexed geometry has ${positions.length} vertices, which is not divisible by 3.`);
    }
    for (let i = 0; i < positions.length; i += 3) {
      faces.push([i, i + 1, i + 2]);
    }
  }

  for (const [faceIndex, face] of faces.entries()) {
    for (const vertexIndex of face) {
      if (vertexIndex < 0 || vertexIndex >= positions.length) {
        throw new Error(`Face ${faceIndex} references missing vertex ${vertexIndex}.`);
      }
    }
  }

  return { positions, faces };
}

export function rawMeshToGeometry(rawMesh: RawMesh): BufferGeometry {
  const geometry = new BufferGeometry();
  const positionArray = new Float32Array(rawMesh.positions.length * 3);
  rawMesh.positions.forEach((position, index) => {
    assertFiniteVector(position, index);
    positionArray[index * 3] = position.x;
    positionArray[index * 3 + 1] = position.y;
    positionArray[index * 3 + 2] = position.z;
  });

  const maxIndex = rawMesh.positions.length - 1;
  const indexArray = maxIndex > 65535
    ? new Uint32Array(rawMesh.faces.length * 3)
    : new Uint16Array(rawMesh.faces.length * 3);

  rawMesh.faces.forEach((face, faceIndex) => {
    const base = faceIndex * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertexIndex = face[corner]!;
      if (vertexIndex < 0 || vertexIndex > maxIndex) {
        throw new Error(`Face ${faceIndex} references missing vertex ${vertexIndex}.`);
      }
      indexArray[base + corner] = vertexIndex;
    }
  });

  geometry.setAttribute('position', new BufferAttribute(positionArray, 3));
  geometry.setIndex(new BufferAttribute(indexArray, 1));
  geometry.computeVertexNormals();
  return geometry;
}
