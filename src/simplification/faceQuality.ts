export const MIN_GENERATED_FACE_QUALITY = 0.03;

export function triangleQualityFromCoordinates(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const bcx = cx - bx;
  const bcy = cy - by;
  const bcz = cz - bz;
  const cax = ax - cx;
  const cay = ay - cy;
  const caz = az - cz;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;

  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  const doubleArea = Math.hypot(nx, ny, nz);
  const edgeLengthSquaredSum =
    abx * abx + aby * aby + abz * abz
    + bcx * bcx + bcy * bcy + bcz * bcz
    + cax * cax + cay * cay + caz * caz;

  if (
    !Number.isFinite(doubleArea)
    || !Number.isFinite(edgeLengthSquaredSum)
    || doubleArea <= 0
    || edgeLengthSquaredSum <= 0
  ) {
    return 0;
  }
  return (2 * Math.sqrt(3) * doubleArea) / edgeLengthSquaredSum;
}
