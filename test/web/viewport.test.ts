import { BoxGeometry, Group, Mesh, MeshBasicMaterial } from 'three';
import { describe, expect, it } from 'vitest';
import {
  applyViewportCameraPoseValues,
  copyViewportCameraPose,
  gridFrameForModel,
  gridFrameForObjectBounds,
} from '../../src/web/viewport';

describe('web viewport grid frame calculation', () => {
  it('places the grid at the world-space bottom of the object bounding box', () => {
    const object = new Mesh(new BoxGeometry(2, 4, 6), new MeshBasicMaterial());
    object.position.set(10, 5, -3);
    object.updateMatrixWorld(true);

    const frame = gridFrameForObjectBounds(object);

    expect(frame.centerX).toBeCloseTo(10);
    expect(frame.centerZ).toBeCloseTo(-3);
    expect(frame.y).toBeCloseTo(3);
    expect(frame.size).toBeCloseTo(12);
    expect(frame.divisions).toBe(12);
    expect(frame.spacing).toBe(1);
  });

  it('uses the larger XZ dimension and includes child transforms', () => {
    const root = new Group();
    const child = new Mesh(new BoxGeometry(3, 2, 5), new MeshBasicMaterial());
    child.position.set(-2, 1, 4);
    child.scale.set(2, 1, 1);
    root.add(child);
    root.position.set(8, -4, 1);
    root.updateMatrixWorld(true);

    const frame = gridFrameForObjectBounds(root);

    expect(frame.centerX).toBeCloseTo(6);
    expect(frame.centerZ).toBeCloseTo(5);
    expect(frame.y).toBeCloseTo(-4);
    expect(frame.size).toBeCloseTo(12);
    expect(frame.divisions).toBe(12);
    expect(frame.spacing).toBe(1);
  });

  it('uses power-of-10 spacing and rounds grid size up for large bounds', () => {
    const object = new Mesh(new BoxGeometry(12, 4, 880), new MeshBasicMaterial());
    object.position.set(0, 2, 0);
    object.updateMatrixWorld(true);

    const frame = gridFrameForObjectBounds(object);

    expect(frame.y).toBeCloseTo(0);
    expect(frame.spacing).toBe(100);
    expect(frame.divisions).toBe(18);
    expect(frame.size).toBe(1800);
  });

  it('keeps a visible fallback grid for empty or flat XZ bounds', () => {
    const empty = new Group();
    const flat = new Mesh(new BoxGeometry(0, 2, 0), new MeshBasicMaterial());
    flat.position.set(1, 2, 3);
    flat.updateMatrixWorld(true);

    expect(gridFrameForObjectBounds(empty)).toEqual({
      centerX: 0,
      centerZ: 0,
      y: -0.001,
      size: 10,
      divisions: 10,
      spacing: 1,
    });

    expect(gridFrameForObjectBounds(flat)).toEqual({
      centerX: 1,
      centerZ: 3,
      y: 1,
      size: 10,
      divisions: 10,
      spacing: 1,
    });
  });

  it('uses the default grid frame after clearing the model when grid updates are enabled', () => {
    expect(gridFrameForModel(null, true)).toEqual({
      centerX: 0,
      centerZ: 0,
      y: -0.001,
      size: 10,
      divisions: 10,
      spacing: 1,
    });
  });

  it('preserves the current grid frame when grid updates are disabled', () => {
    expect(gridFrameForModel(null, false)).toBeNull();
  });
});

describe('web viewport camera pose helpers', () => {
  it('copies camera pose values independently from later mutations', () => {
    const pose = {
      position: { x: 1, y: 2, z: 3 },
      target: { x: 4, y: 5, z: 6 },
      near: 0.1,
      far: 500,
      zoom: 1,
    };

    const copied = copyViewportCameraPose(pose);

    pose.position.x = 99;
    pose.target.z = 99;

    expect(copied.position.x).toBe(1);
    expect(copied.target.z).toBe(6);
  });

  it('applies camera pose values to mutable camera and target objects', () => {
    const camera = {
      position: {
        x: 0,
        y: 0,
        z: 0,
        set(x: number, y: number, z: number): void {
          this.x = x;
          this.y = y;
          this.z = z;
        },
      },
      near: 0.01,
      far: 1000,
      zoom: 1,
      projectionUpdates: 0,
      updateProjectionMatrix(): void {
        this.projectionUpdates += 1;
      },
    };
    const controls = {
      target: {
        x: 0,
        y: 0,
        z: 0,
        set(x: number, y: number, z: number): void {
          this.x = x;
          this.y = y;
          this.z = z;
        },
      },
      updates: 0,
      update(): void {
        this.updates += 1;
      },
    };
    const pose = {
      position: { x: 1, y: 2, z: 3 },
      target: { x: 4, y: 5, z: 6 },
      near: 0.1,
      far: 500,
      zoom: 2,
    };

    applyViewportCameraPoseValues(pose, camera, controls);
    pose.position.x = 99;
    pose.target.z = 99;

    expect(camera.position).toMatchObject({ x: 1, y: 2, z: 3 });
    expect(controls.target).toMatchObject({ x: 4, y: 5, z: 6 });
    expect(camera.near).toBe(0.1);
    expect(camera.far).toBe(500);
    expect(camera.zoom).toBe(2);
    expect(camera.projectionUpdates).toBe(1);
    expect(controls.updates).toBe(1);
  });
});
