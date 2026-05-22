import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { nodePrimitiveWorkerCount } from '../../src/local/primitiveWorkerPool';
import { nodeTextureBakeWorkerCount } from '../../src/local/textureBakeWorkerPool';

function runCliExpectFailure(args: string[]): string {
  try {
    execFileSync(process.execPath, ['--import', 'tsx', 'src/local/main.ts', ...args], {
      encoding: 'utf8',
      stdio: 'pipe',
    });
  } catch (error) {
    const failure = error as { stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
    return [
      failure.stdout?.toString() ?? '',
      failure.stderr?.toString() ?? '',
      failure.message ?? '',
    ].join('\n');
  }
  throw new Error('Expected CLI command to fail.');
}

describe('CLI options', () => {
  it('chooses deterministic Node primitive worker counts', () => {
    expect(nodePrimitiveWorkerCount(0, 8)).toBe(1);
    expect(nodePrimitiveWorkerCount(1, 8)).toBe(1);
    expect(nodePrimitiveWorkerCount(2, 8)).toBe(2);
    expect(nodePrimitiveWorkerCount(20, 16)).toBe(8);
    expect(nodePrimitiveWorkerCount(20, 2)).toBe(1);
  });

  it('chooses deterministic Node texture bake worker counts', () => {
    expect(nodeTextureBakeWorkerCount(0, 8)).toBe(1);
    expect(nodeTextureBakeWorkerCount(1, 8)).toBe(1);
    expect(nodeTextureBakeWorkerCount(4, 8)).toBe(4);
    expect(nodeTextureBakeWorkerCount(20, 16)).toBe(8);
    expect(nodeTextureBakeWorkerCount(20, 2)).toBe(1);
  });

  it('exposes the paper simplification option surface plus the diagnostic cap override', () => {
    const source = readFileSync('src/local/main.ts', 'utf8');

    expect(source).toContain("type CliVirtualEdgeMode = 'auto-local-radius' | 'auto-global-radius' | 'manual-global-radius'");
    expect(source).toContain('parseVirtualEdgeMode');
    expect(source).toContain('--virtual-edge-mode <auto-local-radius|auto-global-radius|manual-global-radius>');
    expect(source).toContain('--virtual-radius <number>');
    expect(source).toContain('--virtual-edge-candidate-cap <integer|none>');
    expect(source).toContain('--no-weld-vertices');
    expect(source).toContain('--no-recompute-normals');
    expect(source).toContain('weldVertices: opts.weldVertices');
    expect(source).toContain('recomputeNormals: opts.recomputeNormals');
    expect(source).toContain("value === 'none'");
    expect(source).not.toContain('--auto-virtual-edge-work-limit');
    expect(source).not.toContain('--area-weight');
    expect(source).not.toContain('--preserve-topology');
    expect(source).not.toContain('--hole-resistant');
  });

  it('reports serialized textured output vertices in baked CLI stats', () => {
    const source = readFileSync('src/local/main.ts', 'utf8');

    expect(source).toContain('countSerializedTexturedVertices');
    expect(source).toContain('outputVertices');
  });

  it('routes processed primitive results through the glTF-Transform adapter', () => {
    const source = readFileSync('src/local/main.ts', 'utf8');
    const legacyTransferImport = `from '../texture/${'uv' + 'Transfer'}'`;
    const legacyTransferFunction = ['transferSourceFace', 'AttributesToOutput'].join('');

    expect(source).toContain('GltfTransformPrimitiveSourceAdapter');
    expect(source).toContain('adapter.extractGroups');
    expect(source).toContain("'geometry-with-texture-metadata'");
    expect(source).toContain('adapter.applyResults');
    expect(source).not.toContain(legacyTransferImport);
    expect(source).not.toContain(legacyTransferFunction);
    expect(source).not.toContain('replaceScenePrimitiveGeometry');
    expect(source).not.toContain('replaceScenePrimitiveGroupGeometry');
  });

  it('shows retained texture options and no removed chart flags in help output', () => {
    const help = execFileSync(process.execPath, ['--import', 'tsx', 'src/local/main.ts', '--help'], {
      encoding: 'utf8',
    });

    expect(help).toContain('--transfer-textures');
    expect(help).toContain('standard PBR material textures and base-color factors');
    expect(help).toContain('using a watlas-generated chart atlas');
    expect(help).toContain('--texture-size <integer>');
    expect(help).toContain('--texture-padding <integer>');
    expect(help).toContain('--texture-filter <nearest|linear>');
    const texturePaddingHelpLine = help
      .split(/\r?\n/)
      .find((line) => line.includes('--texture-padding <integer>'));
    expect(texturePaddingHelpLine).toContain('(default: 2)');
    expect(help).toContain('--primitive-grouping <material-parent|material|none>');
    expect(help).toContain('--virtual-edge-mode <auto-local-radius|auto-global-radius|manual-global-radius>');
    expect(help).toContain('virtual edge radius mode');
    expect(help).toContain('manual global virtual edge radius r; required when --virtual-edge-mode manual-global-radius');
    expect(help).toContain('diagnostic auto-local-radius cap per component pair');
    expect(help).toContain('--no-weld-vertices');
    expect(help).toContain('--no-recompute-normals');
    expect(help).toContain('skip duplicate-position vertex welding during input mesh extraction');
    expect(help).toContain('preserve transferred source normals when available');
    expect(help).toContain('group source primitives before simplification');
    expect(help).toContain('atlas chart padding/gutter size in pixels');
    expect(help).not.toContain('--texture-chart-max-faces');
    expect(help).not.toContain('--texture-chart-normal-angle');
  });

  it('requires manual radius only for manual global virtual-edge mode', () => {
    const output = runCliExpectFailure([
      '--input', 'input.glb',
      '--output', 'output/cli-options-validation.glb',
      '--virtual-edge-mode', 'manual-global-radius',
    ]);

    expect(output).toContain('--virtual-radius is required when --virtual-edge-mode is manual-global-radius.');
  });

  it('rejects manual radius with automatic global virtual-edge mode', () => {
    const output = runCliExpectFailure([
      '--input', 'input.glb',
      '--output', 'output/cli-options-validation.glb',
      '--virtual-edge-mode', 'auto-global-radius',
      '--virtual-radius', '0.05',
    ]);

    expect(output).toContain('--virtual-radius only applies when --virtual-edge-mode is manual-global-radius.');
  });

  it('rejects the candidate cap outside automatic local virtual-edge mode', () => {
    const output = runCliExpectFailure([
      '--input', 'input.glb',
      '--output', 'output/cli-options-validation.glb',
      '--virtual-edge-mode', 'auto-global-radius',
      '--virtual-edge-candidate-cap', '4',
    ]);

    expect(output).toContain('--virtual-edge-candidate-cap only applies when using auto-local-radius virtual edges.');
  });

  it('rejects unknown virtual-edge modes with the expected values', () => {
    const output = runCliExpectFailure([
      '--input', 'input.glb',
      '--output', 'output/cli-options-validation.glb',
      '--virtual-edge-mode', 'auto-local',
    ]);

    expect(output).toContain('Expected auto-local-radius, auto-global-radius, or manual-global-radius');
  });
});
