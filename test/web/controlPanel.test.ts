import { describe, expect, it } from 'vitest';
import { defaultProcessingOptions } from '../../src/pipeline/options';
import { allPbrMaterialProperties } from '../../src/web/pbrControls';
import {
  controlTabs,
  controlTooltips,
  renderControlPanel,
} from '../../src/web/controlPanel';

describe('web control panel markup', () => {
  it('renders the requested tab sections in order', () => {
    expect(controlTabs.map((tab) => tab.id)).toEqual(['controls', 'pbr', 'simplification', 'texture']);
    expect(controlTabs.at(1)).toEqual({ id: 'pbr', label: 'Rendering settings', shortLabel: 'Rendering' });

    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toContain('role="tablist"');
    expect(markup).toContain('data-tab-target="controls"');
    expect(markup).toContain('data-tab-target="simplification"');
    expect(markup).toContain('data-tab-target="texture"');
    expect(markup).toContain('data-tab-target="pbr"');
    expect(markup).toContain('id="panel-controls"');
    expect(markup).toContain('id="panel-simplification"');
    expect(markup).toContain('id="panel-texture"');
    expect(markup).toContain('id="panel-pbr"');
    expect(markup).toMatch(/<section[^>]*id="panel-pbr"[^>]*role="tabpanel"[^>]*>/);
    expect(markup).toContain('aria-labelledby="tab-pbr"');
    expect(markup).toContain('data-tab-panel="pbr" hidden');
    for (const tab of controlTabs) {
      expect(markup).toContain(`aria-label="${tab.label}"`);
      expect(markup).toContain(`title="${tab.label}"`);
    }
  });

  it('renders split simplify and bake actions', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toContain('id="simplify-button"');
    expect(markup).toContain('Simplify geometry');
    expect(markup).toContain('id="bake-button"');
    expect(markup).toContain('Bake texture atlas');
    expect(markup).not.toContain('id="process-button"');
  });

  it('renders rendering mode controls above disabled PBR property checkboxes', () => {
    const markup = renderControlPanel(defaultProcessingOptions());
    const controlsPanel = markup.slice(markup.indexOf('id="panel-controls"'), markup.indexOf('id="panel-simplification"'));
    const renderingPanel = markup.slice(markup.indexOf('id="panel-pbr"'), markup.indexOf('</section>', markup.indexOf('id="panel-pbr"')));

    expect(controlsPanel).not.toContain('id="rendering-mode"');
    expect(renderingPanel).toContain('id="rendering-mode"');
    expect(renderingPanel).toContain('<option value="wireframe">Wireframe</option>');
    expect(renderingPanel).toContain('<option value="geometry">Geometry only</option>');
    expect(renderingPanel).toContain('<option value="pbr" selected>PBR</option>');
    expect(renderingPanel).not.toContain('<option value="textured" selected>Textured</option>');
    expect(renderingPanel.indexOf('id="rendering-mode"')).toBeLessThan(renderingPanel.indexOf('PBR properties'));

    expect(markup).toContain('class="pbr-property-group"');
    expect(markup).toContain('PBR properties');
    expect(markup).toContain('id="pbr-property-help" class="control-help"');
    expect(markup).toContain('Loaded model properties are enabled automatically.');
    expect(markup).toContain('class="pbr-property-list"');
    for (const property of allPbrMaterialProperties) {
      expect(markup).toContain(`class="checkbox-row pbr-property-row"`);
      expect(markup).toContain(`id="pbr-property-${property.id}"`);
      expect(markup).toContain(`data-pbr-property="${property.id}"`);
      expect(markup).toContain(`type="checkbox" disabled`);
      expect(markup).toContain(`<label class="checkbox-label" for="pbr-property-${property.id}">${property.label}</label>`);
    }
  });

  it('removes the texture-transfer checkbox from the texture tab', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).not.toContain('id="transfer-textures"');
    expect(markup).not.toMatch(/<label class="checkbox-label" for="transfer-textures">/);
    expect(markup).not.toContain('Bake base-color texture atlas');
  });

  it('renders a drop zone backed by a hidden native file input', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toContain('id="model-drop-zone"');
    expect(markup).toContain('class="file-drop-zone"');
    expect(markup).toContain('role="button"');
    expect(markup).toContain('tabindex="0"');
    expect(markup).toContain('aria-labelledby="model-drop-zone-label"');
    expect(markup).toContain('id="model-file-name"');
    expect(markup).toMatch(/<input id="model-file" class="file-input-hidden" type="file" multiple tabindex="-1" aria-hidden="true" accept="\.glb,\.gltf,model\/gltf-binary,model\/gltf\+json,\.bin,application\/octet-stream,\.png,\.jpg,\.jpeg,\.webp,image\/png,image\/jpeg,image\/webp"/);
    expect(markup).toContain('Drop GLB/GLTF, .bin, and textures here');
    expect(markup).not.toMatch(/<label for="model-file">Input GLB\/GLTF<\/label>/);
  });

  it('keeps help triggers outside the drop target and checkbox labels so help clicks do not activate controls', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toMatch(/<div class="file-picker">\s*<span class="label-with-help"><span id="model-drop-zone-label">Input GLB\/GLTF<\/span>/);
    expect(markup).not.toContain('class="checkbox-row"');
    expect(markup).not.toMatch(/<div class="checkbox-row">\s*<input id="transfer-textures"/);
  });

  it('omits removed non-paper simplification controls', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).not.toContain('id="auto-virtual-edge-work-limit"');
    expect(markup).not.toContain('id="area-weight"');
    expect(markup).not.toContain('id="preserve-topology"');
    expect(markup).not.toContain('id="hole-resistant"');
  });

  it('renders the CLI-compatible default target ratio', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toMatch(/id="target-ratio"[^>]*value="0\.5"/);
    expect(markup).toMatch(/id="target-face-count"[^>]*min="1"/);
  });

  it('renders the default texture padding from shared processing defaults', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toMatch(/id="texture-padding"[^>]*value="2"/);
  });

  it('renders only the supported texture size options', () => {
    const defaults = defaultProcessingOptions();
    const markup = renderControlPanel(defaults);
    const textureSizeSelect = markup.match(/<select id="texture-size">(?<options>[\s\S]*?)<\/select>/)?.groups?.options;

    expect(textureSizeSelect).toBeDefined();
    expect(Array.from(textureSizeSelect!.matchAll(/<option(?: selected)?>(\d+)<\/option>/g), (match) => Number(match[1]))).toEqual([1024, 2048, 4096, 8192]);
    expect(textureSizeSelect).toContain(`<option selected>${defaults.textureSize}</option>`);
    expect(textureSizeSelect).not.toContain('<option>256</option>');
    expect(textureSizeSelect).not.toContain('<option>512</option>');
  });

  it('renders virtual edge mode controls for local, automatic global, and manual global radius modes', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toContain('id="virtual-edge-mode"');
    expect(markup).toContain('<option value="auto-local-radius" selected>Auto local radius</option>');
    expect(markup).toContain('<option value="auto-global-radius">Auto global radius</option>');
    expect(markup).toContain('<option value="manual-global-radius">Manual global radius</option>');
    expect(markup).toContain('data-virtual-edge-mode="manual-global-radius"');
    expect(markup).toMatch(/id="virtual-edge-radius"[^>]*value="0"/);
  });

  it('renders material grouping modes without a primitive grouping option', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    expect(markup).toContain('id="primitive-grouping"');
    expect(markup).toContain('<option value="material-parent" selected>Material and parent</option>');
    expect(markup).toContain('<option value="material">Material only</option>');
    expect(markup).toContain('<option value="none">None</option>');
    expect(markup).not.toContain('value="primitive"');
    expect(markup.indexOf('id="primitive-grouping"')).toBeLessThan(markup.indexOf('id="target-mode"'));
  });

  it('renders simplification extraction checkboxes from shared defaults', () => {
    const markup = renderControlPanel(defaultProcessingOptions());
    const simplificationPanel = markup.slice(markup.indexOf('id="panel-simplification"'), markup.indexOf('id="panel-texture"'));

    expect(simplificationPanel).toContain('id="weld-vertices"');
    expect(simplificationPanel).toContain('id="recompute-normals"');
    expect(simplificationPanel).toContain('<label class="checkbox-label" for="weld-vertices">Weld vertices</label>');
    expect(simplificationPanel).toContain('<label class="checkbox-label" for="recompute-normals">Recompute normals</label>');
    expect(simplificationPanel).toMatch(/id="weld-vertices"[^>]*type="checkbox"[^>]*checked/);
    expect(simplificationPanel).toMatch(/id="recompute-normals"[^>]*type="checkbox"[^>]*checked/);
    expect(simplificationPanel.indexOf('id="primitive-grouping"')).toBeLessThan(simplificationPanel.indexOf('id="weld-vertices"'));
    expect(simplificationPanel.indexOf('id="recompute-normals"')).toBeLessThan(simplificationPanel.indexOf('id="target-mode"'));
  });

  it('defines help text for every control parameter and action', () => {
    expect(Object.keys(controlTooltips).sort()).toEqual([
      'bake-button',
      'export-button',
      'max-iterations',
      'model-file',
      'pbr-material-properties',
      'primitive-grouping',
      'recompute-normals',
      'rendering-mode',
      'simplify-button',
      'target-face-count',
      'target-mode',
      'target-ratio',
      'texture-filter',
      'texture-padding',
      'texture-size',
      'virtual-edge-mode',
      'virtual-edge-radius',
      'weld-vertices',
    ].sort());

    const markup = renderControlPanel(defaultProcessingOptions());
    for (const helpText of Object.values(controlTooltips)) {
      expect(markup).toContain(helpText);
    }
    expect(controlTooltips['bake-button']).toBe('Bake new standard material texture atlases onto the latest simplified geometry. Disabled until geometry is simplified and source material texture data exists.');
    expect(controlTooltips['rendering-mode']).toBe('Choose how both viewports display models: PBR materials, neutral geometry, or wireframe.');
    expect(controlTooltips['pbr-material-properties']).toBe('Toggle core glTF 2.0 material properties in PBR rendering mode. Unavailable properties are disabled for the loaded model.');
    expect(controlTooltips['virtual-edge-mode']).toBe('Choose automatic local radii, one automatic model-wide radius, or one manual model-wide radius.');
    expect(controlTooltips['virtual-edge-radius']).toBe('Manual global maximum distance for virtual edges between disconnected components. Use 0 to avoid adding virtual edges.');
    expect(controlTooltips['texture-size']).toBe('Square output atlas size in pixels. Larger sizes preserve more material texture detail but cost more memory and processing time.');
    expect(controlTooltips['texture-padding']).toBe('Padding/gutter in pixels around atlas chart islands to reduce filtering seams.');
  });

  it('renders retained texture controls and omits removed chart controls', () => {
    const markup = renderControlPanel(defaultProcessingOptions());

    for (const id of [
      'model-file',
      'simplify-button',
      'bake-button',
      'export-button',
      'target-mode',
      'target-ratio',
      'target-face-count',
      'texture-filter',
      'texture-padding',
      'texture-size',
    ]) {
      expect(markup).toContain(`id="${id}"`);
    }
    expect(markup).not.toContain('id="texture-chart-max-faces"');
    expect(markup).not.toContain('id="texture-chart-normal-angle"');
    expect(markup).not.toContain('texture-chart');
    expect(markup).not.toContain('chart-max-faces');
    expect(markup).not.toContain('chart-normal-angle');
  });

  it('locks the deferred simplification and texture panel control inventory', () => {
    const markup = renderControlPanel(defaultProcessingOptions());
    const simplificationPanel = markup.slice(markup.indexOf('id="panel-simplification"'), markup.indexOf('id="panel-texture"'));
    const texturePanel = markup.slice(markup.indexOf('id="panel-texture"'), markup.indexOf('id="panel-pbr"'));
    const idsIn = (html: string): string[] => Array.from(html.matchAll(/\bid="([^"]+)"/g), (match) => match[1]!)
      .filter((id) => id !== 'panel-simplification' && id !== 'panel-texture');

    expect(idsIn(simplificationPanel)).toEqual([
      'primitive-grouping',
      'weld-vertices',
      'recompute-normals',
      'target-mode',
      'target-ratio',
      'target-face-count',
      'virtual-edge-mode',
      'virtual-edge-radius',
      'max-iterations',
    ]);
    expect(idsIn(texturePanel)).toEqual([
      'texture-size',
      'texture-padding',
      'texture-filter',
    ]);
  });
});
