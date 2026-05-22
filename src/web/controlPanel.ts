import type { ProcessingOptions } from '../pipeline/options';
import type { PbrMaterialPropertyId } from './pbrControls';
import { allPbrMaterialProperties, textureSizeOptions } from './pbrControls';

export const controlTabs = [
  { id: 'controls', label: 'Controls', shortLabel: 'Controls' },
  { id: 'pbr', label: 'Rendering settings', shortLabel: 'Rendering' },
  { id: 'simplification', label: 'Simplification settings', shortLabel: 'Simplification' },
  { id: 'texture', label: 'Texture baking settings', shortLabel: 'Texture' },
] as const;

export const controlTooltips = {
  'model-file': 'Drop or choose a .glb or .gltf model, plus any referenced .bin buffers and PNG, JPEG, or WebP texture images.',
  'rendering-mode': 'Choose how both viewports display models: PBR materials, neutral geometry, or wireframe.',
  'simplify-button': 'Run geometry simplification with the current settings and keep the result available for repeated texture baking.',
  'bake-button': 'Bake new standard material texture atlases onto the latest simplified geometry. Disabled until geometry is simplified and source material texture data exists.',
  'export-button': 'Download the processed output scene as a binary GLB after processing completes.',
  'target-mode': 'Choose whether simplification targets a fraction of source faces or an approximate absolute face count.',
  'target-ratio': 'Fraction of input faces to keep. Lower values simplify more aggressively; valid range is greater than 0 through 1.',
  'target-face-count': 'Approximate number of output faces to target when Target mode is Face count.',
  'primitive-grouping': 'Choose whether simplification groups source triangles by shared material and parent, by material across the whole scene, or leaves primitives ungrouped.',
  'weld-vertices': 'Merge source vertices that share the same position before simplification. Disable to preserve primitive-local vertex streams.',
  'recompute-normals': 'Recompute smooth normals on simplified geometry. Disable to preserve transferred source normals when available.',
  'virtual-edge-mode': 'Choose automatic local radii, one automatic model-wide radius, or one manual model-wide radius.',
  'virtual-edge-radius': 'Manual global maximum distance for virtual edges between disconnected components. Use 0 to avoid adding virtual edges.',
  'max-iterations': 'Optional cap on collapse iterations for debugging or partial simplification. Leave blank for no explicit cap.',
  'texture-size': 'Square output atlas size in pixels. Larger sizes preserve more material texture detail but cost more memory and processing time.',
  'texture-padding': 'Padding/gutter in pixels around atlas chart islands to reduce filtering seams.',
  'texture-filter': 'Source texture sampling mode. Linear smooths samples; nearest preserves hard texel edges.',
  'pbr-material-properties': 'Toggle core glTF 2.0 material properties in PBR rendering mode. Unavailable properties are disabled for the loaded model.',
} as const;

function escapeAttribute(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function helpIcon(key: keyof typeof controlTooltips): string {
  const tooltip = escapeAttribute(controlTooltips[key]);
  return `<span class="help-icon" tabindex="0" title="${tooltip}" aria-label="${tooltip}" data-tooltip="${tooltip}">?</span>`;
}

function labelText(text: string, key: keyof typeof controlTooltips): string {
  return `<span class="label-with-help"><span>${text}</span>${helpIcon(key)}</span>`;
}

function renderTab(tab: typeof controlTabs[number], selected: boolean): string {
  const label = escapeAttribute(tab.label);
  return `<button id="tab-${tab.id}" class="control-tab${selected ? ' is-active' : ''}" type="button" role="tab" aria-selected="${String(selected)}" aria-controls="panel-${tab.id}" aria-label="${label}" title="${label}" data-tab-target="${tab.id}">${tab.shortLabel}</button>`;
}

function renderPbrCheckbox(property: { id: PbrMaterialPropertyId; label: string }): string {
  const id = `pbr-property-${property.id}`;
  return `<div class="checkbox-row pbr-property-row" data-pbr-property="${property.id}"><input id="${id}" type="checkbox" disabled /><label class="checkbox-label" for="${id}">${escapeAttribute(property.label)}</label></div>`;
}

function renderSimplificationCheckbox(id: 'weld-vertices' | 'recompute-normals', label: string, checked: boolean): string {
  return `<div class="checkbox-row simplification-checkbox-row"><input id="${id}" type="checkbox"${checked ? ' checked' : ''} /><label class="checkbox-label" for="${id}">${escapeAttribute(label)}</label>${helpIcon(id)}</div>`;
}

export function renderControlPanel(defaults: ProcessingOptions): string {
  const targetRatio = defaults.target.kind === 'ratio' ? defaults.target.ratio : 0.5;

  return `
<form id="processing-form" class="control-form">
  <div class="control-tabs" role="tablist" aria-label="Control sections">
    ${controlTabs.map((tab, index) => renderTab(tab, index === 0)).join('\n    ')}
  </div>

  <section id="panel-controls" class="control-tab-panel" role="tabpanel" aria-labelledby="tab-controls" data-tab-panel="controls">
    <div class="file-picker">
      <span class="label-with-help"><span id="model-drop-zone-label">Input GLB/GLTF</span>${helpIcon('model-file')}</span>
      <div id="model-drop-zone" class="file-drop-zone" role="button" tabindex="0" aria-labelledby="model-drop-zone-label" aria-describedby="model-drop-zone-help model-file-name">
        <span class="file-drop-primary">Drop GLB/GLTF, .bin, and textures here</span>
        <span id="model-drop-zone-help" class="file-drop-secondary">or click to choose files</span>
        <span id="model-file-name" class="file-name">No model selected</span>
      </div>
      <input id="model-file" class="file-input-hidden" type="file" multiple tabindex="-1" aria-hidden="true" accept=".glb,.gltf,model/gltf-binary,model/gltf+json,.bin,application/octet-stream,.png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp" />
    </div>
    <div class="button-row">
      <button id="simplify-button" type="button" disabled title="${escapeAttribute(controlTooltips['simplify-button'])}">Simplify geometry</button>
      <button id="bake-button" type="button" disabled title="${escapeAttribute(controlTooltips['bake-button'])}">Bake texture atlas</button>
      <button id="export-button" type="button" disabled title="${escapeAttribute(controlTooltips['export-button'])}">Export processed GLB</button>
    </div>
  </section>

  <section id="panel-simplification" class="control-tab-panel" role="tabpanel" aria-labelledby="tab-simplification" data-tab-panel="simplification" hidden>
    <label>
      ${labelText('Primitive grouping', 'primitive-grouping')}
      <select id="primitive-grouping">
        <option value="material-parent" selected>Material and parent</option>
        <option value="material">Material only</option>
        <option value="none">None</option>
      </select>
    </label>
    ${renderSimplificationCheckbox('weld-vertices', 'Weld vertices', defaults.weldVertices)}
    ${renderSimplificationCheckbox('recompute-normals', 'Recompute normals', defaults.recomputeNormals)}
    <label>
      ${labelText('Target mode', 'target-mode')}
      <select id="target-mode">
        <option value="ratio">Face ratio</option>
        <option value="faces">Face count</option>
      </select>
    </label>
    <label data-target-mode="ratio">
      ${labelText('Target ratio', 'target-ratio')}
      <input id="target-ratio" type="number" min="0.001" max="1" step="0.001" value="${targetRatio}" />
    </label>
    <label data-target-mode="faces">
      ${labelText('Target face count', 'target-face-count')}
      <input id="target-face-count" type="number" min="1" step="1" value="1000" />
    </label>
    <label>
      ${labelText('Virtual edge mode', 'virtual-edge-mode')}
      <select id="virtual-edge-mode">
        <option value="auto-local-radius" selected>Auto local radius</option>
        <option value="auto-global-radius">Auto global radius</option>
        <option value="manual-global-radius">Manual global radius</option>
      </select>
    </label>
    <label data-virtual-edge-mode="manual-global-radius">
      ${labelText('Radius', 'virtual-edge-radius')}
      <input id="virtual-edge-radius" type="number" min="0" step="0.001" value="0" />
    </label>
    <label>
      ${labelText('Max iterations (optional)', 'max-iterations')}
      <input id="max-iterations" type="number" min="1" step="1" placeholder="No limit" />
    </label>
  </section>

  <section id="panel-texture" class="control-tab-panel" role="tabpanel" aria-labelledby="tab-texture" data-tab-panel="texture" hidden>
    <label>
      ${labelText('Texture size', 'texture-size')}
      <select id="texture-size">
        ${textureSizeOptions.map((size) => `<option${size === defaults.textureSize ? ' selected' : ''}>${size}</option>`).join('\n        ')}
      </select>
    </label>
    <label>
      ${labelText('Texture padding', 'texture-padding')}
      <input id="texture-padding" type="number" min="0" step="1" value="${defaults.texturePadding}" />
    </label>
    <label>
      ${labelText('Source sampling', 'texture-filter')}
      <select id="texture-filter">
        <option value="linear" selected>Linear</option>
        <option value="nearest">Nearest</option>
      </select>
    </label>
  </section>

  <section id="panel-pbr" class="control-tab-panel" role="tabpanel" aria-labelledby="tab-pbr" data-tab-panel="pbr" hidden>
    <label>
      ${labelText('Rendering mode', 'rendering-mode')}
      <select id="rendering-mode">
        <option value="pbr" selected>PBR</option>
        <option value="geometry">Geometry only</option>
        <option value="wireframe">Wireframe</option>
      </select>
    </label>
    <div class="pbr-property-group">
      <span class="label-with-help"><span>PBR properties</span>${helpIcon('pbr-material-properties')}</span>
      <p id="pbr-property-help" class="control-help">Loaded model properties are enabled automatically.</p>
      <div class="pbr-property-list">
        ${allPbrMaterialProperties.map(renderPbrCheckbox).join('\n        ')}
      </div>
    </div>
  </section>
</form>
`;
}

export function initializeControlTabs(root: ParentNode): void {
  const tabs = Array.from(root.querySelectorAll<HTMLElement>('[role="tab"][data-tab-target]'));
  const panels = Array.from(root.querySelectorAll<HTMLElement>('[role="tabpanel"][data-tab-panel]'));

  function selectTab(target: HTMLElement): void {
    const tabId = target.dataset.tabTarget;
    for (const tab of tabs) {
      const selected = tab.dataset.tabTarget === tabId;
      tab.setAttribute('aria-selected', String(selected));
      tab.classList.toggle('is-active', selected);
    }
    for (const panel of panels) {
      const visible = panel.dataset.tabPanel === tabId;
      panel.hidden = !visible;
    }
  }

  for (const tab of tabs) {
    tab.addEventListener('click', () => selectTab(tab));
    tab.addEventListener('keydown', (event: KeyboardEvent) => {
      const index = tabs.indexOf(tab);
      if (event.key === 'ArrowRight') {
        event.preventDefault();
        tabs[(index + 1) % tabs.length]!.focus();
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        tabs[(index - 1 + tabs.length) % tabs.length]!.focus();
      } else if (event.key === 'Home') {
        event.preventDefault();
        tabs[0]!.focus();
      } else if (event.key === 'End') {
        event.preventDefault();
        tabs[tabs.length - 1]!.focus();
      }
    });
  }
}
