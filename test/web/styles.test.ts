import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const styles = readFileSync(new URL('../../src/web/styles.css', import.meta.url), 'utf8');
const controlTabsBlock = styles.match(/\.control-tabs\s*{(?<block>[\s\S]*?)}/)?.groups?.block ?? '';

describe('web GUI spacing and tooltip styles', () => {
  it('anchors help popovers inside the controls panel instead of centering them over the right-edge icon', () => {
    expect(styles).toMatch(/\.help-icon:hover::after,[\s\S]*\.help-icon:focus-visible::after\s*{[\s\S]*right:\s*0;/);
    expect(styles).toMatch(/\.help-icon:hover::after,[\s\S]*\.help-icon:focus-visible::after\s*{[\s\S]*top:\s*calc\(100% \+ 0\.5rem\);/);
    expect(styles).not.toMatch(/\.help-icon:hover::after,[\s\S]*left:\s*50%[\s\S]*transform:\s*translateX\(-50%\)/);
  });

  it('leaves extra space below the tab buttons and loosens the control layouts', () => {
    expect(styles).toMatch(/\.controls-panel\s*{[\s\S]*padding:\s*1\.15rem;/);
    expect(styles).toMatch(/\.control-form\s*{[\s\S]*gap:\s*1\.15rem;/);
    expect(controlTabsBlock).toMatch(/grid-template-columns:\s*repeat\(2, minmax\(0, 1fr\)\);/);
    expect(controlTabsBlock).not.toMatch(/grid-template-columns:\s*repeat\(3, minmax\(0, 1fr\)\);/);
    expect(styles).toMatch(/\.control-tab-panel\s*{[\s\S]*gap:\s*1\.05rem;[\s\S]*margin-top:\s*0\.55rem;/);
    expect(styles).toMatch(/\.control-tab-panel\[data-tab-panel="controls"\]\s*{[\s\S]*gap:\s*1\.4rem;/);
  });

  it('gives the primary action buttons visibly larger vertical separation', () => {
    expect(styles).toMatch(/\.button-row\s*{[\s\S]*gap:\s*1\.25rem;/);
  });

  it('preserves hidden conditional controls despite grid label display rules', () => {
    expect(styles).toMatch(/\.control-form \[hidden\]\s*{[\s\S]*display:\s*none !important;/);
  });

  it('styles the PBR property controls and helper text', () => {
    expect(styles).toMatch(/\.pbr-property-group\s*{/);
    expect(styles).toMatch(/\.pbr-property-list\s*{/);
    expect(styles).toMatch(/\.pbr-property-row\s*{/);
    expect(styles).toMatch(/\.control-help\s*{/);
  });

  it('styles the model drop zone and hides the native file input', () => {
    expect(styles).toMatch(/\.file-input-hidden\s*{[\s\S]*position:\s*absolute;[\s\S]*width:\s*1px;[\s\S]*height:\s*1px;/);
    expect(styles).toMatch(/\.file-drop-zone\s*{[\s\S]*border:\s*1px dashed rgba\(147, 197, 253, 0\.55\);/);
    expect(styles).toMatch(/\.file-drop-zone:is\(:hover, :focus-visible\),[\s\S]*\.file-drop-zone\.is-drag-over\s*{[\s\S]*border-color:\s*#bfdbfe;/);
    expect(styles).toMatch(/\.file-drop-zone\.is-disabled\s*{[\s\S]*cursor:\s*not-allowed;/);
    expect(styles).toMatch(/\.file-name\s*{[\s\S]*overflow-wrap:\s*anywhere;/);
  });
});
