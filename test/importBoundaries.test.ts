import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const browserSafeTargets = ['src/index.ts', 'src/simplification', 'src/texture', 'src/pipeline'];
const requiredBrowserSafeFiles = ['src/index.ts'];
const forbiddenImportPatterns = [
  /from\s+['"]node:/,
  /import\s+['"]node:/,
  /from\s+['"]commander['"]/, 
  /from\s+['"]@gltf-transform\//,
  /from\s+['"]pngjs['"]/, 
  /from\s+['"]jpeg-js['"]/, 
  /from\s+['"](?:fs|path|os|fs\/promises)['"]/, 
  /from\s+['"]node:(?:fs|path|os|fs\/promises)['"]/, 
];

function tsFiles(root: string): string[] {
  const rootStat = statSync(root);
  if (rootStat.isFile()) return root.endsWith('.ts') ? [root] : [];

  const entries = readdirSync(root);
  const files: string[] = [];
  for (const entry of entries) {
    const path = join(root, entry);
    const stat = statSync(path);
    if (stat.isDirectory()) files.push(...tsFiles(path));
    else if (path.endsWith('.ts')) files.push(path);
  }
  return files;
}

function browserSafeFiles(): string[] {
  return browserSafeTargets.flatMap((target) => tsFiles(target));
}

function normalizePath(file: string): string {
  return file.replace(/\\/g, '/');
}

describe('browser-safe import boundaries', () => {
  it('includes the public entrypoint in boundary coverage', () => {
    expect(browserSafeFiles().map(normalizePath)).toEqual(expect.arrayContaining(requiredBrowserSafeFiles));
  });

  it('keeps reusable modules and the public entrypoint free of local-only dependencies', () => {
    const violations: string[] = [];
    for (const file of browserSafeFiles()) {
      const content = readFileSync(file, 'utf8');
      for (const pattern of forbiddenImportPatterns) {
        if (pattern.test(content)) violations.push(`${file} matches ${pattern}`);
      }
    }
    expect(violations).toEqual([]);
  });
});
