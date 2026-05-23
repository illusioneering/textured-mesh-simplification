# Textured Mesh Simplification

TypeScript implementation of the mesh simplification and texture resampling method described in [*Simplifying Textured Triangle Meshes in the Wild*](https://doi.org/10.1145/3763277). The project provides a reusable browser-safe simplification core, a Node.js CLI, and a Vite/Three.js browser app for loading, previewing, simplifying, texture baking, and exporting GLB/glTF assets.

This implementation was agentically engineered by [Evan Suma Rosenberg](https://scholar.google.com/citations?user=Hg0rPkAAAAAJ) using OpenAI Codex and GPT 5.5. It is released under the MIT License.

A live browser demo is available on [GitHub Pages](https://illusioneering.github.io/textured-mesh-simplification/).

## Overview

This project implements core ideas from:

Hsueh-Ti Derek Liu, Xiaoting Zhang, and Cem Yuksel. 2025. *Simplifying Textured Triangle Meshes in the Wild*. ACM Transactions on Graphics 44, 6, Article 181, 16 pages. https://doi.org/10.1145/3763277

The simplification core includes a simplicial-complex-style representation for triangle meshes in the wild, physical and virtual edge construction, quadric-driven edge collapse, collapse-history tracking, and successive mapping from simplified output surfaces back to the source mesh for texture transfer.

The surrounding pipeline adds the practical pieces needed to process glTF assets end to end:

- GLB/glTF loading and export for both local and browser workflows.
- Standard glTF 2.0 PBR material handling for base-color, normal, metallic-roughness, occlusion, and emissive texture slots.
- Chart atlas generation with [watlas](https://github.com/toji/watlas), a WebAssembly wrapper for [xatlas](https://github.com/jpcy/xatlas).
- Texture resampling into generated PNG atlases, including tangent-frame normal-map rebaking.
- Preservation or transfer of normals, `TEXCOORD_0`, `COLOR_0`, material scalar settings, and derived output tangents where applicable.
- Node worker and Web Worker processing paths for multi-primitive simplification and texture-bake batches.
- A browser interface for loading `.glb`/`.gltf` files, matching uploaded external `.bin` buffers and loose texture images, previewing input/output scenes, and exporting processed GLBs.

## Install

```sh
npm install --include=dev
```

## Command-Line Usage

Geometry-only simplification:

```sh
npm run simplify -- --input input.glb --output simplified.glb --ratio 0.5
```

Simplification with texture-atlas baking:

```sh
npm run simplify -- --input input.glb --output simplified-textured.glb --ratio 0.25 --transfer-textures --texture-size 1024 --texture-padding 2
```

Target an approximate face count instead of a ratio:

```sh
npm run simplify -- --input input.glb --output simplified-1000-faces.glb --target-faces 1000
```

Use a manual global virtual-edge radius:

```sh
npm run simplify -- --input input.glb --output simplified-manual-radius.glb --ratio 0.5 --virtual-edge-mode manual-global-radius --virtual-radius 0.05
```

Command-line options:

- `-i, --input <path>` sets the input `.glb` path.
- `-o, --output <path>` sets the output `.glb` path.
- `-r, --ratio <number>` sets the target output face ratio, such as `0.5`.
- `-f, --target-faces <integer>` sets the target output face count. Use either `--ratio` or `--target-faces`, not both.
- `--virtual-edge-mode <auto-local-radius|auto-global-radius|manual-global-radius>` controls virtual-edge radius selection. Default: `auto-local-radius`.
- `--virtual-radius <number>` sets the manual global virtual-edge radius. This is required when `--virtual-edge-mode manual-global-radius` is used.
- `--virtual-edge-candidate-cap <integer|none>` sets the diagnostic candidate cap per component pair for `auto-local-radius`; `none` disables the cap.
- `--no-weld-vertices` skips duplicate-position welding during input extraction.
- `--no-recompute-normals` preserves transferred source normals when available.
- `--max-iterations <integer>` sets a debug cap on collapse iterations.
- `--progress-interval <integer>` prints progress every N collapses. Default: `1000`.
- `--primitive-grouping <material-parent|material|none>` controls how source primitives are combined before simplification. Default: `material-parent`.
- `--transfer-textures` enables standard PBR texture baking into generated atlases.
- `--texture-size <integer>` sets the output texture width and height in pixels. Default: `1024`.
- `--texture-padding <integer>` sets the atlas chart padding/gutter size in pixels. Default: `2`.
- `--texture-filter <nearest|linear>` controls source texture sampling. Default: `linear`.
- `-h, --help` displays CLI help.

## Browser Usage

Start the development server:

```sh
npm run dev
```

Open the Vite URL shown in the terminal. In the browser app:

1. Drop or choose a `.glb` or `.gltf` file. For `.gltf` assets with external resources, select the `.gltf` file together with referenced `.bin`, PNG, JPEG, or WebP files.
2. Choose simplification settings such as target ratio, target face count, primitive grouping, vertex welding, normal recomputation, and virtual-edge mode.
3. Click **Simplify geometry**.
4. If the loaded asset has supported PBR texture data, click **Bake texture atlas** to resample textures onto generated output UVs.
5. Click **Export processed GLB**.

For production build checks:

```sh
npm run build
npm run preview
```

## Additional Features

This project implements the geometry simplification and texture resampling methods from the original paper, with several useful adaptations for interoperable glTF output:

- The original paper defines the virtual-edge radius as a parameter but leaves the choice of radius open. This implementation provides three radius-selection heuristics:
  - `auto-local-radius` computes a local radius per face from nearby physical edge lengths, with percentile clamping and a bounding-box-based search cap.
  - `auto-global-radius` computes one model-wide radius from the median physical edge length, capped by the model bounding-box diagonal.
  - `manual-global-radius` uses the user-provided `--virtual-radius` value as one model-wide radius.
- Source primitives can be grouped by material and parent, by material only, or processed independently.
- Input extraction can weld duplicate-position vertices, and output normal generation can either recompute smooth normals or prefer transferred authored normals when available.
- Collapse validity rejects newly generated sliver triangles using a scale-invariant quality threshold, improving atlasability for the ordinary-UV output path.
- Baked atlas output copies transferred `COLOR_0` data onto UV seam duplicates without adding topology splits solely for color seams.
- Source tangents are treated as provenance for normal-map interpretation; final glTF tangents are derived from output positions, normals, and UVs.

## Limitations And Future Work

- Texture baking currently supports core glTF 2.0 PBR slots: base-color, normal, metallic-roughness, occlusion, and emissive. Material extension textures and state are not resampled into generated atlases.
- Atlas gutter dilation reduces filtering seams, but runtime mipmapping may still expose chart seams because standard mipmaps average by atlas-space proximity rather than mesh adjacency.
- `COLOR_0` transfer follows simplified vertex provenance and does not preserve every source color seam through extra topology splits.
- Exact scene hierarchy preservation is limited for complex multi-transform inputs; unsupported cases may fall back to world-space output geometry.
- Skinned meshes and morph targets are treated as static base geometry.
- Virtual edges are built during initialization and are not rebuilt as topology changes during simplification.
- Sliver rejection improves robustness for atlas generation but does not guarantee manifold output topology.
