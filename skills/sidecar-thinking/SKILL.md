---
name: sidecar-thinking
description: Use when the user explicitly asks to use Research Sidecar, open/start/connect to the sidecar app, inspect or select a sidecar research graph, install sidecar workspace skills, or run an external/second-opinion sidecar thinking pass.
---

# Sidecar Thinking

Research Sidecar is an opt-in external tool. Do not invoke it just because the task involves research, reports, Markdown, YAML, or graph-like structure.

## Strict Trigger

Use this skill only when the user explicitly asks for at least one of these:

- Use, open, start, run, connect to, or inspect Research Sidecar.
- Use the sidecar graph view, select/save a graph, preview graph-linked documents, or inspect a sidecar research graph.
- Do an external, out-of-band, second-opinion, or sidecar thinking pass.
- Install the sidecar bundled skills into the current workspace.

Do not use this skill for ordinary research reasoning, report writing, code edits, Markdown/HTML generation, YAML editing, or generic graph work unless the user ties the request to Sidecar.

## Current CLI Shape

Start the app only when the user asked for Sidecar to be used or started:

```bash
research-sidecar
```

Useful variants:

```bash
research-sidecar --workspace /path/to/workspace
research-sidecar --graph dingyi/synthetic/graph.yaml
research-sidecar --port 4317
```

Initialize a workspace:

```bash
research-sidecar init --graph research/graph.yaml
research-sidecar init --graph research/graph.yaml --no-skills
```

Install bundled skills into the workspace:

```bash
research-sidecar install-skills
research-sidecar install-skills --force
```

If installed as a project dependency, use the same commands through `npx`:

```bash
npx research-sidecar
npx research-sidecar install-skills
```

For one-shot use without a local install:

```bash
npx @binarycheater/research-sidecar
```

The directory where `research-sidecar` runs is the workspace root unless `--workspace` is supplied. Private state lives in `.side/config.json`. Default URL: `http://localhost:4317`.

## Sessions And API

Use the API only after the Sidecar server is running and the user asked for Sidecar involvement.

Common session sequence:

1. `POST /api/sessions`
2. `PATCH /api/sessions/:id` with manual context
3. `POST /api/sessions/:id/files` for workspace-relative files
4. `POST /api/sessions/:id/messages` to stage a user question
5. `POST /api/sessions/:id/stream` only if the user asked Codex to relay the sidecar answer

## Research Graph

The graph is manifest-first:

- `graph.yaml` is the authority for nodes, edges, UI defaults, and file pointers.
- Markdown and HTML files hold node content.
- Markdown frontmatter can supply local metadata, but `graph.yaml` wins on conflicts.
- Graph file links are relative to the graph manifest directory by default. Use a leading `/` for an explicit workspace-root-relative link.

Default manifest path: `research/graph.yaml`. Override with `--graph`, or save the active graph in `.side/config.json` at `graph.manifestPath`. The UI can discover graph files across the workspace and save the selected graph to `.side/config.json`.

## API Quick Reference

- `GET /api/config`
- `PATCH /api/config`
- `GET /api/graphs`
- `GET /api/workspace`
- `POST /api/workspace/skills/install`
- `GET /api/workspace/file?path=<workspace-relative-path>`
- `GET /api/workspace/raw?path=<workspace-relative-path>`
- `GET /api/research-graph`
- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/:id`
- `PATCH /api/sessions/:id`
- `POST /api/sessions/:id/files`
- `POST /api/sessions/:id/messages`
- `POST /api/sessions/:id/stream`

All workspace paths are constrained to the configured workspace root.
