# Ph.D. Computation Camp — Course Website

Course website for the UW-Madison Computational Bootcamp (Summer 2026), built
with [Quarto](https://quarto.org).

## Structure

| Path | What it is |
|------|------------|
| `_quarto.yml` | Site config: navbar, theme, and which files get published |
| `index.qmd` | Home / syllabus |
| `lectures.qmd` | Slides + code-along links per lecture |
| `homeworks.qmd` | Problem sets, starter code, data, solutions |
| `codealongs.qmd` | Table of code-along files |
| `resources.qmd` | Julia / computational econ references |
| `styles.scss`, `extra.css` | Styling (UW-Madison theme) |
| `lectures/`, `codealongs/`, `homeworks/` | The actual course materials (PDFs, `.jl`, data) |

Content lives in the `.qmd` (Quarto Markdown) files — edit those in any text
editor. Material downloads are the real files already in `lectures/`,
`codealongs/`, and `homeworks/`, linked by relative path.

## Prerequisites

Install [Quarto](https://quarto.org/docs/get-started/) (v1.4+). No R or Python is
required — these pages are plain Markdown.

## Preview locally

```bash
quarto preview
```

This opens a live-reloading preview in your browser. Edit a `.qmd` file and save
to see changes instantly.

## Build

```bash
quarto render
```

Output is written to `_site/`. Open `_site/index.html` to view the built site.

## Deploy to GitHub Pages

A GitHub Actions workflow (`.github/workflows/publish.yml`) renders and deploys
the site automatically.

1. Create a GitHub repo and push this folder to the `main` branch.
2. In the repo, go to **Settings → Pages → Build and deployment** and set
   **Source** to **GitHub Actions**.
3. Every push to `main` rebuilds and publishes the site.

If your repo is named something other than `UW_Econ_Bootcamp_26`, update
`site-url` in `_quarto.yml` and the GitHub link in the navbar.

### Alternative: one-off manual publish

```bash
quarto publish gh-pages
```

## Editing tips

- Add a new page: create `newpage.qmd`, add it to `project.render` and the
  `navbar` in `_quarto.yml`.
- Add materials: drop files into `lectures/`, `codealongs/`, or `homeworks/`,
  link them with a relative path, and make sure the folder glob is listed under
  `project.resources` in `_quarto.yml` so the files get copied into the build.
- Fill in real dates on `homeworks.qmd` and the location/dates on `index.qmd`
  once they're confirmed.
