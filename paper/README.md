# Paper

This folder is **deliberately empty of a manuscript** for now.

## Why it is empty

The study is under submission to **ERAMIA-RS 2026**, which is **double-blind**.
Publishing the manuscript — or linking this repository from it — while the paper
is under review would break anonymity. So, during review:

- the PDF is **not** versioned here;
- the paper does **not** cite this repository, this URL or the author's name;
- the repository stays **private**.

## What goes here after acceptance

Once the paper is accepted, drop the camera-ready PDF in this folder as
`vqc-depth-eramia-2026.pdf` and it becomes the entry point that explains the
numbers this code produces.

## Going public: checklist

The repository is meant to become public *after* acceptance, not before. In
order:

1. **Acceptance confirmed** and the camera-ready is final.
2. Add the PDF to this folder and link it from the root `README.md`.
3. Re-read the whole repository for anything that should not be public — the
   test suite and the CI are already clean, but data caches, output folders and
   personal paths are only kept out by `.gitignore`, so verify with
   `git ls-files` rather than by looking at the working tree.
4. Flip the repository to public.
5. Only then add the repository URL to the paper (camera-ready or a later
   version), since it is no longer blind.
6. Optional but cheap: archive a tagged release on Zenodo to get a DOI, and
   add a `CITATION.cff` pointing at it.

## What this code does and does not contain

Code only. No downloaded datasets, no run outputs, no figures and no tables are
versioned — everything under `--outdir` is generated, and reproducing it is a
matter of running the grid. See the root [README](../README.md).
