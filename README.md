# wgpu-solver-backend

![Rust](https://img.shields.io/badge/Rust-stable-orange)
![wgpu](https://img.shields.io/badge/wgpu-GPU%20backend-blue)
![Linear Algebra](https://img.shields.io/badge/Linear%20Algebra-iterative%20solvers-lightgrey)
![Status](https://img.shields.io/badge/status-learning%20%2F%20building-lightgrey)

A Rust project that ports my **WebGPU-based sparse solver path** to a **native `wgpu` backend**.

This repository is part of the same “project story” as:

- [`fea_app`](https://github.com/RomanShushakov/fea_app) — browser FEA pipeline + WebGL/WebGPU visualization + compute experiments
- [`iterative_solvers`](https://github.com/RomanShushakov/iterative_solvers) — CPU iterative methods (CG / PCG)
- [`colsol`](https://github.com/RomanShushakov/colsol) — direct-solver experiments (LDLᵀ / column-style elimination)
- [`finite_element_method`](https://github.com/RomanShushakov/finite_element_method) — FEM building blocks used by the app

---

## What this repo is

In `fea_app` I built a WebGPU compute path for an iterative solver workflow:
- sparse **SpMV** (CSR)
- vector ops (AXPY / scale)
- dot-product reductions
- a **Block-Jacobi** preconditioner apply (small dense blocks stored as LU)

This repo extracts that idea and **runs it on `wgpu`** (native backend), so the same solver core can be used outside the browser.

The goal is not “fastest possible GPU linear algebra”, but a **clear, inspectable** implementation that keeps the data flow explicit.

---

## Solver building blocks

The GPU side is structured as small kernels/executors, typically along these lines:

- **SpMV (CSR)**  
  `y = A * x`

- **Dot reduction**  
  `dot(a, b)` reduced to a single scalar

- **Vector ops**  
  `y = y + αx` (AXPY), `x = βx` (scale)

- **Block-Jacobi preconditioner apply**  
  For each block, solve a small dense system using stored LU factors:  
  `z = M^{-1} r`

- **PCG loop glue**  
  One iteration wires these pieces together:
  `SpMV → dot → scalar update → axpy/scale → preconditioner → dot → update`

---

## How it relates to the WebGPU version

This is a **backend port**, not a new algorithm:
- the WebGPU version (browser) lives in [`fea_app`](https://github.com/RomanShushakov/fea_app)
- this repo keeps the same conceptual pipeline but uses `wgpu` so it can run:
  - as a native executable
  - in CI
  - in headless environments (when supported)
  - as a building block for other backends

---

## Usage

If the repo exposes a library API:
- build and call it from your own code (typical: pass CSR + RHS + solver params)

If it includes a CLI/example:
- run with an input file describing:
  - CSR matrix
  - RHS vector
  - solver settings (max iters / tolerances / block layout)

> Notes:
> - The implementation assumes consistent dimensions and valid block boundaries.
> - Block-Jacobi LU uses **no pivoting**, so diagonal/conditioning matters.

---

## Design notes

- **Small kernels, explicit buffers.** Each kernel does one thing.
- **Deterministic data movement.** It’s easy to see what buffers are read/written per pass.
- **Educational bias.** Optimizations are added only when they don’t destroy clarity.

---

## Status

Learning / building. APIs and internal layout may evolve as I connect this backend to other parts of the project.

---

## License

MIT OR Apache-2.0 (see `Cargo.toml`).
