# Build Optimization Guide

This document covers strategies to speed up compilation times during development.

## Quick Reference

| Build Profile | Command | Use Case |
|---------------|---------|----------|
| Dev (default) | `cargo build` | Day-to-day development |
| Dev Fast | `cargo build --profile dev-fast` | Faster iteration, moderate optimization |
| Release Fast | `cargo build --profile release-fast` | Testing release-like builds quickly |
| Release | `cargo build --release` | Final distribution, maximum optimization |

## Incremental Build Strategy

### 1. Use the Right Profile

For **quick iteration** (shader changes, UI tweaks):
```bash
cargo build                    # Default dev profile (opt-level=1)
cargo run                      # Auto-builds with dev profile
```

For **testing performance** without full release build:
```bash
cargo build --profile dev-fast # Slightly optimized but fast to build
cargo run --profile dev-fast
```

For **release testing** (when you need LTO but faster):
```bash
cargo build --profile release-fast  # Thin LTO, more codegen units
cargo run --profile release-fast
```

### 2. Avoid cargo clean

Never run `cargo clean` unless absolutely necessary. Incremental compilation is your friend.

If you need a fresh build for only this crate:
```bash
cargo build --bin evolution-sim -Z build-std  # Rebuilds only the binary
```

### 3. Check vs Build

For syntax/type checking only (fastest feedback):
```bash
cargo check       # Much faster than cargo build
cargo check -q    # Quiet mode, just errors
```

### 4. Parallel Builds

The `.cargo/config.toml` already sets `jobs = 0` (use all cores). On a multi-core CPU this helps significantly.

## Faster Linking (Advanced)

Linking is often the slowest part of incremental builds. You can speed it up:

### Windows: LLD Linker

Install LLVM tools:
```bash
rustup component add llvm-tools-preview
```

Then edit `.cargo/config.toml`:
```toml
[target.x86_64-pc-windows-msvc]
linker = "lld-link"
```

### Linux: mold Linker

Install mold (https://github.com/rui314/mold), then:
```toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

## sccache (Build Caching)

For repeated builds across clean builds, use sccache:

1. Install: `cargo install sccache`
2. Set env: `RUSTC_WRAPPER=sccache`
3. Enjoy cached compilation

## Workspace-Specific Tips

### Shader Changes

Shader files (`.wgsl`) are included via `include_str!()` in Rust, so changes trigger recompilation of the compute/render modules. This is relatively fast.

### Config Changes

`config.toml` is loaded at runtime - no recompilation needed! Just restart the app.

### Dependency Updates

When updating dependencies, expect a longer build. Plan these for low-productivity times.

## Measuring Build Time

```bash
cargo build --timings       # Generates HTML report in target/cargo-timings/
cargo build --timings=html  # Same, explicit HTML output
```

Open `target/cargo-timings/cargo-timing.html` to see a waterfall of compile times.

## Profile Comparison

| Profile | Build Time* | Runtime Perf | Debug Info |
|---------|-------------|--------------|------------|
| dev | ~3s incremental | Slow (1x) | Yes (lines only) |
| dev-fast | ~4s incremental | Medium (5x) | No |
| release-fast | ~15s incremental | Fast (15x) | No |
| release | ~45s clean | Fastest (20x) | No |

*Times are approximate for this project on a mid-range CPU.

## Development Workflow

1. **Normal coding**: Use `cargo run` (dev profile)
2. **Testing behavior**: Use `cargo run --profile dev-fast`
3. **Performance testing**: Use `cargo run --profile release-fast`
4. **Final release**: Use `cargo run --release`

## Troubleshooting Slow Builds

1. **Check for full rebuilds**: If everything rebuilds, check if you touched a core file
2. **Disable antivirus scanning**: Exclude `target/` directory from antivirus
3. **Use SSD**: Build on an SSD, not an HDD
4. **RAM disk**: For extreme speed, put `target/` on a RAM disk
5. **Reduce parallelism on low RAM**: If you have <16GB RAM, try `jobs = 4` instead of `0`
