# Contributing

## Getting started

```bash
git clone https://github.com/DeathSurfing/featrs.git
cd featrs
cargo build
cargo test
```

## Code style

- `cargo fmt` before committing
- `cargo clippy --all-targets --all-features -- -D warnings` must pass
- 4-space indentation, no trailing whitespace
- All public items must have doc comments (`///`)
- Module-level `//!` docs for every `.rs` file

## Adding a new transformer

1. Create a struct with `fitted: bool` and parameter fields
2. Implement `Fit<DataFrame, DataFrame>` and `Transform<DataFrame>`
3. Add module docs (`//!`), struct docs (`///`), and constructor docs
4. Add tests in a `#[cfg(test)] mod tests` block
5. Register the module in the parent `mod.rs`

## PR process

1. Branch from `main`: `git checkout -b feat/my-feature`
2. Commit with conventional commits style (`feat:`, `fix:`, `docs:`, etc.)
3. Push and create a PR
4. CI must pass (build, clippy, fmt, test)
5. Squash-merge when approved
