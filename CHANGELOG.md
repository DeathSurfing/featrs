# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Declared Minimum Supported Rust Version (MSRV) as 1.91 in `Cargo.toml` via
  `rust-version = "1.91"`. The floor is set by `polars 0.54`'s use of `_` array
  lengths and `strict_abs` (both stabilized in Rust 1.91), plus the transitive
  `sysinfo` dependency (requires 1.88). Edition 2024 itself only requires 1.85.
- Rewrote CI to cache cargo build artifacts, test on Linux/macOS/Windows,
  check rustdoc with `-D warnings`, verify the MSRV (1.91) in a dedicated job,
  and run `cargo audit` for security advisories.

### Added

- Crate-level lints: `#![forbid(unsafe_code)]` and `#![warn(missing_docs)]`,
  plus a `clippy.toml` pinning `msrv = "1.91"`. `clippy::unwrap_used` is
  explicitly allowed pending a sweep that replaces `unwrap()` with `Result`.
- Doc comments on the 10 previously-undocumented public items
  (`AutoTypeDetector::new`, `CyclicalEncoder::{new,with_periods}`,
  `Difference::new`, `RollingFn` variants, `RollingAggregator::new`).

### Changed

- Consolidated duplicated preprocessing helpers into a new `util` module:
  `numeric_f64_columns`, `require_f64_columns`, and `replace_f64_column`.
  The scaler and polynomial `fit`/`transform` implementations now share one
  copy of the column-discovery, validation, and in-place-replace logic.
- Collapsed the byte-identical crate-root and `prelude` re-export lists into a
  single canonical list in `prelude`, re-exported at the root via
  `pub use crate::prelude::*`. New public types now only need to be added once.

### Fixed

- Replaced `partial_cmp().unwrap()` with `total_cmp()` at the four float-sort
  sites (`RobustScaler`, `SimpleImputer::Median`, `FClassif`, `SelectKBest`),
  so NaN-bearing columns no longer panic during `fit`.
- `AutoTypeDetector::transform` no longer re-fits its `OneHotEncoder` /
  `FeatureHasher` sub-transformers on every call. They are now fit once during
  `fit` and stored, so learned categories and hash mappings are stable across
  `transform` calls (idempotent, O(N) instead of O(N·M) per call).

### Added

- Regression tests: NaN-bearing `RobustScaler` and `SimpleImputer::Median`
  sorts, plus the first unit tests for `AutoTypeDetector` (detection, transform
  idempotency, and not-fitted error).

## [0.2.0] - 2026-07-06

### Added

- Time-series transformers: `Lagger`, `RollingAggregator` (with `RollingFn`),
  `Difference`, `CyclicalEncoder`.
- `FeatureHasher` for hashed categorical encoding.
- `AutoTypeDetector` with `ColumnType` inference and a `PolynomialFeaturesBuilder`.
- `prelude` module re-exporting the public API.
- Integration tests covering end-to-end pipelines, feature selection, encoders,
  and imputation.

### Changed

- Actionable error messages across all transformers.
- Polished README with badges, quick start, and feature matrix.

## [0.1.0] - 2026-07-06

### Added

- Core trait hierarchy: `Fit`, `Transform`, `FitTransform` with `Error`/`Result`.
- Preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`,
  `Binarizer`, `OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`, `SimpleImputer`,
  `PolynomialFeatures`, `MissingIndicator`.
- Feature selection: `SelectKBest` with `FClassif`, `VarianceThreshold`.
- Pipeline primitives: `Pipeline`, `ColumnTransformer` with `Remainder`.
- Comprehensive API docs, module docs, and contributing guide.

[Unreleased]: https://github.com/DeathSurfing/featrs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DeathSurfing/featrs/releases/tag/v0.2.0
[0.1.0]: https://github.com/DeathSurfing/featrs/releases/tag/v0.1.0
