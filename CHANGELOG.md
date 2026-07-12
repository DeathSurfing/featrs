# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LazyFrame support** (`FitLazy` / `TransformLazy` traits, closes #79).
  Every transformer now accepts a Polars `LazyFrame` in addition to `DataFrame`.
  The default implementation collects eagerly and re-lazifies, so all existing
  transformers work out of the box. Six transformers override `transform_lazy`
  with zero-copy Polars expressions that stay inside the query plan:
  `StandardScaler` (`(col - mean) / std`), `MinMaxScaler`, `RobustScaler`,
  `Binarizer` (`when/then/otherwise` with correct `null`/`NaN` propagation),
  `VarianceThreshold`, and `SelectKBest` (both use `select(cols)` for
  projection pushdown). `Pipeline` and `ColumnTransformer` chain lazy
  transforms sequentially / horizontally so an entire pipeline can be
  expressed as a single optimized Polars query plan.
- `FitLazy` and `TransformLazy` exported from `featrs::prelude`.
- `DataFrameTransformer` trait alias now requires `FitLazy + TransformLazy`
  in addition to `Fit + Transform`.

### Fixed

- `Lagger.fit` now rejects duplicate periods at fit time, returning a clear
  `InvalidInput` error instead of silently overwriting lag columns on
  `transform` (Polars `DataFrame::with_column` replaces same-named columns).
- `MissingIndicator.transform` now always adds indicator columns, even
  when the transform data has no nulls. Previously the indicator column
  was conditionally omitted, breaking downstream pipeline schema stability
  (#32).
- `MinMaxScaler.fit` now errors on all-null and all-`f64::NAN` columns
  instead of silently fitting NaN parameters and propagating NaN through
  `transform`. The constant-column guard was bypassed because
  `(NaN).abs() < f64::EPSILON` is `false` per IEEE 754 (#33).
- `SelectKBest.fit` now validates that `k > 0` at fit time, returning a
  clear `InvalidInput` error for `k == 0` instead of failing silently or
  with an opaque downstream error (#44).
- `Binarizer.fit` now rejects empty DataFrames (0 rows) with an
  `InvalidInput` error, matching the convention of every other transformer
  in the crate (#46).
- `StandardScaler.fit` and `RobustScaler.fit` now filter `f64::NAN` values
  before computing statistics (mean, variance, median, IQR) instead of
  silently producing NaN parameters that propagate through `transform`.
  All-null and all-NaN columns now error at fit time (#35).

## [0.3.3] - 2026-07-08

### Fixed

- Corrected `NotFitted` hint messages across all unsupervised transformers
  (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`,
  `Binarizer`, `OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`,
  `SimpleImputer`, `PolynomialFeatures`, `VarianceThreshold`). Each now
  mentions its own name and tells the user to call `.fit(...)` before
  `.transform(...)` instead of the generic "call fit" message (#30).
- `SimpleImputer` `MostFrequent` strategy now deterministically breaks ties
  (the first mode in column order) instead of silently picking a random
  value via unstable sort (#29).

## [0.3.2] - 2026-07-08

### Fixed

- `Normalizer` Max (L∞) norm now uses absolute values (`max(|x_i|)`) instead
  of `max(x_i)`, correcting the normalized output for any row containing
  negative values. The existing `test_max_normalization` test only used
  positive values, so the bug was previously hidden; a regression test
  covering negative values has been added (#9).

## [0.3.1] - 2026-07-07

### Fixed

- `FeatureHasher.fit` now validates that each configured column has `String`
  dtype, surfacing type errors at fit time instead of pushing them to
  `transform`. Non-String columns (e.g. numeric) now fail fast per the
  pipeline contract (#6).

## [0.3.0] - 2026-07-06

### Changed (breaking)

- **Split `Fit` into `Fit` and `FitSupervised`.** Unsupervised transformers now
  implement `Fit<X>` with `fit(&mut self, x: X)` (no target). Only supervised
  transformers (`SelectKBest`) implement `FitSupervised<X, Y>` with
  `fit(&mut self, x: X, y: Y)`. **Migration:** drop the second argument to
  `.fit(...)` on every transformer except `SelectKBest` (e.g.
  `scaler.fit(df, target)` → `scaler.fit(df)`). `use featrs::traits::FitSupervised;`
  where you call `SelectKBest::fit`.
- **`MissingIndicator` moved** from `featrs::traits::missing_indicator` to
  `featrs::preprocessing::missing_indicator`. The prelude re-export is
  unchanged. **Migration:** update any direct `use featrs::traits::missing_indicator`
  paths.
- **`PolynomialFeatures::new` and `PolynomialFeaturesBuilder::build` now return
  `Result<Self>`** instead of panicking on `degree == 0` / missing degree.
  **Migration:** add `.unwrap()` (tests) or `?` (fallible code) at call sites.
- **`Pipeline::new` now returns `Result<Self>`** instead of panicking on empty
  steps. **Migration:** add `.unwrap()` or `?`.
- **`FeatureHasher` uses the signed hashing trick.** Each bucket is now
  incremented by `+1.0` or `-1.0` (determined by a second independent hash),
  so the expected bucket value is zero and collisions no longer bias the mean.
  **Migration:** downstream models trained on unsigned `FeatureHasher` output
  may need retraining; the column count and dtypes are unchanged.

### Added

- `FitTransform::fit_transform(&mut self, x: X) -> Result<Output>` with a
  default implementation of `fit` followed by `transform`. Types may override
  it with a single-pass implementation. `FitSupervised` is re-exported from
  the prelude.
- Tests for the signed hashing trick (`hash_to_bucket` determinism, sign ∈
  `{-1, +1}`, integral bucket values).

### Fixed

- `partial_cmp().unwrap()` → `total_cmp()` at the four float-sort sites
  (NaN-bearing columns no longer panic).
- `AutoTypeDetector::transform` no longer re-fits sub-transformers on every
  call (idempotent, O(N) per call).
- All production `unwrap()`/`expect()` replaced with `Result`-based errors;
  `clippy::unwrap_used`/`expect_used` now denied in production code.

### Changed (non-breaking)

- Declared MSRV 1.91 in `Cargo.toml` (`rust-version = "1.91"`); floor set by
  `polars 0.54` (`_` array lengths, `strict_abs`, transitive `sysinfo`).
- Rewrote CI: cargo caching, Linux/macOS/Windows matrix, rustdoc with
  `-D warnings`, dedicated MSRV job, `cargo audit`, concurrency cancellation.
- Added crate-level lints (`#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`)
  and a `clippy.toml`; fixed the 10 `missing_docs` violations.
- Consolidated duplicated preprocessing helpers into a new `util` module
  (`numeric_f64_columns`, `require_f64_columns`, `replace_f64_column`); the
  scaler/polynomial `fit`/`transform` share one copy of the column logic.
- Collapsed the byte-identical crate-root and `prelude` re-export lists into a
  single canonical list in `prelude` (`pub use crate::prelude::*` at root).
- Regression tests for NaN sorts and the first `AutoTypeDetector` tests.

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

[Unreleased]: https://github.com/DeathSurfing/featrs/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/DeathSurfing/featrs/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/DeathSurfing/featrs/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/DeathSurfing/featrs/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/DeathSurfing/featrs/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/DeathSurfing/featrs/releases/tag/v0.3.0
[0.2.0]: https://github.com/DeathSurfing/featrs/releases/tag/v0.2.0
[0.1.0]: https://github.com/DeathSurfing/featrs/releases/tag/v0.1.0
