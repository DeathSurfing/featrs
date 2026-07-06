# featrs

[![Crates.io](https://img.shields.io/crates/v/featrs)](https://crates.io/crates/featrs)
[![Docs.rs](https://img.shields.io/docsrs/featrs)](https://docs.rs/featrs)
[![CI](https://github.com/anomalyco/featrs/actions/workflows/ci.yml/badge.svg)](https://github.com/anomalyco/featrs/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Feature engineering library for Rust, inspired by scikit-learn.

Built on [Polars](https://pola.rs) — all transformations operate natively on `DataFrame` and preserve column names.

## Installation

```toml
[dependencies]
featrs = "0.1"
```

## Quick start

```rust
use polars::prelude::*;
use featrs::preprocessing::scaler::StandardScaler;
use featrs::traits::{Fit, Transform};

let mut scaler = StandardScaler::new();
scaler.fit(data.clone(), target)?;
let scaled = scaler.transform(data)?;
```

## Features

| Component | Description |
|---|---|
| `StandardScaler` | Z-score normalization (mean 0, variance 1) |
| `MinMaxScaler` | Scale to `[0, 1]` or custom range |
| `RobustScaler` | Scale using median and IQR (outlier-robust) |
| `Normalizer` | Row-wise L1, L2, or Max normalization |
| `OneHotEncoder` | Create binary dummy columns for categories |
| `LabelEncoder` | Encode labels as `0..n_classes-1` integers |
| `OrdinalEncoder` | Per-column category → integer encoding |
| `Binarizer` | Threshold-based binarization |
| `SimpleImputer` | Fill nulls with mean, median, mode, or constant |
| `PolynomialFeatures` | Generate polynomial and interaction features |
| `Pipeline` | Sequentially chain multiple transformers |
| `ColumnTransformer` | Apply different transformers to different columns |
| `VarianceThreshold` | Remove low-variance features |
| `SelectKBest` | Select top-k features by statistical test (ANOVA F) |

## Examples

### StandardScaler

```rust
use featrs::preprocessing::scaler::StandardScaler;
use featrs::traits::{Fit, Transform};

let mut scaler = StandardScaler::new();
scaler.fit(df.clone(), target.clone())?;
let scaled = scaler.transform(df)?;
```

### Pipeline

```rust
use featrs::pipeline::Pipeline;
use featrs::preprocessing::scaler::StandardScaler;
use featrs::preprocessing::polynomial_features::PolynomialFeatures;

let mut pipeline = Pipeline::new(vec![
    ("scaler".into(), Box::new(StandardScaler::new())),
    ("poly".into(), Box::new(PolynomialFeatures::new(2))),
]);
pipeline.fit(df.clone(), target)?;
let result = pipeline.transform(df)?;
```

### ColumnTransformer

```rust
use featrs::pipeline::ColumnTransformer;
use featrs::pipeline::column_transformer::Remainder;
use featrs::preprocessing::scaler::StandardScaler;

let ct = ColumnTransformer::new(
    vec![("scale".into(), Box::new(StandardScaler::new()), vec!["feat_a".into()])],
    Remainder::Passthrough,
);
```

### Feature Selection

```rust
use featrs::feature_selection::VarianceThreshold;
use featrs::feature_selection::SelectKBest;
use featrs::feature_selection::select_kbest::FClassif;

let mut vt = VarianceThreshold::new(0.01);
vt.fit(features.clone(), target.clone())?;
let filtered = vt.transform(features)?;

let mut skb = SelectKBest::new(5, Box::new(FClassif::new()));
skb.fit(features.clone(), target)?;
let selected = skb.transform(features)?;
```

### PolynomialFeatures

```rust
use featrs::preprocessing::polynomial_features::PolynomialFeatures;

let mut pf = PolynomialFeatures::new(3)
    .include_bias(true)
    .interaction_only(false);
pf.fit(df.clone(), target)?;
let result = pf.transform(df)?;
```

## Resources

- [Crates.io](https://crates.io/crates/featrs)
- [Docs.rs](https://docs.rs/featrs)
- [GitHub](https://github.com/anomalyco/featrs)

## License

MIT
