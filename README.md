# featrs

Feature engineering library for Rust, inspired by scikit-learn.

Built on [Polars](https://pola.rs) — works natively with DataFrames.

**Status:** Early development — API is unstable and incomplete.

## Features

| Component | Status |
|---|---|
| `StandardScaler` | ✅ |
| `MinMaxScaler` | 🏗️ Stub |
| `RobustScaler` | 🏗️ Stub |
| `Normalizer` | 🏗️ Stub |
| `OneHotEncoder` | 🏗️ Stub |
| `LabelEncoder` | 🏗️ Stub |
| `OrdinalEncoder` | 🏗️ Stub |
| `Binarizer` | 🏗️ Stub |
| `SimpleImputer` | 🏗️ Stub |
| **`PolynomialFeatures`** | ✅ |
| **`Pipeline`** | ✅ |
| **`ColumnTransformer`** | ✅ |

## Usage

```toml
[dependencies]
featrs = "0.1"
```

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
use featrs::pipeline::Remainder;
use featrs::preprocessing::scaler::StandardScaler;

let ct = ColumnTransformer::new(
    vec![("scale".into(), Box::new(StandardScaler::new()), vec!["feat_a".into()])],
    Remainder::Passthrough,
);
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

## License

MIT
