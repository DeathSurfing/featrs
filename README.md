# featrs

Feature engineering library for Rust, inspired by scikit-learn.

**Status:** 🚧 Early development — API is unstable and incomplete.

## Goals

Provide a scikit-learn-like API for:

- **Preprocessing**: scaling, encoding, normalization, imputation, binarization
- (Future) Feature selection, decomposition, pipeline composition

## Usage

```toml
[dependencies]
featrs = "0.1"
```

```rust
use featrs::preprocessing::scaler::StandardScaler;
use featrs::traits::{Fit, Transform};

let scaler = StandardScaler::new();
// ...
```

## License

MIT
