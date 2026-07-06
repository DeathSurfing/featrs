//! Feature scaling and centering.
//!
//! Analogous to `sklearn.preprocessing.scaler`. Provides:
//! - [`StandardScaler`] — z-score normalization
//! - [`MinMaxScaler`] — min-max scaling to a range
//! - [`RobustScaler`] — scaling robust to outliers via IQR

use polars::prelude::*;

use crate::traits::{Error, Fit, Result, Transform};

fn numeric_f64_columns(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .filter_map(|name| {
            if let Ok(s) = df.column(name) {
                if s.dtype() == &DataType::Float64 {
                    Some(name.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

/// Standardize features by removing the mean and scaling to unit variance.
///
/// For each column `x`, computes `(x - mean) / std` where `mean` and `std`
/// are learned from the training data.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
///
/// let mut scaler = StandardScaler::new();
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // scaler.fit(df.clone(), target)?;
/// // let scaled = scaler.transform(df)?;
/// ```
pub struct StandardScaler {
    fitted: bool,
    params: Option<Vec<ScaleParam>>,
    with_mean: bool,
    with_std: bool,
}

struct ScaleParam {
    name: String,
    mean: f64,
    std: f64,
}

impl StandardScaler {
    /// Create a new `StandardScaler`.
    ///
    /// Both centering and scaling are enabled by default.
    pub fn new() -> Self {
        Self {
            fitted: false,
            params: None,
            with_mean: true,
            with_std: true,
        }
    }

    /// Whether to center the data by subtracting the mean (default: `true`).
    pub fn with_mean(mut self, value: bool) -> Self {
        self.with_mean = value;
        self
    }

    /// Whether to scale the data to unit variance (default: `true`).
    pub fn with_std(mut self, value: bool) -> Self {
        self.with_std = value;
        self
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame, DataFrame> for StandardScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput("data cannot be empty".into()));
        }

        let col_names = numeric_f64_columns(&x);
        if col_names.is_empty() {
            return Err(Error::InvalidInput(
                "no f64 columns found in DataFrame".into(),
            ));
        }

        let mut params = Vec::with_capacity(col_names.len());

        for name in &col_names {
            let s = x
                .column(name.as_str())
                .map_err(|e| Error::InvalidInput(format!("column '{}' not found: {}", name, e)))?;

            let ca = s
                .f64()
                .map_err(|e| Error::Computation(format!("column '{}' is not f64: {}", name, e)))?;

            let col_mean = if self.with_mean {
                ca.mean().unwrap_or(0.0)
            } else {
                0.0
            };

            let col_std = if self.with_std {
                let var = ca
                    .iter()
                    .flatten()
                    .map(|v| (v - col_mean).powi(2))
                    .sum::<f64>()
                    / ca.len() as f64;
                var.sqrt()
            } else {
                1.0
            };

            if col_std < f64::EPSILON {
                return Err(Error::Computation(format!(
                    "column '{}' has zero variance",
                    name
                )));
            }

            params.push(ScaleParam {
                name: name.clone(),
                mean: col_mean,
                std: col_std,
            });
        }

        self.params = Some(params);
        self.fitted = true;

        Ok(())
    }
}

impl Transform<DataFrame> for StandardScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<Self::Output> {
        if !self.fitted {
            return Err(Error::NotFitted("StandardScaler".into()));
        }

        let params = self.params.as_ref().unwrap();
        let mut out = x.clone();

        for p in params {
            let s = out.column(&p.name).map_err(|e| {
                Error::InvalidInput(format!("column '{}' not found: {}", p.name, e))
            })?;

            let ca = s.f64().map_err(|e| {
                Error::Computation(format!("column '{}' is not f64: {}", p.name, e))
            })?;

            let scaled: ChunkedArray<Float64Type> = ca
                .iter()
                .map(|opt_v| opt_v.map(|v| (v - p.mean) / p.std))
                .collect();

            out.replace(&p.name, scaled.into_series().into())
                .map_err(|e| {
                    Error::Computation(format!("failed to replace column '{}': {}", p.name, e))
                })?;
        }

        Ok(out)
    }
}

/// Scale features to a given range (default `[0, 1]`).
///
/// For each column `x`, computes `(x - min) / (max - min) * range + range_min`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::MinMaxScaler;
/// use featrs::traits::{Fit, Transform};
///
/// let mut scaler = MinMaxScaler::new().feature_range((-1.0, 1.0));
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // scaler.fit(df.clone(), target)?;
/// // let scaled = scaler.transform(df)?;
/// ```
pub struct MinMaxScaler {
    fitted: bool,
    params: Option<Vec<MinMaxParam>>,
    feature_range: (f64, f64),
}

struct MinMaxParam {
    name: String,
    min: f64,
    scale: f64,
}

impl MinMaxScaler {
    /// Create a new `MinMaxScaler` that scales to `[0, 1]`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            params: None,
            feature_range: (0.0, 1.0),
        }
    }

    /// Set the output feature range (default `(0.0, 1.0)`).
    pub fn feature_range(mut self, range: (f64, f64)) -> Self {
        self.feature_range = range;
        self
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame, DataFrame> for MinMaxScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        let col_names = numeric_f64_columns(&x);
        let r_min = self.feature_range.0;
        let r_max = self.feature_range.1;
        let mut params = Vec::new();

        for name in &col_names {
            let s = x.column(name.as_str()).unwrap();
            let ca = s.f64().unwrap();
            let vals: Vec<f64> = ca.iter().flatten().collect();
            let col_min = vals.iter().cloned().fold(f64::NAN, f64::min);
            let col_max = vals.iter().cloned().fold(f64::NAN, f64::max);

            if (col_max - col_min).abs() < f64::EPSILON {
                return Err(Error::Computation(format!(
                    "column '{}' has constant values",
                    name
                )));
            }

            let scale = (r_max - r_min) / (col_max - col_min);
            params.push(MinMaxParam {
                name: name.clone(),
                min: col_min,
                scale,
            });
        }

        self.params = Some(params);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for MinMaxScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("MinMaxScaler".into()));
        }
        let r_min = self.feature_range.0;
        let mut out = x.clone();

        for p in self.params.as_ref().unwrap() {
            let s = out.column(&p.name).unwrap();
            let ca = s.f64().unwrap();
            let scaled: ChunkedArray<Float64Type> = ca
                .iter()
                .map(|opt| opt.map(|v| (v - p.min) * p.scale + r_min))
                .collect();
            out.replace(&p.name, scaled.into_series().into()).unwrap();
        }

        Ok(out)
    }
}

/// Scale features using statistics robust to outliers.
///
/// For each column `x`, computes `(x - median) / IQR` using the
/// interquartile range, which is insensitive to outliers.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::RobustScaler;
/// use featrs::traits::{Fit, Transform};
///
/// let mut scaler = RobustScaler::new().with_centering(true);
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // scaler.fit(df.clone(), target)?;
/// // let scaled = scaler.transform(df)?;
/// ```
pub struct RobustScaler {
    fitted: bool,
    params: Option<Vec<RobustParam>>,
    with_centering: bool,
    with_scaling: bool,
}

struct RobustParam {
    name: String,
    center: f64,
    scale: f64,
}

impl RobustScaler {
    /// Create a new `RobustScaler` with centering and scaling enabled.
    pub fn new() -> Self {
        Self {
            fitted: false,
            params: None,
            with_centering: true,
            with_scaling: true,
        }
    }

    /// Whether to center by subtracting the median (default: `true`).
    pub fn with_centering(mut self, value: bool) -> Self {
        self.with_centering = value;
        self
    }

    /// Whether to scale by the IQR (default: `true`).
    pub fn with_scaling(mut self, value: bool) -> Self {
        self.with_scaling = value;
        self
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = (p / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl Fit<DataFrame, DataFrame> for RobustScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        let col_names = numeric_f64_columns(&x);
        let mut params = Vec::new();

        for name in &col_names {
            let s = x.column(name.as_str()).unwrap();
            let ca = s.f64().unwrap();
            let mut vals: Vec<f64> = ca.iter().flatten().collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = percentile_sorted(&vals, 50.0);
            let q1 = percentile_sorted(&vals, 25.0);
            let q3 = percentile_sorted(&vals, 75.0);
            let iqr = q3 - q1;

            if iqr < f64::EPSILON {
                return Err(Error::Computation(format!(
                    "column '{}' has zero IQR",
                    name
                )));
            }

            params.push(RobustParam {
                name: name.clone(),
                center: if self.with_centering { median } else { 0.0 },
                scale: if self.with_scaling { iqr } else { 1.0 },
            });
        }

        self.params = Some(params);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for RobustScaler {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("RobustScaler".into()));
        }
        let mut out = x.clone();

        for p in self.params.as_ref().unwrap() {
            let s = out.column(&p.name).unwrap();
            let ca = s.f64().unwrap();
            let scaled: ChunkedArray<Float64Type> = ca
                .iter()
                .map(|opt| opt.map(|v| (v - p.center) / p.scale))
                .collect();
            out.replace(&p.name, scaled.into_series().into()).unwrap();
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_standard_scaler_fit_transform() {
        let mut scaler = StandardScaler::new();
        let df = make_test_df();
        let y = df.clone();

        scaler.fit(df.clone(), y).unwrap();
        let result = scaler.transform(df).unwrap();

        let scaled_a = result.column("a").unwrap().f64().unwrap();
        let vals: Vec<f64> = scaled_a.iter().flatten().collect();

        assert_relative_eq!(vals[0], -1.22474487, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.22474487, epsilon = 1e-6);
    }

    #[test]
    fn test_min_max_scaler() {
        let mut scaler = MinMaxScaler::new();
        let df = make_test_df();
        let y = df.clone();

        scaler.fit(df.clone(), y).unwrap();
        let result = scaler.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("a")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_robust_scaler() {
        let mut scaler = RobustScaler::new();
        let df = make_test_df();
        let y = df.clone();

        scaler.fit(df.clone(), y).unwrap();
        let result = scaler.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("a")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let df = make_test_df();
        let result = scaler.transform(df);
        assert!(result.is_err());
    }
}
