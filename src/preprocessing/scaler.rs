use polars::prelude::*;
use std::collections::HashMap;

use crate::traits::{Error, Fit, Result, Transform};

/// Standardize features by removing the mean and scaling to unit variance.
///
/// Corresponds to `sklearn.preprocessing.StandardScaler`.
pub struct StandardScaler {
    fitted: bool,
    params: Option<Vec<ScalerParam>>,
    with_mean: bool,
    with_std: bool,
}

struct ScalerParam {
    name: String,
    mean: f64,
    std: f64,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            fitted: false,
            params: None,
            with_mean: true,
            with_std: true,
        }
    }

    pub fn with_mean(mut self, value: bool) -> Self {
        self.with_mean = value;
        self
    }

    pub fn with_std(mut self, value: bool) -> Self {
        self.with_std = value;
        self
    }

    fn numeric_f64_columns(&self, df: &DataFrame) -> Vec<String> {
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
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame, DataFrame> for StandardScaler {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<Self::Output> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput("data cannot be empty".into()));
        }

        let col_names = self.numeric_f64_columns(&x);
        if col_names.is_empty() {
            return Err(Error::InvalidInput(
                "no f64 columns found in DataFrame".into(),
            ));
        }

        let mut params = Vec::with_capacity(col_names.len());

        for name in col_names {
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

            params.push(ScalerParam {
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

            let new_s = scaled.into_series();
            out.replace(&p.name, new_s.into()).map_err(|e| {
                Error::Computation(format!("failed to replace column '{}': {}", p.name, e))
            })?;
        }

        Ok(out)
    }
}

#[allow(dead_code)]
pub struct MinMaxScaler {
    fitted: bool,
    min: Option<HashMap<String, f64>>,
    scale: Option<HashMap<String, f64>>,
    feature_range: (f64, f64),
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self {
            fitted: false,
            min: None,
            scale: None,
            feature_range: (0.0, 1.0),
        }
    }

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

#[allow(dead_code)]
pub struct RobustScaler {
    fitted: bool,
    center: Option<HashMap<String, f64>>,
    scale: Option<HashMap<String, f64>>,
    with_centering: bool,
    with_scaling: bool,
    quantile_range: (f64, f64),
}

impl RobustScaler {
    pub fn new() -> Self {
        Self {
            fitted: false,
            center: None,
            scale: None,
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
        }
    }

    pub fn with_centering(mut self, value: bool) -> Self {
        self.with_centering = value;
        self
    }

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
        let vals: Vec<f64> = scaled_a.iter().filter_map(|v| v).collect();

        assert_eq!(vals.len(), 3);
        assert_relative_eq!(vals[0], -1.22474487, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.22474487, epsilon = 1e-6);
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let df = make_test_df();
        let result = scaler.transform(df);
        assert!(result.is_err());
    }

    #[test]
    fn test_min_max_scaler_new() {
        let s = MinMaxScaler::new();
        assert_eq!(s.feature_range, (0.0, 1.0));
    }

    #[test]
    fn test_robust_scaler_new() {
        let s = RobustScaler::new();
        assert!(s.with_centering);
        assert!(s.with_scaling);
    }
}
