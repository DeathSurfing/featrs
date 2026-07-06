//! Scaling and centering transformations.
//!
//! Analogous to `sklearn.preprocessing` scalers.

use ndarray::Array2;

use crate::traits::{Error, Fit, Result, Transform};

/// Standardize features by removing the mean and scaling to unit variance.
///
/// Corresponds to `sklearn.preprocessing.StandardScaler`.
pub struct StandardScaler {
    fitted: bool,
    mean: Option<Vec<f64>>,
    scale: Option<Vec<f64>>,
    with_mean: bool,
    with_std: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            fitted: false,
            mean: None,
            scale: None,
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
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<f64, Array2<f64>, Array2<f64>> for StandardScaler {
    type Output = ();

    fn fit(&mut self, x: Array2<f64>, _y: Array2<f64>) -> Result<Self::Output> {
        let (n_samples, n_features) = x.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(Error::InvalidInput("data cannot be empty".into()));
        }

        let mut mean = Vec::with_capacity(n_features);
        let mut scale = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = x.column(j);

            let col_mean = if self.with_mean {
                col.mean().unwrap_or(0.0)
            } else {
                0.0
            };

            let col_std = if self.with_std {
                let var = col
                    .iter()
                    .map(|v| (v - col_mean).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;
                var.sqrt()
            } else {
                1.0
            };

            if col_std < f64::EPSILON {
                return Err(Error::Computation(format!(
                    "feature {} has zero variance",
                    j
                )));
            }

            mean.push(col_mean);
            scale.push(col_std);
        }

        self.mean = Some(mean);
        self.scale = Some(scale);
        self.fitted = true;

        Ok(())
    }
}

impl Transform<f64, Array2<f64>> for StandardScaler {
    type Output = Array2<f64>;

    fn transform(&self, x: Array2<f64>) -> Result<Self::Output> {
        if !self.fitted {
            return Err(Error::NotFitted("StandardScaler".into()));
        }
        let (_n_samples, n_features) = x.dim();
        let mean = self.mean.as_ref().unwrap();
        let scale = self.scale.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(Error::InvalidInput(format!(
                "expected {} features, got {}",
                mean.len(),
                n_features
            )));
        }

        let mut out = x;
        for j in 0..n_features {
            let mut col = out.column_mut(j);
            for val in col.iter_mut() {
                *val = (*val - mean[j]) / scale[j];
            }
        }
        Ok(out)
    }
}

/// Scale features to a given range (default [0, 1]).
///
/// Corresponds to `sklearn.preprocessing.MinMaxScaler`.
#[allow(dead_code)]
pub struct MinMaxScaler {
    fitted: bool,
    min: Option<Vec<f64>>,
    scale: Option<Vec<f64>>,
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

/// Scale features using statistics that are robust to outliers.
///
/// Corresponds to `sklearn.preprocessing.RobustScaler`.
#[allow(dead_code)]
pub struct RobustScaler {
    fitted: bool,
    center: Option<Vec<f64>>,
    scale: Option<Vec<f64>>,
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
    use ndarray::arr2;

    #[test]
    fn test_standard_scaler_fit_transform() {
        let mut scaler = StandardScaler::new();
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let y = arr2(&[[0.0; 2]; 3]);
        scaler.fit(x.clone(), y).unwrap();

        let result = scaler.transform(x).unwrap();
        assert_relative_eq!(result[[0, 0]], -1.22474487, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 0]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[[2, 0]], 1.22474487, epsilon = 1e-6);
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let x = arr2(&[[1.0, 2.0]]);
        let result = scaler.transform(x);
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
