//! Variance-based feature selection.
//!
//! [`VarianceThreshold`] removes features whose variance does not meet
//! a threshold, i.e. features that are constant or nearly constant.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Remove features with variance below a threshold.
///
/// Features with variance below `threshold` are removed. By default (threshold `0.0`),
/// only constant features are removed.
///
/// # Example
///
/// ```rust
/// use featrs::feature_selection::VarianceThreshold;
///
/// let mut vt = VarianceThreshold::new(0.01);
/// ```
pub struct VarianceThreshold {
    fitted: bool,
    threshold: f64,
    selected_columns: Option<Vec<String>>,
}

impl VarianceThreshold {
    /// Create a new `VarianceThreshold` transformer.
    ///
    /// Features with variance strictly less than `threshold` are dropped.
    /// Use `0.0` (default) to remove only constant features.
    pub fn new(threshold: f64) -> Self {
        Self {
            fitted: false,
            threshold,
            selected_columns: None,
        }
    }
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Fit<DataFrame, DataFrame> for VarianceThreshold {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "VarianceThreshold.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "VarianceThreshold.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }

        let mut selected = Vec::new();
        for col in x.columns() {
            let name = col.name().to_string();
            let s = x.column(&name).unwrap();
            if s.dtype() != &DataType::Float64 {
                continue;
            }
            let ca = s.f64().unwrap();
            let mean = ca.mean().unwrap_or(0.0);
            let var =
                ca.iter().flatten().map(|v| (v - mean).powi(2)).sum::<f64>() / ca.len() as f64;

            if var >= self.threshold {
                selected.push(name);
            }
        }

        if selected.is_empty() {
            return Err(Error::InvalidInput(format!(
                "VarianceThreshold: no features meet the variance threshold ({}) \
                 out of {} f64 columns. Try lowering the threshold or checking \
                 that your data has variance.",
                self.threshold,
                x.get_column_names().len()
            )));
        }

        self.selected_columns = Some(selected);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for VarianceThreshold {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "VarianceThreshold has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().unwrap();
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("low_var".into(), &[1.0f64, 1.0, 1.0]));
        let b = Column::from(Series::new("high_var".into(), &[1.0f64, 5.0, 9.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_variance_threshold_removes_low_var() {
        let mut vt = VarianceThreshold::new(0.1);
        let df = make_test_df();
        let y = df.clone();

        vt.fit(df.clone(), y).unwrap();
        let result = vt.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "high_var");
    }

    #[test]
    fn test_variance_threshold_zero_keeps_all() {
        let mut vt = VarianceThreshold::new(0.0);
        let df = make_test_df();
        let y = df.clone();

        vt.fit(df.clone(), y).unwrap();
        let result = vt.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }
}
