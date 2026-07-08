//! Feature binarization.
//!
//! [`Binarizer`] thresholds numeric features: values above the threshold
//! become `1.0`, others become `0.0`.

use crate::traits::{Error, Fit, Result, Transform};
use crate::util::replace_f64_column;
use polars::prelude::*;

/// Binarize data according to a threshold.
///
/// Values `> threshold` become `1.0`; all others become `0.0`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::binarizer::Binarizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[-1.0_f64, 0.5, 2.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut b = Binarizer::new(0.5);
/// b.fit(df.clone())?;
/// let binarized = b.transform(df)?;
/// assert_eq!(binarized.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Binarizer {
    fitted: bool,
    threshold: f64,
}

impl Binarizer {
    /// Create a new `Binarizer` with the given threshold.
    ///
    /// Values strictly greater than `threshold` are set to `1.0`.
    pub fn new(threshold: f64) -> Self {
        Self {
            fitted: false,
            threshold,
        }
    }
}

impl Default for Binarizer {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Fit<DataFrame> for Binarizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "Binarizer.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Binarizer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Binarizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let col_names: Vec<String> = x
            .get_column_names()
            .iter()
            .filter_map(|name| {
                let col = x.column(name).ok()?;
                if col.dtype() == &DataType::Float64 {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .collect();

        let mut out = x.clone();
        let threshold = self.threshold;

        for name in &col_names {
            replace_f64_column(&mut out, name.as_str(), "Binarizer", |v| {
                if v > threshold { 1.0 } else { 0.0 }
            })?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_binarizer_default() {
        let mut b = Binarizer::default();
        let a = Column::from(Series::new("x".into(), &[-1.0f64, 0.0, 2.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let mut b = Binarizer::new(5.0);
        let a = Column::from(Series::new("x".into(), &[1.0f64, 5.0, 10.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }
}
