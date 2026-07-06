//! Lag feature generation.
//!
//! [`Lagger`] creates shifted copies of columns for use as features
//! in forecasting models. Analogous to `df.shift()` in pandas.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Create lag features by shifting columns.
///
/// Each specified column gets `periods` additional columns, one for each
/// lag value: `col_name_lag_1`, `col_name_lag_2`, etc.
///
/// # Example
///
/// ```rust
/// use featrs::time_series::lag::Lagger;
/// use featrs::traits::{Fit, Transform};
///
/// let mut lagger = Lagger::new(&["value"], &[1, 3, 7]);
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // lagger.fit(df.clone(), target)?;
/// // let lagged = lagger.transform(df)?;
/// ```
pub struct Lagger {
    fitted: bool,
    columns: Vec<String>,
    periods: Vec<i64>,
}

impl Lagger {
    /// Create a new `Lagger`.
    ///
    /// * `columns` — names of columns to lag
    /// * `periods` — shift amounts (e.g. `[1, 7]` for lag-1 and lag-7)
    pub fn new(columns: &[&str], periods: &[i64]) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            periods: periods.to_vec(),
        }
    }
}

impl Fit<DataFrame, DataFrame> for Lagger {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "Lagger: at least one column name is required.".into(),
            ));
        }
        if self.periods.is_empty() {
            return Err(Error::InvalidInput(
                "Lagger: at least one period is required.".into(),
            ));
        }
        for col in &self.columns {
            if x.column(col.as_str()).is_err() {
                return Err(Error::InvalidInput(format!(
                    "Lagger: column '{}' not found in input. Available columns: {:?}",
                    col,
                    x.get_column_names()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Lagger {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("Lagger has not been fitted.".into()));
        }
        let mut out = x.clone();

        for col in &self.columns {
            let s = out.column(col.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "Lagger.transform: column '{}' not found. {}",
                    col, e
                ))
            })?;
            let s = s.clone();
            for &period in &self.periods {
                let shifted = s.shift(period);
                let lag_name = format!("{}_lag_{}", col, period);
                out.with_column(shifted.with_name(lag_name.as_str().into()))
                    .map_err(|e| Error::Computation(e.to_string()))?;
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagger() {
        let vals = Column::from(Series::new("x".into(), &[1.0f64, 2.0, 3.0, 4.0, 5.0]));
        let df = DataFrame::new(5, vec![vals]).unwrap();
        let mut lagger = Lagger::new(&["x"], &[1, 2]);
        let y = df.clone();

        lagger.fit(df.clone(), y).unwrap();
        let result = lagger.transform(df).unwrap();

        assert_eq!(result.width(), 3); // x, x_lag_1, x_lag_2
        assert_eq!(result.height(), 5);
    }
}
