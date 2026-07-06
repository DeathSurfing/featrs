//! Missing value indicator.
//!
//! [`MissingIndicator`] adds binary columns that mark where values
//! were missing (null / `None`) in the original data. Useful for
//! informing models about missingness patterns.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Add binary indicator columns for missing values.
///
/// For each specified column, creates `{col_name}_missing` with `1.0`
/// where the original value was null and `0.0` otherwise.
///
/// # Example
///
/// ```rust
/// use featrs::traits::missing_indicator::MissingIndicator;
/// use featrs::traits::{Fit, Transform};
///
/// let mut ind = MissingIndicator::new(&["age", "income"]);
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // ind.fit(df.clone(), target)?;
/// // let marked = ind.transform(df)?;
/// ```
pub struct MissingIndicator {
    fitted: bool,
    columns: Vec<String>,
    /// Include missing indicators for all columns (overrides `columns`).
    all_columns: bool,
}

impl MissingIndicator {
    /// Create a new `MissingIndicator` for the specified columns.
    pub fn new(columns: &[&str]) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            all_columns: false,
        }
    }

    /// Mark missing values for all columns in the DataFrame.
    pub fn all() -> Self {
        Self {
            fitted: false,
            columns: vec![],
            all_columns: true,
        }
    }
}

impl Fit<DataFrame, DataFrame> for MissingIndicator {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if self.all_columns {
            self.columns = x.get_column_names().iter().map(|s| s.to_string()).collect();
        }
        if self.columns.is_empty() {
            return Err(Error::InvalidInput(
                "MissingIndicator: no columns to check. Provide column names or use .all().".into(),
            ));
        }
        for col in &self.columns {
            if x.column(col.as_str()).is_err() {
                return Err(Error::InvalidInput(format!(
                    "MissingIndicator: column '{}' not found.",
                    col
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for MissingIndicator {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("MissingIndicator".into()));
        }
        let mut out = x.clone();

        for col in &self.columns {
            let s = out
                .column(col.as_str())
                .map_err(|e| {
                    Error::InvalidInput(format!(
                        "MissingIndicator.transform: column '{}' not found. {}",
                        col, e
                    ))
                })?
                .as_materialized_series()
                .clone();
            let has_missing = s.null_count() > 0;

            let null_mask = s.is_null();
            let indicator_vals: ChunkedArray<Float64Type> = null_mask
                .iter()
                .map(|opt_b| opt_b.map(|b| if b { 1.0 } else { 0.0 }))
                .collect();

            let ind_name = format!("{}_missing", col);
            if has_missing {
                out.with_column(
                    indicator_vals
                        .into_series()
                        .with_name(ind_name.as_str().into())
                        .into(),
                )
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
    fn test_missing_indicator() {
        let a = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(3.0)]));
        let df = DataFrame::new(3, vec![a]).unwrap();
        let mut ind = MissingIndicator::new(&["x"]);
        let y = df.clone();

        ind.fit(df.clone(), y).unwrap();
        let result = ind.transform(df).unwrap();

        assert_eq!(result.width(), 2); // x, x_missing
        let missing = result.column("x_missing").unwrap().f64().unwrap();
        assert_eq!(missing.get(0).unwrap(), 0.0);
        assert_eq!(missing.get(1).unwrap(), 1.0);
        assert_eq!(missing.get(2).unwrap(), 0.0);
    }
}
