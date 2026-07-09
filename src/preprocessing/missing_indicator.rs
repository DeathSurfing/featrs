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
/// use featrs::preprocessing::missing_indicator::MissingIndicator;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[Some(1.0_f64), None, Some(3.0)]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut ind = MissingIndicator::new(&["x"]);
/// ind.fit(df.clone())?;
/// let marked = ind.transform(df)?;
/// assert_eq!(marked.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
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

impl Fit<DataFrame> for MissingIndicator {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
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
            let null_mask = s.is_null();
            let indicator_vals: ChunkedArray<Float64Type> = null_mask
                .iter()
                .map(|opt_b| opt_b.map(|b| if b { 1.0 } else { 0.0 }))
                .collect();

            let ind_name = format!("{}_missing", col);
            out.with_column(
                indicator_vals
                    .into_series()
                    .with_name(ind_name.as_str().into())
                    .into(),
            )
            .map_err(|e| Error::Computation(e.to_string()))?;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_indicator_schema_stability() {
        // Fit on data with nulls, transform on data without nulls.
        // Output should always include indicator columns.
        let with_nulls = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(3.0)]));
        let df_nulls = DataFrame::new(3, vec![with_nulls]).unwrap();

        let without_nulls = Column::from(Series::new(
            "x".into(),
            &[Some(4.0f64), Some(5.0), Some(6.0)],
        ));
        let df_no_nulls = DataFrame::new(3, vec![without_nulls]).unwrap();

        let mut ind = MissingIndicator::new(&["x"]);
        ind.fit(df_nulls).unwrap();

        let result = ind.transform(df_no_nulls).unwrap();
        assert_eq!(result.width(), 2); // x, x_missing
        let missing = result.column("x_missing").unwrap().f64().unwrap();
        assert_eq!(missing.get(0).unwrap(), 0.0);
        assert_eq!(missing.get(1).unwrap(), 0.0);
        assert_eq!(missing.get(2).unwrap(), 0.0);
    }

    #[test]
    fn test_missing_indicator() {
        let a = Column::from(Series::new("x".into(), &[Some(1.0f64), None, Some(3.0)]));
        let df = DataFrame::new(3, vec![a]).unwrap();
        let mut ind = MissingIndicator::new(&["x"]);

        ind.fit(df.clone()).unwrap();
        let result = ind.transform(df).unwrap();

        assert_eq!(result.width(), 2); // x, x_missing
        let missing = result.column("x_missing").unwrap().f64().unwrap();
        assert_eq!(missing.get(0).unwrap(), 0.0);
        assert_eq!(missing.get(1).unwrap(), 1.0);
        assert_eq!(missing.get(2).unwrap(), 0.0);
    }
}
