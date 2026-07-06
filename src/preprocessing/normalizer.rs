//! Row-wise normalization.
//!
//! [`Normalizer`] scales each row (sample) to unit norm independently.
//! Supports L1, L2, and Max norms. Analogous to `sklearn.preprocessing.Normalizer`.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Normalize samples individually to unit norm.
///
/// Each row is divided by its norm so that the row has unit length.
/// Note: this is a **row-wise** operation, not column-wise like the scalers.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::normalizer::Normalizer;
/// use featrs::preprocessing::normalizer::Norm;
/// use featrs::traits::{Fit, Transform};
///
/// let mut n = Normalizer::l2();
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // n.fit(df.clone(), target)?;
/// // let normalized = n.transform(df)?;
/// ```
pub struct Normalizer {
    fitted: bool,
    norm: Norm,
}

/// The norm to use for normalization.
#[derive(Clone, Copy)]
pub enum Norm {
    /// L1 norm — sum of absolute values.
    L1,
    /// L2 norm — Euclidean (sqrt of sum of squares).
    L2,
    /// Max norm — maximum absolute value.
    Max,
}

impl Normalizer {
    /// Create a new `Normalizer` with the given norm.
    pub fn new(norm: Norm) -> Self {
        Self {
            fitted: false,
            norm,
        }
    }

    /// Normalize using the L1 norm (sum of absolute values).
    pub fn l1() -> Self {
        Self::new(Norm::L1)
    }

    /// Normalize using the L2 norm (Euclidean).
    pub fn l2() -> Self {
        Self::new(Norm::L2)
    }

    /// Normalize using the Max norm (maximum absolute value).
    pub fn max() -> Self {
        Self::new(Norm::Max)
    }

    fn row_norm(values: &[f64], norm: Norm) -> f64 {
        match norm {
            Norm::L1 => values.iter().map(|v| v.abs()).sum(),
            Norm::L2 => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
            Norm::Max => values.iter().cloned().fold(0.0f64, f64::max),
        }
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::l2()
    }
}

impl Fit<DataFrame, DataFrame> for Normalizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "Normalizer.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        for col in x.columns() {
            if col.dtype() != &DataType::Float64 {
                return Err(Error::InvalidInput(format!(
                    "Normalizer.fit: column '{}' has dtype {}; expected Float64. \
                     Normalizer operates on f64 columns only.",
                    col.name(),
                    col.dtype()
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Normalizer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Normalizer has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }

        let n_cols = x.width();
        let n_rows = x.height();
        let col_names: Vec<&str> = x.get_column_names().iter().map(|s| s.as_str()).collect();

        let mut col_data: Vec<Vec<f64>> = Vec::with_capacity(n_cols);
        for name in &col_names {
            let s = x.column(name).map_err(|e| {
                Error::InvalidInput(format!(
                    "Normalizer.transform: column '{}' not found. {}",
                    name, e
                ))
            })?;
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "Normalizer.transform: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;
            col_data.push(ca.iter().flatten().collect());
        }

        for i in 0..n_rows {
            let row_vals: Vec<f64> = col_data.iter().map(|col| col[i]).collect();
            let norm = Self::row_norm(&row_vals, self.norm);
            if norm > f64::EPSILON {
                for col in &mut col_data {
                    col[i] /= norm;
                }
            }
        }

        let mut out_cols: Vec<Column> = Vec::with_capacity(n_cols);
        for (j, name) in col_names.iter().enumerate() {
            let new_ca: ChunkedArray<Float64Type> =
                ChunkedArray::from_slice(name.to_string().as_str().into(), &col_data[j]);
            out_cols.push(new_ca.into_series().into());
        }

        DataFrame::new(n_rows, out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[3.0f64, 0.0, 1.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 0.0, 2.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_l2_normalization() {
        let mut n = Normalizer::l2();
        let df = make_test_df();
        let y = df.clone();

        n.fit(df.clone(), y).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();
        assert_relative_eq!(col_a.get(0).unwrap(), 0.6, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 0.8, epsilon = 1e-6);
        assert_relative_eq!(col_a.get(1).unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_l1_normalization() {
        let mut n = Normalizer::l1();
        let df = make_test_df();
        let y = df.clone();

        n.fit(df.clone(), y).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();
        assert_relative_eq!(col_a.get(0).unwrap(), 3.0 / 7.0, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 4.0 / 7.0, epsilon = 1e-6);
    }

    #[test]
    fn test_max_normalization() {
        let mut n = Normalizer::max();
        let df = make_test_df();
        let y = df.clone();

        n.fit(df.clone(), y).unwrap();
        let result = n.transform(df).unwrap();

        let col_a = result.column("a").unwrap().f64().unwrap();
        let col_b = result.column("b").unwrap().f64().unwrap();
        assert_relative_eq!(col_a.get(0).unwrap(), 0.75, epsilon = 1e-6);
        assert_relative_eq!(col_b.get(0).unwrap(), 1.0, epsilon = 1e-6);
    }
}
