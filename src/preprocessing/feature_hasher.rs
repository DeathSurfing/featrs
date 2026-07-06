//! Feature hashing (the hashing trick).
//!
//! [`FeatureHasher`] maps categorical features into a fixed-size vector
//! using a hash function, enabling memory-efficient encoding of
//! high-cardinality categories.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Map categorical columns to a fixed number of hash buckets.
///
/// Each string cell is hashed to `0..n_features` and the corresponding
/// bucket is incremented by 1.0. This avoids storing a category mapping
/// and works with unseen categories at transform time.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::feature_hasher::FeatureHasher;
/// use featrs::traits::{Fit, Transform};
///
/// let mut fh = FeatureHasher::new(&["text", "category"], 100);
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // fh.fit(df.clone(), target)?;
/// // let hashed = fh.transform(df)?;
/// ```
pub struct FeatureHasher {
    fitted: bool,
    columns: Vec<String>,
    n_features: usize,
}

impl FeatureHasher {
    /// Create a new `FeatureHasher`.
    ///
    /// * `columns` — names of string columns to hash
    /// * `n_features` — number of hash buckets (output dimension)
    pub fn new(columns: &[&str], n_features: usize) -> Self {
        Self {
            fitted: false,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            n_features,
        }
    }

    fn hash_to_index(s: &str, n: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        (hasher.finish() as usize) % n
    }
}

impl Fit<DataFrame, DataFrame> for FeatureHasher {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if self.n_features == 0 {
            return Err(Error::InvalidInput(
                "FeatureHasher: n_features must be >= 1.".into(),
            ));
        }
        for col in &self.columns {
            if x.column(col.as_str()).is_err() {
                return Err(Error::InvalidInput(format!(
                    "FeatureHasher: column '{}' not found.",
                    col
                )));
            }
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for FeatureHasher {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("FeatureHasher".into()));
        }
        let n_rows = x.height();
        let mut buckets = vec![vec![0.0f64; n_rows]; self.n_features];

        for col in &self.columns {
            let s = x.column(col.as_str()).unwrap().as_materialized_series();
            let ca = s.str().map_err(|_| {
                Error::InvalidInput(format!(
                    "FeatureHasher: column '{}' is not a string column.",
                    col
                ))
            })?;
            for (i, opt) in ca.iter().enumerate() {
                if let Some(val) = opt {
                    let idx = Self::hash_to_index(val, self.n_features);
                    buckets[idx][i] += 1.0;
                }
            }
        }

        let mut out_cols: Vec<Column> = Vec::with_capacity(self.n_features);
        for (idx, bucket) in buckets.iter().enumerate() {
            let name = format!("hashed_{}", idx);
            out_cols.push(Column::from(Series::new(name.as_str().into(), bucket)));
        }

        DataFrame::new(n_rows, out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_hasher() {
        let c = Column::from(Series::new("color".into(), &["red", "blue", "red"]));
        let df = DataFrame::new(3, vec![c]).unwrap();
        let mut fh = FeatureHasher::new(&["color"], 10);
        let y = df.clone();

        fh.fit(df.clone(), y).unwrap();
        let result = fh.transform(df).unwrap();

        assert_eq!(result.width(), 10);
        assert_eq!(result.height(), 3);
    }
}
