//! Automatic feature type detection.
//!
//! [`AutoTypeDetector`] infers the semantic type of each column and
//! applies appropriate transformations (e.g., one-hot for low-cardinality
//! strings, pass-through for floats).

use crate::pipeline::DataFrameTransformer;
use crate::preprocessing::encoder::OneHotEncoder;
use crate::preprocessing::feature_hasher::FeatureHasher;
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// How to treat each detected column type.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnType {
    /// Pass through unchanged.
    Numeric,
    /// One-hot encode (low-cardinality strings).
    Categorical,
    /// Feature hashing (high-cardinality strings).
    HighCardinality,
}

/// Auto-detect column types and apply default transformations.
///
/// Detection rules:
/// - Float64 → `Numeric` (passthrough)
/// - String with < 20 unique values → `Categorical` (one-hot)
/// - String with ≥ 20 unique values → `HighCardinality` (hash to 100 buckets)
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::auto_type::AutoTypeDetector;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let num = Column::from(Series::new("num".into(), &[1.0_f64, 2.0, 3.0]));
/// let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
/// let df = DataFrame::new(3, vec![num, cat])?;
///
/// let mut atd = AutoTypeDetector::new();
/// atd.fit(df.clone())?;
/// let typed = atd.transform(df)?;
/// assert_eq!(typed.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AutoTypeDetector {
    fitted: bool,
    cat_threshold: usize,
    hash_buckets: usize,
    column_types: Option<Vec<(String, ColumnType)>>,
    /// Fitted sub-transformers for the non-numeric columns, in frame order.
    /// Learned once during `fit` and reused on every `transform` call so the
    /// learned categories / hash mapping do not drift with the transform data.
    encoders: Option<Vec<(String, Box<dyn DataFrameTransformer>)>>,
}

impl AutoTypeDetector {
    /// Create a new detector with defaults: `cat_threshold = 20`, `hash_buckets = 100`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            cat_threshold: 20,
            hash_buckets: 100,
            column_types: None,
            encoders: None,
        }
    }

    /// Set the maximum unique values for a string column to be treated
    /// as categorical (one-hot). Default: 20.
    pub fn cat_threshold(mut self, value: usize) -> Self {
        self.cat_threshold = value;
        self
    }

    /// Set the number of hash buckets for high-cardinality columns. Default: 100.
    pub fn hash_buckets(mut self, value: usize) -> Self {
        self.hash_buckets = value;
        self
    }

    /// Return the inferred column types.
    pub fn column_types(&self) -> Option<&[(String, ColumnType)]> {
        self.column_types.as_deref()
    }
}

impl Default for AutoTypeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame> for AutoTypeDetector {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        let mut types = Vec::new();
        let mut encoders: Vec<(String, Box<dyn DataFrameTransformer>)> = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            let dtype = col.dtype();

            let detected = match dtype {
                dt if dt == &DataType::Float64
                    || dt == &DataType::Int64
                    || dt == &DataType::Int32 =>
                {
                    ColumnType::Numeric
                }
                dt if dt == &DataType::String => {
                    let ca = col.as_materialized_series().str().map_err(|_| {
                        Error::Computation(format!("could not read string column '{}'", name))
                    })?;
                    let n_unique = ca
                        .iter()
                        .flatten()
                        .collect::<std::collections::HashSet<_>>()
                        .len();
                    if n_unique < self.cat_threshold {
                        ColumnType::Categorical
                    } else {
                        ColumnType::HighCardinality
                    }
                }
                _ => ColumnType::Numeric,
            };

            // For non-numeric columns, fit the corresponding sub-transformer on
            // the fit data now, so transform() never has to re-fit.
            match detected {
                ColumnType::Categorical => {
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    let mut enc = OneHotEncoder::new();
                    enc.fit(subset.clone()).map_err(|e| {
                        Error::Computation(format!(
                            "AutoType.fit: one-hot failed on '{}': {}",
                            name, e
                        ))
                    })?;
                    encoders.push((name.clone(), Box::new(enc)));
                }
                ColumnType::HighCardinality => {
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    let mut fh = FeatureHasher::new(&[name.as_str()], self.hash_buckets);
                    fh.fit(subset.clone()).map_err(|e| {
                        Error::Computation(format!(
                            "AutoType.fit: hashing failed on '{}': {}",
                            name, e
                        ))
                    })?;
                    encoders.push((name.clone(), Box::new(fh)));
                }
                ColumnType::Numeric => {}
            }

            types.push((name, detected));
        }

        self.column_types = Some(types);
        self.encoders = Some(encoders);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for AutoTypeDetector {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("AutoTypeDetector".into()));
        }
        let types = self
            .column_types
            .as_ref()
            .ok_or_else(|| Error::NotFitted("AutoTypeDetector has not been fitted.".into()))?;
        let encoders = self
            .encoders
            .as_ref()
            .ok_or_else(|| Error::NotFitted("AutoTypeDetector has not been fitted.".into()))?;
        let mut parts: Vec<DataFrame> = Vec::new();
        let mut numeric_cols: Vec<Column> = Vec::new();

        for (name, ctype) in types {
            match ctype {
                ColumnType::Numeric => {
                    if let Ok(col) = x.column(name.as_str()) {
                        numeric_cols.push(col.clone());
                    }
                }
                ColumnType::Categorical | ColumnType::HighCardinality => {
                    // Sub-transformers were fitted during `fit`; reuse them so
                    // learned categories / hash mapping are stable across calls.
                    let enc = encoders
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, e)| e)
                        .ok_or_else(|| {
                            Error::Computation(format!(
                                "AutoTypeDetector: no fitted encoder for column '{}'",
                                name
                            ))
                        })?;
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    let out = enc.transform(subset).map_err(|e| {
                        Error::Computation(format!(
                            "AutoTypeDetector.transform: '{}' failed: {}",
                            name, e
                        ))
                    })?;
                    if !out.columns().is_empty() {
                        parts.push(out);
                    }
                }
            }
        }

        if !numeric_cols.is_empty() {
            let h = x.height();
            parts.push(
                DataFrame::new(h, numeric_cols).map_err(|e| Error::Computation(e.to_string()))?,
            );
        }

        if parts.is_empty() {
            return Err(Error::Computation(
                "AutoTypeDetector produced no output columns.".into(),
            ));
        }

        let mut result = parts.remove(0);
        for other in &parts {
            let cols = other.columns().to_vec();
            result = result
                .hstack(&cols)
                .map_err(|e| Error::Computation(e.to_string()))?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_df() -> DataFrame {
        let num = Column::from(Series::new("num".into(), &[1.0f64, 2.0, 3.0]));
        let cat = Column::from(Series::new("cat".into(), &["a", "b", "a"]));
        let high = Column::from(Series::new("high".into(), &["x1", "x2", "x3"]));
        DataFrame::new(3, vec![num, cat, high]).unwrap()
    }

    #[test]
    fn test_auto_type_detect_and_transform() {
        let mut atd = AutoTypeDetector::new().cat_threshold(5).hash_buckets(8);
        let df = make_test_df();
        atd.fit(df.clone()).unwrap();

        // `cat` has 2 uniques (< threshold 5) -> Categorical; `high` has 3
        // uniques but threshold is 5 so also Categorical here. Bump threshold
        // down to push `high` into HighCardinality.
        let types: std::collections::HashMap<&str, ColumnType> = atd
            .column_types()
            .unwrap()
            .iter()
            .map(|(n, t)| (n.as_str(), t.clone()))
            .collect();
        assert_eq!(types.get("num"), Some(&ColumnType::Numeric));
        assert_eq!(types.get("cat"), Some(&ColumnType::Categorical));

        let out = atd.transform(df).unwrap();
        assert!(out.width() >= 1);
        assert_eq!(out.height(), 3);
    }

    /// Regression: `transform` used to re-fit its sub-transformers on every
    /// call, so learned categories could drift. Verify two consecutive
    /// transforms on the same data produce identical output schemas.
    #[test]
    fn test_auto_type_transform_is_idempotent() {
        let mut atd = AutoTypeDetector::new().cat_threshold(5).hash_buckets(8);
        let df = make_test_df();
        atd.fit(df.clone()).unwrap();

        let out1 = atd.transform(df.clone()).unwrap();
        let out2 = atd.transform(df).unwrap();

        let names1: Vec<String> = out1
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let names2: Vec<String> = out2
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            names1, names2,
            "transform schema must be stable across calls"
        );
        assert_eq!(out1.height(), out2.height());
    }

    #[test]
    fn test_auto_type_not_fitted() {
        let atd = AutoTypeDetector::new();
        let df = make_test_df();
        assert!(atd.transform(df).is_err());
    }
}
