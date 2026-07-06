//! Automatic feature type detection.
//!
//! [`AutoTypeDetector`] infers the semantic type of each column and
//! applies appropriate transformations (e.g., one-hot for low-cardinality
//! strings, pass-through for floats).

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
///
/// let mut atd = AutoTypeDetector::new();
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // atd.fit(df.clone(), target)?;
/// // let typed = atd.transform(df)?;
/// ```
pub struct AutoTypeDetector {
    fitted: bool,
    cat_threshold: usize,
    hash_buckets: usize,
    column_types: Option<Vec<(String, ColumnType)>>,
}

impl AutoTypeDetector {
    pub fn new() -> Self {
        Self {
            fitted: false,
            cat_threshold: 20,
            hash_buckets: 100,
            column_types: None,
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

impl Fit<DataFrame, DataFrame> for AutoTypeDetector {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        let mut types = Vec::new();

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

            types.push((name, detected));
        }

        self.column_types = Some(types);
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
        let types = self.column_types.as_ref().unwrap();
        let mut parts: Vec<DataFrame> = Vec::new();
        let mut numeric_cols: Vec<Column> = Vec::new();

        for (name, ctype) in types {
            match ctype {
                ColumnType::Numeric => {
                    if let Ok(col) = x.column(name.as_str()) {
                        numeric_cols.push(col.clone());
                    }
                }
                ColumnType::Categorical => {
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    use crate::preprocessing::encoder::OneHotEncoder;
                    let mut enc = OneHotEncoder::new();
                    enc.fit(subset.clone(), subset.clone()).map_err(|e| {
                        Error::Computation(format!("AutoType: one-hot failed on '{}': {}", name, e))
                    })?;
                    let encoded = enc.transform(subset).map_err(|e| {
                        Error::Computation(format!(
                            "AutoType: one-hot transform failed on '{}': {}",
                            name, e
                        ))
                    })?;
                    if !encoded.columns().is_empty() {
                        parts.push(encoded);
                    }
                }
                ColumnType::HighCardinality => {
                    let subset = x
                        .clone()
                        .select([name.as_str()])
                        .map_err(|e| Error::Computation(e.to_string()))?;
                    use crate::preprocessing::feature_hasher::FeatureHasher;
                    let mut fh = FeatureHasher::new(&[name.as_str()], self.hash_buckets);
                    fh.fit(subset.clone(), subset.clone()).map_err(|e| {
                        Error::Computation(format!("AutoType: hashing failed on '{}': {}", name, e))
                    })?;
                    let hashed = fh.transform(subset).map_err(|e| {
                        Error::Computation(format!(
                            "AutoType: hashing transform failed on '{}': {}",
                            name, e
                        ))
                    })?;
                    if !hashed.columns().is_empty() {
                        parts.push(hashed);
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
