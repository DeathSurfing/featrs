//! Column-wise transformation routing.
//!
//! [`ColumnTransformer`] applies different preprocessing pipelines to
//! different column subsets and combines the results into a single
//! [`DataFrame`](polars::prelude::DataFrame).

use crate::pipeline::DataFrameTransformer;
use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::HashSet;

/// How to handle columns not specified in any transformer.
pub enum Remainder {
    /// Drop unspecified columns from the output.
    Drop,
    /// Pass unspecified columns through unchanged.
    Passthrough,
}

/// Apply different transformers to different subsets of columns.
///
/// Each transformer receives only its designated columns and produces
/// transformed columns. The results are horizontally stacked.
///
/// # Example
///
/// ```rust
/// use featrs::pipeline::ColumnTransformer;
/// use featrs::pipeline::column_transformer::Remainder;
/// use featrs::preprocessing::scaler::StandardScaler;
///
/// let ct = ColumnTransformer::new(
///     vec![("scale".into(), Box::new(StandardScaler::new()), vec!["feat_a".into()])],
///     Remainder::Passthrough,
/// );
/// ```
pub struct ColumnTransformer {
    transformers: Vec<(String, Box<dyn DataFrameTransformer>, Vec<String>)>,
    remainder: Remainder,
}

impl ColumnTransformer {
    /// Create a new `ColumnTransformer`.
    ///
    /// Each entry in `transformers` is `(name, transformer, columns)`:
    /// - `name`: identifier for debugging
    /// - `transformer`: any [`DataFrameTransformer`]
    /// - `columns`: column names to apply the transformer to
    ///
    /// `remainder` controls how columns not listed in any transformer are handled.
    pub fn new(
        transformers: Vec<(String, Box<dyn DataFrameTransformer>, Vec<String>)>,
        remainder: Remainder,
    ) -> Self {
        Self {
            transformers,
            remainder,
        }
    }

    fn all_specified_columns(&self) -> HashSet<&str> {
        let mut cols = HashSet::new();
        for (_, _, columns) in &self.transformers {
            for c in columns {
                cols.insert(c.as_str());
            }
        }
        cols
    }
}

impl Fit<DataFrame, DataFrame> for ColumnTransformer {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        for (_, transformer, columns) in &mut self.transformers {
            let subset = x
                .select(columns)
                .map_err(|e| Error::InvalidInput(format!("column selection failed: {}", e)))?;
            let y_dummy = subset.clone();
            transformer.fit(subset, y_dummy)?;
        }
        Ok(())
    }
}

impl Transform<DataFrame> for ColumnTransformer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let mut parts: Vec<DataFrame> = Vec::new();

        for (_, transformer, columns) in &self.transformers {
            let subset = x
                .clone()
                .select(columns)
                .map_err(|e| Error::InvalidInput(format!("column selection failed: {}", e)))?;
            let transformed = transformer.transform(subset)?;
            parts.push(transformed);
        }

        let specified = self.all_specified_columns();
        match self.remainder {
            Remainder::Passthrough => {
                let remaining_cols: Vec<&str> = x
                    .get_column_names()
                    .iter()
                    .filter(|name| !specified.contains(name.as_str()))
                    .map(|s| s.as_str())
                    .collect();
                if !remaining_cols.is_empty() {
                    let remaining = x.select(remaining_cols).map_err(|e| {
                        Error::InvalidInput(format!("remainder selection failed: {}", e))
                    })?;
                    parts.push(remaining);
                }
            }
            Remainder::Drop => {}
        }

        if parts.is_empty() {
            return Err(Error::InvalidInput(
                "ColumnTransformer produced no output columns".into(),
            ));
        }

        let mut result = parts.remove(0);
        for other in &parts {
            let cols = other.columns().to_vec();
            result = result
                .hstack(&cols)
                .map_err(|e| Error::Computation(format!("failed to concatenate columns: {}", e)))?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::scaler::StandardScaler;
    use crate::traits::Transform;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        let c = Column::from(Series::new("c".into(), &[10.0f64, 20.0, 30.0]));
        DataFrame::new(3, vec![a, b, c]).unwrap()
    }

    #[test]
    fn test_column_transformer_selective() {
        let scaler = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![("scale_a".into(), Box::new(scaler), vec!["a".into()])],
            Remainder::Passthrough,
        );
        let df = make_test_df();
        let y = df.clone();

        ct.fit(df.clone(), y).unwrap();
        let result = ct.transform(df).unwrap();
        assert_eq!(result.width(), 3);
    }
}
