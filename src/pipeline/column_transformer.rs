//! Column-wise transformation routing.
//!
//! [`ColumnTransformer`] applies different preprocessing pipelines to
//! different column subsets and combines the results into a single
//! [`DataFrame`].

use crate::pipeline::DataFrameTransformer;
use crate::traits::{Error, Fit, FitLazy, Result, Transform, TransformLazy};
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
/// let _ct = ColumnTransformer::new(
///     vec![("scale".into(), Box::new(StandardScaler::new()), vec!["feat_a".into()])],
///     Remainder::Passthrough,
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
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

impl Fit<DataFrame> for ColumnTransformer {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "ColumnTransformer.fit received a DataFrame with 0 columns.".into(),
            ));
        }
        for (t_name, transformer, columns) in &mut self.transformers {
            let col_names = columns.clone();
            let subset = x.select(&col_names).map_err(|e| {
                Error::InvalidInput(format!(
                    "ColumnTransformer: transformer '{}' requested columns {:?} \
                     but one or more don't exist in the input. Available columns: {:?}. {}",
                    t_name,
                    col_names,
                    x.get_column_names()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                    e
                ))
            })?;
            transformer.fit(subset).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: transformer '{}' failed during fit: {}",
                    t_name, e
                ))
            })?;
        }
        Ok(())
    }
}

impl Transform<DataFrame> for ColumnTransformer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let mut parts: Vec<DataFrame> = Vec::new();

        for (t_name, transformer, columns) in &self.transformers {
            let subset = x.clone().select(columns).map_err(|e| {
                Error::InvalidInput(format!(
                    "ColumnTransformer: transformer '{}' requested columns {:?} \
                     but one or more don't exist in the input. Available columns: {:?}. {}",
                    t_name,
                    columns,
                    x.get_column_names()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                    e
                ))
            })?;
            let transformed = transformer.transform(subset).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: transformer '{}' failed during transform: {}",
                    t_name, e
                ))
            })?;
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
                    let rem = remaining_cols.clone();
                    let remaining = x.select(remaining_cols).map_err(|e| {
                        Error::InvalidInput(format!(
                            "ColumnTransformer: failed to select remainder columns {:?}: {}",
                            rem, e
                        ))
                    })?;
                    parts.push(remaining);
                }
            }
            Remainder::Drop => {}
        }

        if parts.is_empty() {
            return Err(Error::InvalidInput(
                "ColumnTransformer produced no output columns. \
                 Check that at least one transformer has matching input columns \
                 or use Remainder::Passthrough to keep unspecified columns."
                    .into(),
            ));
        }

        let mut result = parts.remove(0);
        for other in &parts {
            let cols = other.columns().to_vec();
            result = result.hstack(&cols).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: failed to stack transformed columns: {}",
                    e
                ))
            })?;
        }

        Ok(result)
    }
}

impl FitLazy for ColumnTransformer {
    fn fit_lazy(&mut self, mut x: LazyFrame) -> Result<()> {
        let schema = x
            .collect_schema()
            .map_err(|e| Error::Computation(e.to_string()))?;
        if schema.is_empty() {
            return Err(Error::InvalidInput(
                "ColumnTransformer.fit received a LazyFrame with 0 columns.".into(),
            ));
        }
        for (t_name, transformer, columns) in &mut self.transformers {
            let exprs: Vec<Expr> = columns.iter().map(|c| col(c.as_str())).collect();
            let subset = x.clone().select(exprs);
            transformer.fit_lazy(subset).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: transformer '{}' failed during lazy fit: {}",
                    t_name, e
                ))
            })?;
        }
        Ok(())
    }
}

impl TransformLazy for ColumnTransformer {
    fn transform_lazy(&self, mut x: LazyFrame) -> Result<LazyFrame> {
        let mut parts: Vec<LazyFrame> = Vec::new();

        for (t_name, transformer, columns) in &self.transformers {
            let exprs: Vec<Expr> = columns.iter().map(|c| col(c.as_str())).collect();
            let subset = x.clone().select(exprs);
            let transformed = transformer.transform_lazy(subset).map_err(|e| {
                Error::Computation(format!(
                    "ColumnTransformer: transformer '{}' failed during lazy transform: {}",
                    t_name, e
                ))
            })?;
            parts.push(transformed);
        }

        let specified = self.all_specified_columns();
        match self.remainder {
            Remainder::Passthrough => {
                let schema = x
                    .collect_schema()
                    .map_err(|e| Error::Computation(e.to_string()))?;
                let remaining_cols: Vec<Expr> = schema
                    .iter()
                    .filter(|(name, _)| !specified.contains(name.as_str()))
                    .map(|(name, _)| col(name.as_str()))
                    .collect();
                if !remaining_cols.is_empty() {
                    let remaining = x.clone().select(remaining_cols);
                    parts.push(remaining);
                }
            }
            Remainder::Drop => {}
        }

        if parts.is_empty() {
            return Err(Error::InvalidInput(
                "ColumnTransformer produced no output columns. \
                 Check that at least one transformer has matching input columns \
                 or use Remainder::Passthrough to keep unspecified columns."
                    .into(),
            ));
        }

        polars::prelude::concat_lf_horizontal(parts, HConcatOptions::default()).map_err(|e| {
            Error::Computation(format!(
                "ColumnTransformer: failed to horizontally concatenate: {}",
                e
            ))
        })
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

        ct.fit(df.clone()).unwrap();
        let result = ct.transform(df).unwrap();
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_column_transformer_drop_remainder() {
        let scaler = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![("scale_a".into(), Box::new(scaler), vec!["a".into()])],
            Remainder::Drop,
        );
        let df = make_test_df();

        ct.fit(df.clone()).unwrap();
        let result = ct.transform(df).unwrap();
        // Only the transformed "a" column is kept; b and c are dropped.
        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_column_transformer_multiple() {
        let scaler_a = StandardScaler::new();
        let scaler_b = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![
                ("scale_a".into(), Box::new(scaler_a), vec!["a".into()]),
                ("scale_b".into(), Box::new(scaler_b), vec!["b".into()]),
            ],
            Remainder::Passthrough,
        );
        let df = make_test_df();

        ct.fit(df.clone()).unwrap();
        let result = ct.transform(df).unwrap();
        // a (scaled) + b (scaled) + c (passthrough) = 3 columns.
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_column_transformer_not_fitted() {
        let scaler = StandardScaler::new();
        let ct = ColumnTransformer::new(
            vec![("scale_a".into(), Box::new(scaler), vec!["a".into()])],
            Remainder::Passthrough,
        );
        let df = make_test_df();
        assert!(ct.transform(df).is_err());
    }

    #[test]
    fn test_column_transformer_lazy() {
        let scaler_a = StandardScaler::new();
        let scaler_b = StandardScaler::new();
        let mut ct = ColumnTransformer::new(
            vec![
                ("scale_a".into(), Box::new(scaler_a), vec!["a".into()]),
                ("scale_b".into(), Box::new(scaler_b), vec!["b".into()]),
            ],
            Remainder::Passthrough,
        );
        let df = make_test_df();

        ct.fit_lazy(df.clone().lazy()).unwrap();
        let eager_out = ct.transform(df.clone()).unwrap();
        let lazy_out = ct
            .transform_lazy(df.clone().lazy())
            .unwrap()
            .collect()
            .unwrap();

        assert_eq!(eager_out, lazy_out);
    }
}
