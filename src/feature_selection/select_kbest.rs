//! Select top-k features using statistical tests.
//!
//! Provides [`SelectKBest`] and the [`FClassif`] scoring function
//! (ANOVA F-value between each feature and the target).

use crate::traits::{Error, FitSupervised, Result, Transform};
use polars::prelude::*;

/// Scoring function for [`SelectKBest`].
///
/// Implementors compute a score for each feature column indicating
/// how relevant it is for predicting the target. Higher scores are better.
pub trait ScoreFunction: Send + Sync {
    /// Score each feature in `x` against the target `y`.
    ///
    /// Returns a list of `(column_name, score)` pairs for numeric columns.
    fn score(&self, x: &DataFrame, y: &Column) -> Result<Vec<(String, f64)>>;
}

/// ANOVA F-value scoring function.
///
/// Computes the F-statistic between each feature and the target labels:
///
/// ```text
/// F = (SS_between / df_between) / (SS_within / df_within)
/// ```
///
/// Where `SS_between` is the between-group sum of squares and `SS_within`
/// is the within-group sum of squares. Higher F-values indicate stronger
/// class separation.
///
/// Requires the target column to be [`Float64`](DataType::Float64).
pub struct FClassif;

impl FClassif {
    /// Create a new `FClassif` scorer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FClassif {
    fn default() -> Self {
        Self::new()
    }
}

impl ScoreFunction for FClassif {
    fn score(&self, x: &DataFrame, y: &Column) -> Result<Vec<(String, f64)>> {
        let y_ca = y.as_materialized_series().f64().map_err(|_| {
            Error::InvalidInput(format!(
                "FClassif: target column '{}' has dtype {}; expected Float64. \
                     The target must be numeric (0, 1, 2, ...) for ANOVA F-test.",
                y.name(),
                y.dtype()
            ))
        })?;
        let y_vals: Vec<f64> = y_ca.iter().flatten().collect();
        let n = y_vals.len() as f64;
        let y_mean = y_vals.iter().sum::<f64>() / n;

        let mut classes: Vec<f64> = y_ca.iter().flatten().collect();
        classes.sort_by(|a, b| a.total_cmp(b));
        classes.dedup();

        if classes.len() < 2 {
            return Err(Error::InvalidInput(format!(
                "FClassif: target has only {} unique class(es); need at least 2 \
                 to compute ANOVA F-statistic.",
                classes.len()
            )));
        }

        if n != x.height() as f64 {
            return Err(Error::InvalidInput(format!(
                "FClassif: feature rows ({}) and target rows ({}) don't match.",
                x.height(),
                n
            )));
        }

        let mut scores = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            if col.dtype() != &DataType::Float64 {
                continue;
            }
            let ca = col.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "FClassif: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    col.dtype(),
                    e
                ))
            })?;
            let vals: Vec<Option<f64>> = ca.iter().collect();

            let mut ss_between = 0.0;
            let mut ss_within = 0.0;

            for &cls in &classes {
                let group_vals: Vec<f64> = vals
                    .iter()
                    .zip(&y_vals)
                    .filter_map(|(xv, &yv)| if (yv - cls).abs() < 1e-10 { *xv } else { None })
                    .collect();

                if group_vals.is_empty() {
                    continue;
                }
                let g_mean = group_vals.iter().sum::<f64>() / group_vals.len() as f64;
                let g_n = group_vals.len() as f64;

                ss_between += g_n * (g_mean - y_mean).powi(2);

                for &v in &group_vals {
                    ss_within += (v - g_mean).powi(2);
                }
            }

            let n_classes = classes.len() as f64;
            let df_between = n_classes - 1.0;
            let df_within = n - n_classes;

            let f_stat = if ss_within > 1e-15 && df_within > 0.0 {
                (ss_between / df_between) / (ss_within / df_within)
            } else {
                0.0
            };

            scores.push((name, f_stat));
        }

        Ok(scores)
    }
}

/// Select the top `k` features according to a [`ScoreFunction`].
///
/// `SelectKBest` is supervised: it implements [`FitSupervised`] and requires a
/// target `y` at `fit` time. Only `Float64` feature columns are scored; columns
/// of other dtypes are silently skipped. The target `y` must be a single
/// `Float64` column with at least two distinct classes.
///
/// # Example
///
/// ```rust
/// use featrs::feature_selection::SelectKBest;
/// use featrs::feature_selection::select_kbest::FClassif;
/// use featrs::traits::{FitSupervised, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let a = Column::from(Series::new("noise".into(), &[1.0_f64, 2.0, 3.0, 4.0]));
/// let b = Column::from(Series::new("signal".into(), &[0.0_f64, 1.0, 2.0, 10.0]));
/// let features = DataFrame::new(4, vec![a, b])?;
///
/// let target = Column::from(Series::new("y".into(), &[0.0_f64, 0.0, 1.0, 1.0]));
/// let y = DataFrame::new(4, vec![target])?;
///
/// let mut skb = SelectKBest::new(1, Box::new(FClassif::new()));
/// skb.fit(features.clone(), y.clone())?;
/// let selected = skb.transform(features)?;
/// assert_eq!(selected.width(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SelectKBest {
    fitted: bool,
    k: usize,
    score_fn: Box<dyn ScoreFunction>,
    selected_columns: Option<Vec<String>>,
    scores: Option<Vec<(String, f64)>>,
}

impl SelectKBest {
    /// Create a new `SelectKBest` transformer.
    ///
    /// * `k` — number of top features to keep
    /// * `score_fn` — scoring function (e.g. [`FClassif`])
    pub fn new(k: usize, score_fn: Box<dyn ScoreFunction>) -> Self {
        Self {
            fitted: false,
            k,
            score_fn,
            selected_columns: None,
            scores: None,
        }
    }

    /// Returns the scores for each feature from the last `fit`.
    ///
    /// Returns `None` if not fitted yet. The list is sorted highest-score first.
    pub fn scores(&self) -> Option<&[(String, f64)]> {
        self.scores.as_deref()
    }
}

impl FitSupervised<DataFrame, DataFrame> for SelectKBest {
    type Output = ();

    fn fit(&mut self, x: DataFrame, y: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "SelectKBest.fit received a DataFrame with 0 columns.".into(),
            ));
        }
        if self.k == 0 {
            return Err(Error::InvalidInput(
                "SelectKBest: k must be greater than 0, got 0. \n\
                 Choose k >= 1 to select at least one feature."
                    .into(),
            ));
        }
        if y.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "SelectKBest.fit: target must have exactly 1 column but got {} columns. \
                 Select a single target column.",
                y.width()
            )));
        }
        let y_col = &y.columns()[0];
        let mut scores = self.score_fn.score(&x, y_col)?;

        if scores.is_empty() {
            return Err(Error::InvalidInput(
                "SelectKBest: no f64 columns found to score. \
                 SelectKBest operates on Float64 columns only."
                    .into(),
            ));
        }

        scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let k = self.k.min(scores.len());
        let selected: Vec<String> = scores.iter().take(k).map(|(n, _)| n.clone()).collect();

        self.scores = Some(scores);
        self.selected_columns = Some(selected);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for SelectKBest {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "SelectKBest has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "SelectKBest has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            )
        })?;
        if cols.is_empty() {
            // Should not happen if fit succeeded, but handle gracefully
            return Err(Error::Computation(
                "SelectKBest: no columns were selected. \
                 This may mean the scoring function returned no valid scores."
                    .into(),
            ));
        }
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features() -> DataFrame {
        let a = Column::from(Series::new(
            "noise".into(),
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let b = Column::from(Series::new(
            "signal".into(),
            &[0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0],
        ));
        DataFrame::new(6, vec![a, b]).unwrap()
    }

    fn make_target_col() -> Column {
        Column::from(Series::new(
            "target".into(),
            &[0.0f64, 0.0, 0.0, 1.0, 1.0, 1.0],
        ))
    }

    #[test]
    fn test_select_kbest_f_classif() {
        let mut skb = SelectKBest::new(1, Box::new(FClassif::new()));
        let features = make_features();
        let y = DataFrame::new(6, vec![make_target_col()]).unwrap();

        skb.fit(features.clone(), y).unwrap();
        let result = skb.transform(features).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "signal");
    }

    #[test]
    fn test_f_classif_scores() {
        let f = FClassif::new();
        let features = make_features();
        let y_col = make_target_col();

        let scores = f.score(&features, &y_col).unwrap();
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_select_kbest_k_zero_rejected() {
        let mut skb = SelectKBest::new(0, Box::new(FClassif::new()));
        let features = make_features();
        let y = DataFrame::new(6, vec![make_target_col()]).unwrap();
        let result = skb.fit(features, y);
        assert!(result.is_err(), "k=0 should be rejected at fit time");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("k must be greater than 0"),
            "error message should mention k"
        );
    }
}
