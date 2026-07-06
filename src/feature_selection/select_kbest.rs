use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Scoring function for SelectKBest.
pub trait ScoreFunction: Send + Sync {
    fn score(&self, x: &DataFrame, y: &Column) -> Result<Vec<(String, f64)>>;
}

/// ANOVA F-value between each feature and the target.
///
/// For each f64 column in X, computes F = (between-group variance) / (within-group variance).
pub struct FClassif;

impl FClassif {
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
        let y_ca = y
            .as_materialized_series()
            .f64()
            .map_err(|_| Error::InvalidInput("target must be f64".into()))?;
        let y_vals: Vec<f64> = y_ca.iter().flatten().collect();
        let n = y_vals.len() as f64;
        let y_mean = y_vals.iter().sum::<f64>() / n;

        // Determine unique classes
        let mut classes: Vec<f64> = y_ca.iter().flatten().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();

        let mut scores = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            if col.dtype() != &DataType::Float64 {
                continue;
            }
            let ca = col.f64().unwrap();
            let vals: Vec<Option<f64>> = ca.iter().collect();

            // Between-group sum of squares
            let mut ss_between = 0.0;
            // Within-group sum of squares
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

/// Select top k features according to a scoring function.
///
/// Corresponds to `sklearn.feature_selection.SelectKBest`.
pub struct SelectKBest {
    fitted: bool,
    k: usize,
    score_fn: Box<dyn ScoreFunction>,
    selected_columns: Option<Vec<String>>,
    scores: Option<Vec<(String, f64)>>,
}

impl SelectKBest {
    pub fn new(k: usize, score_fn: Box<dyn ScoreFunction>) -> Self {
        Self {
            fitted: false,
            k,
            score_fn,
            selected_columns: None,
            scores: None,
        }
    }

    pub fn scores(&self) -> Option<&[(String, f64)]> {
        self.scores.as_deref()
    }
}

impl Fit<DataFrame, DataFrame> for SelectKBest {
    type Output = ();

    fn fit(&mut self, x: DataFrame, y: DataFrame) -> Result<()> {
        if y.width() != 1 {
            return Err(Error::InvalidInput(
                "target must be a single-column DataFrame".into(),
            ));
        }
        let y_col = &y.columns()[0];
        let mut scores = self.score_fn.score(&x, y_col)?;

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
            return Err(Error::NotFitted("SelectKBest".into()));
        }
        let cols = self.selected_columns.as_ref().unwrap();
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features() -> DataFrame {
        // 6 rows, 2 classes (0 and 1), 3 samples per class
        // noise has weak class separation, signal has strong separation
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
}
