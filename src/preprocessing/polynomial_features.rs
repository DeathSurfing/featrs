//! Polynomial feature generation.
//!
//! [`PolynomialFeatures`] generates all polynomial combinations of the
//! input features up to a specified degree. Analogous to
//! `sklearn.preprocessing.PolynomialFeatures`.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

fn series_pow(s: &Series, exp: usize) -> Series {
    let ca = s.f64().expect("expected f64 series");
    let exp_f64 = exp as f64;
    let result: ChunkedArray<Float64Type> =
        ca.iter().map(|opt| opt.map(|v| v.powf(exp_f64))).collect();
    result.into_series()
}

fn series_mul(a: &Series, b: &Series) -> Series {
    let ca_a = a.f64().expect("expected f64 series");
    let ca_b = b.f64().expect("expected f64 series");
    let result: ChunkedArray<Float64Type> = ca_a
        .iter()
        .zip(ca_b.iter())
        .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
            (Some(va), Some(vb)) => Some(va * vb),
            _ => None,
        })
        .collect();
    result.into_series()
}

/// Generate polynomial and interaction features.
///
/// Creates new features that are polynomial combinations of the original
/// features. For example, with `degree=2` and inputs `[a, b]`, the output
/// includes `[a, b, a², a·b, b²]` plus an optional bias term.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::polynomial_features::PolynomialFeatures;
/// use featrs::traits::{Fit, Transform};
///
/// let mut pf = PolynomialFeatures::new(2)
///     .include_bias(true)
///     .interaction_only(false);
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // pf.fit(df.clone(), target)?;
/// // let result = pf.transform(df)?;
/// ```
pub struct PolynomialFeatures {
    fitted: bool,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
    input_columns: Option<Vec<String>>,
}

impl PolynomialFeatures {
    /// Create a new `PolynomialFeatures` with the given maximum degree.
    ///
    /// # Panics
    ///
    /// Panics if `degree` is `0`.
    pub fn new(degree: usize) -> Self {
        assert!(
            degree >= 1,
            "PolynomialFeatures::new: degree must be >= 1, got {degree}"
        );
        Self {
            fitted: false,
            degree,
            interaction_only: false,
            include_bias: true,
            input_columns: None,
        }
    }

    /// Whether to include only interaction features (`a·b`, `a·b·c`, ...)
    /// and exclude pure powers (`a²`, `a³`, ...). Default: `false`.
    pub fn interaction_only(mut self, value: bool) -> Self {
        self.interaction_only = value;
        self
    }

    /// Whether to include a bias column (all ones). Default: `true`.
    pub fn include_bias(mut self, value: bool) -> Self {
        self.include_bias = value;
        self
    }

    fn numeric_f64_column_names(&self, df: &DataFrame) -> Vec<String> {
        df.get_column_names()
            .iter()
            .filter_map(|name| {
                if let Ok(s) = df.column(name) {
                    if s.dtype() == &DataType::Float64 {
                        Some(name.to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    fn generate_powers(
        n_features: usize,
        degree: usize,
        interaction_only: bool,
    ) -> Vec<Vec<usize>> {
        let mut result = Vec::new();

        if interaction_only {
            let max_mask = 1usize << n_features;
            for mask in 0..max_mask {
                let sum_bits = mask.count_ones() as usize;
                if sum_bits >= 2 && sum_bits <= degree {
                    let mut powers = vec![0usize; n_features];
                    for (j, cell) in powers.iter_mut().enumerate() {
                        if mask & (1 << j) != 0 {
                            *cell = 1;
                        }
                    }
                    result.push(powers);
                }
            }
            result.sort_by_key(|p| p.iter().sum::<usize>());
        } else {
            fn recurse(
                result: &mut Vec<Vec<usize>>,
                current: &mut Vec<usize>,
                idx: usize,
                remaining: usize,
                n: usize,
                max_degree: usize,
            ) {
                if idx == n {
                    let total: usize = current.iter().sum();
                    if total >= 1 && total <= max_degree {
                        result.push(current.clone());
                    }
                    return;
                }
                let max_power = remaining.min(max_degree);
                for p in 0..=max_power {
                    current.push(p);
                    let new_remaining = remaining.saturating_sub(p);
                    recurse(result, current, idx + 1, new_remaining, n, max_degree);
                    current.pop();
                }
            }

            let mut current = Vec::with_capacity(n_features);
            recurse(&mut result, &mut current, 0, degree, n_features, degree);

            result.sort_by_key(|p| (p.iter().sum::<usize>(), p.clone()));
        }

        result
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Fit<DataFrame, DataFrame> for PolynomialFeatures {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "PolynomialFeatures.fit received an empty DataFrame (0 rows or 0 columns).".into(),
            ));
        }
        let col_names = self.numeric_f64_column_names(&x);
        if col_names.is_empty() {
            let all_types: Vec<String> = x
                .get_column_names()
                .iter()
                .filter_map(|n| x.column(n).ok().map(|c| format!("'{}' ({})", n, c.dtype())))
                .collect();
            return Err(Error::InvalidInput(format!(
                "PolynomialFeatures.fit: no Float64 columns found. \
                 Available columns: [{}]. PolynomialFeatures operates on f64 columns.",
                all_types.join(", ")
            )));
        }
        self.input_columns = Some(col_names);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for PolynomialFeatures {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "PolynomialFeatures has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let input_columns = self.input_columns.as_ref().unwrap();
        let powers = Self::generate_powers(input_columns.len(), self.degree, self.interaction_only);

        let mut columns: Vec<Column> = Vec::new();
        let n_rows = x.height();

        if self.include_bias {
            let bias = Column::from(Series::new("bias".into(), vec![1.0f64; n_rows]));
            columns.push(bias);
        }

        for power in &powers {
            let mut col_name = String::new();
            let mut has_terms = false;
            let mut series_vec: Option<Series> = None;

            for (j, &p) in power.iter().enumerate() {
                if p == 0 {
                    continue;
                }

                let name = &input_columns[j];
                let orig_series = x
                    .column(name)
                    .map_err(|e| {
                        Error::InvalidInput(format!(
                            "PolynomialFeatures.transform: column '{}' not found. \
                             The transformer was fitted on columns: {:?}. {}",
                            name, input_columns, e
                        ))
                    })?
                    .as_materialized_series()
                    .clone();

                if !has_terms {
                    series_vec = if p > 1 {
                        Some(series_pow(&orig_series, p))
                    } else {
                        Some(orig_series)
                    };
                    col_name = name.clone();
                    has_terms = true;
                } else {
                    let powered = if p > 1 {
                        series_pow(&orig_series, p)
                    } else {
                        orig_series
                    };
                    series_vec = Some(series_mul(&series_vec.unwrap(), &powered));
                    col_name.push('_');
                    col_name.push_str(name);
                }
            }

            if let Some(s) = series_vec {
                if power.iter().filter(|&&p| p > 0).count() > 1 {
                    col_name = format!("{}_inter", col_name);
                } else if power.iter().any(|&p| p > 1) {
                    col_name = format!("{}^", col_name);
                }

                let mut renamed = s.clone();
                renamed.rename(col_name.as_str().into());
                columns.push(Column::from(renamed));
            }
        }

        if columns.is_empty() {
            return Err(Error::Computation(
                "PolynomialFeatures: no features were generated. \
                 Ensure degree >= 1 and the input has at least one f64 column."
                    .into(),
            ));
        }

        DataFrame::new(n_rows, columns)
            .map_err(|e| Error::Computation(format!("failed to create polynomial features: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 2.0, 3.0]));
        let b = Column::from(Series::new("b".into(), &[4.0f64, 5.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_generate_powers_degree2_2features() {
        let powers = PolynomialFeatures::generate_powers(2, 2, false);
        assert_eq!(powers.len(), 5);
        assert_eq!(powers[0].iter().sum::<usize>(), 1);
        assert_eq!(powers[1].iter().sum::<usize>(), 1);
        assert_eq!(powers[2].iter().sum::<usize>(), 2);
        assert_eq!(powers[3].iter().sum::<usize>(), 2);
        assert_eq!(powers[4].iter().sum::<usize>(), 2);
    }

    #[test]
    fn test_polynomial_features_fit_transform() {
        let mut pf = PolynomialFeatures::new(2).include_bias(true);
        let df = make_test_df();
        let y = df.clone();

        pf.fit(df.clone(), y).unwrap();
        let result = pf.transform(df).unwrap();

        assert_eq!(result.width(), 6);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        let mut pf = PolynomialFeatures::new(2).include_bias(false);
        let df = make_test_df();
        let y = df.clone();

        pf.fit(df.clone(), y).unwrap();
        let result = pf.transform(df).unwrap();

        assert_eq!(result.width(), 5);
    }

    #[test]
    fn test_polynomial_features_interaction_only() {
        let mut pf = PolynomialFeatures::new(2)
            .include_bias(false)
            .interaction_only(true);
        let df = make_test_df();
        let y = df.clone();

        pf.fit(df.clone(), y).unwrap();
        let result = pf.transform(df).unwrap();

        assert_eq!(result.width(), 1);
    }
}
