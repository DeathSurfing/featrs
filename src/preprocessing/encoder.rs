//! Categorical encoding.
//!
//! Analogous to `sklearn.preprocessing`. Provides:
//! - [`OneHotEncoder`] — create dummy/binary columns for each category
//! - [`LabelEncoder`] — encode labels as `0..n_classes-1` integers
//! - [`OrdinalEncoder`] — encode categorical features as integer columns

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;
use std::collections::HashMap;

fn column_unique_strings(col: &Column) -> Result<Vec<String>> {
    let s = col.as_materialized_series();
    let ca = s.str().map_err(|e| {
        Error::InvalidInput(format!(
            "Encoder: column '{}' has dtype {}; expected String. \
             Only string columns can be encoded. {}",
            col.name(),
            col.dtype(),
            e
        ))
    })?;
    let mut unique: Vec<String> = ca
        .iter()
        .flatten()
        .map(|s| s.to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique.sort();
    Ok(unique)
}

/// Encode categorical features as a one-hot numeric array.
///
/// Creates a binary column for each category value. Non-string columns
/// are ignored.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::OneHotEncoder;
/// use featrs::traits::{Fit, Transform};
///
/// let mut enc = OneHotEncoder::new().drop_first(false);
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // enc.fit(df.clone(), target)?;
/// // let encoded = enc.transform(df)?;
/// ```
pub struct OneHotEncoder {
    fitted: bool,
    categories: Option<Vec<OneHotCategory>>,
    drop_first: bool,
}

struct OneHotCategory {
    column: String,
    categories: Vec<String>,
}

impl OneHotEncoder {
    /// Create a new `OneHotEncoder`.
    ///
    /// By default, a column is created for every category. Use
    /// [`drop_first`](Self::drop_first) to drop the first category
    /// and avoid multicollinearity.
    pub fn new() -> Self {
        Self {
            fitted: false,
            categories: None,
            drop_first: false,
        }
    }

    /// Whether to drop the first category of each feature (default: `false`).
    ///
    /// When `true`, the first category (alphabetically) is omitted from the
    /// output, producing `k-1` columns for a feature with `k` categories.
    pub fn drop_first(mut self, value: bool) -> Self {
        self.drop_first = value;
        self
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame, DataFrame> for OneHotEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "OneHotEncoder.fit received a DataFrame with 0 rows. \
                 Provide at least 1 row."
                    .into(),
            ));
        }
        let mut cats = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            let unique = column_unique_strings(col)?;

            if !unique.is_empty() {
                cats.push(OneHotCategory {
                    column: name,
                    categories: unique,
                });
            }
        }

        if cats.is_empty() {
            return Err(Error::InvalidInput(
                "OneHotEncoder.fit: no string columns found. \
                 OneHotEncoder operates on String columns only. \
                 Cast categorical columns to String first."
                    .into(),
            ));
        }

        self.categories = Some(cats);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for OneHotEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "OneHotEncoder has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let cats = self.categories.as_ref().unwrap();
        let mut new_cols: Vec<Column> = Vec::new();
        let n_rows = x.height();

        for cat in cats {
            let s = x.column(&cat.column).map_err(|e| {
                Error::InvalidInput(format!(
                    "OneHotEncoder.transform: column '{}' not found in input. \
                     The encoder was fitted on columns: {:?}. {}",
                    cat.column,
                    cats.iter().map(|c| &c.column).collect::<Vec<_>>(),
                    e
                ))
            })?;
            let ca = s.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "OneHotEncoder.transform: column '{}' has dtype {}; expected String. {}",
                    cat.column,
                    s.dtype(),
                    e
                ))
            })?;
            let start_idx = if self.drop_first { 1 } else { 0 };

            for (_j, category) in cat.categories.iter().enumerate().skip(start_idx) {
                let mut vals = vec![0.0f64; n_rows];
                for (i, opt) in ca.iter().enumerate() {
                    if let Some(v) = opt
                        && v == *category
                    {
                        vals[i] = 1.0;
                    }
                }
                let col_name = format!("{}_{}", cat.column, category);
                new_cols.push(Column::from(Series::new(col_name.as_str().into(), &vals)));
            }
        }

        DataFrame::new(n_rows, new_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Encode labels as integers `0` to `n_classes - 1`.
///
/// Operates on a single string column. The mapping is sorted alphabetically.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::LabelEncoder;
/// use featrs::traits::{Fit, Transform};
///
/// let mut enc = LabelEncoder::new();
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // enc.fit(df.clone(), target)?;
/// // let encoded = enc.transform(df)?;
/// ```
pub struct LabelEncoder {
    fitted: bool,
    classes: Option<Vec<String>>,
    mapping: Option<HashMap<String, usize>>,
}

impl LabelEncoder {
    /// Create a new `LabelEncoder`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            classes: None,
            mapping: None,
        }
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame, DataFrame> for LabelEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.width() != 1 {
            return Err(Error::InvalidInput(format!(
                "LabelEncoder.fit expects a single column but got {} columns. \
                 Select one column before calling fit, e.g. df.select(['target']).",
                x.width()
            )));
        }
        let col = &x.columns()[0];
        let classes = column_unique_strings(col)?;

        if classes.is_empty() {
            return Err(Error::InvalidInput(format!(
                "LabelEncoder.fit: column '{}' contains no unique values. \
                 Provide data with at least one non-null string.",
                col.name()
            )));
        }

        let mapping: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        self.classes = Some(classes);
        self.mapping = Some(mapping);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for LabelEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "LabelEncoder has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let mapping = self.mapping.as_ref().unwrap();
        let s = x.columns()[0].as_materialized_series();
        let ca = s.str().map_err(|e| {
            Error::InvalidInput(format!(
                "LabelEncoder.transform: column '{}' has dtype {}; expected String. {}",
                s.name(),
                s.dtype(),
                e
            ))
        })?;

        let encoded: ChunkedArray<UInt32Type> = ca
            .iter()
            .map(|opt| opt.and_then(|v| mapping.get(v).copied().map(|x| x as u32)))
            .collect();

        let mut series = encoded.into_series();
        series.rename(s.name().clone());
        DataFrame::new(x.height(), vec![Column::from(series)])
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

/// Encode categorical features as integer columns.
///
/// Similar to [`LabelEncoder`] but operates on multiple columns at once.
/// Each column receives its own `0..n_categories` mapping.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::encoder::OrdinalEncoder;
/// use featrs::traits::{Fit, Transform};
///
/// let mut enc = OrdinalEncoder::new();
/// # let df = polars::prelude::DataFrame::new(0usize, vec![]).unwrap();
/// // enc.fit(df.clone(), target)?;
/// // let encoded = enc.transform(df)?;
/// ```
pub struct OrdinalEncoder {
    fitted: bool,
    categories: Option<Vec<(String, HashMap<String, u32>)>>,
}

impl OrdinalEncoder {
    /// Create a new `OrdinalEncoder`.
    pub fn new() -> Self {
        Self {
            fitted: false,
            categories: None,
        }
    }
}

impl Default for OrdinalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<DataFrame, DataFrame> for OrdinalEncoder {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "OrdinalEncoder.fit received a DataFrame with 0 columns. \
                 Provide at least one string column to encode."
                    .into(),
            ));
        }
        let mut cats = Vec::new();

        for col in x.columns() {
            let name = col.name().to_string();
            let classes = column_unique_strings(col)?;

            if classes.is_empty() {
                continue;
            }

            let mapping: HashMap<String, u32> = classes
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i as u32))
                .collect();

            cats.push((name, mapping));
        }

        if cats.is_empty() {
            return Err(Error::InvalidInput(
                "OrdinalEncoder.fit: no string columns found. \
                 OrdinalEncoder operates on String columns only."
                    .into(),
            ));
        }

        self.categories = Some(cats);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for OrdinalEncoder {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "OrdinalEncoder has not been fitted. \
                 Call .fit(dataframe, target) before .transform()."
                    .into(),
            ));
        }
        let mut out_cols = Vec::new();

        for (name, mapping) in self.categories.as_ref().unwrap() {
            let s = x.column(name.as_str()).map_err(|e| {
                Error::InvalidInput(format!(
                    "OrdinalEncoder.transform: column '{}' not found. \
                     The encoder was fitted on columns: {:?}. {}",
                    name,
                    self.categories
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|(n, _)| n)
                        .collect::<Vec<_>>(),
                    e
                ))
            })?;
            let ca = s.as_materialized_series().str().map_err(|e| {
                Error::InvalidInput(format!(
                    "OrdinalEncoder.transform: column '{}' has dtype {}; expected String. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;

            let encoded: ChunkedArray<UInt32Type> = ca
                .iter()
                .map(|opt| opt.and_then(|v| mapping.get(v).copied()))
                .collect();

            let mut series = encoded.into_series();
            series.rename(name.as_str().into());
            out_cols.push(Column::from(series));
        }

        DataFrame::new(x.height(), out_cols).map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_categorical_df() -> DataFrame {
        let a = Column::from(Series::new(
            "color".into(),
            &["red", "blue", "red", "green"],
        ));
        let b = Column::from(Series::new("size".into(), &["S", "M", "L", "M"]));
        DataFrame::new(4, vec![a, b]).unwrap()
    }

    #[test]
    fn test_one_hot_encoder() {
        let mut enc = OneHotEncoder::new();
        let df = make_categorical_df();
        let y = df.clone();

        enc.fit(df.clone(), y).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 6);
        assert_eq!(result.height(), 4);
    }

    #[test]
    fn test_one_hot_encoder_drop_first() {
        let mut enc = OneHotEncoder::new().drop_first(true);
        let df = make_categorical_df();
        let y = df.clone();

        enc.fit(df.clone(), y).unwrap();
        let result = enc.transform(df).unwrap();

        assert_eq!(result.width(), 4);
    }

    #[test]
    fn test_label_encoder() {
        let mut enc = LabelEncoder::new();
        let colors = Column::from(Series::new(
            "color".into(),
            &["red", "blue", "red", "green"],
        ));
        let df = DataFrame::new(4, vec![colors]).unwrap();
        let y = df.clone();

        enc.fit(df.clone(), y).unwrap();
        let result = enc.transform(df).unwrap();

        let vals: Vec<u32> = result
            .column("color")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(vals, vec![2, 0, 2, 1]);
    }

    #[test]
    fn test_ordinal_encoder() {
        let mut enc = OrdinalEncoder::new();
        let df = make_categorical_df();
        let y = df.clone();

        enc.fit(df.clone(), y).unwrap();
        let result = enc.transform(df).unwrap();

        let color_vals: Vec<u32> = result
            .column("color")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(color_vals, vec![2, 0, 2, 1]);

        let size_vals: Vec<u32> = result
            .column("size")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(size_vals, vec![2, 1, 0, 1]);
    }
}
