use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Binarize data (set feature values to 0 or 1) according to a threshold.
///
/// Corresponds to `sklearn.preprocessing.Binarizer`.
pub struct Binarizer {
    fitted: bool,
    threshold: f64,
}

impl Binarizer {
    pub fn new(threshold: f64) -> Self {
        Self {
            fitted: false,
            threshold,
        }
    }
}

impl Default for Binarizer {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Fit<DataFrame, DataFrame> for Binarizer {
    type Output = ();

    fn fit(&mut self, x: DataFrame, _y: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput("empty DataFrame".into()));
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Binarizer {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted("Binarizer".into()));
        }

        let col_names: Vec<String> = x
            .get_column_names()
            .iter()
            .filter_map(|name| {
                let col = x.column(name).ok()?;
                if col.dtype() == &DataType::Float64 {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .collect();

        let mut out = x.clone();

        for name in &col_names {
            let s = out.column(name.as_str()).unwrap();
            let ca = s.f64().unwrap();
            let binarized: ChunkedArray<Float64Type> = ca
                .iter()
                .map(|opt| opt.map(|v| if v > self.threshold { 1.0 } else { 0.0 }))
                .collect();
            out.replace(name.as_str(), binarized.into_series().into())
                .unwrap();
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_binarizer_default() {
        let mut b = Binarizer::default();
        let a = Column::from(Series::new("x".into(), &[-1.0f64, 0.0, 2.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();
        let y = df.clone();

        b.fit(df.clone(), y).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let mut b = Binarizer::new(5.0);
        let a = Column::from(Series::new("x".into(), &[1.0f64, 5.0, 10.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();
        let y = df.clone();

        b.fit(df.clone(), y).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }
}
