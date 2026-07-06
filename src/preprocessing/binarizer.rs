use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Binarize data (set feature values to 0 or 1) according to a threshold.
///
/// Corresponds to `sklearn.preprocessing.Binarizer`.
#[allow(dead_code)]
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

    fn fit(&mut self, _x: DataFrame, _y: DataFrame) -> Result<Self::Output> {
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Binarizer {
    type Output = DataFrame;

    fn transform(&self, _x: DataFrame) -> Result<Self::Output> {
        Err(Error::NotFitted("Binarizer".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binarizer_new() {
        let b = Binarizer::new(0.5);
        assert!(!b.fitted);
    }

    #[test]
    fn test_binarizer_default() {
        let b = Binarizer::default();
        assert_eq!(b.threshold, 0.0);
    }
}
