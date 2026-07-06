//! Binarization transformations.
//!
//! Analogous to `sklearn.preprocessing.Binarizer`.

use crate::traits::{Error, Fit, Result, Transform};
use ndarray::Array2;

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

impl Fit<f64, Array2<f64>, Array2<f64>> for Binarizer {
    type Output = ();

    fn fit(&mut self, _x: Array2<f64>, _y: Array2<f64>) -> Result<Self::Output> {
        Ok(())
    }
}

impl Transform<f64, Array2<f64>> for Binarizer {
    type Output = Array2<f64>;

    fn transform(&self, _x: Array2<f64>) -> Result<Self::Output> {
        Err(Error::NotFitted("Binarizer".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binarizer_new() {
        let b = Binarizer::new(0.5);
        assert_eq!(b.threshold, 0.5);
    }

    #[test]
    fn test_binarizer_default() {
        let b = Binarizer::default();
        assert_eq!(b.threshold, 0.0);
    }
}
