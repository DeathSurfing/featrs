use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Normalize samples individually to unit norm.
///
/// Corresponds to `sklearn.preprocessing.Normalizer`.
#[allow(dead_code)]
pub struct Normalizer {
    fitted: bool,
    norm: Norm,
}

#[derive(Clone, Copy)]
pub enum Norm {
    L1,
    L2,
    Max,
}

impl Normalizer {
    pub fn new(norm: Norm) -> Self {
        Self {
            fitted: false,
            norm,
        }
    }

    pub fn l1() -> Self {
        Self::new(Norm::L1)
    }

    pub fn l2() -> Self {
        Self::new(Norm::L2)
    }

    pub fn max() -> Self {
        Self::new(Norm::Max)
    }
}

impl Fit<DataFrame, DataFrame> for Normalizer {
    type Output = ();

    fn fit(&mut self, _x: DataFrame, _y: DataFrame) -> Result<Self::Output> {
        Ok(())
    }
}

impl Transform<DataFrame> for Normalizer {
    type Output = DataFrame;

    fn transform(&self, _x: DataFrame) -> Result<Self::Output> {
        Err(Error::NotFitted("Normalizer".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer_new() {
        let n = Normalizer::l2();
        assert!(matches!(n.norm, Norm::L2));
    }
}
