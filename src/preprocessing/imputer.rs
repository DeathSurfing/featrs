//! Imputation transformations.
//!
//! Analogous to `sklearn.preprocessing.SimpleImputer`.



/// Imputation strategy.
#[derive(Clone, Copy)]
pub enum Strategy {
    Mean,
    Median,
    MostFrequent,
    Constant(f64),
}

/// Impute missing values using basic statistics.
///
/// Corresponds to `sklearn.preprocessing.SimpleImputer`.
#[allow(dead_code)]
pub struct SimpleImputer {
    fitted: bool,
    strategy: Strategy,
    fill_values: Option<Vec<f64>>,
}

impl SimpleImputer {
    pub fn new(strategy: Strategy) -> Self {
        Self {
            fitted: false,
            strategy,
            fill_values: None,
        }
    }

    pub fn mean() -> Self {
        Self::new(Strategy::Mean)
    }

    pub fn median() -> Self {
        Self::new(Strategy::Median)
    }

    pub fn most_frequent() -> Self {
        Self::new(Strategy::MostFrequent)
    }

    pub fn constant(value: f64) -> Self {
        Self::new(Strategy::Constant(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_imputer_new() {
        let imp = SimpleImputer::mean();
        assert!(matches!(imp.strategy, Strategy::Mean));
    }

    #[test]
    fn test_simple_imputer_constant() {
        let imp = SimpleImputer::constant(0.0);
        assert!(matches!(imp.strategy, Strategy::Constant(v) if v == 0.0));
    }
}
