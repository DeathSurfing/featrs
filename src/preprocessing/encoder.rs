//! Categorical encoding transformations.
//!
//! Analogous to `sklearn.preprocessing` encoders.

use std::collections::HashMap;

/// Encode categorical features as a one-hot numeric array.
///
/// Corresponds to `sklearn.preprocessing.OneHotEncoder`.
#[allow(dead_code)]
pub struct OneHotEncoder {
    fitted: bool,
    categories: Option<Vec<Vec<String>>>,
    sparse_output: bool,
}

impl OneHotEncoder {
    pub fn new() -> Self {
        Self {
            fitted: false,
            categories: None,
            sparse_output: false,
        }
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode categorical labels with value between 0 and n_classes-1.
///
/// Corresponds to `sklearn.preprocessing.LabelEncoder`.
#[allow(dead_code)]
pub struct LabelEncoder {
    fitted: bool,
    classes: Option<Vec<String>>,
    mapping: Option<HashMap<String, usize>>,
}

impl LabelEncoder {
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

/// Encode categorical features as an integer array.
///
/// Corresponds to `sklearn.preprocessing.OrdinalEncoder`.
#[allow(dead_code)]
pub struct OrdinalEncoder {
    fitted: bool,
    categories: Option<Vec<Vec<String>>>,
}

impl OrdinalEncoder {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encoder_new() {
        let enc = OneHotEncoder::new();
        assert!(!enc.fitted);
    }

    #[test]
    fn test_label_encoder_new() {
        let enc = LabelEncoder::new();
        assert!(!enc.fitted);
    }

    #[test]
    fn test_ordinal_encoder_new() {
        let enc = OrdinalEncoder::new();
        assert!(!enc.fitted);
    }
}
