use featrs::time_series::cyclical::CyclicalEncoder;
use featrs::time_series::diff::Difference;
use featrs::time_series::lag::Lagger;
use featrs::time_series::rolling::{RollingAggregator, RollingFn};
use featrs::traits::{Fit, Transform};
use polars::prelude::{Column, DataFrame, NamedFrom, Series};

fn sales_df() -> DataFrame {
    let col = Column::from(Series::new("sales".into(), &[1.0_f64, 2.0, 3.0, 4.0, 5.0]));
    DataFrame::new(5, vec![col]).unwrap()
}

#[test]
fn test_lagger_integration() {
    let mut lagger = Lagger::new(&["sales"], &[1]);
    let df = sales_df();

    lagger.fit(df.clone()).unwrap();
    let result = lagger.transform(df).unwrap();

    let names: Vec<String> = result
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("lag")),
        "expected a lag column, got {names:?}"
    );
    assert_eq!(result.height(), 5);
}

#[test]
fn test_rolling_aggregator_integration() {
    let mut rolling = RollingAggregator::new(&["sales"], 3, RollingFn::Mean);
    let df = sales_df();

    rolling.fit(df.clone()).unwrap();
    let result = rolling.transform(df).unwrap();

    assert_eq!(result.height(), 5);
}

#[test]
fn test_difference_integration() {
    let mut diff = Difference::diff(&["sales"], 1);
    let df = sales_df();

    diff.fit(df.clone()).unwrap();
    let result = diff.transform(df).unwrap();

    assert_eq!(result.height(), 5);
    let names: Vec<String> = result
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("diff")),
        "expected a diff column, got {names:?}"
    );
}

#[test]
fn test_cyclical_encoder_integration() {
    let mut enc = CyclicalEncoder::new(&["sales"], 7);
    let df = sales_df();

    enc.fit(df.clone()).unwrap();
    let result = enc.transform(df).unwrap();

    // CyclicalEncoder adds sin + cos columns for each input column.
    assert!(result.width() >= 2);
    assert_eq!(result.height(), 5);
}
