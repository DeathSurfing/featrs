//! End-to-end example: scale a small DataFrame, expand polynomial features,
//! and print the result. Run with `cargo run --example scaling_pipeline`.

use featrs::prelude::*;
use featrs::traits::{Fit, Transform};
use polars::prelude::{Column, DataFrame, NamedFrom, Series};

fn main() -> featrs::Result<()> {
    let a = Column::from(Series::new("a".into(), &[1.0_f64, 2.0, 3.0, 4.0]));
    let b = Column::from(Series::new("b".into(), &[2.0_f64, 4.0, 6.0, 8.0]));
    let df = DataFrame::new(4, vec![a, b]).unwrap();

    let mut pipeline = Pipeline::new(vec![
        ("scale".into(), Box::new(StandardScaler::new())),
        ("poly".into(), Box::new(PolynomialFeatures::new(2)?)),
    ])?;

    pipeline.fit(df.clone())?;
    let result = pipeline.transform(df)?;

    println!(
        "Output shape: {} cols x {} rows",
        result.width(),
        result.height()
    );
    println!("{result}");
    Ok(())
}
