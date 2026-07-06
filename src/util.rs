//! Shared helpers for transformers operating on `Float64` columns.
//!
//! These utilities exist to keep the per-transformer code free of duplicated
//! column-discovery, validation, and in-place replacement boilerplate.

use polars::prelude::{ChunkedArray, DataFrame, DataType, Float64Type, IntoSeries};

use crate::traits::{Error, Result};

/// Return the names of all `Float64` columns in `df`, in frame order.
///
/// Non-`Float64` columns are silently skipped. Transformers that need to
/// surface this as an error should use [`require_f64_columns`] instead.
pub fn numeric_f64_columns(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .filter_map(|name| {
            df.column(name)
                .ok()
                .filter(|s| s.dtype() == &DataType::Float64)
                .map(|_| name.to_string())
        })
        .collect()
}

/// Return the names of all `Float64` columns, or an `InvalidInput` error that
/// lists every column and its dtype when none are `Float64`.
///
/// `who` is the transformer name used in the error message
/// (e.g. `"StandardScaler"`), so failures are easy to trace back to the
/// transformer that produced them.
pub fn require_f64_columns(df: &DataFrame, who: &str) -> Result<Vec<String>> {
    let cols = numeric_f64_columns(df);
    if cols.is_empty() {
        let all_types: Vec<String> = df
            .get_column_names()
            .iter()
            .filter_map(|n| df.column(n).ok().map(|c| format!("'{n}' ({})", c.dtype())))
            .collect();
        return Err(Error::InvalidInput(format!(
            "{who}: no Float64 columns found. This transformer only operates on f64 columns. \
             Available columns: [{}]. Cast non-f64 columns before fitting.",
            all_types.join(", ")
        )));
    }
    Ok(cols)
}

/// Apply a per-element f64 transform to a single named column of `df`,
/// replacing the column in place.
///
/// `f` maps each non-null `f64` value to its replacement; nulls are preserved.
/// `who` names the calling transformer for error context.
///
/// This collapses the `column(name).unwrap()` → `f64().unwrap()` →
/// `replace(...).unwrap()` boilerplate that would otherwise be duplicated in
/// every f64-based transformer's `transform` implementation.
pub fn replace_f64_column<F>(df: &mut DataFrame, name: &str, who: &str, f: F) -> Result<()>
where
    F: Fn(f64) -> f64,
{
    let s = df.column(name).map_err(|e| {
        Error::InvalidInput(format!("{who}.transform: column '{name}' not found. {e}"))
    })?;
    let ca = s.f64().map_err(|e| {
        Error::InvalidInput(format!(
            "{who}.transform: column '{name}' has dtype {}; expected Float64. {e}",
            s.dtype()
        ))
    })?;
    let mapped: ChunkedArray<Float64Type> = ca.iter().map(|opt| opt.map(&f)).collect();
    df.replace(name, mapped.into_series().into()).map_err(|e| {
        Error::Computation(format!(
            "{who}.transform: failed to replace column '{name}'. {e}"
        ))
    })?;
    Ok(())
}
