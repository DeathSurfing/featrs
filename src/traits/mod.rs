use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not fitted: {0}")]
    NotFitted(String),
    #[error("computation error: {0}")]
    Computation(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Fit<X, Y = X> {
    type Output;
    fn fit(&mut self, x: X, y: Y) -> Result<Self::Output>;
}

pub trait Transform<X> {
    type Output;
    fn transform(&self, x: X) -> Result<Self::Output>;
}

pub trait FitTransform<X, Y = X>: Fit<X, Y> + Transform<X> {}

impl<T, X, Y> FitTransform<X, Y> for T where T: Fit<X, Y> + Transform<X> {}
