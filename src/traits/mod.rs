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

pub trait Fit<A, X, Y = X> {
    type Output;
    fn fit(&mut self, x: X, y: Y) -> Result<Self::Output>;
}

pub trait Transform<A, X> {
    type Output;
    fn transform(&self, x: X) -> Result<Self::Output>;
}

pub trait FitTransform<A, X, Y = X>: Fit<A, X, Y> + Transform<A, X> {}

impl<T, A, X, Y> FitTransform<A, X, Y> for T where T: Fit<A, X, Y> + Transform<A, X> {}
