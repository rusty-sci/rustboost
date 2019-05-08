use super::types::dtype;

#[cfg(test)]
use pretty_assertions::{assert_eq};

pub const EPS: f64 = 10_000_000f64;

/// Checl almost equal, v1 rounded by EPS must be equal to v2
/// also rounded by EPS. EPS - is the small number.
pub fn almost_eq(v1: dtype, v2: dtype, eps: dtype) {
  assert_eq!((v1 * eps).round() / eps,
    (v2 * eps).round() / eps);
}