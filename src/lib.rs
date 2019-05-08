#![crate_name = "rustboost"]

#[macro_use]
extern crate log;
extern crate env_logger;

pub mod tree;
pub mod rustboost;
pub mod config;

mod dataset;
mod types;
mod loss;
mod utils;

#[cfg(test)]
mod tests {
  extern crate pretty_assertions;
  mod loss_tests;
}