use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod engine;
pub mod nn;
pub mod util;

pub mod python;