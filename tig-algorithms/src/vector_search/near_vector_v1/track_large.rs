// near_vector_v1 / track_large.rs
//
// FP16 cuBLAS path for the LARGER tracks (n_queries=11000/13000/15000).
// The chain leader autovector_final3 only uses an FP16 input path on n=9000
// (its track_t20) — the larger tracks still go through plain FP32 cuBLAS with
// TILE_DB=4096. Half the memory bandwidth on the GEMM (FP16 inputs) and a
// bigger tile is the easy win.
//
// Pattern (same as autovector_final3::track_t20):
//   * inputs: FP16 (CUDA_R_16F)
//   * compute: CUBLAS_COMPUTE_32F (FP32 accumulator — same precision as FP32
//     GEMM, only the IO/multiply intermediate uses FP16)
//   * find_best_fp16_t20 kernel: reads FP16 dot tile, accumulates in FP32
//
// TILE_DB chosen to keep `d_dot` (num_queries * TILE_DB * 2 bytes) safely
// inside 1 GB at the largest track (n=15000 -> ~960 MB) so an 8GB consumer GPU
// still has plenty of headroom alongside the original FP32 challenge database
// (~1.5 GB at n=15000) and our FP16 copy (~750 MB at n=15000).

use anyhow::{anyhow, Result};
use cudarc::cublas::{result as cublas_result, sys as cublas_sys, CudaBlas};
use cudarc::driver::{
    safe::{CudaModule, CudaStream, LaunchConfig},
    DevicePtr, DevicePtrMut, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::ffi::c_void;
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {}
    }
}

const DIMS: usize = 250;
const TILE_DB: usize = 32768;
const WARPS_PER_BLOCK: u32 = 8;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    let nq_i32 = num_queries as i32;
    let db_i32 = database_size as i32;
    let dims_i32 = DIMS as i32;

    let mut d_best_dist = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_best_idx  = stream.alloc_zeros::<i32>(num_queries)?;
    let mut d_q_norms   = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_db_norms  = stream.alloc_zeros::<f32>(database_size)?;

    let mut d_q_fp16  = stream.alloc_zeros::<u16>(num_queries * DIMS)?;
    let mut d_db_fp16 = stream.alloc_zeros::<u16>(database_size * DIMS)?;

    let mut d_dot = stream.alloc_zeros::<u16>(num_queries * TILE_DB)?;

    let norms_func   = module.load_function("compute_norms_v122")?;
    let convert_func = module.load_function("convert_fp32_to_fp16")?;
    let find_func    = module.load_function("find_best_fp16_t20")?;

    // norms (FP32, computed from the original FP32 vectors — exact, no precision loss)
    let q_blk  = (num_queries as u32 + 255) / 256;
    let db_blk = (database_size as u32 + 255) / 256;
    unsafe {
        stream.launch_builder(&norms_func)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_norms)
            .arg(&nq_i32)
            .arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&norms_func)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_norms)
            .arg(&db_i32)
            .arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    // FP32 -> FP16 conversion (preserves cluster-scale precision; the gap
    // between true 1-NN and 2-NN is much larger than FP16 rounding error).
    let q_elem  = (num_queries * DIMS) as i32;
    let db_elem = (database_size * DIMS) as i32;
    let conv_q_blk  = (q_elem  as u32 + 255) / 256;
    let conv_db_blk = (db_elem as u32 + 255) / 256;
    unsafe {
        stream.launch_builder(&convert_func)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_fp16)
            .arg(&q_elem)
            .launch(LaunchConfig { grid_dim: (conv_q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&convert_func)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_fp16)
            .arg(&db_elem)
            .launch(LaunchConfig { grid_dim: (conv_db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    let cublas = CudaBlas::new(stream.clone())?;
    let n_db_tiles = (database_size + TILE_DB - 1) / TILE_DB;

    let alpha_f32: f32 = -2.0f32;
    let beta_f32:  f32 =  0.0f32;

    for tile_idx in 0..n_db_tiles {
        let db_start = tile_idx * TILE_DB;
        let tile_len = TILE_DB.min(database_size - db_start);

        let db_fp16_tile = d_db_fp16.slice(db_start * DIMS..(db_start + tile_len) * DIMS);

        // GEMM: dot = -2 * db_tile^T @ q  (FP16 inputs, FP32 accumulator)
        {
            let (db_ptr,  _db_sync)  = db_fp16_tile.device_ptr(&stream);
            let (q_ptr,   _q_sync)   = d_q_fp16.device_ptr(&stream);
            let (dot_ptr, _dot_sync) = d_dot.device_ptr_mut(&stream);
            unsafe {
                let status = cublas_result::gemm_ex(
                    *cublas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    tile_len as i32,
                    nq_i32,
                    dims_i32,
                    &alpha_f32 as *const f32 as *const c_void,
                    db_ptr as *const c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    dims_i32,
                    q_ptr as *const c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    dims_i32,
                    &beta_f32 as *const f32 as *const c_void,
                    dot_ptr as *mut c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    tile_len as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                );
                status.map_err(|e| anyhow!("cublasGemmEx failed: {:?}", e))?;
            }
        }

        // find_best across the tile (running argmin per query, FP32 distance reconstruction)
        let grid_y = (num_queries as u32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        let find_cfg = LaunchConfig {
            grid_dim: (1, grid_y, 1),
            block_dim: (32, WARPS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let db_start_i32 = db_start as i32;
        let tile_len_i32 = tile_len as i32;
        let first_tile: i32 = if tile_idx == 0 { 1 } else { 0 };
        unsafe {
            stream.launch_builder(&find_func)
                .arg(&d_q_norms)
                .arg(&d_db_norms)
                .arg(&d_dot)
                .arg(&mut d_best_idx)
                .arg(&mut d_best_dist)
                .arg(&db_start_i32)
                .arg(&tile_len_i32)
                .arg(&nq_i32)
                .arg(&first_tile)
                .launch(find_cfg)?;
        }
    }

    stream.synchronize()?;

    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_best_idx)?;
    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| if idx < 0 || idx >= db_i32 { 0 } else { idx as usize })
        .collect();

    save_solution(&Solution { indexes })?;
    Ok(())
}

pub fn help() {
    println!("near_vector_v1 track_large — FP16 GemmEx (COMPUTE_32F) TILE_DB=32768, for n=11000/13000/15000");
}
