use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, CommandEncoder, ComputePassDescriptor};

use crate::compute::block_jacobi::{
    BlockJacobiPipeline, create_block_jacobi_bind_group, create_block_jacobi_pipeline,
};
use crate::gpu::context::GpuContext;

/// BlockJacobiExecutor
///
/// Owns the immutable GPU resources for the Block-Jacobi preconditioner:
///   - `lu_blocks_buffer`: packed LU blocks (one dense 6x6 per block, row-major)
///   - `block_starts_buffer`: block ranges (length num_blocks + 1)
///   - `params_buffer`: uniform [n, num_blocks, 0, 0]
///
/// Apply usage per iteration:
///   encode_apply(ctx, encoder, r_gpu, z_gpu)
/// where:
///   - `r_gpu`: input residual vector r
///   - `z_gpu`: output vector z = M^{-1} r
///
/// Dispatch:
///   - 1 workgroup per block (WGSL workgroup_size = 1),
///     so dispatch `num_blocks` workgroups in X.
pub struct BlockJacobiExecutor {
    n: u32,
    num_blocks: u32,

    // Pipeline + layout (immutable)
    block_jacobi_pipeline: BlockJacobiPipeline,

    // Persistent GPU buffers (immutable)
    params_buffer: Buffer,
    lu_blocks_buffer: Buffer,
    block_starts_buffer: Buffer,
}

impl BlockJacobiExecutor {
    /// Create Block-Jacobi executor.
    ///
    /// Inputs:
    /// - `n` length of vectors r/z (in f32)
    /// - `lu_blocks_host`: packed LU blocks, one 6x6 per block (36 f32 per block)
    /// - `block_starts_u32`: length num_blocks + 1, defines offsets into vector (in entries)
    ///
    /// NOTE:
    /// This executor assumes BLOCK_SIZE = 6 and LU_STRIDE = 36 in WGSL.
    pub fn create(
        ctx: &GpuContext,
        n: u32,
        lu_blocks_host: &[f32],
        block_starts_u32: &[u32],
    ) -> Self {
        let device = &ctx.device;

        let num_blocks = (block_starts_u32.len() as u32).saturating_sub(1);

        // 1) Pipeline (once)
        let block_jacobi_pipeline = create_block_jacobi_pipeline(ctx);

        // 2) Params uniform (once): [n, num_blocks, 0, 0]
        let params_words: [u32; 4] = [n, num_blocks, 0, 0];
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("block_jacobi params"),
            contents: bytemuck::cast_slice(&params_words),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // 3) LU blocks buffer (once)
        let lu_blocks_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("block_jacobi lu_blocks"),
            contents: bytemuck::cast_slice(lu_blocks_host),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // 4) block_starts buffer (once)
        let block_starts_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("block_jacobi block_starts"),
            contents: bytemuck::cast_slice(block_starts_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        Self {
            n,
            num_blocks,
            block_jacobi_pipeline,
            params_buffer,
            lu_blocks_buffer,
            block_starts_buffer,
        }
    }

    /// Encode: z = M^{-1} r
    ///
    /// `r_gpu` and `z_gpu` are per-call because they vary per iteration.
    pub fn encode_apply(
        &self,
        ctx: &GpuContext,
        encoder: &mut CommandEncoder,
        r_gpu: &Buffer,
        z_gpu: &Buffer,
    ) {
        // Bind group depends on per-call buffers r/z.
        let bind_group = create_block_jacobi_bind_group(
            &ctx.device,
            &self.block_jacobi_pipeline.block_jacobi_bind_group_layout,
            &self.params_buffer,
            &self.lu_blocks_buffer,
            &self.block_starts_buffer,
            r_gpu,
            z_gpu,
        );

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("block_jacobi apply pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.block_jacobi_pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // One workgroup per block (WGSL workgroup_size = 1).
        pass.dispatch_workgroups(self.num_blocks, 1, 1);
    }

    // pub fn n(&self) -> u32 {
    //     self.n
    // }

    // pub fn num_blocks(&self) -> u32 {
    //     self.num_blocks
    // }
}
