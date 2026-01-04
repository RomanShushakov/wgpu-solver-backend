use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::gpu::context::GpuContext;

pub struct BlockJacobiPipeline {
    pub pipeline: ComputePipeline,
    pub block_jacobi_bind_group_layout: BindGroupLayout,
}

fn create_uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_storage_entry(binding: u32, is_read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage {
                read_only: is_read_only,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn create_block_jacobi_pipeline(ctx: &GpuContext) -> BlockJacobiPipeline {
    let device = &ctx.device;

    // Shader module
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("block_jacobi.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/block_jacobi.wgsl").into()),
    });

    // Bind group layout (group 0), matches block_jacobi.wgsl:
    //  0: params (uniform)
    //  1: lu_blocks (RO storage)
    //  2: block_starts (RO storage)
    //  3: r (RO storage)
    //  4: z (RW storage)
    let block_jacobi_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("block_jacobi bgl0"),
            entries: &[
                create_uniform_entry(0),
                create_storage_entry(1, true),
                create_storage_entry(2, true),
                create_storage_entry(3, true),
                create_storage_entry(4, false),
            ],
        });

    // Pipeline layout (newer wgpu uses immediate_size)
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("block_jacobi pipeline layout"),
        bind_group_layouts: &[&block_jacobi_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("block_jacobi pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    BlockJacobiPipeline {
        pipeline,
        block_jacobi_bind_group_layout,
    }
}

pub fn create_block_jacobi_bind_group(
    device: &Device,
    block_jacobi_bind_group_layout: &BindGroupLayout,
    params_buffer: &Buffer,
    lu_blocks_buffer: &Buffer,
    block_starts_buffer: &Buffer,
    r_buffer: &Buffer,
    z_buffer: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("block_jacobi bind group 0"),
        layout: block_jacobi_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: lu_blocks_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: block_starts_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: r_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: z_buffer.as_entire_binding(),
            },
        ],
    })
}
