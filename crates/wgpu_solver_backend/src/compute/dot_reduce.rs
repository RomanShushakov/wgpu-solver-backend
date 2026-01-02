use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::gpu::context::GpuContext;

pub struct DotReducePipeline {
    pub pipeline: ComputePipeline,
    pub dot_reduce_bind_group_layout: BindGroupLayout,
}

fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
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

fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn create_dot_reduce_pipeline(ctx: &GpuContext) -> DotReducePipeline {
    let device = &ctx.device;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("dot_reduce.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/dot_reduce.wgsl").into()),
    });

    // WGSL bindings:
    //  @binding(0) params (uniform)  -> params.n = current_len
    //  @binding(1) input  (storage read)
    //  @binding(2) output (storage write)
    let dot_reduce_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dot_reduce bgl0"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("dot_reduce pipeline layout"),
        bind_group_layouts: &[&dot_reduce_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("dot_reduce pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    DotReducePipeline {
        pipeline,
        dot_reduce_bind_group_layout,
    }
}

pub fn create_dot_reduce_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    params_buffer: &Buffer, // binding(0)
    input_buffer: &Buffer,  // binding(1)
    output_buffer: &Buffer, // binding(2)
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("dot_reduce bind group 0"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: input_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    })
}
