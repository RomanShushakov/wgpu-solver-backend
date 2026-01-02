use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::gpu::context::GpuContext;

pub struct DotPartialsPipeline {
    pub pipeline: ComputePipeline,
    pub dot_partials_bind_group_layout: BindGroupLayout,
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

pub fn create_dot_partials_pipeline(ctx: &GpuContext) -> DotPartialsPipeline {
    let device = &ctx.device;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("dot_partials.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/dot_partials.wgsl").into()),
    });

    // WGSL bindings:
    //  @binding(0) params (uniform)
    //  @binding(1) a (storage read)
    //  @binding(2) b (storage read)
    //  @binding(3) partial (storage write)
    let dot_partials_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dot_partials bgl0"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("dot_partials pipeline layout"),
        bind_group_layouts: &[&dot_partials_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("dot_partials pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    DotPartialsPipeline {
        pipeline,
        dot_partials_bind_group_layout,
    }
}

pub fn create_dot_partials_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    params_buffer: &Buffer,   // binding(0)
    a_buffer: &Buffer,        // binding(1)
    b_buffer: &Buffer,        // binding(2)
    partials_buffer: &Buffer, // binding(3)
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("dot_partials bind group 0"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: a_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: b_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: partials_buffer.as_entire_binding(),
            },
        ],
    })
}
