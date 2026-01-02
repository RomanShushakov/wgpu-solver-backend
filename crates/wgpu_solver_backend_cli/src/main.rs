use clap::{Parser, Subcommand};
use futures::executor::block_on;
use serde::Serialize;
use serde_json::to_string_pretty;
use std::process::exit;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use wgpu_solver_backend::gpu::context::{GpuBackend, GpuContext};

#[derive(Parser, Debug)]
#[command(
    name = "wgpu-solver-backend",
    version,
    about = "Compute-first wgpu backend for iterative solvers"
)]
struct Cli {
    #[arg(long, default_value = "auto")]
    backend: String,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print GPU adapter info and emit a metrics JSON blob (stdout).
    Info,
}

#[derive(Serialize)]
struct Metrics {
    run_id: String,
    command: String,
    gpu: GpuMetrics,
    build: BuildMetrics,
}

#[derive(Serialize)]
struct GpuMetrics {
    adapter_name: String,
    backend: String,
    device_type: String,
    vendor: u32,
    device: u32,
}

#[derive(Serialize)]
struct BuildMetrics {
    crate_version: String,
    git_rev: Option<String>,
}

fn parse_backend(s: &str) -> GpuBackend {
    match s.to_lowercase().as_str() {
        "auto" => GpuBackend::Auto,
        "vulkan" => GpuBackend::Vulkan,
        "dx12" => GpuBackend::Dx12,
        "metal" => GpuBackend::Metal,
        other => {
            eprintln!("Unknown backend '{other}', using auto");
            GpuBackend::Auto
        }
    }
}

fn now_utc_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown-time".to_string())
}

fn main() {
    let cli = Cli::parse();
    let gpu_backend = parse_backend(&cli.backend);

    match cli.cmd {
        Command::Info => {
            let ctx = block_on(GpuContext::create(gpu_backend)).unwrap_or_else(|e| {
                eprintln!("Failed to init GPU context: {e}");
                exit(2);
            });

            // Human-readable (nice in logs)
            println!("{}", ctx.describe());

            // Machine-readable (Slurm-friendly)
            let m = Metrics {
                run_id: now_utc_rfc3339(),
                command: "info".to_string(),
                gpu: GpuMetrics {
                    adapter_name: ctx.adapter_info.name.clone(),
                    backend: format!("{:?}", ctx.adapter_info.backend),
                    device_type: format!("{:?}", ctx.adapter_info.device_type),
                    vendor: ctx.adapter_info.vendor,
                    device: ctx.adapter_info.device,
                },
                build: BuildMetrics {
                    crate_version: env!("CARGO_PKG_VERSION").to_string(),
                    git_rev: option_env!("GIT_REV").map(|s| s.to_string()),
                },
            };

            println!("{}", to_string_pretty(&m).unwrap());
        }
    }
}
