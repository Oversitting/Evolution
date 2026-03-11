//! Performance benchmark for identifying bottlenecks
//! 
//! Tests each compute shader in isolation to identify performance issues.
//! Run with: cargo run --example perf_test --release

use std::time::{Duration, Instant};
use pollster::block_on;

const WORLD_WIDTH: u32 = 512;
const WORLD_HEIGHT: u32 = 512;
const NUM_ORGANISMS: u32 = 1000;
const NUM_TICKS: u32 = 100;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("=== Evolution Simulator Performance Benchmark ===\n");
    println!("World: {}x{} = {} cells", WORLD_WIDTH, WORLD_HEIGHT, WORLD_WIDTH * WORLD_HEIGHT);
    println!("Organisms: {}", NUM_ORGANISMS);
    println!("Ticks: {}\n", NUM_TICKS);
    
    block_on(run_benchmark());
}

async fn run_benchmark() {
    // Initialize GPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find adapter");
    
    println!("GPU: {}", adapter.get_info().name);
    println!("Backend: {:?}\n", adapter.get_info().backend);
    
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("Failed to create device");
    
    // Run benchmarks
    println!("--- Benchmark Results ---\n");
    
    let world_time = benchmark_world_shader(&device, &queue).await;
    let world_simple_time = benchmark_world_shader_simple(&device, &queue).await;
    let sense_time = benchmark_sense_shader(&device, &queue).await;
    let think_time = benchmark_think_shader(&device, &queue).await;
    let act_time = benchmark_act_shader(&device, &queue).await;
    
    println!("\n--- Summary (per tick avg) ---");
    println!("World (with hash):     {:.3} ms", world_time.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    println!("World (simple):        {:.3} ms", world_simple_time.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    println!("Sense (8 rays):        {:.3} ms", sense_time.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    println!("Think (NN):            {:.3} ms", think_time.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    println!("Act:                   {:.3} ms", act_time.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    
    let total = world_time + sense_time + think_time + act_time;
    let total_simple = world_simple_time + sense_time + think_time + act_time;
    println!("\nTotal (hash):          {:.3} ms/tick", total.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    println!("Total (simple):        {:.3} ms/tick", total_simple.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    println!("Potential speedup:     {:.1}x", total.as_secs_f64() / total_simple.as_secs_f64());
}

async fn benchmark_world_shader(device: &wgpu::Device, queue: &wgpu::Queue) -> Duration {
    // Create minimal buffers for world shader
    let food_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Food Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Obstacles Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Buffer"),
        size: 256, // Enough for SimConfig
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Dummy buffers for layout compatibility
    let organisms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Organisms Buffer"),
        size: (NUM_ORGANISMS * 56) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sensory Buffer"),
        size: (NUM_ORGANISMS * 80) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let actions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Actions Buffer"),
        size: (NUM_ORGANISMS * 24) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let nn_weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NN Weights Buffer"),
        size: (NUM_ORGANISMS * 438 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    // Bind group layout matching world.wgsl
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("World Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("World Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: organisms_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: food_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: obstacles_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sensory_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: actions_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: nn_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: config_buffer.as_entire_binding() },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("World Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("World Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/compute/shaders/world.wgsl").into()),
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("World Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    
    // Initialize config with proper values
    let config_data = create_config_uniform(0);
    queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
    
    // Initialize food with patches (so growth logic runs)
    let mut food_data = vec![0.0f32; (WORLD_WIDTH * WORLD_HEIGHT) as usize];
    for i in 0..50 {
        let cx = (i * 73 % WORLD_WIDTH) as usize;
        let cy = (i * 97 % WORLD_HEIGHT) as usize;
        for dy in 0..25 {
            for dx in 0..25 {
                let x = (cx + dx).min((WORLD_WIDTH - 1) as usize);
                let y = (cy + dy).min((WORLD_HEIGHT - 1) as usize);
                food_data[y * WORLD_WIDTH as usize + x] = 5.0;
            }
        }
    }
    queue.write_buffer(&food_buffer, 0, bytemuck::cast_slice(&food_data));
    
    // Warm up
    for tick in 0..10 {
        let config_data = create_config_uniform(tick);
        queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((WORLD_WIDTH + 7) / 8, (WORLD_HEIGHT + 7) / 8, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    
    // Benchmark
    let start = Instant::now();
    for tick in 0..NUM_TICKS {
        let config_data = create_config_uniform(tick + 10);
        queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((WORLD_WIDTH + 7) / 8, (WORLD_HEIGHT + 7) / 8, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    let elapsed = start.elapsed();
    
    println!("World shader (with hash): {:.1} ms total, {:.3} ms/tick", 
             elapsed.as_secs_f64() * 1000.0,
             elapsed.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    
    elapsed
}

async fn benchmark_world_shader_simple(device: &wgpu::Device, queue: &wgpu::Queue) -> Duration {
    // Test simplified world shader without per-cell hash
    let shader_source = r#"
struct SimConfig {
    world_width: u32,
    world_height: u32,
    num_organisms: u32,
    tick: u32,
    vision_range: f32,
    vision_fov: f32,
    vision_rays: u32,
    _pad1: u32,
    max_energy: f32,
    passive_drain: f32,
    movement_cost_forward: f32,
    movement_cost_rotate: f32,
    max_age: u32,
    crowding_factor: f32,
    max_organisms: u32,
    age_drain_factor: f32,
    max_speed: f32,
    max_rotation: f32,
    organism_radius: f32,
    _pad2: u32,
    food_growth_rate: f32,
    food_max_per_cell: f32,
    food_energy_value: f32,
    food_effectiveness: f32,
    reproduction_threshold: f32,
    reproduction_signal_min: f32,
    reproduction_cost: f32,
    reproduction_min_age: u32,
}

struct Organism {
    position: vec2<f32>,
    velocity: vec2<f32>,
    rotation: f32,
    energy: f32,
    age: u32,
    flags: u32,
    genome_id: u32,
    generation: u32,
    offspring_count: u32,
    parent_id: u32,
}

@group(0) @binding(0) var<storage, read_write> organisms: array<Organism>;
@group(0) @binding(1) var<storage, read_write> food: array<f32>;
@group(0) @binding(2) var<storage, read> obstacles: array<u32>;
@group(0) @binding(3) var<storage, read_write> sensory: array<f32>;
@group(0) @binding(4) var<storage, read_write> actions: array<f32>;
@group(0) @binding(5) var<storage, read> nn_weights: array<f32>;
@group(0) @binding(6) var<uniform> config: SimConfig;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    
    if x >= config.world_width || y >= config.world_height {
        return;
    }
    
    let idx = y * config.world_width + x;
    
    if obstacles[idx] != 0u {
        return;
    }
    
    var current_food = food[idx];
    
    // Simple growth only - no hash calculation
    if current_food > 0.1 {
        let growth = config.food_growth_rate * current_food * (1.0 - current_food / config.food_max_per_cell);
        current_food += growth;
    }
    
    current_food = clamp(current_food, 0.0, config.food_max_per_cell);
    food[idx] = current_food;
}
"#;
    
    // Create buffers (same as above)
    let food_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Food Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Obstacles Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Buffer"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let organisms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Organisms Buffer"),
        size: (NUM_ORGANISMS * 56) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sensory Buffer"),
        size: (NUM_ORGANISMS * 80) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let actions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Actions Buffer"),
        size: (NUM_ORGANISMS * 24) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let nn_weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NN Weights Buffer"),
        size: (NUM_ORGANISMS * 438 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Simple Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Simple Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: organisms_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: food_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: obstacles_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sensory_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: actions_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: nn_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: config_buffer.as_entire_binding() },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Simple Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Simple World Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Simple World Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    
    // Initialize
    let config_data = create_config_uniform(0);
    queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
    
    let mut food_data = vec![0.0f32; (WORLD_WIDTH * WORLD_HEIGHT) as usize];
    for i in 0..50 {
        let cx = (i * 73 % WORLD_WIDTH) as usize;
        let cy = (i * 97 % WORLD_HEIGHT) as usize;
        for dy in 0..25 {
            for dx in 0..25 {
                let x = (cx + dx).min((WORLD_WIDTH - 1) as usize);
                let y = (cy + dy).min((WORLD_HEIGHT - 1) as usize);
                food_data[y * WORLD_WIDTH as usize + x] = 5.0;
            }
        }
    }
    queue.write_buffer(&food_buffer, 0, bytemuck::cast_slice(&food_data));
    
    // Warm up
    for _ in 0..10 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((WORLD_WIDTH + 7) / 8, (WORLD_HEIGHT + 7) / 8, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_TICKS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((WORLD_WIDTH + 7) / 8, (WORLD_HEIGHT + 7) / 8, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    let elapsed = start.elapsed();
    
    println!("World shader (simple):    {:.1} ms total, {:.3} ms/tick", 
             elapsed.as_secs_f64() * 1000.0,
             elapsed.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    
    elapsed
}

async fn benchmark_sense_shader(device: &wgpu::Device, queue: &wgpu::Queue) -> Duration {
    // Minimal benchmark for sense shader
    let food_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Food Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Obstacles Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Buffer"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let organisms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Organisms Buffer"),
        size: (NUM_ORGANISMS * 56) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sensory Buffer"),
        size: (NUM_ORGANISMS * 80) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let actions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Actions Buffer"),
        size: (NUM_ORGANISMS * 24) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let nn_weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NN Weights Buffer"),
        size: (NUM_ORGANISMS * 438 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Sense Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sense Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: organisms_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: food_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: obstacles_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sensory_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: actions_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: nn_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: config_buffer.as_entire_binding() },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Sense Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sense Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/compute/shaders/sense.wgsl").into()),
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Sense Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    
    // Initialize
    let config_data = create_config_uniform(0);
    queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
    
    // Initialize organisms with valid positions
    let mut org_data = vec![0u8; (NUM_ORGANISMS * 56) as usize];
    for i in 0..NUM_ORGANISMS {
        let offset = (i * 56) as usize;
        let x = (i % WORLD_WIDTH) as f32;
        let y = (i / WORLD_WIDTH % WORLD_HEIGHT) as f32;
        org_data[offset..offset+4].copy_from_slice(&x.to_le_bytes());
        org_data[offset+4..offset+8].copy_from_slice(&y.to_le_bytes());
        org_data[offset+20..offset+24].copy_from_slice(&100.0f32.to_le_bytes()); // energy
        org_data[offset+28..offset+32].copy_from_slice(&1u32.to_le_bytes()); // flags = alive
    }
    queue.write_buffer(&organisms_buffer, 0, &org_data);
    
    // Warm up
    for _ in 0..10 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((NUM_ORGANISMS + 63) / 64, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_TICKS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((NUM_ORGANISMS + 63) / 64, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    let elapsed = start.elapsed();
    
    println!("Sense shader:             {:.1} ms total, {:.3} ms/tick", 
             elapsed.as_secs_f64() * 1000.0,
             elapsed.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    
    elapsed
}

async fn benchmark_think_shader(device: &wgpu::Device, queue: &wgpu::Queue) -> Duration {
    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Buffer"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let organisms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Organisms Buffer"),
        size: (NUM_ORGANISMS * 56) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let food_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Food Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Obstacles Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sensory Buffer"),
        size: (NUM_ORGANISMS * 80) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let actions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Actions Buffer"),
        size: (NUM_ORGANISMS * 24) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let nn_weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NN Weights Buffer"),
        size: (NUM_ORGANISMS * 438 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Think Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Think Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: organisms_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: food_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: obstacles_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sensory_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: actions_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: nn_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: config_buffer.as_entire_binding() },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Think Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Think Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/compute/shaders/think.wgsl").into()),
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Think Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    
    // Initialize
    let config_data = create_config_uniform(0);
    queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
    
    // Initialize organisms
    let mut org_data = vec![0u8; (NUM_ORGANISMS * 56) as usize];
    for i in 0..NUM_ORGANISMS {
        let offset = (i * 56) as usize;
        org_data[offset+20..offset+24].copy_from_slice(&100.0f32.to_le_bytes());
        org_data[offset+28..offset+32].copy_from_slice(&1u32.to_le_bytes());
        org_data[offset+32..offset+36].copy_from_slice(&i.to_le_bytes()); // genome_id
    }
    queue.write_buffer(&organisms_buffer, 0, &org_data);
    
    // Initialize NN weights with random values
    let weights: Vec<f32> = (0..(NUM_ORGANISMS * 438))
        .map(|i| (i as f32 * 0.001).sin() * 0.5)
        .collect();
    queue.write_buffer(&nn_weights_buffer, 0, bytemuck::cast_slice(&weights));
    
    // Warm up
    for _ in 0..10 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((NUM_ORGANISMS + 63) / 64, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_TICKS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((NUM_ORGANISMS + 63) / 64, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    let elapsed = start.elapsed();
    
    println!("Think shader:             {:.1} ms total, {:.3} ms/tick", 
             elapsed.as_secs_f64() * 1000.0,
             elapsed.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    
    elapsed
}

async fn benchmark_act_shader(device: &wgpu::Device, queue: &wgpu::Queue) -> Duration {
    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Buffer"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let organisms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Organisms Buffer"),
        size: (NUM_ORGANISMS * 56) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let food_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Food Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Obstacles Buffer"),
        size: (WORLD_WIDTH * WORLD_HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sensory Buffer"),
        size: (NUM_ORGANISMS * 80) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let actions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Actions Buffer"),
        size: (NUM_ORGANISMS * 24) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let nn_weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NN Weights Buffer"),
        size: (NUM_ORGANISMS * 438 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Act Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Act Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: organisms_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: food_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: obstacles_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sensory_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: actions_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: nn_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: config_buffer.as_entire_binding() },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Act Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Act Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/compute/shaders/act.wgsl").into()),
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Act Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    
    // Initialize
    let config_data = create_config_uniform(0);
    queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
    
    // Initialize organisms
    let mut org_data = vec![0u8; (NUM_ORGANISMS * 56) as usize];
    for i in 0..NUM_ORGANISMS {
        let offset = (i * 56) as usize;
        let x = ((i * 7) % WORLD_WIDTH) as f32 + 0.5;
        let y = ((i * 13) % WORLD_HEIGHT) as f32 + 0.5;
        org_data[offset..offset+4].copy_from_slice(&x.to_le_bytes());
        org_data[offset+4..offset+8].copy_from_slice(&y.to_le_bytes());
        org_data[offset+20..offset+24].copy_from_slice(&100.0f32.to_le_bytes());
        org_data[offset+28..offset+32].copy_from_slice(&1u32.to_le_bytes());
    }
    queue.write_buffer(&organisms_buffer, 0, &org_data);
    
    // Initialize food
    let food_data = vec![5.0f32; (WORLD_WIDTH * WORLD_HEIGHT) as usize];
    queue.write_buffer(&food_buffer, 0, bytemuck::cast_slice(&food_data));
    
    // Initialize actions
    let actions: Vec<f32> = (0..(NUM_ORGANISMS * 6))
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    queue.write_buffer(&actions_buffer, 0, bytemuck::cast_slice(&actions));
    
    // Warm up
    for _ in 0..10 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((NUM_ORGANISMS + 63) / 64, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_TICKS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((NUM_ORGANISMS + 63) / 64, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    let elapsed = start.elapsed();
    
    println!("Act shader:               {:.1} ms total, {:.3} ms/tick", 
             elapsed.as_secs_f64() * 1000.0,
             elapsed.as_secs_f64() * 1000.0 / NUM_TICKS as f64);
    
    elapsed
}

fn create_config_uniform(tick: u32) -> [u32; 36] {
    let mut data = [0u32; 36];
    data[0] = WORLD_WIDTH;
    data[1] = WORLD_HEIGHT;
    data[2] = NUM_ORGANISMS;
    data[3] = tick;
    data[4] = 80.0f32.to_bits();  // vision_range
    data[5] = 2.0944f32.to_bits(); // vision_fov (120 degrees)
    data[6] = 8; // vision_rays
    data[7] = 0; // pad
    data[8] = 150.0f32.to_bits();  // max_energy
    data[9] = 0.15f32.to_bits();   // passive_drain
    data[10] = 0.1f32.to_bits();   // movement_cost_forward
    data[11] = 0.02f32.to_bits();  // movement_cost_rotate
    data[12] = 3000; // max_age
    data[13] = 0.5f32.to_bits();   // crowding_factor
    data[14] = 1000; // max_organisms
    data[15] = 0.3f32.to_bits();   // age_drain_factor
    data[16] = 2.5f32.to_bits();   // max_speed
    data[17] = 0.25f32.to_bits();  // max_rotation
    data[18] = 3.0f32.to_bits();   // organism_radius
    data[19] = 0; // pad
    data[20] = 0.08f32.to_bits();  // food_growth_rate
    data[21] = 10.0f32.to_bits();  // food_max_per_cell
    data[22] = 8.0f32.to_bits();   // food_energy_value
    data[23] = 1.0f32.to_bits();   // food_effectiveness
    data[24] = 80.0f32.to_bits();  // reproduction_threshold
    data[25] = 0.3f32.to_bits();   // reproduction_signal_min
    data[26] = 50.0f32.to_bits();  // reproduction_cost
    data[27] = 100; // reproduction_min_age
    data
}
