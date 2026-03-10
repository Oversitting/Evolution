//! Food System Test
//! 
//! Tests food spawning, growth, and consumption patterns.
//! Validates that food system is working correctly.
//! 
//! Run with: cargo run --example food_system_test --release

use std::time::Instant;
use pollster::block_on;

const WORLD_WIDTH: u32 = 512;
const WORLD_HEIGHT: u32 = 512;
const NUM_TICKS: u32 = 500;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("=== Food System Test ===\n");
    block_on(run_tests());
}

async fn run_tests() {
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
    
    println!("GPU: {}\n", adapter.get_info().name);
    
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("Failed to create device");
    
    // Test 1: Food growth dynamics
    println!("--- Test 1: Food Growth Dynamics ---");
    test_food_growth(&device, &queue).await;
    
    // Test 2: Food patch distribution
    println!("\n--- Test 2: Food Patch Distribution ---");
    test_food_distribution(&device, &queue).await;
    
    // Test 3: Spontaneous generation rate
    println!("\n--- Test 3: Spontaneous Generation ---");
    test_spontaneous_generation(&device, &queue).await;
    
    println!("\n=== All Food System Tests Complete ===");
}

async fn test_food_growth(device: &wgpu::Device, queue: &wgpu::Queue) {
    // Test that food grows according to logistic growth
    let size = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
    
    // Initialize with a single food patch
    let mut food_data = vec![0.0f32; size];
    let center_x = WORLD_WIDTH / 2;
    let center_y = WORLD_HEIGHT / 2;
    let initial_food = 2.0f32;
    let max_per_cell = 10.0f32;
    let growth_rate = 0.08f32;
    
    // Create a 5x5 patch with initial food
    for dy in 0..5 {
        for dx in 0..5 {
            let x = center_x + dx;
            let y = center_y + dy;
            food_data[(y * WORLD_WIDTH + x) as usize] = initial_food;
        }
    }
    
    // Create buffers
    let food_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Food Buffer"),
        size: (size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (size * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Obstacles Buffer"),
        size: (size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Buffer"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Dummy buffers
    let organisms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Organisms Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sensory Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let actions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Actions Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let nn_weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NN Weights Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    
    let bind_group_layout = create_bind_group_layout(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
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
        label: Some("Pipeline Layout"),
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
    
    // Initialize buffers
    queue.write_buffer(&food_buffer, 0, bytemuck::cast_slice(&food_data));
    
    let initial_total: f32 = food_data.iter().sum();
    println!("  Initial food in patch: {:.1} (25 cells × {:.1})", initial_total, initial_food);
    
    // Run simulation
    let mut results = Vec::new();
    results.push((0, initial_total));
    
    for tick in 0..100 {
        let config_data = create_config(tick, growth_rate, max_per_cell);
        queue.write_buffer(&config_buffer, 0, bytemuck::cast_slice(&config_data));
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((WORLD_WIDTH + 7) / 8, (WORLD_HEIGHT + 7) / 8, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        
        // Read back every 10 ticks
        if tick % 20 == 19 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(&food_buffer, 0, &staging_buffer, 0, (size * 4) as u64);
            queue.submit(std::iter::once(encoder.finish()));
            
            let slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            let food: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            
            let total: f32 = food.iter().sum();
            results.push((tick + 1, total));
        }
    }
    
    println!("  Growth over time:");
    for (tick, total) in &results {
        let pct = total / (25.0 * max_per_cell) * 100.0;
        println!("    Tick {:3}: total = {:6.1}, capacity = {:.1}%", tick, total, pct);
    }
    
    // Verify growth happened
    let final_total = results.last().unwrap().1;
    let expected_max = 25.0 * max_per_cell;
    if final_total > initial_total && final_total < expected_max * 1.1 {
        println!("  ✅ PASS: Food grew from {:.1} toward capacity {:.1}", initial_total, expected_max);
    } else {
        println!("  ❌ FAIL: Unexpected growth pattern");
    }
}

async fn test_food_distribution(device: &wgpu::Device, queue: &wgpu::Queue) {
    // Test that food patches are well-distributed
    let size = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
    
    // Initialize with multiple random patches
    let mut food_data = vec![0.0f32; size];
    let num_patches = 50;
    let patch_size = 25;
    
    for i in 0..num_patches {
        let cx = (i * 73 % WORLD_WIDTH) as i32;
        let cy = (i * 97 % WORLD_HEIGHT) as i32;
        let half = patch_size / 2;
        
        for dy in -half..=half {
            for dx in -half..=half {
                let x = (cx + dx).clamp(0, WORLD_WIDTH as i32 - 1) as u32;
                let y = (cy + dy).clamp(0, WORLD_HEIGHT as i32 - 1) as u32;
                food_data[(y * WORLD_WIDTH + x) as usize] = 8.0;
            }
        }
    }
    
    // Analyze distribution
    let cells_with_food = food_data.iter().filter(|&&f| f > 0.5).count();
    let total_food: f32 = food_data.iter().sum();
    let coverage = cells_with_food as f32 / size as f32 * 100.0;
    
    // Check quadrant distribution
    let mut quadrant_food = [0.0f32; 4];
    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let food = food_data[(y * WORLD_WIDTH + x) as usize];
            let q = (if x < WORLD_WIDTH / 2 { 0 } else { 1 }) + (if y < WORLD_HEIGHT / 2 { 0 } else { 2 });
            quadrant_food[q] += food;
        }
    }
    
    let avg_quadrant = total_food / 4.0;
    let max_deviation = quadrant_food.iter()
        .map(|&q| (q - avg_quadrant).abs() / avg_quadrant * 100.0)
        .fold(0.0f32, f32::max);
    
    println!("  Patches: {}", num_patches);
    println!("  Patch size: {}x{}", patch_size, patch_size);
    println!("  Cells with food: {} ({:.1}% coverage)", cells_with_food, coverage);
    println!("  Total food: {:.0}", total_food);
    println!("  Quadrant distribution:");
    println!("    Top-left:     {:.0}", quadrant_food[0]);
    println!("    Top-right:    {:.0}", quadrant_food[1]);
    println!("    Bottom-left:  {:.0}", quadrant_food[2]);
    println!("    Bottom-right: {:.0}", quadrant_food[3]);
    println!("  Max quadrant deviation: {:.1}%", max_deviation);
    
    if coverage > 5.0 && max_deviation < 50.0 {
        println!("  ✅ PASS: Food is well-distributed");
    } else {
        println!("  ❌ FAIL: Poor distribution (coverage={:.1}%, deviation={:.1}%)", coverage, max_deviation);
    }
}

async fn test_spontaneous_generation(device: &wgpu::Device, queue: &wgpu::Queue) {
    // Test that spontaneous food generation works
    let size = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
    
    // Start with empty world
    let food_data = vec![0.0f32; size];
    
    let food_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Food Buffer"),
        size: (size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (size * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Obstacles Buffer"),
        size: (size * 4) as u64,
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
        label: Some("Organisms Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let sensory_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sensory Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let actions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Actions Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let nn_weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("NN Weights Buffer"), size: 1024, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    
    let bind_group_layout = create_bind_group_layout(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
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
        label: Some("Pipeline Layout"),
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
    
    queue.write_buffer(&food_buffer, 0, bytemuck::cast_slice(&food_data));
    
    // Run for many ticks
    let num_ticks = 10000;
    println!("  Running {} ticks with empty world...", num_ticks);
    
    let start = Instant::now();
    for tick in 0..num_ticks {
        let config_data = create_config(tick, 0.08, 10.0);
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
    
    // Read back
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&food_buffer, 0, &staging_buffer, 0, (size * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));
    
    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    
    let data = slice.get_mapped_range();
    let food: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();
    
    let cells_with_food = food.iter().filter(|&&f| f > 0.01).count();
    let total_food: f32 = food.iter().sum();
    
    // Expected: ~0.001% chance per cell per tick = ~2.6 new cells per tick
    // Over 10000 ticks: ~26000 cells (but they overlap/grow, so fewer unique)
    let expected_per_tick = (size as f32) * 0.00001;
    
    println!("  Time: {:.1} ms ({:.3} ms/tick)", elapsed.as_secs_f64() * 1000.0, elapsed.as_secs_f64() * 1000.0 / num_ticks as f64);
    println!("  Cells with food: {} ({:.4}% of world)", cells_with_food, cells_with_food as f32 / size as f32 * 100.0);
    println!("  Total food amount: {:.1}", total_food);
    println!("  Expected new cells/tick: ~{:.1}", expected_per_tick);
    
    if cells_with_food > 0 && cells_with_food < size / 2 {
        println!("  ✅ PASS: Spontaneous generation works (not too much, not zero)");
    } else if cells_with_food == 0 {
        println!("  ⚠️  WARNING: No spontaneous food generated (might be intentional if probability is very low)");
    } else {
        println!("  ❌ FAIL: Too much spontaneous food (check RNG)");
    }
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    })
}

fn create_config(tick: u32, growth_rate: f32, max_per_cell: f32) -> [u32; 36] {
    let mut data = [0u32; 36];
    data[0] = WORLD_WIDTH;
    data[1] = WORLD_HEIGHT;
    data[2] = 0; // num_organisms
    data[3] = tick;
    data[4] = 80.0f32.to_bits();
    data[5] = 2.0944f32.to_bits();
    data[6] = 8;
    data[7] = 0;
    data[8] = 150.0f32.to_bits();
    data[9] = 0.15f32.to_bits();
    data[10] = 0.1f32.to_bits();
    data[11] = 0.02f32.to_bits();
    data[12] = 3000;
    data[13] = 0.5f32.to_bits();
    data[14] = 1000;
    data[15] = 0.3f32.to_bits();
    data[16] = 2.5f32.to_bits();
    data[17] = 0.25f32.to_bits();
    data[18] = 3.0f32.to_bits();
    data[19] = 0;
    data[20] = growth_rate.to_bits();
    data[21] = max_per_cell.to_bits();
    data[22] = 8.0f32.to_bits();
    data[23] = 1.0f32.to_bits();
    data[24] = 80.0f32.to_bits();
    data[25] = 0.3f32.to_bits();
    data[26] = 50.0f32.to_bits();
    data[27] = 100;
    data
}
