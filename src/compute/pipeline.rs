//! GPU compute pipeline management

use anyhow::Result;
use crate::config::{SimulationConfig, SimUniform};
use crate::simulation::Simulation;
use super::buffers::GpuBuffers;

/// Main compute pipeline for simulation
#[allow(dead_code)]
pub struct ComputePipeline {
    // Pipelines
    sense_pipeline: wgpu::ComputePipeline,
    think_pipeline: wgpu::ComputePipeline,
    act_pipeline: wgpu::ComputePipeline,
    world_pipeline: wgpu::ComputePipeline,
    
    // Single bind group shared by all pipelines
    bind_group: wgpu::BindGroup,
    
    // Buffers
    pub buffers: GpuBuffers,
    
    // Layout
    bind_group_layout: wgpu::BindGroupLayout,
    
    // Diagnostics
    readback_count: u32,
}

impl ComputePipeline {
    pub fn new(device: &wgpu::Device, config: &SimulationConfig, simulation: &Simulation) -> Result<Self> {
        let buffers = GpuBuffers::new(device, config, simulation);
        
        // Create bind group layout for all compute shaders
        // Using 6 storage buffers + 1 uniform to stay under the 8 storage buffer limit
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Organisms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Food
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Obstacles
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sensory
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Actions
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Combined NN Weights (weights_l1 + biases_l1 + weights_l2 + biases_l2)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Config uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Biomes
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffers.organisms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffers.food.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buffers.obstacles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buffers.sensory.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buffers.actions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buffers.nn_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buffers.config_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buffers.biomes.as_entire_binding() },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Load shaders
        let sense_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sense Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sense.wgsl").into()),
        });
        
        let think_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Think Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/think.wgsl").into()),
        });
        
        let act_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Act Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/act.wgsl").into()),
        });
        
        let world_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("World Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/world.wgsl").into()),
        });
        
        // Create pipelines
        let sense_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sense Pipeline"),
            layout: Some(&pipeline_layout),
            module: &sense_shader,
            entry_point: "main",
        });
        
        let think_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Think Pipeline"),
            layout: Some(&pipeline_layout),
            module: &think_shader,
            entry_point: "main",
        });
        
        let act_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Act Pipeline"),
            layout: Some(&pipeline_layout),
            module: &act_shader,
            entry_point: "main",
        });
        
        let world_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("World Pipeline"),
            layout: Some(&pipeline_layout),
            module: &world_shader,
            entry_point: "main",
        });
        
        Ok(Self {
            sense_pipeline,
            think_pipeline,
            act_pipeline,
            world_pipeline,
            bind_group,
            buffers,
            bind_group_layout,
            readback_count: 0,
        })
    }
    
    
    pub fn read_gpu_state(
        &mut self,
        device: &wgpu::Device,
        simulation: &mut Simulation,
        config: &SimulationConfig,
        tick: u32,
    ) -> f32 {
        let readback_start = std::time::Instant::now();
        
        let do_readback = tick % config.system.readback_interval == 0;
        
        // First, try to get any pending readback from previous frame
        if do_readback {
            if let Some(gpu_organisms) = self.buffers.try_get_organism_data(device) {
                self.readback_count += 1;
                let dead_species_ids = simulation.organisms.update_from_gpu_buffer(&gpu_organisms);
                
                // Track deaths in species manager
                for species_id in dead_species_ids {
                    simulation.species_manager.on_organism_death(species_id);
                }
            }

            // Try to get food data if pending
            if let Some(food_data) = self.buffers.try_get_food_data(device, config.world.width, config.world.height) {
                simulation.world.food = food_data;
            }
        }

        // Keep GPU readback sizes aligned with the current organism count after any deaths
        self.buffers.set_organism_count(simulation.organism_count());
        
        // Log periodic tick info with more detail
        if tick % config.system.diagnostic_interval == 0 {
            // Sample reproduce signals
            let repro_signals: Vec<f32> = simulation.organisms.iter()
                .filter(|o| o.is_alive())
                .take(10)
                .map(|o| o.reproduce_signal)
                .collect();
            
            // Count organisms wanting to reproduce
            let repro_ready: usize = simulation.organisms.iter()
                .filter(|o| o.is_alive() && o.reproduce_signal > 0.3 && o.energy >= 100.0 && o.age >= 100)
                .count();
            
            // Sample ages
            let ages: Vec<u32> = simulation.organisms.iter()
                .filter(|o| o.is_alive())
                .take(5)
                .map(|o| o.age)
                .collect();
            
            log::info!(
                "Tick {}: pop={}, avg_energy={:.1}, max_gen={}, repro_ready={}, ages={:?}, repro_signals={:?}",
                tick, 
                simulation.organisms.count(), 
                simulation.avg_energy(),
                simulation.max_generation(),
                repro_ready,
                ages,
                repro_signals
            );
        }

        readback_start.elapsed().as_secs_f32() * 1000.0
    }
    
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        simulation: &Simulation,
        config: &SimulationConfig,
        tick: u32,
    ) -> ExecutionTiming {
        let mut timing = ExecutionTiming::default();
        let total_start = std::time::Instant::now();
        
        let do_readback = tick % config.system.readback_interval == 0;
        let do_food_readback = tick % config.system.food_readback_interval == 0;
        
        // Update uniform buffer
        let upload_start = std::time::Instant::now();
        let uniform = SimUniform::from_config(config, simulation.organism_count(), tick);
        self.buffers.update_config(queue, &uniform);
        timing.upload_ms = upload_start.elapsed().as_secs_f32() * 1000.0;
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        
        // Ensure buffers know the correct count for dispatch
        // (Note: read_gpu_state called set_organism_count, but simulation might have changed in between)
        // We can't easily access mutable buffers here if we only have immutable simulation
        // But buffers is self.buffers (mutable).
        // self.buffers.set_organism_count(simulation.organism_count());
        
        let workgroup_count = (simulation.organism_count() + 63) / 64;
        let world_workgroups_x = (config.world.width + 7) / 8;
        let world_workgroups_y = (config.world.height + 7) / 8;
        
        // Only dispatch if we have organisms
        if workgroup_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Simulation Compute Pass"),
                timestamp_writes: None,
            });
            
            // Sense pass
            pass.set_pipeline(&self.sense_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // Think pass
            pass.set_pipeline(&self.think_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // Act pass
            pass.set_pipeline(&self.act_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // World pass (food growth, world updates)
            pass.set_pipeline(&self.world_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(world_workgroups_x, world_workgroups_y, 1);
        }
        
        // Only initiate readback at intervals (reduces GPU stalls)
        if do_readback {
            self.buffers.start_readback(&mut encoder);
        }
        
        if do_food_readback {
            self.buffers.start_food_readback(&mut encoder, config.world.width, config.world.height);
        }
        
        let submit_start = std::time::Instant::now();
        queue.submit(std::iter::once(encoder.finish()));
        timing.submit_ms = submit_start.elapsed().as_secs_f32() * 1000.0;
        
        timing.total_ms = total_start.elapsed().as_secs_f32() * 1000.0;
        timing
    }
}

/// Timing information for profiling
#[derive(Debug, Default, Clone, Copy)]
pub struct ExecutionTiming {
    pub readback_ms: f32,
    pub upload_ms: f32,
    pub submit_ms: f32,
    pub total_ms: f32,
}
