//! GPU buffer management

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;
use crate::config::SimulationConfig;
use crate::simulation::{Simulation, OrganismPool};
use crate::simulation::organism::OrganismGpu;
use crate::simulation::genome::{INPUT_DIM, OUTPUT_DIM};

/// All GPU buffers for the simulation
pub struct GpuBuffers {
    // Organism state
    pub organisms: wgpu::Buffer,
    
    // Staging buffer for readback (double buffered)
    pub organisms_staging: [wgpu::Buffer; 2],
    pub staging_index: usize,
    pub readback_pending: bool,
    pub organism_count: u32,
    
    // Combined neural network weights
    pub nn_weights: wgpu::Buffer,
    
    // IO buffers
    pub sensory: wgpu::Buffer,
    pub actions: wgpu::Buffer,
    
    // World
    pub food: wgpu::Buffer,
    pub food_staging: wgpu::Buffer,
    pub food_readback_pending: bool,
    pub obstacles: wgpu::Buffer,
    pub biomes: wgpu::Buffer,
    
    // Uniform
    pub config_uniform: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(device: &wgpu::Device, config: &SimulationConfig, simulation: &Simulation) -> Self {
        let max_organisms = config.population.max_organisms;
        
        // Calculate buffer sizes
        let sensory_size = max_organisms as usize * INPUT_DIM * 4;
        let actions_size = max_organisms as usize * OUTPUT_DIM * 4;
        let organism_buffer_size = max_organisms as usize * std::mem::size_of::<OrganismGpu>();
        
        // Create organism buffer - pad to max_organisms capacity
        let mut organism_data = simulation.organisms.to_gpu_buffer();
        organism_data.resize(max_organisms as usize, OrganismGpu::zeroed());
        let organisms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organisms Buffer"),
            contents: bytemuck::cast_slice(&organism_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        
        // Staging buffers for GPU->CPU readback (double buffered for async)
        let organisms_staging_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organisms Staging Buffer 0"),
            size: organism_buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let organisms_staging_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organisms Staging Buffer 1"),
            size: organism_buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create combined neural network weights buffer - preallocate for max genomes
        let _nn_weights_size = max_organisms as usize * crate::simulation::genome::TOTAL_PARAMS * 4;
        let mut nn_weights_data = simulation.genomes.nn_weights_buffer();
        nn_weights_data.resize(max_organisms as usize * crate::simulation::genome::TOTAL_PARAMS, 0.0);
        let nn_weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NN Weights Buffer"),
            contents: bytemuck::cast_slice(&nn_weights_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create IO buffers
        let sensory = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sensory Buffer"),
            size: sensory_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let actions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Actions Buffer"),
            size: actions_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create world buffers
        let food = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Food Buffer"),
            contents: bytemuck::cast_slice(&simulation.world.food),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let food_size = config.world.width as u64 * config.world.height as u64 * 4;
        let food_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Food Staging Buffer"),
            size: food_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let obstacles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Obstacles Buffer"),
            contents: bytemuck::cast_slice(&simulation.world.obstacles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Biomes buffer (u32 to match GPU alignment - shader reads as u32 array)
        let biomes_u32: Vec<u32> = simulation.world.biomes.iter().map(|&b| b as u32).collect();
        let biomes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Biomes Buffer"),
            contents: bytemuck::cast_slice(&biomes_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create uniform buffer
        let config_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config Uniform Buffer"),
            size: std::mem::size_of::<crate::config::SimUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            organisms,
            organisms_staging: [organisms_staging_0, organisms_staging_1],
            staging_index: 0,
            readback_pending: false,
            organism_count: simulation.organisms.count(),
            nn_weights,
            sensory,
            actions,
            food,
            food_staging,
            food_readback_pending: false,
            obstacles,
            biomes,
            config_uniform,
        }
    }
    
    #[allow(dead_code)]
    pub fn update_organisms(&mut self, queue: &wgpu::Queue, organisms: &OrganismPool) {
        self.organism_count = organisms.count();
        let data = organisms.to_gpu_buffer();
        queue.write_buffer(&self.organisms, 0, bytemuck::cast_slice(&data));
    }
    
    /// Update a single organism slot (for newly spawned organisms)
    pub fn update_organism_at(&mut self, queue: &wgpu::Queue, index: u32, organism: &crate::simulation::organism::OrganismGpu) {
        let offset = index as u64 * std::mem::size_of::<crate::simulation::organism::OrganismGpu>() as u64;
        queue.write_buffer(&self.organisms, offset, bytemuck::bytes_of(organism));
    }
    
    /// Sync organism count for uniform
    pub fn set_organism_count(&mut self, count: u32) {
        self.organism_count = count;
    }
    
    /// Update a single genome's neural network weights in the GPU buffer
    pub fn update_nn_weights_for_genome(&self, queue: &wgpu::Queue, genome_id: u32, weights: &[f32]) {
        let offset = genome_id as u64 * crate::simulation::genome::TOTAL_PARAMS as u64 * 4; // 4 bytes per f32
        queue.write_buffer(&self.nn_weights, offset, bytemuck::cast_slice(weights));
    }
    
    /// Update all neural network weights from GenomePool
    #[allow(dead_code)]
    pub fn update_all_nn_weights(&self, queue: &wgpu::Queue, genomes: &crate::simulation::genome::GenomePool) {
        let data = genomes.nn_weights_buffer();
        queue.write_buffer(&self.nn_weights, 0, bytemuck::cast_slice(&data));
    }
    
    pub fn update_config(&self, queue: &wgpu::Queue, uniform: &crate::config::SimUniform) {
        queue.write_buffer(&self.config_uniform, 0, bytemuck::bytes_of(uniform));
    }
    
    /// Update food grid on GPU
    pub fn update_food(&self, queue: &wgpu::Queue, food: &[f32]) {
        queue.write_buffer(&self.food, 0, bytemuck::cast_slice(food));
    }
    
    /// Update all neural network weights from flat buffer
    pub fn update_nn_weights(&self, queue: &wgpu::Queue, weights: &[f32]) {
        queue.write_buffer(&self.nn_weights, 0, bytemuck::cast_slice(weights));
    }
    
    /// Initiate async copy from GPU to staging buffer
    pub fn start_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let size = self.organism_count as u64 * std::mem::size_of::<OrganismGpu>() as u64;
        if size == 0 {
            return;
        }
        encoder.copy_buffer_to_buffer(
            &self.organisms, 0,
            &self.organisms_staging[self.staging_index], 0,
            size
        );
        self.readback_pending = true;
    }
    
    /// Get readback data - waits for completion
    pub fn try_get_organism_data(&mut self, device: &wgpu::Device) -> Option<Vec<OrganismGpu>> {
        if !self.readback_pending {
            return None;
        }
        
        let size = self.organism_count as usize * std::mem::size_of::<OrganismGpu>();
        if size == 0 {
            self.readback_pending = false;
            return Some(Vec::new());
        }
        
        let staging = &self.organisms_staging[self.staging_index];
        let buffer_slice = staging.slice(0..size as u64);
        
        // Map the buffer
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        // Poll until complete
        device.poll(wgpu::Maintain::Wait);
        
        match rx.recv() {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let organisms: Vec<OrganismGpu> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging.unmap();
                
                // Swap staging buffer for next frame
                self.staging_index = 1 - self.staging_index;
                self.readback_pending = false;
                
                Some(organisms)
            }
            _ => {
                self.readback_pending = false;
                None
            }
        }
    }

    /// Initiate async copy of food from GPU to staging buffer
    pub fn start_food_readback(&mut self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let size = width as u64 * height as u64 * 4;
        encoder.copy_buffer_to_buffer(
            &self.food, 0,
            &self.food_staging, 0,
            size
        );
        self.food_readback_pending = true;
    }

    /// Get food readback data - waits for completion
    pub fn try_get_food_data(&mut self, device: &wgpu::Device, width: u32, height: u32) -> Option<Vec<f32>> {
        if !self.food_readback_pending {
            return None;
        }

        let size = width as u64 * height as u64 * 4;
        let buffer_slice = self.food_staging.slice(0..size);

        // Map the buffer
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll until complete
        // Note: this might block if try_get_organism_data hasn't already polled
        device.poll(wgpu::Maintain::Wait);

        match rx.recv() {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let food: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                self.food_staging.unmap();
                self.food_readback_pending = false;
                Some(food)
            }
            _ => {
                self.food_readback_pending = false;
                None
            }
        }
    }
}
