//! Organism data structures

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

/// GPU-compatible organism data (64 bytes with morphology)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OrganismGpu {
    // Position & Movement (16 bytes)
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    
    // Rotation (4 bytes)
    pub rotation: f32,
    
    // State (12 bytes)
    pub energy: f32,
    pub age: u32,
    pub flags: u32, // Bit 0: alive
    
    // Genome reference (8 bytes)
    pub genome_id: u32,
    pub generation: u32,
    
    // Stats (8 bytes)
    pub offspring_count: u32,
    pub parent_id: u32,
    
    // Neural network outputs for CPU reproduction check (8 bytes)
    pub reproduce_signal: f32,
    pub species_id: u32, // Species cluster ID for coloring
    
    // Morphology traits (16 bytes) - copied from genome for GPU access
    pub morph_size: f32,
    pub morph_speed_mult: f32,
    pub morph_vision_mult: f32,
    pub morph_metabolism: f32,
}

impl OrganismGpu {
    pub fn is_alive(&self) -> bool {
        (self.flags & 1) != 0
    }
}

/// CPU-side organism with additional state
#[derive(Clone, Debug)]
pub struct Organism {
    pub position: Vec2,
    pub velocity: Vec2,
    pub rotation: f32,
    pub energy: f32,
    pub age: u32,
    pub alive: bool,
    pub genome_id: u32,
    pub generation: u32,
    pub offspring_count: u32,
    pub parent_id: u32,
    pub cooldown: u32,
    pub reproduce_signal: f32,
    pub species_id: u32,
    // Morphology traits (cached from genome)
    pub morph_size: f32,
    pub morph_speed_mult: f32,
    pub morph_vision_mult: f32,
    pub morph_metabolism: f32,
}

impl Organism {
    /// Create a new organism with deterministic rotation from provided RNG
    pub fn new<R: rand::Rng>(position: Vec2, energy: f32, genome_id: u32, generation: u32, rng: &mut R) -> Self {
        Self {
            position,
            velocity: Vec2::ZERO,
            rotation: rng.gen::<f32>() * std::f32::consts::TAU,
            energy,
            age: 0,
            alive: true,
            genome_id,
            generation,
            offspring_count: 0,
            parent_id: u32::MAX,
            cooldown: 0,
            reproduce_signal: 0.0,
            species_id: 0,
            // Default morphology (will be set from genome after spawn)
            morph_size: 1.0,
            morph_speed_mult: 1.0,
            morph_vision_mult: 1.0,
            morph_metabolism: 1.0,
        }
    }
    
    /// Set morphology traits from genome
    pub fn set_morphology(&mut self, size: f32, speed_mult: f32, vision_mult: f32, metabolism: f32) {
        self.morph_size = size;
        self.morph_speed_mult = speed_mult;
        self.morph_vision_mult = vision_mult;
        self.morph_metabolism = metabolism;
    }
    
    pub fn is_alive(&self) -> bool {
        self.alive && self.energy > 0.0
    }
    
    pub fn to_gpu(&self) -> OrganismGpu {
        OrganismGpu {
            position: self.position.into(),
            velocity: self.velocity.into(),
            rotation: self.rotation,
            energy: self.energy,
            age: self.age,
            flags: if self.alive { 1 } else { 0 },
            genome_id: self.genome_id,
            generation: self.generation,
            offspring_count: self.offspring_count,
            parent_id: self.parent_id,
            reproduce_signal: self.reproduce_signal,
            species_id: self.species_id,
            morph_size: self.morph_size,
            morph_speed_mult: self.morph_speed_mult,
            morph_vision_mult: self.morph_vision_mult,
            morph_metabolism: self.morph_metabolism,
        }
    }
    
    pub fn update_from_gpu(&mut self, gpu: &OrganismGpu) {
        self.position = Vec2::from(gpu.position);
        self.velocity = Vec2::from(gpu.velocity);
        self.rotation = gpu.rotation;
        self.energy = gpu.energy;
        self.age = gpu.age;
        self.alive = gpu.is_alive();
        self.reproduce_signal = gpu.reproduce_signal;
        self.species_id = gpu.species_id;
        // Note: morphology is stable, no need to update from GPU
    }
}

/// Pool of organisms with free list management
pub struct OrganismPool {
    organisms: Vec<Organism>,
    free_list: Vec<u32>,
    count: u32,
    max_size: u32,
}

impl OrganismPool {
    pub fn new(max_size: u32) -> Self {
        Self {
            organisms: Vec::with_capacity(max_size as usize),
            free_list: Vec::new(),
            count: 0,
            max_size,
        }
    }
    
    pub fn spawn<R: rand::Rng>(&mut self, position: Vec2, energy: f32, genome_id: u32, generation: u32, rng: &mut R) -> Option<u32> {
        if self.count >= self.max_size {
            return None;
        }
        
        let organism = Organism::new(position, energy, genome_id, generation, rng);
        
        let id = if let Some(id) = self.free_list.pop() {
            self.organisms[id as usize] = organism;
            id
        } else {
            // Can only push if we haven't reached max_size
            if self.organisms.len() >= self.max_size as usize {
                return None; // No room for new slots
            }
            let id = self.organisms.len() as u32;
            self.organisms.push(organism);
            id
        };
        
        self.count += 1;
        Some(id)
    }
    
    pub fn count(&self) -> u32 {
        self.count
    }
    
    pub fn get(&self, id: u32) -> Option<&Organism> {
        self.organisms.get(id as usize)
    }
    
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Organism> {
        self.organisms.get_mut(id as usize)
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &Organism> {
        self.organisms.iter()
    }
    
    pub fn iter_mut_indexed(&mut self) -> impl Iterator<Item = (u32, &mut Organism)> {
        self.organisms.iter_mut().enumerate().map(|(i, o)| (i as u32, o))
    }
    
    pub fn to_gpu_buffer(&self) -> Vec<OrganismGpu> {
        self.organisms.iter().map(|o| o.to_gpu()).collect()
    }
    
    /// Update organisms from GPU data. Returns Vec of species_ids for organisms that died.
    pub fn update_from_gpu_buffer(&mut self, buffer: &[OrganismGpu]) -> Vec<u32> {
        let mut alive_count = 0u32;
        let mut dead_indices = Vec::new();
        let mut dead_species_ids = Vec::new();
        
        for (idx, (org, gpu)) in self.organisms.iter_mut().zip(buffer.iter()).enumerate() {
            let was_alive = org.alive;
            let species_id = org.species_id; // Capture before update
            org.update_from_gpu(gpu);
            
            // Track deaths from GPU (energy depleted)
            if was_alive && !org.alive {
                dead_indices.push(idx as u32);
                dead_species_ids.push(species_id);
            }
            
            if org.alive {
                alive_count += 1;
            }
        }
        
        // Add dead indices to free list
        self.free_list.extend(dead_indices);
        self.count = alive_count;
        
        dead_species_ids
    }
    
    pub fn cleanup_dead(&mut self, _genomes: &mut super::GenomePool) {
        // Note: Deaths are primarily tracked via update_from_gpu_buffer().
        // This function only handles cooldown ticks and any edge cases.
        // We don't add to free_list here to avoid duplicate entries.
        for (_id, org) in self.organisms.iter_mut().enumerate() {
            // Tick down cooldown
            if org.cooldown > 0 {
                org.cooldown -= 1;
            }
        }
    }
    
    /// Restore an organism during loading (clears and rebuilds pool)
    pub fn restore(&mut self, org: Organism) {
        // Check bounds before adding
        if self.organisms.len() >= self.max_size as usize {
            log::warn!("Cannot restore organism: pool at max capacity {}", self.max_size);
            return;
        }
        if org.alive {
            self.count += 1;
        }
        self.organisms.push(org);
    }
}
