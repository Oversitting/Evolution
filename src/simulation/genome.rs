//! Genome (neural network weights) management

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use crate::config::MorphologyConfig;

/// Neural network dimensions
pub const INPUT_DIM: usize = 20;  // 8 rays * 2 + 4 internal
pub const HIDDEN_DIM: usize = 16;
pub const OUTPUT_DIM: usize = 6;

/// Weights per genome
pub const WEIGHTS_L1: usize = INPUT_DIM * HIDDEN_DIM;  // 320
pub const BIASES_L1: usize = HIDDEN_DIM;               // 16
pub const WEIGHTS_L2: usize = HIDDEN_DIM * OUTPUT_DIM; // 96
pub const BIASES_L2: usize = OUTPUT_DIM;               // 6
pub const TOTAL_PARAMS: usize = WEIGHTS_L1 + BIASES_L1 + WEIGHTS_L2 + BIASES_L2; // 438

/// Morphology trait count (size, speed_mult, vision_mult, metabolism)
pub const MORPH_TRAITS: usize = 4;

/// Morphology traits - evolvable physical characteristics
#[derive(Clone, Copy, Debug, Default)]
pub struct MorphTraits {
    /// Size multiplier (affects rendering, collision, energy capacity)
    /// Range: [min_size, max_size] from config
    pub size: f32,
    /// Speed multiplier (affects max_speed)
    /// Range: [min_speed_mult, max_speed_mult]
    pub speed_mult: f32,
    /// Vision range multiplier
    /// Range: [min_vision_mult, max_vision_mult]
    pub vision_mult: f32,
    /// Metabolism efficiency (lower passive drain when higher)
    /// Range: [min_metabolism, max_metabolism]
    pub metabolism: f32,
}

impl MorphTraits {
    /// Create random morphology traits
    pub fn new_random<R: Rng>(rng: &mut R, config: &MorphologyConfig) -> Self {
        if !config.enabled {
            return Self::default_traits();
        }
        Self {
            size: rng.gen_range(config.min_size..=config.max_size),
            speed_mult: rng.gen_range(config.min_speed_mult..=config.max_speed_mult),
            vision_mult: rng.gen_range(config.min_vision_mult..=config.max_vision_mult),
            metabolism: rng.gen_range(config.min_metabolism..=config.max_metabolism),
        }
    }
    
    /// Default traits (all 1.0 = baseline)
    pub fn default_traits() -> Self {
        Self {
            size: 1.0,
            speed_mult: 1.0,
            vision_mult: 1.0,
            metabolism: 1.0,
        }
    }
    
    /// Mutate traits
    pub fn mutate<R: Rng>(&mut self, config: &MorphologyConfig, rng: &mut R) {
        if !config.enabled {
            return;
        }
        let normal = Normal::new(0.0, config.mutation_strength as f64).unwrap();
        
        if rng.gen::<f32>() < config.mutation_rate {
            self.size = (self.size + normal.sample(rng) as f32)
                .clamp(config.min_size, config.max_size);
        }
        if rng.gen::<f32>() < config.mutation_rate {
            self.speed_mult = (self.speed_mult + normal.sample(rng) as f32)
                .clamp(config.min_speed_mult, config.max_speed_mult);
        }
        if rng.gen::<f32>() < config.mutation_rate {
            self.vision_mult = (self.vision_mult + normal.sample(rng) as f32)
                .clamp(config.min_vision_mult, config.max_vision_mult);
        }
        if rng.gen::<f32>() < config.mutation_rate {
            self.metabolism = (self.metabolism + normal.sample(rng) as f32)
                .clamp(config.min_metabolism, config.max_metabolism);
        }
    }
    
    /// Convert to flat array for GPU upload
    pub fn to_array(&self) -> [f32; MORPH_TRAITS] {
        [self.size, self.speed_mult, self.vision_mult, self.metabolism]
    }
    
    /// Create from flat array
    pub fn from_array(arr: [f32; MORPH_TRAITS]) -> Self {
        Self {
            size: arr[0],
            speed_mult: arr[1],
            vision_mult: arr[2],
            metabolism: arr[3],
        }
    }
}

/// A single genome's neural network weights
#[derive(Clone, Debug)]
pub struct Genome {
    pub weights_l1: Vec<f32>,  // [INPUT_DIM, HIDDEN_DIM]
    pub biases_l1: Vec<f32>,   // [HIDDEN_DIM]
    pub weights_l2: Vec<f32>,  // [HIDDEN_DIM, OUTPUT_DIM]
    pub biases_l2: Vec<f32>,   // [OUTPUT_DIM]
    pub morphology: MorphTraits,
    #[allow(dead_code)]
    pub alive: bool,
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            weights_l1: vec![0.0; WEIGHTS_L1],
            biases_l1: vec![0.0; BIASES_L1],
            weights_l2: vec![0.0; WEIGHTS_L2],
            biases_l2: vec![0.0; BIASES_L2],
            morphology: MorphTraits::default_traits(),
            alive: false,
        }
    }
}

impl Genome {
    pub fn new_random<R: Rng>(rng: &mut R, morph_config: &MorphologyConfig) -> Self {
        Self {
            weights_l1: xavier_init(INPUT_DIM, HIDDEN_DIM, rng),
            biases_l1: vec![0.0; HIDDEN_DIM],
            weights_l2: xavier_init(HIDDEN_DIM, OUTPUT_DIM, rng),
            biases_l2: vec![0.0; OUTPUT_DIM],
            morphology: MorphTraits::new_random(rng, morph_config),
            alive: true,
        }
    }
    
    /// Create random genome without morphology config (uses defaults)
    pub fn new_random_simple<R: Rng>(rng: &mut R) -> Self {
        Self {
            weights_l1: xavier_init(INPUT_DIM, HIDDEN_DIM, rng),
            biases_l1: vec![0.0; HIDDEN_DIM],
            weights_l2: xavier_init(HIDDEN_DIM, OUTPUT_DIM, rng),
            biases_l2: vec![0.0; OUTPUT_DIM],
            morphology: MorphTraits::default_traits(),
            alive: true,
        }
    }
    
    pub fn clone_and_mutate<R: Rng>(&self, rate: f32, strength: f32, morph_config: &MorphologyConfig, rng: &mut R) -> Self {
        // Mutate morphology first (before closure captures rng)
        let mut new_morphology = self.morphology;
        new_morphology.mutate(morph_config, rng);
        
        // Now mutate neural network weights
        let normal = Normal::new(0.0, strength as f64).unwrap();
        
        let mutate_vec = |v: &Vec<f32>, rng: &mut R| -> Vec<f32> {
            v.iter()
                .map(|&w| {
                    if rng.gen::<f32>() < rate {
                        w + normal.sample(rng) as f32
                    } else {
                        w
                    }
                })
                .collect()
        };
        
        Self {
            weights_l1: mutate_vec(&self.weights_l1, rng),
            biases_l1: mutate_vec(&self.biases_l1, rng),
            weights_l2: mutate_vec(&self.weights_l2, rng),
            biases_l2: mutate_vec(&self.biases_l2, rng),
            morphology: new_morphology,
            alive: true,
        }
    }
    
    /// Sexual reproduction: crossover with another genome and mutate
    /// Uses uniform crossover - each gene randomly chosen from either parent
    pub fn crossover_and_mutate<R: Rng>(
        &self,
        mate: &Genome,
        crossover_ratio: f32,
        mutation_rate: f32,
        mutation_strength: f32,
        morph_config: &MorphologyConfig,
        rng: &mut R,
    ) -> Self {
        // Crossover morphology first
        let new_morphology = if morph_config.enabled {
            MorphTraits {
                size: if rng.gen::<f32>() < crossover_ratio { self.morphology.size } else { mate.morphology.size },
                speed_mult: if rng.gen::<f32>() < crossover_ratio { self.morphology.speed_mult } else { mate.morphology.speed_mult },
                vision_mult: if rng.gen::<f32>() < crossover_ratio { self.morphology.vision_mult } else { mate.morphology.vision_mult },
                metabolism: if rng.gen::<f32>() < crossover_ratio { self.morphology.metabolism } else { mate.morphology.metabolism },
            }
        } else {
            MorphTraits::default_traits()
        };
        
        let mut final_morphology = new_morphology;
        final_morphology.mutate(morph_config, rng);
        
        // Crossover neural network weights with uniform crossover
        let normal = Normal::new(0.0, mutation_strength as f64).unwrap();
        
        let crossover_and_mutate_vec = |v1: &Vec<f32>, v2: &Vec<f32>, rng: &mut R| -> Vec<f32> {
            v1.iter().zip(v2.iter())
                .map(|(&w1, &w2)| {
                    // Select from parent or mate based on crossover_ratio
                    let selected = if rng.gen::<f32>() < crossover_ratio { w1 } else { w2 };
                    // Then potentially mutate
                    if rng.gen::<f32>() < mutation_rate {
                        selected + normal.sample(rng) as f32
                    } else {
                        selected
                    }
                })
                .collect()
        };
        
        Self {
            weights_l1: crossover_and_mutate_vec(&self.weights_l1, &mate.weights_l1, rng),
            biases_l1: crossover_and_mutate_vec(&self.biases_l1, &mate.biases_l1, rng),
            weights_l2: crossover_and_mutate_vec(&self.weights_l2, &mate.weights_l2, rng),
            biases_l2: crossover_and_mutate_vec(&self.biases_l2, &mate.biases_l2, rng),
            morphology: final_morphology,
            alive: true,
        }
    }
    
    /// Calculate genetic distance (Euclidean) to another genome
    /// Uses a subset of weights for efficiency (first 64 weights from each layer)
    pub fn distance_to(&self, other: &Genome) -> f32 {
        // Sample key weights from each layer for efficient comparison
        let sample_size = 64.min(WEIGHTS_L1).min(WEIGHTS_L2);
        
        let mut sum_sq = 0.0;
        
        // Sample from layer 1 weights
        for i in 0..sample_size {
            let diff = self.weights_l1[i] - other.weights_l1[i];
            sum_sq += diff * diff;
        }
        
        // Sample from layer 2 weights
        for i in 0..sample_size {
            let diff = self.weights_l2[i] - other.weights_l2[i];
            sum_sq += diff * diff;
        }
        
        // Include biases
        for i in 0..BIASES_L1 {
            let diff = self.biases_l1[i] - other.biases_l1[i];
            sum_sq += diff * diff;
        }
        
        for i in 0..BIASES_L2 {
            let diff = self.biases_l2[i] - other.biases_l2[i];
            sum_sq += diff * diff;
        }
        
        sum_sq.sqrt()
    }
    
    /// Flatten all weights into a single vector for GPU upload
    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(TOTAL_PARAMS);
        flat.extend(&self.weights_l1);
        flat.extend(&self.biases_l1);
        flat.extend(&self.weights_l2);
        flat.extend(&self.biases_l2);
        flat
    }
}

/// Xavier uniform initialization
fn xavier_init<R: Rng>(fan_in: usize, fan_out: usize, rng: &mut R) -> Vec<f32> {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let dist = Uniform::new(-limit, limit);
    (0..fan_in * fan_out).map(|_| dist.sample(rng)).collect()
}

/// Pool of genomes with ID management
pub struct GenomePool {
    genomes: Vec<Genome>,
}

impl GenomePool {
    pub fn new(max_size: u32) -> Self {
        // Preallocate with empty genomes to allow indexed access
        let genomes = vec![Genome::default(); max_size as usize];
        Self {
            genomes,
        }
    }
    
    /// Create a random genome at a specific index (used when organism slot is known)
    pub fn create_random_at<R: Rng>(&mut self, id: u32, morph_config: &MorphologyConfig, rng: &mut R) {
        let genome = Genome::new_random(rng, morph_config);
        self.genomes[id as usize] = genome;
    }
    
    /// Clone parent genome, mutate, and store at specific index
    pub fn clone_and_mutate_at<R: Rng>(
        &mut self,
        target_id: u32,
        parent_id: u32,
        rate: f32,
        strength: f32,
        morph_config: &MorphologyConfig,
        rng: &mut R,
    ) {
        let parent = &self.genomes[parent_id as usize];
        let child = parent.clone_and_mutate(rate, strength, morph_config, rng);
        self.genomes[target_id as usize] = child;
    }
    
    /// Sexual reproduction: crossover two genomes, mutate, and store at specific index
    pub fn crossover_and_mutate_at<R: Rng>(
        &mut self,
        target_id: u32,
        parent_id: u32,
        mate_id: u32,
        crossover_ratio: f32,
        mutation_rate: f32,
        mutation_strength: f32,
        morph_config: &MorphologyConfig,
        rng: &mut R,
    ) {
        let parent = &self.genomes[parent_id as usize];
        let mate = &self.genomes[mate_id as usize];
        let child = parent.crossover_and_mutate(
            mate,
            crossover_ratio,
            mutation_rate,
            mutation_strength,
            morph_config,
            rng,
        );
        self.genomes[target_id as usize] = child;
    }
    
    #[allow(dead_code)]
    pub fn free(&mut self, id: u32) {
        if let Some(genome) = self.genomes.get_mut(id as usize) {
            genome.alive = false;
        }
    }
    
    pub fn get(&self, id: u32) -> Option<&Genome> {
        self.genomes.get(id as usize)
    }
    
    /// Get morphology traits for an organism
    pub fn get_morphology(&self, id: u32) -> Option<MorphTraits> {
        self.genomes.get(id as usize).map(|g| g.morphology)
    }
    
    /// Get the flat weights for a single genome (for GPU upload)
    pub fn get_weights_flat(&self, id: u32) -> Option<Vec<f32>> {
        self.genomes.get(id as usize).map(|g| g.to_flat())
    }
    
    /// Get all weights for GPU buffer (layer 1 weights)
    #[allow(dead_code)]
    pub fn weights_l1_buffer(&self) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(self.genomes.len() * WEIGHTS_L1);
        for genome in &self.genomes {
            buffer.extend(&genome.weights_l1);
        }
        buffer
    }
    
    /// Get all biases for GPU buffer (layer 1)
    #[allow(dead_code)]
    pub fn biases_l1_buffer(&self) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(self.genomes.len() * BIASES_L1);
        for genome in &self.genomes {
            buffer.extend(&genome.biases_l1);
        }
        buffer
    }
    
    /// Get all weights for GPU buffer (layer 2)
    #[allow(dead_code)]
    pub fn weights_l2_buffer(&self) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(self.genomes.len() * WEIGHTS_L2);
        for genome in &self.genomes {
            buffer.extend(&genome.weights_l2);
        }
        buffer
    }
    
    /// Get all biases for GPU buffer (layer 2)
    #[allow(dead_code)]
    pub fn biases_l2_buffer(&self) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(self.genomes.len() * BIASES_L2);
        for genome in &self.genomes {
            buffer.extend(&genome.biases_l2);
        }
        buffer
    }
    
    /// Get combined neural network weights for GPU buffer
    /// Layout per genome: weights_l1 (320) + biases_l1 (16) + weights_l2 (96) + biases_l2 (6) = 438 floats
    pub fn nn_weights_buffer(&self) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(self.genomes.len() * TOTAL_PARAMS);
        for genome in &self.genomes {
            buffer.extend(&genome.weights_l1);
            buffer.extend(&genome.biases_l1);
            buffer.extend(&genome.weights_l2);
            buffer.extend(&genome.biases_l2);
        }
        buffer
    }
    
    /// Get all morphology traits for GPU buffer
    /// Layout per genome: size, speed_mult, vision_mult, metabolism (4 floats)
    pub fn morphology_buffer(&self) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(self.genomes.len() * MORPH_TRAITS);
        for genome in &self.genomes {
            let arr = genome.morphology.to_array();
            buffer.extend_from_slice(&arr);
        }
        buffer
    }
    
    /// Get morphology traits for a single genome
    pub fn get_morphology_flat(&self, id: u32) -> Option<[f32; MORPH_TRAITS]> {
        self.genomes.get(id as usize).map(|g| g.morphology.to_array())
    }
    
    /// Restore a genome at a specific index during loading
    pub fn restore_at(&mut self, id: u32, genome: Genome) {
        if (id as usize) < self.genomes.len() {
            self.genomes[id as usize] = genome;
        }
    }
    
    /// Iterate over all genomes
    pub fn iter(&self) -> impl Iterator<Item = &Genome> {
        self.genomes.iter()
    }
    
    /// Count alive genomes
    pub fn count(&self) -> u32 {
        self.genomes.iter().filter(|g| g.alive).count() as u32
    }
}
