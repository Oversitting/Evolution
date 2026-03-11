//! Simulation module - organism and world state management

pub mod organism;
pub mod world;
pub mod genome;
pub mod save_load;
pub mod species;

pub use organism::{OrganismPool, OrganismGpu};
#[allow(unused_imports)]
pub use world::{World, BiomeType};
#[allow(unused_imports)]
pub use genome::GenomePool;
#[allow(unused_imports)]
pub use save_load::{FounderPool, FounderRecord, SaveState, SavedOrganism, SavedGenome, SurvivorBank, SurvivorEntry};
pub use species::{SpeciesManager, SpeciesConfig};

use crate::config::SimulationConfig;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Reproduction result containing organisms and genomes that need GPU sync
pub struct ReproductionResult {
    /// Organisms that changed (parent energy deducted + new spawns)
    pub organism_changes: Vec<(u32, OrganismGpu)>,
    /// New genome IDs created during reproduction (need weight sync)
    pub new_genome_ids: Vec<u32>,
}

/// Main simulation state
pub struct Simulation {
    pub organisms: OrganismPool,
    pub genomes: GenomePool,
    pub world: World,
    pub rng: Xoshiro256PlusPlus,
    pub species_manager: SpeciesManager,
}

impl Simulation {
    fn load_survivor_bank(config: &SimulationConfig) -> Vec<SurvivorEntry> {
        if !config.bootstrap.enabled || !config.bootstrap.load_on_start || config.bootstrap.founder_count == 0 {
            return Vec::new();
        }

        match save_load::load_bootstrap_entries(&config.bootstrap.path, config.bootstrap.founder_count as usize) {
            Ok(founders) => {

                if !founders.is_empty() {
                    log::info!(
                        "Loaded {} bootstrap founders from {:?}",
                        founders.len(),
                        config.bootstrap.path
                    );
                }

                founders
            }
            Err(error) => {
                log::info!(
                    "No survivor bank loaded from {:?}: {}",
                    config.bootstrap.path,
                    error
                );
                Vec::new()
            }
        }
    }

    fn survivor_score(org: &organism::Organism) -> f32 {
        org.generation as f32 * 5_000.0
            + org.age as f32 * 8.0
            + org.energy * 40.0
            + org.offspring_count as f32 * 600.0
    }

    fn wrap_position(position: glam::Vec2, world_width: f32, world_height: f32) -> glam::Vec2 {
        glam::Vec2::new(
            position.x.rem_euclid(world_width),
            position.y.rem_euclid(world_height),
        )
    }

    pub fn new(config: &SimulationConfig) -> Self {
        let mut rng = if let Some(seed) = config.seed {
            Xoshiro256PlusPlus::seed_from_u64(seed)
        } else {
            Xoshiro256PlusPlus::from_entropy()
        };
        
        let mut organisms = OrganismPool::new(config.population.max_organisms);
        let mut genomes = GenomePool::new(config.population.max_organisms);
        
        // Create world with seeded RNG for deterministic food patch generation
        let world = World::new_with_rng(config, &mut rng);
        
        // Collect positions with significant food for spawning (above baseline)
        let baseline = config.food.baseline_food;
        let food_positions: Vec<(f32, f32)> = world.food.iter()
            .enumerate()
            .filter(|(_, &food)| food > baseline + 1.0)  // Only patches, not baseline
            .map(|(idx, _)| {
                let x = (idx as u32 % config.world.width) as f32 + 0.5;
                let y = (idx as u32 / config.world.width) as f32 + 0.5;
                (x, y)
            })
            .collect();
        
        log::info!("Found {} cells with food patches for spawning", food_positions.len());
        
        let survivor_founders = Self::load_survivor_bank(config);

        // Spawn initial organisms - DISTRIBUTE across patches, not clustered
        // Each organism picks a different random food position to encourage dispersal
        let num_organisms = config.population.initial_organisms;
        let positions_per_organism = if !food_positions.is_empty() {
            food_positions.len() / (num_organisms as usize).max(1)
        } else {
            1
        };

        let seeded_founders = survivor_founders.len().min(num_organisms as usize);
        
        for i in 0..num_organisms {
            let pos = if !food_positions.is_empty() {
                // Distribute organisms across different regions of food
                // Use stride-based selection to spread across patches
                let base_idx = (i as usize * positions_per_organism) % food_positions.len();
                let food_idx = (base_idx + rand::Rng::gen_range(&mut rng, 0..positions_per_organism.max(1))) 
                    % food_positions.len();
                let (x, y) = food_positions[food_idx];
                // Add small random offset to avoid stacking
                Self::wrap_position(
                    glam::Vec2::new(
                    x + rand::Rng::gen_range(&mut rng, -3.0..3.0),
                    y + rand::Rng::gen_range(&mut rng, -3.0..3.0),
                    ),
                    config.world.width as f32,
                    config.world.height as f32,
                )
            } else {
                // Fallback to random position
                glam::Vec2::new(
                    rand::Rng::gen_range(&mut rng, 0.0..config.world.width as f32),
                    rand::Rng::gen_range(&mut rng, 0.0..config.world.height as f32),
                )
            };
            
            // Create genome at this slot index first (with morphology config)
            let founder_generation = if let Some(entry) = survivor_founders.get(i as usize) {
                genomes.restore_at(i, genome::Genome::from(&entry.genome));
                entry.generation
            } else {
                genomes.create_random_at(i, &config.morphology, &mut rng);
                0
            };
            // Spawn organism with genome_id = slot index
            organisms.spawn(pos, config.energy.starting, i, founder_generation, &mut rng);
            
            // Set morphology traits on organism from genome
            if let Some(morph) = genomes.get_morphology(i) {
                if let Some(org) = organisms.get_mut(i) {
                    org.set_morphology(morph.size, morph.speed_mult, morph.vision_mult, morph.metabolism);
                }
            }
        }
        
        log::info!(
            "Spawned {} initial organisms distributed across {} food patches ({} loaded from bootstrap store)",
            organisms.count(),
            config.food.initial_patches,
            seeded_founders
        );
        
        // Create species manager and assign initial species
        let mut species_manager = SpeciesManager::new(SpeciesConfig::default());
        
        // Collect alive organism info first to avoid borrow issues
        let alive_info: Vec<(u32, u32, u32)> = organisms
            .iter()
            .enumerate()
            .filter(|(_, org)| org.is_alive())
            .map(|(idx, org)| (idx as u32, org.genome_id, org.generation))
            .collect();
        
        // Assign species to all initial organisms
        for (idx, genome_id, generation) in alive_info {
            let species_id = species_manager.assign_species(genome_id, generation, &genomes);
            if let Some(org_mut) = organisms.get_mut(idx) {
                org_mut.species_id = species_id;
            }
        }
        
        log::info!("Assigned {} initial species", species_manager.species_count());
        
        Self {
            organisms,
            genomes,
            world,
            rng,
            species_manager,
        }
    }
    
    pub fn organism_count(&self) -> u32 {
        self.organisms.count()
    }
    
    /// Calculate average energy of alive organisms
    pub fn avg_energy(&self) -> f32 {
        let alive: Vec<_> = self.organisms.iter().filter(|o| o.is_alive()).collect();
        if alive.is_empty() {
            return 0.0;
        }
        alive.iter().map(|o| o.energy).sum::<f32>() / alive.len() as f32
    }
    
    /// Get maximum generation reached
    pub fn max_generation(&self) -> u32 {
        self.organisms.iter()
            .filter(|o| o.is_alive())
            .map(|o| o.generation)
            .max()
            .unwrap_or(0)
    }
    
    /// Get total food in world
    pub fn total_food(&self) -> f32 {
        self.world.food.iter().sum()
    }
    
    /// Handle reproduction, returns organisms and genomes that need GPU sync
    pub fn handle_reproduction(&mut self, config: &SimulationConfig) -> ReproductionResult {
        let max_organisms = config.population.max_organisms;
        let current_count = self.organisms.count();
        let world_width = config.world.width as f32;
        let world_height = config.world.height as f32;
        
        // Struct to hold reproduction intent
        struct ReproductionIntent {
            parent_genome_id: u32,
            parent_generation: u32,
            parent_species_id: u32,
            child_pos: glam::Vec2,
            mate_genome_id: Option<u32>, // Some() for sexual, None for asexual
        }
        
        let mut intents = Vec::new();
        let mut parent_indices = Vec::new();
        
        // First pass: collect organisms that want to reproduce
        let organisms_snapshot: Vec<_> = self.organisms.iter()
            .enumerate()
            .filter(|(_, org)| org.is_alive())
            .map(|(idx, org)| (idx as u32, org.position, org.genome_id, org.generation, org.species_id, org.energy, org.age, org.cooldown, org.reproduce_signal))
            .collect();
        
        // Check each organism for reproduction
        for (idx, org) in self.organisms.iter_mut_indexed() {
            if !org.is_alive() {
                continue;
            }
            
            // Check reproduction conditions
            let can_reproduce = 
                org.energy >= config.reproduction.threshold &&
                org.age >= config.reproduction.min_age &&
                org.cooldown == 0 &&
                org.reproduce_signal > config.reproduction.signal_min;
            
            if can_reproduce && (current_count + intents.len() as u32) < max_organisms {
                // Deduct energy
                org.energy -= config.reproduction.cost;
                org.cooldown = config.reproduction.cooldown;
                org.offspring_count += 1;
                
                parent_indices.push(idx);
                
                // Calculate spawn position
                let offset = glam::Vec2::new(
                    rand::Rng::gen_range(&mut self.rng, -5.0..5.0),
                    rand::Rng::gen_range(&mut self.rng, -5.0..5.0),
                );
                let child_pos = Self::wrap_position(org.position + offset, world_width, world_height);
                
                // Find mate for sexual reproduction
                let mate_genome_id = if config.reproduction.sexual_enabled {
                    // Find nearest willing mate within range
                    let mut best_mate: Option<(u32, f32)> = None;
                    
                    for &(other_idx, other_pos, other_genome_id, _, _, other_energy, other_age, other_cooldown, other_reproduce_signal) in &organisms_snapshot {
                        if other_idx == idx {
                            continue; // Skip self
                        }
                        
                        // Check if potential mate meets reproduction conditions
                        let mate_willing = 
                            other_energy >= config.reproduction.threshold * 0.5 && // Lower threshold for mates
                            other_age >= config.reproduction.min_age &&
                            other_cooldown == 0 &&
                            other_reproduce_signal > config.reproduction.mate_signal_min;
                        
                        if !mate_willing {
                            continue;
                        }
                        
                        // Calculate distance with world wrap
                        let mut dx = other_pos.x - org.position.x;
                        let mut dy = other_pos.y - org.position.y;
                        if dx > world_width / 2.0 { dx -= world_width; }
                        else if dx < -world_width / 2.0 { dx += world_width; }
                        if dy > world_height / 2.0 { dy -= world_height; }
                        else if dy < -world_height / 2.0 { dy += world_height; }
                        
                        let dist_sq = dx * dx + dy * dy;
                        let mate_range_sq = config.reproduction.mate_range * config.reproduction.mate_range;
                        
                        if dist_sq < mate_range_sq {
                            match best_mate {
                                Some((_, best_dist_sq)) if dist_sq < best_dist_sq => {
                                    best_mate = Some((other_genome_id, dist_sq));
                                }
                                None => {
                                    best_mate = Some((other_genome_id, dist_sq));
                                }
                                _ => {}
                            }
                        }
                    }
                    
                    best_mate.map(|(id, _)| id)
                } else {
                    None // Asexual reproduction
                };
                
                intents.push(ReproductionIntent {
                    parent_genome_id: org.genome_id,
                    parent_generation: org.generation,
                    parent_species_id: org.species_id,
                    child_pos,
                    mate_genome_id,
                });
            }
        }
        
        let mut organism_changes = Vec::new();
        let mut new_genome_ids = Vec::new();
        
        // Add parent changes to sync list
        for idx in parent_indices {
            if let Some(org) = self.organisms.get(idx) {
                let org_gpu = org.to_gpu();
                organism_changes.push((idx, org_gpu));
            }
        }
        
        // Spawn offspring
        for intent in intents {
            // First spawn organism to get slot index (with placeholder genome_id)
            if let Some(child_idx) = self.organisms.spawn(
                intent.child_pos,
                config.reproduction.cost, // Child starts with reproduction cost as energy
                0, // placeholder, will be set to child_idx
                intent.parent_generation + 1,
                &mut self.rng,
            ) {
                // Create genome based on reproduction mode
                match intent.mate_genome_id {
                    Some(mate_genome_id) => {
                        // Sexual reproduction: crossover + mutation
                        self.genomes.crossover_and_mutate_at(
                            child_idx,
                            intent.parent_genome_id,
                            mate_genome_id,
                            config.reproduction.crossover_ratio,
                            config.mutation.rate,
                            config.mutation.strength,
                            &config.morphology,
                            &mut self.rng,
                        );
                    }
                    None => {
                        // Asexual reproduction: clone + mutation
                        self.genomes.clone_and_mutate_at(
                            child_idx,
                            intent.parent_genome_id,
                            config.mutation.rate,
                            config.mutation.strength,
                            &config.morphology,
                            &mut self.rng,
                        );
                    }
                }
                
                // Update organism's genome_id to match its slot
                if let Some(org) = self.organisms.get_mut(child_idx) {
                    org.genome_id = child_idx;
                }
                
                // Assign species to the child
                let child_species_id = self.species_manager.assign_child_species(
                    child_idx,
                    intent.parent_species_id,
                    intent.parent_generation + 1,
                    &self.genomes,
                );
                
                // Set species and morphology on the child
                if let Some(morph) = self.genomes.get_morphology(child_idx) {
                    if let Some(org) = self.organisms.get_mut(child_idx) {
                        org.species_id = child_species_id;
                        org.set_morphology(morph.size, morph.speed_mult, morph.vision_mult, morph.metabolism);
                    }
                }
                
                // Track the new genome for GPU sync
                new_genome_ids.push(child_idx);
                
                // Get the GPU representation for the new organism
                if let Some(org) = self.organisms.get(child_idx) {
                    let org_gpu = org.to_gpu();
                    organism_changes.push((child_idx, org_gpu));
                }
            }
        }
        
        // Clean up dead organisms
        self.organisms.cleanup_dead(&mut self.genomes);
        
        ReproductionResult {
            organism_changes,
            new_genome_ids,
        }
    }
    
    /// Update species assignments periodically
    /// Should be called after GPU state is synced back to CPU
    pub fn update_species(&mut self) {
        if self.species_manager.should_update() {
            self.species_manager.recalculate_all(&mut self.organisms, &self.genomes);
        }
    }
    
    /// Get the current number of species
    pub fn species_count(&self) -> usize {
        self.species_manager.species_count()
    }
    
    /// Create a SaveState from current simulation
    pub fn to_save_state(&self, tick: u64, config: &SimulationConfig) -> SaveState {
        SaveState {
            version: SaveState::VERSION,
            tick,
            config: config.clone(),
            organisms: self.organisms.iter()
                .map(|o| SavedOrganism::from(o))
                .collect(),
            genomes: self.genomes.iter()
                .map(|g| SavedGenome::from(g))
                .collect(),
            food: self.world.food.clone(),
            world_width: self.world.width,
            world_height: self.world.height,
        }
    }

    /// Create a persistent survivor bank from the best currently living organisms.
    pub fn to_survivor_bank(&self, tick: u64, max_entries: usize) -> Option<SurvivorBank> {
        if max_entries == 0 {
            return None;
        }

        let mut entries: Vec<SurvivorEntry> = self
            .organisms
            .iter()
            .filter(|org| org.is_alive())
            .filter_map(|org| {
                self.genomes.get(org.genome_id).map(|genome| SurvivorEntry {
                    genome: SavedGenome::from(genome),
                    generation: org.generation,
                    offspring_count: org.offspring_count,
                    age: org.age,
                    energy: org.energy,
                    species_id: org.species_id,
                    score: Self::survivor_score(org),
                })
            })
            .collect();

        if entries.is_empty() {
            return None;
        }

        entries.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(right.generation.cmp(&left.generation))
                .then(right.age.cmp(&left.age))
                .then_with(|| right.energy.partial_cmp(&left.energy).unwrap_or(std::cmp::Ordering::Equal))
        });
        entries.truncate(max_entries);

        Some(SurvivorBank {
            version: SurvivorBank::VERSION,
            source_tick: tick,
            entries,
        })
    }
    
    /// Restore simulation from SaveState
    pub fn from_save_state(state: &SaveState) -> Self {
        use rand::SeedableRng;
        
        // Create new organism pool and restore organisms
        let mut organisms = organism::OrganismPool::new(state.config.population.max_organisms);
        for saved in &state.organisms {
            organisms.restore(organism::Organism::from(saved));
        }
        
        // Create new genome pool and restore genomes
        let mut genomes = genome::GenomePool::new(state.config.population.max_organisms);
        for (idx, saved) in state.genomes.iter().enumerate() {
            genomes.restore_at(idx as u32, genome::Genome::from(saved));
        }
        
        // Create RNG for biome generation (re-seed since we don't save RNG state)
        let mut biome_rng = if let Some(seed) = state.config.seed {
            Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(state.tick + 12345))
        } else {
            Xoshiro256PlusPlus::from_entropy()
        };
        
        // Generate biomes for restored world
        let biomes = World::generate_biomes_static(
            state.world_width,
            state.world_height,
            state.config.biomes.biome_count,
            state.config.biomes.enabled,
            &mut biome_rng,
        );
        
        // Create world with restored food
        let world = World {
            width: state.world_width,
            height: state.world_height,
            food: state.food.clone(),
            obstacles: vec![0; (state.world_width * state.world_height) as usize],
            biomes,
        };
        
        // Create RNG (re-seed since we don't save RNG state)
        let rng = if let Some(seed) = state.config.seed {
            Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(state.tick))
        } else {
            Xoshiro256PlusPlus::from_entropy()
        };
        
        log::info!(
            "Restored simulation: {} organisms, {} genomes, tick {}",
            organisms.count(),
            genomes.count(),
            state.tick
        );
        
        // Create species manager and immediately recalculate species for loaded organisms
        let mut species_manager = SpeciesManager::new(SpeciesConfig::default());
        species_manager.recalculate_all(&mut organisms, &genomes);
        
        log::info!("Recalculated {} species for loaded organisms", species_manager.species_count());
        
        Self {
            organisms,
            genomes,
            world,
            rng,
            species_manager,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use std::fs;

    fn temp_bank_path(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}.bin", name, unique))
    }

    #[test]
    fn survivor_bank_round_trip_seeds_future_founders() {
        let bank_path = temp_bank_path("survivor_bank_test");

        let mut source_config = SimulationConfig::default();
        source_config.seed = Some(11);
        source_config.population.max_organisms = 8;
        source_config.population.initial_organisms = 2;
        source_config.bootstrap.path = bank_path.clone();
        source_config.bootstrap.founder_count = 2;
        source_config.bootstrap.survivor_count = 2;
        source_config.bootstrap.load_on_start = false;

        let mut source_sim = Simulation::new(&source_config);
        {
            let first = source_sim.organisms.get_mut(0).unwrap();
            first.age = 400;
            first.energy = 150.0;
            first.offspring_count = 3;
        }
        {
            let second = source_sim.organisms.get_mut(1).unwrap();
            second.age = 250;
            second.energy = 120.0;
            second.offspring_count = 1;
            second.generation = 2;
        }

        let bank = source_sim.to_survivor_bank(123, 2).unwrap();
        bank.save_to_file(&bank_path).unwrap();

        let mut seeded_config = SimulationConfig::default();
        seeded_config.seed = Some(12);
        seeded_config.population.max_organisms = 4;
        seeded_config.population.initial_organisms = 2;
        seeded_config.bootstrap.path = bank_path.clone();
        seeded_config.bootstrap.founder_count = 2;
        seeded_config.bootstrap.survivor_count = 2;
        seeded_config.bootstrap.load_on_start = true;

        let seeded_sim = Simulation::new(&seeded_config);

        let expected = bank.entries[0].genome.weights_l1[0];
        let actual = seeded_sim.genomes.get(0).unwrap().weights_l1[0];
        assert_eq!(actual, expected);
        assert_eq!(seeded_sim.organisms.get(0).unwrap().generation, bank.entries[0].generation);

        let _ = fs::remove_file(bank_path);
    }
}
