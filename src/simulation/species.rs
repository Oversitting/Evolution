//! Species detection and clustering based on genetic distance
//!
//! This module implements a simple representative-based clustering algorithm:
//! 1. Each species has a representative genome (the first member)
//! 2. New organisms are compared against all species representatives
//! 3. If distance < threshold, join that species; otherwise create new species
//! 4. Species without members are removed periodically

use super::genome::GenomePool;
use super::organism::OrganismPool;
use std::collections::HashMap;

/// Configuration for species detection
#[derive(Clone, Debug)]
pub struct SpeciesConfig {
    /// Enable species detection
    pub enabled: bool,
    /// Maximum genetic distance to be considered same species
    pub distance_threshold: f32,
    /// How often to run full species recalculation (in ticks)
    pub update_interval: u32,
    /// Maximum number of species to track (prevents unbounded growth)
    pub max_species: u32,
}

impl Default for SpeciesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            distance_threshold: 8.0,  // Tuned for typical genome weight distributions
            update_interval: 60,      // Every 60 ticks (1 second at 60 updates/sec)
            max_species: 64,          // Reasonable limit for visualization
        }
    }
}

/// Represents a species cluster
#[derive(Clone, Debug)]
pub struct Species {
    /// Unique species ID
    pub id: u32,
    /// Representative genome ID (used for distance comparisons)
    pub representative_genome_id: u32,
    /// Count of current members
    pub member_count: u32,
    /// Generation when species first appeared (for future UI display)
    #[allow(dead_code)]
    pub origin_generation: u32,
}

/// Manages species detection and assignment
pub struct SpeciesManager {
    /// All active species
    species: HashMap<u32, Species>,
    /// Next species ID to assign
    next_species_id: u32,
    /// Ticks since last full update
    ticks_since_update: u32,
    /// Configuration
    config: SpeciesConfig,
}

impl SpeciesManager {
    pub fn new(config: SpeciesConfig) -> Self {
        Self {
            species: HashMap::new(),
            next_species_id: 1,  // Start at 1, 0 means "unassigned"
            ticks_since_update: 0,
            config,
        }
    }
    
    /// Returns true if species detection is enabled (for future conditional logic)
    #[allow(dead_code)]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
    
    /// Get the number of active species
    pub fn species_count(&self) -> usize {
        self.species.len()
    }
    
    /// Get all species for UI display (for future species breakdown panel)
    #[allow(dead_code)]
    pub fn iter_species(&self) -> impl Iterator<Item = &Species> {
        self.species.values()
    }
    
    /// Check if it's time for a full species update
    pub fn should_update(&mut self) -> bool {
        self.ticks_since_update += 1;
        if self.ticks_since_update >= self.config.update_interval {
            self.ticks_since_update = 0;
            true
        } else {
            false
        }
    }
    
    /// Assign a species ID to a new organism based on its genome
    /// Returns the assigned species ID (creates new species if needed)
    pub fn assign_species(
        &mut self,
        genome_id: u32,
        generation: u32,
        genomes: &GenomePool,
    ) -> u32 {
        if !self.config.enabled {
            return 0;
        }
        
        let genome = match genomes.get(genome_id) {
            Some(g) => g,
            None => return 0,
        };
        
        // Find the closest species representative
        let mut best_species_id = 0u32;
        let mut best_distance = f32::MAX;
        
        for species in self.species.values() {
            if let Some(rep_genome) = genomes.get(species.representative_genome_id) {
                let dist = genome.distance_to(rep_genome);
                if dist < best_distance {
                    best_distance = dist;
                    best_species_id = species.id;
                }
            }
        }
        
        // If close enough to an existing species, join it
        if best_distance < self.config.distance_threshold {
            if let Some(species) = self.species.get_mut(&best_species_id) {
                species.member_count += 1;
            }
            return best_species_id;
        }
        
        // Create new species if we haven't hit the limit
        if (self.species.len() as u32) < self.config.max_species {
            let new_id = self.next_species_id;
            self.next_species_id += 1;
            
            self.species.insert(new_id, Species {
                id: new_id,
                representative_genome_id: genome_id,
                member_count: 1,
                origin_generation: generation,
            });
            
            return new_id;
        }
        
        // At limit - assign to closest species even if above threshold
        if best_species_id > 0 {
            if let Some(species) = self.species.get_mut(&best_species_id) {
                species.member_count += 1;
            }
            return best_species_id;
        }
        
        0 // Fallback
    }
    
    /// Full species recalculation - reassigns all organisms to species
    /// This is more expensive but helps keep species assignments consistent
    pub fn recalculate_all(
        &mut self,
        organisms: &mut OrganismPool,
        genomes: &GenomePool,
    ) {
        if !self.config.enabled {
            return;
        }
        
        // Reset member counts
        for species in self.species.values_mut() {
            species.member_count = 0;
        }
        
        // Collect alive organism indices and their genome IDs
        let alive_organisms: Vec<(u32, u32, u32)> = organisms
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_alive())
            .map(|(idx, o)| (idx as u32, o.genome_id, o.generation))
            .collect();
        
        // Assign species to each organism
        for (org_idx, genome_id, generation) in alive_organisms {
            let species_id = self.assign_species_internal(genome_id, generation, genomes);
            organisms.get_mut(org_idx).map(|o| o.species_id = species_id);
        }
        
        // Remove species with no members
        self.species.retain(|_, s| s.member_count > 0);
    }
    
    /// Internal species assignment (doesn't create new species during recalculation)
    fn assign_species_internal(
        &mut self,
        genome_id: u32,
        generation: u32,
        genomes: &GenomePool,
    ) -> u32 {
        let genome = match genomes.get(genome_id) {
            Some(g) => g,
            None => return 0,
        };
        
        // Find the closest species representative
        let mut best_species_id = 0u32;
        let mut best_distance = f32::MAX;
        
        for species in self.species.values() {
            if let Some(rep_genome) = genomes.get(species.representative_genome_id) {
                let dist = genome.distance_to(rep_genome);
                if dist < best_distance {
                    best_distance = dist;
                    best_species_id = species.id;
                }
            }
        }
        
        // If close enough, join existing species
        if best_distance < self.config.distance_threshold && best_species_id > 0 {
            if let Some(species) = self.species.get_mut(&best_species_id) {
                species.member_count += 1;
            }
            return best_species_id;
        }
        
        // Create new species during recalculation
        if (self.species.len() as u32) < self.config.max_species {
            let new_id = self.next_species_id;
            self.next_species_id += 1;
            
            self.species.insert(new_id, Species {
                id: new_id,
                representative_genome_id: genome_id,
                member_count: 1,
                origin_generation: generation,
            });
            
            return new_id;
        }
        
        // At limit - assign to closest
        if best_species_id > 0 {
            if let Some(species) = self.species.get_mut(&best_species_id) {
                species.member_count += 1;
            }
        }
        
        best_species_id
    }
    
    /// Assign species to a newly born organism (child of parent)
    /// Children typically inherit parent's species unless they've mutated significantly
    pub fn assign_child_species(
        &mut self,
        child_genome_id: u32,
        parent_species_id: u32,
        generation: u32,
        genomes: &GenomePool,
    ) -> u32 {
        if !self.config.enabled || parent_species_id == 0 {
            return self.assign_species(child_genome_id, generation, genomes);
        }
        
        let child_genome = match genomes.get(child_genome_id) {
            Some(g) => g,
            None => return parent_species_id,
        };
        
        // Check distance to parent's species representative
        if let Some(parent_species) = self.species.get(&parent_species_id) {
            if let Some(rep_genome) = genomes.get(parent_species.representative_genome_id) {
                let dist = child_genome.distance_to(rep_genome);
                
                if dist < self.config.distance_threshold {
                    // Child stays in parent's species
                    if let Some(species) = self.species.get_mut(&parent_species_id) {
                        species.member_count += 1;
                    }
                    return parent_species_id;
                }
            }
        }
        
        // Child has diverged - assign to best matching or new species
        self.assign_species(child_genome_id, generation, genomes)
    }
    
    /// Call when an organism dies
    pub fn on_organism_death(&mut self, species_id: u32) {
        if let Some(species) = self.species.get_mut(&species_id) {
            species.member_count = species.member_count.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    #[test]
    fn test_species_creation() {
        let config = SpeciesConfig::default();
        let mut manager = SpeciesManager::new(config);
        let mut genomes = GenomePool::new(10);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        genomes.create_random_at(0, &mut rng);
        
        let species_id = manager.assign_species(0, 1, &genomes);
        
        assert!(species_id > 0);
        assert_eq!(manager.species_count(), 1);
    }
    
    #[test]
    fn test_species_clustering() {
        let config = SpeciesConfig {
            distance_threshold: 5.0,
            ..Default::default()
        };
        let mut manager = SpeciesManager::new(config);
        let mut genomes = GenomePool::new(10);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        // Create two genomes
        genomes.create_random_at(0, &mut rng);
        genomes.create_random_at(1, &mut rng);
        
        let sp1 = manager.assign_species(0, 1, &genomes);
        let sp2 = manager.assign_species(1, 1, &genomes);
        
        // Random genomes should be different species (distance > threshold)
        // unless they happen to be similar by chance
        assert!(sp1 > 0);
        assert!(sp2 > 0);
    }
}
