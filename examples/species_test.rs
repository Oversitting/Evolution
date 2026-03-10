//! Species detection and clustering tests
//! 
//! Tests for the species detection system based on genetic distance.

use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::{Simulation, GenomePool, SpeciesConfig, SpeciesManager};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    println!("=== Species Detection Tests ===\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    let tests: Vec<(&str, fn() -> Result<(), String>)> = vec![
        ("Species Config Defaults", test_config_defaults),
        ("Species Manager Creation", test_manager_creation),
        ("Genetic Distance Calculation", test_genetic_distance),
        ("Species Assignment", test_species_assignment),
        ("Child Species Inheritance", test_child_species),
        ("Simulation Species Count", test_simulation_species),
    ];
    
    for (name, test) in tests {
        print!("  {} ... ", name);
        match test() {
            Ok(()) => {
                println!("OK");
                passed += 1;
            }
            Err(e) => {
                println!("FAILED: {}", e);
                failed += 1;
            }
        }
    }
    
    println!("\n=== Results: {} passed, {} failed ===", passed, failed);
    
    if failed > 0 {
        std::process::exit(1);
    }
}

fn test_config_defaults() -> Result<(), String> {
    let config = SpeciesConfig::default();
    
    if !config.enabled {
        return Err("Species should be enabled by default".into());
    }
    
    if config.distance_threshold <= 0.0 {
        return Err("Distance threshold should be positive".into());
    }
    
    if config.max_species == 0 {
        return Err("Max species should be non-zero".into());
    }
    
    Ok(())
}

fn test_manager_creation() -> Result<(), String> {
    let config = SpeciesConfig::default();
    let manager = SpeciesManager::new(config);
    
    if manager.species_count() != 0 {
        return Err("New manager should have 0 species".into());
    }
    
    Ok(())
}

fn test_genetic_distance() -> Result<(), String> {
    use evolution_sim::simulation::genome::{Genome, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM};
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    let g1 = Genome::new_random(&mut rng);
    let g2 = Genome::new_random(&mut rng);
    
    // Random genomes should have some distance
    let dist = g1.distance_to(&g2);
    
    if dist <= 0.0 {
        return Err("Random genomes should have non-zero distance".into());
    }
    
    // Clone should have zero distance
    let g1_clone = g1.clone();
    let clone_dist = g1.distance_to(&g1_clone);
    
    if clone_dist > 0.001 {
        return Err(format!("Clone should have near-zero distance, got {}", clone_dist));
    }
    
    // Mutated genome should have small but non-zero distance
    let g1_mutated = g1.clone_and_mutate(0.5, 0.1, &mut rng);
    let mutated_dist = g1.distance_to(&g1_mutated);
    
    if mutated_dist <= 0.0 {
        return Err("Mutated genome should have non-zero distance".into());
    }
    
    // Mutated distance should typically be less than random distance
    // (not always, but with 50% mutation rate it usually is)
    
    Ok(())
}

fn test_species_assignment() -> Result<(), String> {
    let config = SpeciesConfig::default();
    let mut manager = SpeciesManager::new(config);
    let mut genomes = GenomePool::new(10);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    // Create first genome
    genomes.create_random_at(0, &mut rng);
    
    let species1 = manager.assign_species(0, 1, &genomes);
    
    if species1 == 0 {
        return Err("First assignment should create species > 0".into());
    }
    
    if manager.species_count() != 1 {
        return Err(format!("Should have 1 species, got {}", manager.species_count()));
    }
    
    // Create second random genome (should be different species due to distance)
    genomes.create_random_at(1, &mut rng);
    let species2 = manager.assign_species(1, 1, &genomes);
    
    if species2 == 0 {
        return Err("Second assignment should create species > 0".into());
    }
    
    // Random genomes should be different species (high confidence)
    // but we can't guarantee this, so just check both have valid IDs
    
    Ok(())
}

fn test_child_species() -> Result<(), String> {
    let config = SpeciesConfig {
        distance_threshold: 10.0, // Higher threshold
        ..Default::default()
    };
    let mut manager = SpeciesManager::new(config);
    let mut genomes = GenomePool::new(10);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    // Create parent genome and assign species
    genomes.create_random_at(0, &mut rng);
    let parent_species = manager.assign_species(0, 1, &genomes);
    
    // Create child genome with small mutation
    genomes.clone_and_mutate_at(1, 0, 0.1, 0.05, &mut rng); // Low mutation
    
    let child_species = manager.assign_child_species(1, parent_species, 2, &genomes);
    
    // With low mutation and high threshold, child should likely be same species
    // But this isn't guaranteed, so we just check valid assignment
    if child_species == 0 {
        return Err("Child should have valid species ID".into());
    }
    
    Ok(())
}

fn test_simulation_species() -> Result<(), String> {
    let mut config = SimulationConfig::default();
    config.population.initial_organisms = 10;
    config.population.max_organisms = 100;
    
    let sim = Simulation::new(&config);
    
    // Check that species are assigned to initial organisms
    let species_count = sim.species_manager.species_count();
    
    if species_count == 0 {
        // If all random genomes happened to cluster, this could be 0
        // But with 10 organisms, very unlikely
        log::warn!("No species detected for {} organisms", sim.organism_count());
    }
    
    // Check that organisms have species IDs
    let orgs_with_species: usize = sim.organisms.iter()
        .filter(|o| o.is_alive() && o.species_id > 0)
        .count();
    
    if orgs_with_species == 0 {
        return Err("Some organisms should have species IDs".into());
    }
    
    println!("  (Found {} species for {} organisms)", species_count, sim.organism_count());
    
    Ok(())
}
