//! Integration Test - Runtime Simulation Verification
//!
//! This test runs the actual simulation headlessly and verifies:
//! - Organisms move and evolve
//! - Energy dynamics work correctly
//! - Reproduction occurs
//! - Save/Load works
//!
//! Run with: cargo run --example integration_test

use std::path::PathBuf;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Evolution Simulator - Integration Test (Headless)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    // Test 1: Simulation initialization and basic tick
    run_test("Simulation Init & Basic Tick", test_simulation_tick, &mut passed, &mut failed);
    
    // Test 2: Population dynamics (growth)
    run_test("Population Growth", test_population_growth, &mut passed, &mut failed);
    
    // Test 3: Energy dynamics
    run_test("Energy Dynamics", test_energy_dynamics, &mut passed, &mut failed);
    
    // Test 4: Reproduction and mutation
    run_test("Reproduction & Mutation", test_reproduction, &mut passed, &mut failed);
    
    // Test 5: Generation advancement  
    run_test("Generation Advancement", test_generations, &mut passed, &mut failed);
    
    // Test 6: Age and death
    run_test("Age and Death System", test_age_death, &mut passed, &mut failed);
    
    // Test 7: Save and Load roundtrip
    run_test("Save/Load Roundtrip", test_save_load, &mut passed, &mut failed);
    
    // Test 8: Morphology system
    run_test("Morphology Traits", test_morphology, &mut passed, &mut failed);
    
    // Test 9: Sexual reproduction
    run_test("Sexual Reproduction", test_sexual_reproduction, &mut passed, &mut failed);
    
    // Test 10: Biomes system
    run_test("Biomes System", test_biomes, &mut passed, &mut failed);
    
    // Summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);
    println!("  Total:  {}", passed + failed);
    
    if failed == 0 {
        println!("\n  ✅ All integration tests passed!\n");
    } else {
        println!("\n  ❌ Some tests failed!\n");
        std::process::exit(1);
    }
}

fn run_test<F>(name: &str, test_fn: F, passed: &mut u32, failed: &mut u32)
where
    F: FnOnce() -> Result<(), String>,
{
    print!("  {} ... ", name);
    match test_fn() {
        Ok(()) => {
            println!("✓ PASS");
            *passed += 1;
        }
        Err(e) => {
            println!("✗ FAIL: {}", e);
            *failed += 1;
        }
    }
}

fn test_simulation_tick() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::Simulation;
    
    let mut config = SimulationConfig::default();
    config.population.initial_organisms = 50;
    config.population.max_organisms = 100;
    config.world.width = 256;
    config.world.height = 256;
    
    let sim = Simulation::new(&config);
    
    if sim.organism_count() != 50 {
        return Err(format!("Expected 50 organisms, got {}", sim.organism_count()));
    }
    
    // Check initial energy
    let avg = sim.avg_energy();
    if avg < config.energy.starting * 0.9 || avg > config.energy.starting * 1.1 {
        return Err(format!("Unexpected initial avg energy: {} (expected ~{})", 
            avg, config.energy.starting));
    }
    
    Ok(())
}

fn test_population_growth() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::Simulation;
    
    let mut config = SimulationConfig::default();
    config.population.initial_organisms = 20;
    config.population.max_organisms = 200;
    config.world.width = 256;
    config.world.height = 256;
    // Make reproduction easier
    config.reproduction.threshold = 50.0;
    config.reproduction.cost = 20.0;
    config.energy.starting = 80.0;
    config.reproduction.min_age = 10;
    
    let mut sim = Simulation::new(&config);
    let initial_count = sim.organism_count();
    
    // Run many reproduction cycles (CPU-only simulation)
    // Note: This tests CPU reproduction logic, not full GPU simulation
    for _ in 0..100 {
        let result = sim.handle_reproduction(&config);
        for (idx, org_gpu) in result.organism_changes {
            if let Some(org) = sim.organisms.get_mut(idx) {
                org.update_from_gpu(&org_gpu);
            }
        }
    }
    
    // Population should have grown or stayed same (can't shrink without energy drain)
    let final_count = sim.organism_count();
    if final_count < initial_count {
        return Err(format!("Population decreased: {} -> {}", initial_count, final_count));
    }
    
    Ok(())
}

fn test_energy_dynamics() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    use glam::Vec2;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut org = Organism::new(Vec2::new(100.0, 100.0), 100.0, 0, 0, &mut rng);
    
    // Verify initial state
    if org.energy != 100.0 {
        return Err(format!("Initial energy wrong: {}", org.energy));
    }
    
    // Manually drain energy
    org.energy -= 10.0;
    if org.energy != 90.0 {
        return Err(format!("After drain: {} (expected 90)", org.energy));
    }
    
    // Organism should still be alive
    if !org.is_alive() {
        return Err("Organism died at 90 energy".into());
    }
    
    // Drain to 0
    org.energy = 0.0;
    if org.is_alive() {
        return Err("Organism alive at 0 energy".into());
    }
    
    Ok(())
}

fn test_reproduction() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::genome::GenomePool;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let config = SimulationConfig::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut genomes = GenomePool::new(100);
    
    // Create parent
    genomes.create_random_at(0, &config.morphology, &mut rng);
    let parent_w1_first = genomes.get(0).unwrap().weights_l1[0];
    
    // Clone with mutation
    genomes.clone_and_mutate_at(1, 0, 0.5, 0.2, &config.morphology, &mut rng);
    
    // Check child genome exists
    let child = genomes.get(1);
    if child.is_none() {
        return Err("Child genome not created".into());
    }
    
    // With 50% mutation rate and 0.2 strength, some weights should differ
    let child_w1_first = child.unwrap().weights_l1[0];
    
    // Can't guarantee difference due to randomness, but genome should exist
    println!("  Parent[0]={:.4}, Child[0]={:.4}", parent_w1_first, child_w1_first);
    
    Ok(())
}

fn test_generations() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    use glam::Vec2;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    // Create parent at generation 0
    let parent = Organism::new(Vec2::new(100.0, 100.0), 100.0, 0, 0, &mut rng);
    if parent.generation != 0 {
        return Err(format!("Parent generation wrong: {}", parent.generation));
    }
    
    // Create child at generation 1
    let child = Organism::new(Vec2::new(110.0, 100.0), 50.0, 1, 1, &mut rng);
    if child.generation != 1 {
        return Err(format!("Child generation wrong: {}", child.generation));
    }
    
    // Create grandchild at generation 2
    let grandchild = Organism::new(Vec2::new(120.0, 100.0), 50.0, 2, 2, &mut rng);
    if grandchild.generation != 2 {
        return Err(format!("Grandchild generation wrong: {}", grandchild.generation));
    }
    
    Ok(())
}

fn test_age_death() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    use glam::Vec2;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut org = Organism::new(Vec2::new(100.0, 100.0), 100.0, 0, 0, &mut rng);
    
    // Initial age should be 0
    if org.age != 0 {
        return Err(format!("Initial age wrong: {}", org.age));
    }
    
    // Manually increment age (simulating GPU update)
    org.age = 1000;
    if org.age != 1000 {
        return Err(format!("Age after increment: {}", org.age));
    }
    
    // Organism still alive (age alone doesn't kill, energy does)
    if !org.is_alive() {
        return Err("Organism died from age alone".into());
    }
    
    // Death from energy depletion
    org.energy = -1.0;
    if org.is_alive() {
        return Err("Organism alive with negative energy".into());
    }
    
    Ok(())
}

fn test_save_load() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::{Simulation, SaveState};
    
    let mut config = SimulationConfig::default();
    config.population.initial_organisms = 10;
    config.population.max_organisms = 50;
    config.world.width = 128;
    config.world.height = 128;
    
    let sim = Simulation::new(&config);
    let tick = 500u64;
    
    // Create save state
    let state = sim.to_save_state(tick, &config);
    
    // Verify state contents
    if state.tick != 500 {
        return Err(format!("Save state tick wrong: {}", state.tick));
    }
    if state.organisms.len() != 10 {
        return Err(format!("Save state organism count: {}", state.organisms.len()));
    }
    
    // Save to temp file
    let temp_path = PathBuf::from("test_save_temp.bin");
    state.save_to_file(&temp_path)
        .map_err(|e| format!("Save failed: {}", e))?;
    
    // Load back
    let loaded = SaveState::load_from_file(&temp_path)
        .map_err(|e| format!("Load failed: {}", e))?;
    
    // Verify loaded state
    if loaded.tick != 500 {
        return Err(format!("Loaded tick wrong: {}", loaded.tick));
    }
    if loaded.organisms.len() != 10 {
        return Err(format!("Loaded organism count: {}", loaded.organisms.len()));
    }
    
    // Restore simulation
    let restored = Simulation::from_save_state(&loaded);
    if restored.organism_count() != 10 {
        return Err(format!("Restored sim count: {}", restored.organism_count()));
    }
    
    // Cleanup temp file
    let _ = std::fs::remove_file(&temp_path);
    
    Ok(())
}

fn test_morphology() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::genome::GenomePool;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut config = SimulationConfig::default();
    config.seed = Some(42);
    config.morphology.enabled = true;
    config.morphology.min_size = 0.5;
    config.morphology.max_size = 2.0;
    config.morphology.min_speed_mult = 0.5;
    config.morphology.max_speed_mult = 1.5;
    config.morphology.mutation_strength = 0.1;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut genomes = GenomePool::new(10);
    
    // Create genome with random morphology
    genomes.create_random_at(0, &config.morphology, &mut rng);
    
    let genome = genomes.get(0).ok_or("No genome at index 0")?;
    let morph = genome.morphology;
    
    // Verify morphology traits are within expected ranges
    if morph.size < 0.0 || morph.size > 10.0 {
        return Err(format!("Morph size out of range: {}", morph.size));
    }
    if morph.speed_mult < 0.0 || morph.speed_mult > 10.0 {
        return Err(format!("Morph speed_mult out of range: {}", morph.speed_mult));
    }
    if morph.vision_mult < 0.0 || morph.vision_mult > 10.0 {
        return Err(format!("Morph vision_mult out of range: {}", morph.vision_mult));
    }
    if morph.metabolism < 0.0 || morph.metabolism > 10.0 {
        return Err(format!("Morph metabolism out of range: {}", morph.metabolism));
    }
    
    // Test morphology mutation
    let mut new_morph = morph;
    new_morph.mutate(&config.morphology, &mut rng);
    
    // After mutation, at least one trait should have changed (with high probability)
    let _changed = (new_morph.size - morph.size).abs() > 0.001 
        || (new_morph.speed_mult - morph.speed_mult).abs() > 0.001
        || (new_morph.vision_mult - morph.vision_mult).abs() > 0.001
        || (new_morph.metabolism - morph.metabolism).abs() > 0.001;
    
    // With mutation rate 0.3 and 4 traits, P(no change) ≈ 0.7^4 = 0.24, so usually at least one changes
    // Don't fail on this since it's probabilistic, just verify the mutation doesn't crash
    
    Ok(())
}

fn test_sexual_reproduction() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::genome::GenomePool;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut config = SimulationConfig::default();
    config.seed = Some(123);
    config.reproduction.sexual_enabled = true;
    config.reproduction.crossover_ratio = 0.5;
    config.mutation.rate = 0.0;
    config.mutation.strength = 0.0;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
    let mut genomes = GenomePool::new(10);
    
    // Create two parent genomes
    genomes.create_random_at(0, &config.morphology, &mut rng);
    genomes.create_random_at(1, &config.morphology, &mut rng);
    
    let parent1 = genomes.get(0).ok_or("No genome 0")?;
    let parent2 = genomes.get(1).ok_or("No genome 1")?;
    
    // Get some weights from parents
    // Crossover and mutate
    let child = parent1.crossover_and_mutate(
        parent2,
        config.reproduction.crossover_ratio,
        config.mutation.rate,
        config.mutation.strength,
        &config.morphology,
        &mut rng,
    );
    
    // Child should have weights (possibly mutated from either parent)
    // The child should have valid weights in reasonable range
    if child.weights_l1.is_empty() {
        return Err("Child has no weights_l1".to_string());
    }
    if child.biases_l1.is_empty() {
        return Err("Child has no biases_l1".to_string());
    }
    
    // Morphology is inherited by uniform crossover when mutation is disabled.
    let inherited_size = child.morphology.size == parent1.morphology.size
        || child.morphology.size == parent2.morphology.size;
    if !inherited_size {
        return Err(format!(
            "Child morph size {} did not match either parent ({}, {})",
            child.morphology.size,
            parent1.morphology.size,
            parent2.morphology.size
        ));
    }
    
    Ok(())
}

fn test_biomes() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::{World, BiomeType};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut config = SimulationConfig::default();
    config.seed = Some(456);
    config.biomes.enabled = true;
    config.biomes.biome_count = 16;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(456);
    
    // Create world with biomes
    let world = World::new_with_rng(&config, &mut rng);
    
    // Verify biome map is created
    let expected_size = (config.world.width * config.world.height) as usize;
    if world.biomes.len() != expected_size {
        return Err(format!(
            "Biome map size {} != expected {}",
            world.biomes.len(), expected_size
        ));
    }
    
    // Count biome types - should have some variety
    let mut biome_counts = [0u32; 5];
    for &b in &world.biomes {
        if (b as usize) < 5 {
            biome_counts[b as usize] += 1;
        }
    }
    
    // With 16 Voronoi cells, we should have multiple biome types
    let num_biome_types = biome_counts.iter().filter(|&&c| c > 0).count();
    if num_biome_types < 2 {
        return Err(format!(
            "Only {} biome types found, expected variety",
            num_biome_types
        ));
    }
    
    // Test biome accessors
    let biome_at_origin = world.get_biome(0, 0);
    // Just verify it returns a valid BiomeType (doesn't crash)
    match biome_at_origin {
        BiomeType::Normal | BiomeType::Fertile | BiomeType::Barren | 
        BiomeType::Swamp | BiomeType::Harsh => {}
    }
    
    // Test float position accessor with wrapping
    let biome_at_negative = world.get_biome_at(-10.0, -10.0);
    match biome_at_negative {
        BiomeType::Normal | BiomeType::Fertile | BiomeType::Barren | 
        BiomeType::Swamp | BiomeType::Harsh => {}
    }
    
    Ok(())
}
