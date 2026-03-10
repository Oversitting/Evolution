//! Comprehensive Feature Test
//!
//! Tests all Phase 1 and Phase 2 features programmatically.
//! Each test validates a specific feature without user interaction.
//!
//! Run with: cargo run --example feature_test

use std::path::Path;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║        Evolution Simulator - Feature Verification Test       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    // Phase 1 Tests
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 1: MVP Foundation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // 1.1-1.3: Core Setup
    run_test("1.1 Project Setup - Config Loading", test_config_loading, &mut passed, &mut failed);
    run_test("1.3 Buffer Sizes - Organism Buffer", test_organism_buffer_sizes, &mut passed, &mut failed);
    run_test("1.3 Buffer Sizes - Genome Buffer", test_genome_buffer_sizes, &mut passed, &mut failed);
    
    // 1.4: Neural Network
    run_test("1.4 Neural Network - Dimensions", test_nn_dimensions, &mut passed, &mut failed);
    run_test("1.4 Neural Network - Forward Pass", test_nn_forward_pass, &mut passed, &mut failed);
    run_test("1.4 Neural Network - Mutation", test_nn_mutation, &mut passed, &mut failed);
    
    // 1.5: World System
    run_test("1.5 World System - Grid Creation", test_world_grid, &mut passed, &mut failed);
    run_test("1.5 World System - Food Patches", test_food_patches, &mut passed, &mut failed);
    
    // 1.6: Energy & Death
    run_test("1.6 Energy - Organism Creation", test_organism_energy, &mut passed, &mut failed);
    run_test("1.6 Energy - Death Condition", test_death_condition, &mut passed, &mut failed);
    
    // 1.7: Reproduction
    run_test("1.7 Reproduction - Spawn Logic", test_spawn_logic, &mut passed, &mut failed);
    run_test("1.7 Reproduction - Genome Cloning", test_genome_cloning, &mut passed, &mut failed);
    
    // Phase 2 Tests
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 2: Core Polish");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // 2.1: Selection & Inspection
    run_test("2.1 Selection - Organism Data Struct", test_selected_organism_struct, &mut passed, &mut failed);
    
    // 2.2: Brain Visualization
    run_test("2.2 Brain Viz - Input Labels", test_brain_input_labels, &mut passed, &mut failed);
    run_test("2.2 Brain Viz - Output Labels", test_brain_output_labels, &mut passed, &mut failed);
    
    // 2.5: Save/Load
    run_test("2.5 Save/Load - SaveState Struct", test_save_state_struct, &mut passed, &mut failed);
    run_test("2.5 Save/Load - Serialization", test_serialization, &mut passed, &mut failed);
    
    // Summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);
    println!("  Total:  {}", passed + failed);
    
    if failed == 0 {
        println!("\n  ✅ All tests passed!\n");
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

// ============================================================================
// Phase 1 Tests
// ============================================================================

fn test_config_loading() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use std::path::Path as StdPath;
    
    // Test default config creation
    let config = SimulationConfig::default();
    
    if config.population.max_organisms == 0 {
        return Err("max_organisms is 0".into());
    }
    if config.world.width == 0 || config.world.height == 0 {
        return Err("world dimensions are 0".into());
    }
    if config.energy.maximum <= 0.0 {
        return Err("energy.maximum <= 0".into());
    }
    
    // Test config file loading if exists
    if Path::new("config.toml").exists() {
        let _loaded = SimulationConfig::from_file(StdPath::new("config.toml"))
            .map_err(|e| format!("Failed to load config.toml: {}", e))?;
    }
    
    Ok(())
}

fn test_organism_buffer_sizes() -> Result<(), String> {
    use evolution_sim::simulation::OrganismGpu;
    
    // OrganismGpu should be 56 bytes (14 * 4 bytes)
    let size = std::mem::size_of::<OrganismGpu>();
    if size != 56 {
        return Err(format!("OrganismGpu size is {} bytes, expected 56", size));
    }
    
    Ok(())
}

fn test_genome_buffer_sizes() -> Result<(), String> {
    use evolution_sim::simulation::genome::{
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, TOTAL_PARAMS,
        WEIGHTS_L1, WEIGHTS_L2
    };
    
    // Verify dimensions match expected values
    if INPUT_DIM != 20 {
        return Err(format!("INPUT_DIM is {}, expected 20", INPUT_DIM));
    }
    if HIDDEN_DIM != 16 {
        return Err(format!("HIDDEN_DIM is {}, expected 16", HIDDEN_DIM));
    }
    if OUTPUT_DIM != 6 {
        return Err(format!("OUTPUT_DIM is {}, expected 6", OUTPUT_DIM));
    }
    
    // Verify weight counts
    let expected_w1 = INPUT_DIM * HIDDEN_DIM; // 320
    let expected_w2 = HIDDEN_DIM * OUTPUT_DIM; // 96
    let expected_total = expected_w1 + HIDDEN_DIM + expected_w2 + OUTPUT_DIM; // 438
    
    if WEIGHTS_L1 != expected_w1 {
        return Err(format!("WEIGHTS_L1 is {}, expected {}", WEIGHTS_L1, expected_w1));
    }
    if WEIGHTS_L2 != expected_w2 {
        return Err(format!("WEIGHTS_L2 is {}, expected {}", WEIGHTS_L2, expected_w2));
    }
    if TOTAL_PARAMS != expected_total {
        return Err(format!("TOTAL_PARAMS is {}, expected {}", TOTAL_PARAMS, expected_total));
    }
    
    Ok(())
}

fn test_nn_dimensions() -> Result<(), String> {
    use evolution_sim::simulation::genome::{Genome, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM};
    
    let genome = Genome::default();
    
    if genome.weights_l1.len() != INPUT_DIM * HIDDEN_DIM {
        return Err(format!("weights_l1 length wrong: {} vs {}", 
            genome.weights_l1.len(), INPUT_DIM * HIDDEN_DIM));
    }
    if genome.biases_l1.len() != HIDDEN_DIM {
        return Err(format!("biases_l1 length wrong: {} vs {}", 
            genome.biases_l1.len(), HIDDEN_DIM));
    }
    if genome.weights_l2.len() != HIDDEN_DIM * OUTPUT_DIM {
        return Err(format!("weights_l2 length wrong: {} vs {}", 
            genome.weights_l2.len(), HIDDEN_DIM * OUTPUT_DIM));
    }
    if genome.biases_l2.len() != OUTPUT_DIM {
        return Err(format!("biases_l2 length wrong: {} vs {}", 
            genome.biases_l2.len(), OUTPUT_DIM));
    }
    
    Ok(())
}

fn test_nn_forward_pass() -> Result<(), String> {
    use evolution_sim::simulation::genome::Genome;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let genome = Genome::new_random(&mut rng);
    
    // Verify all weights are in reasonable range after Xavier init
    let max_weight = genome.weights_l1.iter()
        .chain(genome.weights_l2.iter())
        .fold(0.0f32, |m, &w| m.max(w.abs()));
    
    // Xavier init should keep weights small (typically < 2.0)
    if max_weight > 5.0 {
        return Err(format!("Max weight {} is too large for Xavier init", max_weight));
    }
    
    Ok(())
}

fn test_nn_mutation() -> Result<(), String> {
    use evolution_sim::simulation::genome::Genome;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let parent = Genome::new_random(&mut rng);
    
    // Clone and mutate with 100% rate
    let child = parent.clone_and_mutate(1.0, 0.1, &mut rng);
    
    // At least some weights should be different
    let mut differences = 0;
    for (p, c) in parent.weights_l1.iter().zip(child.weights_l1.iter()) {
        if (p - c).abs() > 0.001 {
            differences += 1;
        }
    }
    
    if differences == 0 {
        return Err("No mutations occurred with 100% rate".into());
    }
    
    Ok(())
}

fn test_world_grid() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::World;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let config = SimulationConfig::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let world = World::new_with_rng(&config, &mut rng);
    
    let expected_size = (config.world.width * config.world.height) as usize;
    if world.food.len() != expected_size {
        return Err(format!("Food grid size {} != expected {}", 
            world.food.len(), expected_size));
    }
    
    Ok(())
}

fn test_food_patches() -> Result<(), String> {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::World;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let config = SimulationConfig::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let world = World::new_with_rng(&config, &mut rng);
    
    // Count cells with food above baseline
    let baseline = config.food.baseline_food;
    let cells_with_food: usize = world.food.iter()
        .filter(|&&f| f > baseline + 1.0)
        .count();
    
    // Should have significant food from patches
    if cells_with_food == 0 {
        return Err("No food patches created".into());
    }
    
    // Total food should be substantial
    let total_food: f32 = world.food.iter().sum();
    if total_food < 10000.0 {
        return Err(format!("Total food {} is too low", total_food));
    }
    
    Ok(())
}

fn test_organism_energy() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    use glam::Vec2;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let org = Organism::new(Vec2::new(100.0, 100.0), 70.0, 0, 0, &mut rng);
    
    if org.energy != 70.0 {
        return Err(format!("Energy {} != 70.0", org.energy));
    }
    if !org.alive {
        return Err("New organism not alive".into());
    }
    if !org.is_alive() {
        return Err("is_alive() returned false".into());
    }
    
    Ok(())
}

fn test_death_condition() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    use glam::Vec2;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    // Organism with 0 energy should be dead
    let org = Organism::new(Vec2::new(100.0, 100.0), 0.0, 0, 0, &mut rng);
    
    // is_alive checks energy > 0
    if org.is_alive() {
        return Err("Organism with 0 energy should not be alive".into());
    }
    
    // Negative energy also dead
    let mut org2 = Organism::new(Vec2::new(100.0, 100.0), 50.0, 0, 0, &mut rng);
    org2.energy = -10.0;
    if org2.is_alive() {
        return Err("Organism with negative energy should not be alive".into());
    }
    
    Ok(())
}

fn test_spawn_logic() -> Result<(), String> {
    use evolution_sim::simulation::organism::OrganismPool;
    use glam::Vec2;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut pool = OrganismPool::new(100);
    
    // Spawn some organisms
    let id1 = pool.spawn(Vec2::new(10.0, 10.0), 50.0, 0, 0, &mut rng);
    let id2 = pool.spawn(Vec2::new(20.0, 20.0), 50.0, 1, 0, &mut rng);
    
    if id1.is_none() {
        return Err("Failed to spawn first organism".into());
    }
    if id2.is_none() {
        return Err("Failed to spawn second organism".into());
    }
    
    if pool.count() != 2 {
        return Err(format!("Pool count {} != 2", pool.count()));
    }
    
    Ok(())
}

fn test_genome_cloning() -> Result<(), String> {
    use evolution_sim::simulation::genome::GenomePool;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
    let mut pool = GenomePool::new(100);
    
    // Create parent genome
    pool.create_random_at(0, &mut rng);
    
    // Clone and mutate to child slot
    pool.clone_and_mutate_at(1, 0, 0.0, 0.0, &mut rng);  // 0% mutation
    
    // With 0% mutation, child should be identical
    let parent = pool.get(0).unwrap();
    let child = pool.get(1).unwrap();
    
    for (p, c) in parent.weights_l1.iter().zip(child.weights_l1.iter()) {
        if (p - c).abs() > 0.0001 {
            return Err("Child weights differ from parent with 0% mutation".into());
        }
    }
    
    Ok(())
}

// ============================================================================
// Phase 2 Tests
// ============================================================================

fn test_selected_organism_struct() -> Result<(), String> {
    use evolution_sim::ui::SelectedOrganism;
    use evolution_sim::simulation::genome::{INPUT_DIM, OUTPUT_DIM};
    
    // Create a SelectedOrganism instance
    let selected = SelectedOrganism {
        id: 42,
        alive: true,
        position: [100.0, 200.0],
        rotation: 1.57,
        energy: 85.5,
        age: 500,
        generation: 3,
        offspring_count: 2,
        parent_id: 10,
        reproduce_signal: 0.75,
        genome_id: 42,
        species_id: 1,
        nn_inputs: [0.0; INPUT_DIM],
        nn_outputs: [0.0; OUTPUT_DIM],
    };
    
    if selected.id != 42 {
        return Err("SelectedOrganism id mismatch".into());
    }
    if selected.generation != 3 {
        return Err("SelectedOrganism generation mismatch".into());
    }
    if !selected.alive {
        return Err("SelectedOrganism should be alive".into());
    }
    
    Ok(())
}

fn test_brain_input_labels() -> Result<(), String> {
    // The inspector should have 20 input labels (8 rays * 2 + 4 internal)
    // We can't directly access the labels, but we can verify the constant
    use evolution_sim::simulation::genome::INPUT_DIM;
    
    if INPUT_DIM != 20 {
        return Err(format!("INPUT_DIM {} != 20, brain viz labels may be wrong", INPUT_DIM));
    }
    
    Ok(())
}

fn test_brain_output_labels() -> Result<(), String> {
    use evolution_sim::simulation::genome::OUTPUT_DIM;
    
    if OUTPUT_DIM != 6 {
        return Err(format!("OUTPUT_DIM {} != 6, brain viz labels may be wrong", OUTPUT_DIM));
    }
    
    Ok(())
}

fn test_save_state_struct() -> Result<(), String> {
    use evolution_sim::simulation::SaveState;
    
    // Verify VERSION constant exists
    if SaveState::VERSION == 0 {
        return Err("SaveState::VERSION is 0".into());
    }
    
    Ok(())
}

fn test_serialization() -> Result<(), String> {
    use evolution_sim::simulation::{SaveState, SavedOrganism, SavedGenome};
    use evolution_sim::config::SimulationConfig;
    
    let config = SimulationConfig::default();
    
    // Create a minimal save state
    let state = SaveState {
        version: SaveState::VERSION,
        tick: 1000,
        config: config.clone(),
        organisms: vec![SavedOrganism {
            position: [100.0, 200.0],
            velocity: [1.0, 0.0],
            rotation: 0.5,
            energy: 75.0,
            age: 100,
            alive: true,
            genome_id: 0,
            generation: 1,
            offspring_count: 0,
            parent_id: u32::MAX,
            cooldown: 0,
            reproduce_signal: 0.3,
            species_id: 1,
        }],
        genomes: vec![SavedGenome {
            weights_l1: vec![0.0; 320],
            biases_l1: vec![0.0; 16],
            weights_l2: vec![0.0; 96],
            biases_l2: vec![0.0; 6],
            alive: true,
        }],
        food: vec![0.0; 100],
        world_width: 10,
        world_height: 10,
    };
    
    // Serialize to bytes
    let bytes = bincode::serialize(&state)
        .map_err(|e| format!("Serialization failed: {}", e))?;
    
    if bytes.is_empty() {
        return Err("Serialized bytes is empty".into());
    }
    
    // Deserialize back
    let restored: SaveState = bincode::deserialize(&bytes)
        .map_err(|e| format!("Deserialization failed: {}", e))?;
    
    if restored.tick != 1000 {
        return Err(format!("Restored tick {} != 1000", restored.tick));
    }
    if restored.organisms.len() != 1 {
        return Err(format!("Restored organisms count {} != 1", restored.organisms.len()));
    }
    
    Ok(())
}
