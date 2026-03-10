//! Boundary condition and edge case tests
//! 
//! Tests for edge cases identified in the code analysis.

use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::{Simulation, OrganismPool};
use glam::Vec2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    println!("=== Boundary Condition Tests ===\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    let tests: Vec<(&str, fn() -> Result<(), String>)> = vec![
        ("Zero Organisms", test_zero_organisms),
        ("Single Organism", test_single_organism),
        ("Max Organisms Limit", test_max_organisms_limit),
        ("Energy At Zero", test_energy_at_zero),
        ("Energy At Maximum", test_energy_at_maximum),
        ("World Wrapping X", test_world_wrapping_x),
        ("World Wrapping Y", test_world_wrapping_y),
        ("Age At Max Age", test_age_at_max_age),
        ("OrganismPool Restore Bounds", test_organism_pool_restore_bounds),
        ("Count Saturating Sub", test_count_saturating),
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

fn test_zero_organisms() -> Result<(), String> {
    let mut config = SimulationConfig::default();
    config.population.initial_organisms = 0;
    
    let sim = Simulation::new(&config);
    
    if sim.organism_count() != 0 {
        return Err(format!("Expected 0 organisms, got {}", sim.organism_count()));
    }
    
    // Check stats don't panic on empty
    let avg = sim.avg_energy();
    let max_gen = sim.max_generation();
    
    if avg != 0.0 {
        return Err(format!("Expected avg energy 0.0, got {}", avg));
    }
    
    if max_gen != 0 {
        return Err(format!("Expected max generation 0, got {}", max_gen));
    }
    
    Ok(())
}

fn test_single_organism() -> Result<(), String> {
    let mut config = SimulationConfig::default();
    config.population.initial_organisms = 1;
    config.population.max_organisms = 1;
    
    let sim = Simulation::new(&config);
    
    if sim.organism_count() != 1 {
        return Err(format!("Expected 1 organism, got {}", sim.organism_count()));
    }
    
    Ok(())
}

fn test_max_organisms_limit() -> Result<(), String> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut pool = OrganismPool::new(5);  // Very small limit
    
    // Try to spawn 10 organisms
    let mut spawned = 0;
    for i in 0..10 {
        let result = pool.spawn(
            Vec2::new(i as f32 * 10.0, 0.0),
            50.0,
            i as u32,
            0,
            &mut rng,
        );
        if result.is_some() {
            spawned += 1;
        }
    }
    
    // Should only spawn 5 (the limit)
    if spawned != 5 {
        return Err(format!("Expected 5 spawned, got {}", spawned));
    }
    
    if pool.count() != 5 {
        return Err(format!("Expected count 5, got {}", pool.count()));
    }
    
    Ok(())
}

fn test_energy_at_zero() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    // Create organism with exactly 0 energy
    let org = Organism::new(Vec2::new(0.0, 0.0), 0.0, 0, 0, &mut rng);
    
    // Should not be considered alive (energy must be > 0)
    if org.is_alive() {
        return Err("Organism with 0 energy should not be alive".into());
    }
    
    Ok(())
}

fn test_energy_at_maximum() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    
    let config = SimulationConfig::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    // Create organism at max energy
    let mut org = Organism::new(Vec2::new(0.0, 0.0), config.energy.maximum, 0, 0, &mut rng);
    
    // Should be alive
    if !org.is_alive() {
        return Err("Organism at max energy should be alive".into());
    }
    
    // Try to add more energy - should cap at maximum (in simulation, not in Organism struct directly)
    org.energy += 100.0;
    
    // The raw energy can exceed max, but act.wgsl clamps it
    // This test just verifies the struct handles large values
    if org.energy < config.energy.maximum {
        return Err("Energy should be at least at max".into());
    }
    
    Ok(())
}

fn test_world_wrapping_x() -> Result<(), String> {
    let config = SimulationConfig::default();
    
    // Test wrapping math (same as act.wgsl)
    let world_width = config.world.width as f32;
    
    // Position slightly over the edge
    let mut pos_x = world_width + 5.0;
    pos_x = ((pos_x % world_width) + world_width) % world_width;
    
    if pos_x < 0.0 || pos_x >= world_width {
        return Err(format!("Wrapped X position {} out of bounds", pos_x));
    }
    
    if (pos_x - 5.0).abs() > 0.001 {
        return Err(format!("Wrapped X should be 5.0, got {}", pos_x));
    }
    
    // Test negative wrapping
    let mut neg_x = -10.0;
    neg_x = ((neg_x % world_width) + world_width) % world_width;
    
    let expected = world_width - 10.0;
    if (neg_x - expected).abs() > 0.001 {
        return Err(format!("Negative wrapped X should be {}, got {}", expected, neg_x));
    }
    
    Ok(())
}

fn test_world_wrapping_y() -> Result<(), String> {
    let config = SimulationConfig::default();
    
    let world_height = config.world.height as f32;
    
    // Position at exact boundary
    let mut pos_y = world_height;
    pos_y = ((pos_y % world_height) + world_height) % world_height;
    
    if (pos_y).abs() > 0.001 {
        return Err(format!("Wrapped Y at boundary should be 0.0, got {}", pos_y));
    }
    
    // Very large negative
    let mut big_neg = -world_height * 3.0 - 100.0;
    big_neg = ((big_neg % world_height) + world_height) % world_height;
    
    if big_neg < 0.0 || big_neg >= world_height {
        return Err(format!("Large negative wrapped Y {} out of bounds", big_neg));
    }
    
    Ok(())
}

fn test_age_at_max_age() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    
    let config = SimulationConfig::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    let mut org = Organism::new(Vec2::new(0.0, 0.0), 100.0, 0, 0, &mut rng);
    
    // Set age to exactly max_age
    org.age = config.energy.max_age;
    
    // Organism is still alive on CPU side (death check is in GPU shader)
    // But we can verify the age value is handled
    if org.age != config.energy.max_age {
        return Err(format!("Age should be {}, got {}", config.energy.max_age, org.age));
    }
    
    // Set age beyond max_age
    org.age = config.energy.max_age + 100;
    
    if org.age <= config.energy.max_age {
        return Err("Age should be able to exceed max_age".into());
    }
    
    Ok(())
}

fn test_organism_pool_restore_bounds() -> Result<(), String> {
    use evolution_sim::simulation::organism::Organism;
    
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut pool = OrganismPool::new(3);  // Very small
    
    // Restore organisms up to max
    for i in 0..3 {
        let org = Organism::new(Vec2::new(i as f32, 0.0), 50.0, i as u32, 0, &mut rng);
        pool.restore(org);
    }
    
    if pool.count() != 3 {
        return Err(format!("Expected 3 organisms, got {}", pool.count()));
    }
    
    // Try to restore beyond max - should be ignored (with warning)
    let extra = Organism::new(Vec2::new(100.0, 0.0), 50.0, 99, 0, &mut rng);
    pool.restore(extra);
    
    // Count should still be 3 (extra was not added)
    if pool.count() != 3 {
        return Err(format!("Count should still be 3 after overflow attempt, got {}", pool.count()));
    }
    
    Ok(())
}

fn test_count_saturating() -> Result<(), String> {
    // Test that count operations don't underflow
    // This is a regression test for the fix to cleanup_dead
    
    let config = SimulationConfig::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut pool = OrganismPool::new(10);
    
    // Spawn one organism
    pool.spawn(Vec2::new(0.0, 0.0), 50.0, 0, 0, &mut rng);
    
    if pool.count() != 1 {
        return Err(format!("Expected count 1, got {}", pool.count()));
    }
    
    // The cleanup_dead function no longer modifies count directly,
    // so there's no risk of underflow. This test just verifies
    // the current behavior is stable.
    
    let mut genomes = evolution_sim::simulation::GenomePool::new(10);
    pool.cleanup_dead(&mut genomes);
    
    // Count should remain unchanged since the organism is alive
    if pool.count() != 1 {
        return Err(format!("Count changed unexpectedly: {}", pool.count()));
    }
    
    Ok(())
}
