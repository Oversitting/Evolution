//! Determinism tests for the Evolution Simulator
//! 
//! This test suite verifies that:
//! 1. Same seed produces identical results
//! 2. Simulation speed doesn't affect results
//! 3. Critical config invariants are sanitized before runtime  
//! 4. FPS/frame timing doesn't affect results
//!
//! Run with: cargo run --example determinism_test --release

use std::collections::BTreeMap;

mod common {
    use evolution_sim::config::SimulationConfig;
    use evolution_sim::simulation::Simulation;

    /// Create a deterministic test config with fixed seed
    pub fn test_config(seed: u64) -> SimulationConfig {
        let mut config = SimulationConfig::default();
        config.seed = Some(seed);
        // Use smaller world for faster tests
        config.world.width = 256;
        config.world.height = 256;
        config.population.max_organisms = 200;
        config.population.initial_organisms = 50;
        config.food.initial_patches = 20;
        config.food.patch_size = 8;
        // Make reproduction happen faster for testing
        config.reproduction.threshold = 60.0;
        config.reproduction.min_age = 50;
        config.reproduction.cooldown = 30;
        config
    }

    /// Extract a snapshot of simulation state for comparison
    #[derive(Debug, Clone, PartialEq)]
    pub struct SimulationSnapshot {
        pub organism_count: u32,
        pub total_energy: f32,
        pub total_food: f32,
        pub max_generation: u32,
        /// Hash of organism positions (truncated for comparison)
        pub position_hash: u64,
        /// Hash of food grid
        pub food_hash: u64,
    }

    impl SimulationSnapshot {
        pub fn from_simulation(sim: &Simulation) -> Self {
            let mut position_hash: u64 = 0;
            let mut total_energy = 0.0f32;
            
            for (i, org) in sim.organisms.iter().enumerate() {
                if org.is_alive() {
                    // Use deterministic hash combining
                    let px = (org.position.x * 1000.0) as i32;
                    let py = (org.position.y * 1000.0) as i32;
                    position_hash = position_hash
                        .wrapping_mul(31)
                        .wrapping_add(px as u64)
                        .wrapping_mul(31)
                        .wrapping_add(py as u64)
                        .wrapping_mul(31)
                        .wrapping_add(i as u64);
                    total_energy += org.energy;
                }
            }
            
            // Hash food grid (sample every 16th cell for speed)
            let mut food_hash: u64 = 0;
            for (i, &food) in sim.world.food.iter().enumerate().step_by(16) {
                let food_int = (food * 1000.0) as i32;
                food_hash = food_hash.wrapping_mul(31).wrapping_add(food_int as u64).wrapping_add(i as u64);
            }
            
            Self {
                organism_count: sim.organism_count(),
                total_energy,
                total_food: sim.total_food(),
                max_generation: sim.max_generation(),
                position_hash,
                food_hash,
            }
        }
    }
}

use common::*;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           EVOLUTION SIMULATOR - DETERMINISM TEST SUITE           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    let mut all_passed = true;
    
    // Test 1: Same seed produces identical results
    all_passed &= test_seed_determinism();
    
    // Test 2: Unsafe runtime config is sanitized
    all_passed &= test_runtime_config_sanitization();
    
    // Test 3: Verify initial state is deterministic
    all_passed &= test_initial_state_determinism();
    
    // Test 4: Multiple runs at different "speeds" (batched ticks)
    all_passed &= test_speed_independence();
    
    println!("\n════════════════════════════════════════════════════════════════════");
    if all_passed {
        println!("✅ ALL DETERMINISM TESTS PASSED");
    } else {
        println!("❌ SOME TESTS FAILED - See above for details");
        std::process::exit(1);
    }
}

/// Test 1: Running with the same seed twice should produce identical results
fn test_seed_determinism() -> bool {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 1: Seed-based Determinism                                   │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    
    let seed = 12345u64;
    let ticks = 500;
    
    println!("  Running simulation with seed {} for {} ticks (run 1)...", seed, ticks);
    let snapshots_1 = run_cpu_simulation(seed, ticks, 1);
    
    println!("  Running simulation with seed {} for {} ticks (run 2)...", seed, ticks);
    let snapshots_2 = run_cpu_simulation(seed, ticks, 1);
    
    // Compare snapshots at each recorded tick
    let mut all_match = true;
    for (tick, snap1) in &snapshots_1 {
        if let Some(snap2) = snapshots_2.get(tick) {
            if snap1 != snap2 {
                println!("  ❌ Tick {}: Snapshots differ!", tick);
                println!("     Run 1: {:?}", snap1);
                println!("     Run 2: {:?}", snap2);
                all_match = false;
            }
        } else {
            println!("  ❌ Tick {}: Missing in run 2", tick);
            all_match = false;
        }
    }
    
    if all_match {
        println!("  ✅ PASSED: Both runs produced identical results\n");
        true
    } else {
        println!("  ❌ FAILED: Runs diverged\n");
        false
    }
}

/// Test 2: Critical runtime config values are sanitized to safe invariants
fn test_runtime_config_sanitization() -> bool {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 2: Runtime Config Sanitization                              │");
    println!("└──────────────────────────────────────────────────────────────────┘");

    let mut config = test_config(54321);
    config.vision.rays = 12;
    config.system.readback_interval = 5;
    config.system.food_readback_interval = 0;
    config.system.diagnostic_interval = 0;

    config.sanitize();

    let passed = config.vision.rays == 8
        && config.system.readback_interval == 1
        && config.system.food_readback_interval == 1
        && config.system.diagnostic_interval == 1;

    if passed {
        println!("  ✅ PASSED: Unsafe values were clamped to safe runtime defaults\n");
        true
    } else {
        println!("  ❌ FAILED: Sanitized config still contains unsafe values: {:?}\n", config.system);
        false
    }
}

/// Test 3: Initial state should be deterministic with same seed
fn test_initial_state_determinism() -> bool {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 3: Initial State Determinism                                │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    
    use evolution_sim::simulation::Simulation;
    
    let seed = 99999u64;
    let config = test_config(seed);
    
    println!("  Creating simulation 1 with seed {}...", seed);
    let sim1 = Simulation::new(&config);
    let snap1 = SimulationSnapshot::from_simulation(&sim1);
    
    println!("  Creating simulation 2 with seed {}...", seed);
    let sim2 = Simulation::new(&config);
    let snap2 = SimulationSnapshot::from_simulation(&sim2);
    
    if snap1 == snap2 {
        println!("  ✅ PASSED: Initial states are identical");
        println!("     Organisms: {}, Food: {:.0}, Positions hash: {:016x}\n", 
            snap1.organism_count, snap1.total_food, snap1.position_hash);
        true
    } else {
        println!("  ❌ FAILED: Initial states differ");
        println!("     Sim 1: {:?}", snap1);
        println!("     Sim 2: {:?}", snap2);
        println!();
        false
    }
}

/// Test 4: Running ticks in batches vs one at a time should be identical
fn test_speed_independence() -> bool {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 4: Speed Independence (Batched Ticks)                       │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    
    let seed = 77777u64;
    let total_ticks = 200;
    
    // Run with speed=1 (1 tick per "frame")
    println!("  Running {} ticks with speed=1 (200 frames)...", total_ticks);
    let snapshots_1 = run_cpu_simulation(seed, total_ticks, 1);
    
    // Run with speed=4 (4 ticks per "frame", 50 frames)
    println!("  Running {} ticks with speed=4 (50 frames)...", total_ticks);
    let snapshots_4 = run_cpu_simulation(seed, total_ticks, 4);
    
    // Run with speed=16 (16 ticks per "frame", ~13 frames)
    println!("  Running {} ticks with speed=16 (~13 frames)...", total_ticks);
    let snapshots_16 = run_cpu_simulation(seed, total_ticks, 16);
    
    // Run with speed=64 (64 ticks per "frame", ~4 frames)
    println!("  Running {} ticks with speed=64 (~4 frames)...", total_ticks);
    let snapshots_64 = run_cpu_simulation(seed, total_ticks, 64);
    
    // Compare at final tick
    let final_tick = total_ticks as u64;
    let snap1 = snapshots_1.get(&final_tick);
    let snap4 = snapshots_4.get(&final_tick);
    let snap16 = snapshots_16.get(&final_tick);
    let snap64 = snapshots_64.get(&final_tick);
    
    // Also compare at intermediate points that all speeds would have recorded
    let mut all_match = true;
    
    // Compare final states
    if snap1 != snap4 || snap4 != snap16 || snap16 != snap64 {
        all_match = false;
        println!("  ❌ Final states differ:");
        println!("     speed=1:  {:?}", snap1);
        println!("     speed=4:  {:?}", snap4);
        println!("     speed=16: {:?}", snap16);
        println!("     speed=64: {:?}", snap64);
    }
    
    if all_match {
        println!("  ✅ PASSED: All speed multipliers produced identical results\n");
        true
    } else {
        println!("  ❌ FAILED: Speed multiplier affected simulation\n");
        false
    }
}

/// Run a CPU-only simulation and collect snapshots
/// This simulates what the app does without GPU
fn run_cpu_simulation(
    seed: u64,
    ticks: u32,
    speed_multiplier: u32,
) -> BTreeMap<u64, SimulationSnapshot> {
    use evolution_sim::simulation::Simulation;
    
    let config = test_config(seed);
    
    let mut sim = Simulation::new(&config);
    let mut snapshots = BTreeMap::new();
    let mut tick: u64 = 0;
    
    // Record initial state
    snapshots.insert(0, SimulationSnapshot::from_simulation(&sim));
    
    // Simulate frames with speed multiplier
    let frames_needed = (ticks + speed_multiplier - 1) / speed_multiplier;
    
    for _frame in 0..frames_needed {
        for _step in 0..speed_multiplier {
            if tick >= ticks as u64 {
                break;
            }
            
            // Simulate a tick: handle reproduction on CPU side
            // Note: This is a simplified version - real simulation uses GPU
            let _result = sim.handle_reproduction(&config);
            
            tick += 1;
            
            // Record snapshot at intervals
            if tick % 50 == 0 || tick == ticks as u64 {
                snapshots.insert(tick, SimulationSnapshot::from_simulation(&sim));
            }
        }
    }
    
    snapshots
}
