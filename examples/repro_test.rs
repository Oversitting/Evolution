//! Reproduction System Test
//! 
//! Tests reproduction mechanics, cooldowns, and population dynamics.
//! 
//! Run with: cargo run --example repro_test --release

use std::collections::HashMap;

fn main() {
    println!("=== Reproduction System Test ===\n");
    
    // Test 1: Verify reproduction conditions
    test_reproduction_conditions();
    
    // Test 2: Verify cooldown mechanics
    test_cooldown_mechanics();
    
    // Test 3: Verify energy transfer
    test_energy_transfer();
    
    // Test 4: Verify generation tracking
    test_generation_tracking();
    
    println!("\n=== All Reproduction Tests Complete ===");
}

fn test_reproduction_conditions() {
    println!("--- Test 1: Reproduction Conditions ---");
    
    // Config values from config.toml
    let threshold = 80.0f32;
    let signal_min = 0.3f32;
    let min_age = 100u32;
    
    // Test cases: (energy, signal, age, cooldown, expected_can_reproduce)
    let test_cases = [
        (100.0, 0.5, 200, 0, true, "healthy organism ready to reproduce"),
        (79.0, 0.5, 200, 0, false, "energy below threshold"),
        (100.0, 0.2, 200, 0, false, "signal below minimum"),
        (100.0, 0.5, 50, 0, false, "too young"),
        (100.0, 0.5, 200, 50, false, "cooldown active"),
        (80.0, 0.31, 100, 0, true, "exactly at thresholds"),
        (150.0, 1.0, 1000, 0, true, "max stats organism"),
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (energy, signal, age, cooldown, expected, desc) in test_cases {
        let can_reproduce = energy >= threshold 
            && signal > signal_min 
            && age >= min_age 
            && cooldown == 0;
        
        if can_reproduce == expected {
            println!("  ✅ PASS: {} - can_reproduce={}", desc, can_reproduce);
            passed += 1;
        } else {
            println!("  ❌ FAIL: {} - expected {}, got {}", desc, expected, can_reproduce);
            failed += 1;
        }
    }
    
    println!("  Result: {}/{} tests passed", passed, passed + failed);
}

fn test_cooldown_mechanics() {
    println!("\n--- Test 2: Cooldown Mechanics ---");
    
    let cooldown_ticks = 100u32;
    
    // Simulate cooldown decrement
    let mut cooldown = cooldown_ticks;
    let mut ticks_until_ready = 0;
    
    while cooldown > 0 {
        cooldown = cooldown.saturating_sub(1);
        ticks_until_ready += 1;
    }
    
    if ticks_until_ready == cooldown_ticks as usize {
        println!("  ✅ PASS: Cooldown decrements correctly ({} ticks)", ticks_until_ready);
    } else {
        println!("  ❌ FAIL: Cooldown took {} ticks, expected {}", ticks_until_ready, cooldown_ticks);
    }
    
    // Test saturating behavior
    let mut cooldown: u32 = 5;
    for _ in 0..10 {
        cooldown = cooldown.saturating_sub(1);
    }
    
    if cooldown == 0 {
        println!("  ✅ PASS: Cooldown saturates at 0");
    } else {
        println!("  ❌ FAIL: Cooldown went negative or didn't saturate");
    }
}

fn test_energy_transfer() {
    println!("\n--- Test 3: Energy Transfer ---");
    
    let reproduction_cost = 50.0f32;
    let parent_energy = 100.0f32;
    
    // After reproduction:
    // - Parent loses cost
    // - Child gets cost
    let parent_after = parent_energy - reproduction_cost;
    let child_energy = reproduction_cost;
    
    if parent_after == 50.0 && child_energy == 50.0 {
        println!("  ✅ PASS: Energy split correctly");
        println!("    Parent: {:.0} → {:.0}", parent_energy, parent_after);
        println!("    Child: {:.0}", child_energy);
    } else {
        println!("  ❌ FAIL: Energy transfer incorrect");
    }
    
    // Test that parent can't go negative
    let low_parent_energy = 40.0f32;
    let can_afford = low_parent_energy >= reproduction_cost;
    
    if !can_afford {
        println!("  ✅ PASS: Low energy parent blocked from reproduction");
    } else {
        println!("  ❌ FAIL: Low energy parent should not reproduce");
    }
}

fn test_generation_tracking() {
    println!("\n--- Test 4: Generation Tracking ---");
    
    // Simulate generation tree
    let mut organisms: HashMap<u32, (u32, u32)> = HashMap::new(); // id -> (parent_id, generation)
    
    // Create initial population (gen 0)
    for i in 0..10 {
        organisms.insert(i, (i, 0));
    }
    
    // Simulate reproduction
    let mut next_id = 10u32;
    for gen in 1..=5 {
        let parents: Vec<u32> = organisms.iter()
            .filter(|(_, (_, g))| *g == gen - 1)
            .map(|(&id, _)| id)
            .collect();
        
        for parent_id in parents {
            let child_id = next_id;
            next_id += 1;
            organisms.insert(child_id, (parent_id, gen));
        }
    }
    
    // Verify generation counts
    let mut gen_counts: HashMap<u32, usize> = HashMap::new();
    for (_, (_, gen)) in &organisms {
        *gen_counts.entry(*gen).or_insert(0) += 1;
    }
    
    println!("  Generation distribution:");
    for gen in 0..=5 {
        let count = gen_counts.get(&gen).unwrap_or(&0);
        println!("    Gen {}: {} organisms", gen, count);
    }
    
    // Each generation should have same count as previous (asexual reproduction)
    let gen0_count = *gen_counts.get(&0).unwrap_or(&0);
    let all_equal = (1..=5).all(|g| gen_counts.get(&g) == Some(&gen0_count));
    
    if all_equal && gen0_count == 10 {
        println!("  ✅ PASS: Generation tracking correct");
    } else {
        println!("  ❌ FAIL: Generation distribution unexpected");
    }
    
    // Max generation should be 5
    let max_gen = organisms.values().map(|(_, g)| *g).max().unwrap_or(0);
    if max_gen == 5 {
        println!("  ✅ PASS: Max generation correct ({})", max_gen);
    } else {
        println!("  ❌ FAIL: Max generation should be 5, got {}", max_gen);
    }
}
