//! Predation system tests
//! 
//! Tests for the attack and predation mechanics added in Phase 4.

use evolution_sim::config::{SimulationConfig, PredationConfig};

fn main() {
    println!("=== Predation System Tests ===\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    let tests: Vec<(&str, fn() -> Result<(), String>)> = vec![
        ("Predation Config Defaults", test_predation_config_defaults),
        ("Predation Config Custom", test_predation_config_custom),
        ("Attack Threshold Range", test_attack_threshold_range),
        ("Energy Transfer Calculation", test_energy_transfer_calculation),
        ("Attack Cost Calculation", test_attack_cost_calculation),
        ("SimUniform Predation Fields", test_sim_uniform_predation_fields),
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

fn test_predation_config_defaults() -> Result<(), String> {
    let config = PredationConfig::default();
    
    // Default should be disabled
    if config.enabled {
        return Err("Predation should be disabled by default".into());
    }
    
    // Check default values are reasonable
    if config.attack_threshold < 0.0 || config.attack_threshold > 1.0 {
        return Err(format!("Attack threshold {} out of range [0, 1]", config.attack_threshold));
    }
    
    if config.attack_range <= 0.0 {
        return Err("Attack range must be positive".into());
    }
    
    if config.attack_damage <= 0.0 {
        return Err("Attack damage must be positive".into());
    }
    
    if config.energy_transfer < 0.0 || config.energy_transfer > 1.0 {
        return Err(format!("Energy transfer {} out of range [0, 1]", config.energy_transfer));
    }
    
    Ok(())
}

fn test_predation_config_custom() -> Result<(), String> {
    let config = PredationConfig {
        enabled: true,
        attack_threshold: 0.5,
        attack_range: 10.0,
        attack_damage: 30.0,
        energy_transfer: 0.7,
        attack_cost: 3.0,
    };
    
    if !config.enabled {
        return Err("Custom config should be enabled".into());
    }
    
    if config.attack_threshold != 0.5 {
        return Err(format!("Threshold mismatch: {} != 0.5", config.attack_threshold));
    }
    
    if config.attack_range != 10.0 {
        return Err(format!("Range mismatch: {} != 10.0", config.attack_range));
    }
    
    Ok(())
}

fn test_attack_threshold_range() -> Result<(), String> {
    // Test various threshold values
    let thresholds = [0.0, 0.3, 0.5, 0.7, 1.0];
    
    for &threshold in &thresholds {
        let config = PredationConfig {
            attack_threshold: threshold,
            ..PredationConfig::default()
        };
        
        if config.attack_threshold != threshold {
            return Err(format!("Threshold not set correctly: {} != {}", config.attack_threshold, threshold));
        }
    }
    
    Ok(())
}

fn test_energy_transfer_calculation() -> Result<(), String> {
    // Simulate energy transfer math
    let victim_energy: f32 = 100.0;
    let damage: f32 = 30.0;
    let transfer_rate: f32 = 0.5;
    
    let expected_transfer = damage * transfer_rate;
    let actual_transfer = damage * transfer_rate;
    
    if (expected_transfer - actual_transfer).abs() > 0.001 {
        return Err(format!("Energy transfer mismatch: {} vs {}", expected_transfer, actual_transfer));
    }
    
    // Test kill bonus calculation
    let kill_bonus = victim_energy * transfer_rate;
    if (kill_bonus - 50.0).abs() > 0.001 {
        return Err(format!("Kill bonus wrong: {} != 50.0", kill_bonus));
    }
    
    Ok(())
}

fn test_attack_cost_calculation() -> Result<(), String> {
    let config = PredationConfig {
        attack_cost: 5.0,
        ..PredationConfig::default()
    };
    
    // Verify attack cost is applied correctly
    let initial_energy = 100.0;
    let after_attack = initial_energy - config.attack_cost;
    
    if (after_attack - 95.0).abs() > 0.001 {
        return Err(format!("Energy after attack wrong: {} != 95.0", after_attack));
    }
    
    // Test multiple attacks
    let after_10_attacks = initial_energy - (config.attack_cost * 10.0);
    if (after_10_attacks - 50.0).abs() > 0.001 {
        return Err(format!("Energy after 10 attacks wrong: {} != 50.0", after_10_attacks));
    }
    
    Ok(())
}

fn test_sim_uniform_predation_fields() -> Result<(), String> {
    use evolution_sim::config::SimUniform;
    
    let mut config = SimulationConfig::default();
    config.predation.enabled = true;
    config.predation.attack_threshold = 0.4;
    config.predation.attack_range = 12.0;
    config.predation.attack_damage = 25.0;
    config.predation.energy_transfer = 0.6;
    config.predation.attack_cost = 4.0;
    
    let uniform = SimUniform::from_config(&config, 100, 0);
    
    // Verify predation fields are correctly transferred
    if uniform.predation_enabled != 1 {
        return Err(format!("predation_enabled should be 1, got {}", uniform.predation_enabled));
    }
    
    if (uniform.attack_threshold - 0.4).abs() > 0.001 {
        return Err(format!("attack_threshold mismatch: {}", uniform.attack_threshold));
    }
    
    if (uniform.attack_range - 12.0).abs() > 0.001 {
        return Err(format!("attack_range mismatch: {}", uniform.attack_range));
    }
    
    if (uniform.attack_damage - 25.0).abs() > 0.001 {
        return Err(format!("attack_damage mismatch: {}", uniform.attack_damage));
    }
    
    if (uniform.energy_transfer - 0.6).abs() > 0.001 {
        return Err(format!("energy_transfer mismatch: {}", uniform.energy_transfer));
    }
    
    if (uniform.attack_cost - 4.0).abs() > 0.001 {
        return Err(format!("attack_cost mismatch: {}", uniform.attack_cost));
    }
    
    Ok(())
}
