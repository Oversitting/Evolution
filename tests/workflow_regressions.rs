use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use evolution_sim::config::{SimUniform, SimulationConfig};
use evolution_sim::simulation::save_load::{load_bootstrap_entries, load_bootstrap_quality_score};
use evolution_sim::simulation::{FounderPool, Simulation};

fn temp_path(prefix: &str, extension: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}.{}", prefix, unique, extension))
}

#[test]
fn config_file_sanitization_produces_runtime_safe_values() {
    let path = temp_path("invalid_config", "toml");
    let toml = r#"
        [population]
        max_organisms = 0
        initial_organisms = 9

        [world]
        width = 0
        height = 0

        [energy]
        maximum = -2.0
        starting = 500.0
        max_age = 80

        [reproduction]
        threshold = 900.0
        cost = 1200.0
        min_age = 140
        signal_min = 1.5

        [food]
        max_per_cell = 0.0
        patch_size = 0
        baseline_food = 12.0
        spawn_amount = 25.0
        seasonal_period = 0
        seasonal_amplitude = 3.0
        hotspots_enabled = true
        hotspot_count = 0

        [biomes]
        enabled = true
        biome_count = 0

        [bootstrap]
        enabled = true
        path = ""

        [system]
        readback_interval = 5
        food_readback_interval = 0
        diagnostic_interval = 0
    "#;

    fs::write(&path, toml).unwrap();

    let config = SimulationConfig::from_file(&path).unwrap();
    let uniform = SimUniform::from_config(&config, 3, 99);

    assert_eq!(config.population.max_organisms, 1);
    assert_eq!(config.population.initial_organisms, 1);
    assert_eq!(config.world.width, 1);
    assert_eq!(config.world.height, 1);
    assert_eq!(config.reproduction.min_age, config.energy.max_age);
    assert_eq!(config.food.max_per_cell, 10.0);
    assert_eq!(config.food.patch_size, 1);
    assert_eq!(config.food.baseline_food, config.food.max_per_cell);
    assert_eq!(config.food.spawn_amount, config.food.max_per_cell);
    assert_eq!(config.food.seasonal_period, 6000);
    assert_eq!(config.food.seasonal_amplitude, 1.0);
    assert_eq!(config.food.hotspot_count, 1);
    assert_eq!(config.biomes.biome_count, 1);
    assert_eq!(config.system.readback_interval, 1);
    assert_eq!(config.system.food_readback_interval, 1);
    assert_eq!(config.system.diagnostic_interval, 1);
    assert_eq!(config.bootstrap.path, PathBuf::from("founder_pool.json"));

    assert_eq!(uniform.world_width, 1);
    assert_eq!(uniform.world_height, 1);
    assert_eq!(uniform.food_max_per_cell, 10.0);
    assert_eq!(uniform.seasonal_period, 6000);
    assert_eq!(uniform.hotspot_count, 1);

    let _ = fs::remove_file(path);
}

#[test]
fn founder_pool_json_bootstrap_filters_disabled_entries_and_sorts_by_score() {
    let path = temp_path("founder_pool", "json");

    let mut config = SimulationConfig::default();
    config.seed = Some(17);
    config.population.max_organisms = 4;
    config.population.initial_organisms = 3;
    config.bootstrap.load_on_start = false;

    let mut simulation = Simulation::new(&config);
    {
        let organism = simulation.organisms.get_mut(0).unwrap();
        organism.age = 100;
        organism.energy = 60.0;
        organism.offspring_count = 0;
    }
    {
        let organism = simulation.organisms.get_mut(1).unwrap();
        organism.age = 250;
        organism.energy = 140.0;
        organism.offspring_count = 4;
    }
    {
        let organism = simulation.organisms.get_mut(2).unwrap();
        organism.age = 400;
        organism.energy = 90.0;
        organism.offspring_count = 1;
    }

    let bank = simulation.to_survivor_bank(77, 3).unwrap();
    let mut pool = FounderPool::from_survivor_bank(&bank, "integration", "workflow regression");
    pool.entries[0].score = 10.0;
    pool.entries[1].score = 80.0;
    pool.entries[2].score = 40.0;
    pool.entries[2].enabled = false;
    pool.save_to_file(&path).unwrap();

    let entries = load_bootstrap_entries(&path, 3).unwrap();
    let quality_score = load_bootstrap_quality_score(&path).unwrap();

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].score, 80.0);
    assert_eq!(entries[1].score, 10.0);
    assert_eq!(quality_score, 90.0);

    let _ = fs::remove_file(path);
}