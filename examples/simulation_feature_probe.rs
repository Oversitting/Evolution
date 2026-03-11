//! Scenario-based feature probes for longer headless runs.
//! 
//! Exercises reproduction, predation, and food-balance behavior with tuned scenarios
//! and reports suspicious outcomes as failures.

#[path = "support/headless.rs"]
mod headless;

use anyhow::Result;
use evolution_sim::config::SimulationConfig;
use headless::HeadlessRunner;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("Feature probe failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let reproduction = run_reproduction_probe().await?;
    let predation = run_predation_probe().await?;
    let food_balance = run_food_balance_probe().await?;

    println!("\nScenario probe summaries:");
    for summary in [reproduction, predation, food_balance] {
        println!(
            "  {} -> pop {} -> {}, births={}, deaths={}, max_gen={}, min_food={:.1}, max_food={:.1}, min_energy={:.1}, max_energy={:.1}",
            summary.name,
            summary.initial_population,
            summary.final_population,
            summary.total_births,
            summary.total_deaths,
            summary.max_generation,
            summary.min_food,
            summary.max_food,
            summary.min_avg_energy,
            summary.max_avg_energy,
        );
    }

    Ok(())
}

#[derive(Debug)]
struct ScenarioSummary {
    name: &'static str,
    initial_population: u32,
    final_population: u32,
    total_births: u32,
    total_deaths: u32,
    max_generation: u32,
    initial_food: f32,
    min_food: f32,
    max_food: f32,
    min_avg_energy: f32,
    max_avg_energy: f32,
}

async fn run_reproduction_probe() -> Result<ScenarioSummary> {
    let mut config = base_config(1001);
    config.population.initial_organisms = 64;
    config.reproduction.threshold = 52.0;
    config.reproduction.cost = 24.0;
    config.reproduction.min_age = 24;
    config.reproduction.cooldown = 20;
    config.energy.starting = 90.0;
    config.food.initial_patches = 40;
    config.food.patch_size = 10;

    let summary = run_scenario("reproduction", config, 320).await?;

    if summary.total_births == 0 || summary.max_generation == 0 {
        anyhow::bail!(
            "Reproduction probe did not produce offspring: births={}, max_generation={}",
            summary.total_births,
            summary.max_generation
        );
    }

    Ok(summary)
}

async fn run_predation_probe() -> Result<ScenarioSummary> {
    let mut config = base_config(2002);
    config.population.initial_organisms = 96;
    config.population.max_organisms = 320;
    config.world.width = 96;
    config.world.height = 96;
    config.food.initial_patches = 14;
    config.food.patch_size = 6;
    config.predation.enabled = true;
    config.predation.attack_threshold = -0.25;
    config.predation.attack_range = 18.0;
    config.predation.attack_damage = 14.0;
    config.predation.attack_cost = 0.8;
    config.predation.energy_transfer = 0.35;

    let summary = run_scenario("predation", config, 260).await?;

    if summary.total_deaths == 0 {
        anyhow::bail!("Predation probe saw zero deaths despite aggressive attack settings");
    }

    Ok(summary)
}

async fn run_food_balance_probe() -> Result<ScenarioSummary> {
    let mut config = base_config(3003);
    config.population.initial_organisms = 72;
    config.food.growth_rate = 0.005;
    config.food.initial_patches = 28;
    config.food.patch_size = 8;
    config.food.effectiveness = 0.9;
    config.energy.passive_drain = 0.12;

    let summary = run_scenario("food-balance", config, 480).await?;

    if summary.final_population == 0 {
        anyhow::bail!("Food balance probe ended with extinction");
    }
    if summary.min_food <= 0.0 {
        anyhow::bail!("Food balance probe depleted food to zero");
    }
    if summary.max_food > summary.initial_food * 1.15 {
        anyhow::bail!(
            "Food balance probe inflated food too far: initial_food={:.1}, max_food={:.1}",
            summary.initial_food,
            summary.max_food
        );
    }

    Ok(summary)
}

async fn run_scenario(name: &'static str, config: SimulationConfig, ticks: u32) -> Result<ScenarioSummary> {
    let mut runner = HeadlessRunner::new(config).await?;
    let initial_population = runner.simulation().organism_count();
    let initial_food = runner.simulation().world.total_food();
    let mut total_births = 0u32;
    let mut total_deaths = 0u32;
    let mut min_food = f32::INFINITY;
    let mut max_food = 0.0f32;
    let mut min_avg_energy = f32::INFINITY;
    let mut max_avg_energy = 0.0f32;
    let mut max_generation = 0u32;

    for _ in 0..ticks {
        let metrics = runner.step()?;
        total_births += metrics.births;
        total_deaths += metrics.deaths;
        min_food = min_food.min(metrics.total_food);
        max_food = max_food.max(metrics.total_food);
        min_avg_energy = min_avg_energy.min(metrics.avg_energy);
        max_avg_energy = max_avg_energy.max(metrics.avg_energy);
        max_generation = max_generation.max(metrics.max_generation);
    }

    let final_metrics = runner.flush()?;
    total_deaths += final_metrics.deaths;
    min_food = min_food.min(final_metrics.total_food);
    max_food = max_food.max(final_metrics.total_food);
    min_avg_energy = min_avg_energy.min(final_metrics.avg_energy);
    max_avg_energy = max_avg_energy.max(final_metrics.avg_energy);
    max_generation = max_generation.max(final_metrics.max_generation);

    Ok(ScenarioSummary {
        name,
        initial_population,
        final_population: final_metrics.population,
        total_births,
        total_deaths,
        max_generation,
        initial_food,
        min_food,
        max_food,
        min_avg_energy,
        max_avg_energy,
    })
}

fn base_config(seed: u64) -> SimulationConfig {
    let mut config = SimulationConfig::default();
    config.seed = Some(seed);
    config.population.max_organisms = 256;
    config.population.initial_organisms = 64;
    config.world.width = 128;
    config.world.height = 128;
    config.food.initial_patches = 24;
    config.food.patch_size = 8;
    config.system.diagnostic_interval = 10_000;
    config
}