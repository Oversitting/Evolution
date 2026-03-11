//! Curriculum trainer and runtime stability search.
//!
//! Trains survivor banks through progressively harsher environments, then
//! evaluates a set of runtime presets across multiple seeds and validates the
//! best bank + preset combination for long 10,000 tick runs.

#[path = "support/headless.rs"]
mod headless;

use anyhow::Result;
use clap::Parser;
use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::SurvivorBank;
use headless::HeadlessRunner;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    #[arg(long, default_value = "trained_survivor_bank.bin")]
    bank_path: PathBuf,

    #[arg(long, default_value_t = 4)]
    epochs: u32,

    #[arg(long, default_value_t = 1200)]
    phase1_ticks: u32,

    #[arg(long, default_value_t = 1800)]
    phase2_ticks: u32,

    #[arg(long, default_value_t = 2600)]
    phase3_ticks: u32,

    #[arg(long, default_value_t = 3000)]
    search_ticks: u32,

    #[arg(long, default_value_t = 10_000)]
    validate_ticks: u32,

    #[arg(long, default_value_t = 3)]
    search_seeds: u32,

    #[arg(long, default_value_t = 3)]
    validate_seeds: u32,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Clone, Debug)]
struct TrainingPhase {
    name: &'static str,
    world_size: u32,
    initial_organisms: u32,
    max_organisms: u32,
    founder_count: u32,
    starting_energy: f32,
    passive_drain: f32,
    reproduction_threshold: f32,
    reproduction_cost: f32,
    reproduction_min_age: u32,
    reproduction_cooldown: u32,
    food_growth_rate: f32,
    food_effectiveness: f32,
    initial_patches: u32,
    patch_size: u32,
}

#[derive(Clone, Debug)]
struct RuntimePreset {
    name: String,
    initial_organisms: u32,
    founder_count: u32,
    passive_drain: f32,
    reproduction_threshold: f32,
    reproduction_min_age: u32,
    reproduction_cooldown: u32,
    food_growth_rate: f32,
    food_effectiveness: f32,
    initial_patches: u32,
}

#[derive(Clone, Debug, Default)]
struct EvalSummary {
    final_population: u32,
    min_population: u32,
    max_population: u32,
    max_generation: u32,
    avg_energy: f32,
    total_food: f32,
}

#[derive(Clone, Debug)]
struct CandidateResult {
    preset: RuntimePreset,
    score: f32,
    worst_final_population: u32,
    worst_min_population: u32,
    avg_final_population: f32,
    avg_max_generation: f32,
    avg_total_food: f32,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("Stability search failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Args::parse();
    let base_config = SimulationConfig::from_file(&args.config)?;
    if args.bank_path.exists() {
        std::fs::remove_file(&args.bank_path)?;
    }

    let phases = curriculum_phases();
    let candidates = build_runtime_candidates(&base_config, args.seed);
    let mut best_bank: Option<SurvivorBank> = None;
    let mut best_candidate: Option<CandidateResult> = None;
    let mut best_epoch_score = f32::MIN;

    println!(
        "Curriculum search: epochs={}, search_ticks={}, validate_ticks={}, bank={}",
        args.epochs,
        args.search_ticks,
        args.validate_ticks,
        args.bank_path.display()
    );

    for epoch in 0..args.epochs {
        let epoch_seed = args.seed + epoch as u64 * 1_000;
        let mut epoch_bank: Option<SurvivorBank> = None;

        for (phase_index, phase) in phases.iter().enumerate() {
            let load_bank = phase_index > 0 || epoch > 0;
            let config = training_config_from_phase(
                &base_config,
                phase,
                epoch_seed + phase_index as u64,
                &args.bank_path,
                load_bank,
            );
            let ticks = match phase_index {
                0 => args.phase1_ticks,
                1 => args.phase2_ticks,
                _ => args.phase3_ticks,
            };

            let (summary, bank) = run_training_phase(config, ticks).await?;
            println!(
                "epoch {:2} phase {:>8} | final_pop {:4} | min_pop {:4} | max_pop {:4} | max_gen {:3} | avg_energy {:6.1} | food {:9.0}",
                epoch + 1,
                phase.name,
                summary.final_population,
                summary.min_population,
                summary.max_population,
                summary.max_generation,
                summary.avg_energy,
                summary.total_food,
            );

            if let Some(bank) = bank {
                bank.save_to_file(&args.bank_path)?;
                epoch_bank = Some(bank);
            }
        }

        let Some(bank) = epoch_bank.clone() else {
            continue;
        };

        let candidate = evaluate_candidates(
            &base_config,
            &args.bank_path,
            &candidates,
            epoch_seed + 50_000,
            args.search_ticks,
            args.search_seeds,
        )
        .await?;

        println!(
            "epoch {:2} best preset {:>14} | score {:10.1} | worst_final {:4} | worst_min {:4} | avg_final {:6.1} | avg_gen {:5.1} | avg_food {:9.0}",
            epoch + 1,
            candidate.preset.name,
            candidate.score,
            candidate.worst_final_population,
            candidate.worst_min_population,
            candidate.avg_final_population,
            candidate.avg_max_generation,
            candidate.avg_total_food,
        );

        if candidate.score > best_epoch_score {
            best_epoch_score = candidate.score;
            best_bank = Some(bank);
            best_candidate = Some(candidate);
        }
    }

    let Some(best_bank) = best_bank else {
        anyhow::bail!("Curriculum training never produced a usable survivor bank");
    };
    let Some(best_candidate) = best_candidate else {
        anyhow::bail!("No runtime candidate produced a valid score");
    };

    best_bank.save_to_file(&args.bank_path)?;
    println!(
        "\nSelected runtime preset: {}\n  initial={} founders={} passive_drain={:.3} repro_threshold={:.1} min_age={} cooldown={} growth_rate={:.4} effectiveness={:.2} patches={}",
        best_candidate.preset.name,
        best_candidate.preset.initial_organisms,
        best_candidate.preset.founder_count,
        best_candidate.preset.passive_drain,
        best_candidate.preset.reproduction_threshold,
        best_candidate.preset.reproduction_min_age,
        best_candidate.preset.reproduction_cooldown,
        best_candidate.preset.food_growth_rate,
        best_candidate.preset.food_effectiveness,
        best_candidate.preset.initial_patches,
    );

    println!("\nLong-run validation:");
    let mut validation_results = Vec::new();
    for seed_index in 0..args.validate_seeds {
        let mut config = apply_runtime_preset(&base_config, &best_candidate.preset, &args.bank_path, true);
        config.seed = Some(args.seed + 200_000 + seed_index as u64);
        let summary = run_eval(config, args.validate_ticks).await?;
        println!(
            "  seed {:2} | final_pop {:4} | min_pop {:4} | max_pop {:4} | max_gen {:3} | avg_energy {:6.1} | food {:9.0}",
            seed_index + 1,
            summary.final_population,
            summary.min_population,
            summary.max_population,
            summary.max_generation,
            summary.avg_energy,
            summary.total_food,
        );
        validation_results.push(summary);
    }

    let validated = summarize_candidate(&best_candidate.preset, &validation_results);
    println!(
        "\nValidation summary | score {:10.1} | worst_final {:4} | worst_min {:4} | avg_final {:6.1} | avg_gen {:5.1} | avg_food {:9.0}",
        validated.score,
        validated.worst_final_population,
        validated.worst_min_population,
        validated.avg_final_population,
        validated.avg_max_generation,
        validated.avg_total_food,
    );

    Ok(())
}

fn curriculum_phases() -> Vec<TrainingPhase> {
    vec![
        TrainingPhase {
            name: "rich",
            world_size: 224,
            initial_organisms: 80,
            max_organisms: 384,
            founder_count: 32,
            starting_energy: 84.0,
            passive_drain: 0.085,
            reproduction_threshold: 62.0,
            reproduction_cost: 36.0,
            reproduction_min_age: 60,
            reproduction_cooldown: 55,
            food_growth_rate: 0.0085,
            food_effectiveness: 1.00,
            initial_patches: 46,
            patch_size: 9,
        },
        TrainingPhase {
            name: "bridge",
            world_size: 288,
            initial_organisms: 88,
            max_organisms: 416,
            founder_count: 40,
            starting_energy: 78.0,
            passive_drain: 0.105,
            reproduction_threshold: 68.0,
            reproduction_cost: 40.0,
            reproduction_min_age: 85,
            reproduction_cooldown: 75,
            food_growth_rate: 0.0052,
            food_effectiveness: 0.97,
            initial_patches: 42,
            patch_size: 8,
        },
        TrainingPhase {
            name: "stress",
            world_size: 384,
            initial_organisms: 96,
            max_organisms: 448,
            founder_count: 48,
            starting_energy: 74.0,
            passive_drain: 0.116,
            reproduction_threshold: 70.0,
            reproduction_cost: 44.0,
            reproduction_min_age: 105,
            reproduction_cooldown: 95,
            food_growth_rate: 0.0040,
            food_effectiveness: 0.96,
            initial_patches: 42,
            patch_size: 8,
        },
    ]
}

fn training_config_from_phase(
    base: &SimulationConfig,
    phase: &TrainingPhase,
    seed: u64,
    bank_path: &PathBuf,
    load_bank: bool,
) -> SimulationConfig {
    let mut config = SimulationConfig::default();
    config.seed = Some(seed);
    config.population.max_organisms = phase.max_organisms;
    config.population.initial_organisms = phase.initial_organisms;
    config.world.width = phase.world_size;
    config.world.height = phase.world_size;
    config.energy.starting = phase.starting_energy;
    config.energy.maximum = base.energy.maximum;
    config.energy.passive_drain = phase.passive_drain;
    config.energy.max_age = base.energy.max_age;
    config.energy.crowding_factor = base.energy.crowding_factor;
    config.energy.age_drain_factor = base.energy.age_drain_factor;
    config.reproduction.threshold = phase.reproduction_threshold;
    config.reproduction.cost = phase.reproduction_cost;
    config.reproduction.min_age = phase.reproduction_min_age;
    config.reproduction.cooldown = phase.reproduction_cooldown;
    config.reproduction.signal_min = base.reproduction.signal_min;
    config.mutation = base.mutation.clone();
    config.physics = base.physics.clone();
    config.vision = base.vision.clone();
    config.food.growth_rate = phase.food_growth_rate;
    config.food.max_per_cell = base.food.max_per_cell;
    config.food.energy_value = base.food.energy_value;
    config.food.effectiveness = phase.food_effectiveness;
    config.food.initial_patches = phase.initial_patches;
    config.food.patch_size = phase.patch_size;
    config.food.baseline_food = base.food.baseline_food;
    config.food.spawn_chance = base.food.spawn_chance;
    config.food.spawn_amount = base.food.spawn_amount;
    config.bootstrap.enabled = true;
    config.bootstrap.path = bank_path.clone();
    config.bootstrap.founder_count = if load_bank { phase.founder_count } else { 0 };
    config.bootstrap.survivor_count = 256;
    config.bootstrap.load_on_start = load_bank;
    config.bootstrap.save_on_exit = true;
    config.system.diagnostic_interval = 10_000;
    config.system.food_readback_interval = base.system.food_readback_interval;
    config.system.readback_interval = 1;
    config.sanitize();
    config
}

fn build_runtime_candidates(base: &SimulationConfig, seed: u64) -> Vec<RuntimePreset> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + 900_000);
    let mut candidates = vec![RuntimePreset {
        name: "base".to_string(),
        initial_organisms: base.population.initial_organisms,
        founder_count: base.bootstrap.founder_count,
        passive_drain: base.energy.passive_drain,
        reproduction_threshold: base.reproduction.threshold,
        reproduction_min_age: base.reproduction.min_age,
        reproduction_cooldown: base.reproduction.cooldown,
        food_growth_rate: base.food.growth_rate,
        food_effectiveness: base.food.effectiveness,
        initial_patches: base.food.initial_patches,
    }];

    for index in 0..11 {
        candidates.push(RuntimePreset {
            name: format!("sample-{:02}", index + 1),
            initial_organisms: rng.gen_range(180..=260),
            founder_count: rng.gen_range(96..=160),
            passive_drain: rng.gen_range(0.108..=0.124),
            reproduction_threshold: rng.gen_range(64.0..=74.0),
            reproduction_min_age: rng.gen_range(150..=230),
            reproduction_cooldown: rng.gen_range(130..=220),
            food_growth_rate: rng.gen_range(0.0030..=0.0044),
            food_effectiveness: rng.gen_range(0.92..=1.00),
            initial_patches: rng.gen_range(230..=320),
        });
    }

    candidates
}

async fn run_training_phase(config: SimulationConfig, ticks: u32) -> Result<(EvalSummary, Option<SurvivorBank>)> {
    let survivor_limit = config.bootstrap.survivor_count as usize;
    let mut runner = HeadlessRunner::new(config).await?;
    let mut min_population = runner.simulation().organism_count();
    let mut max_population = min_population;
    let mut max_generation = 0;

    for _ in 0..ticks {
        let metrics = runner.step()?;
        min_population = min_population.min(metrics.population);
        max_population = max_population.max(metrics.population);
        max_generation = max_generation.max(metrics.max_generation);
    }

    let final_metrics = runner.flush()?;
    min_population = min_population.min(final_metrics.population);
    max_population = max_population.max(final_metrics.population);
    max_generation = max_generation.max(final_metrics.max_generation);

    let bank = runner.simulation().to_survivor_bank(runner.tick() as u64, survivor_limit);
    Ok((
        EvalSummary {
            final_population: final_metrics.population,
            min_population,
            max_population,
            max_generation,
            avg_energy: final_metrics.avg_energy,
            total_food: final_metrics.total_food,
        },
        bank,
    ))
}

async fn evaluate_candidates(
    base_config: &SimulationConfig,
    bank_path: &PathBuf,
    candidates: &[RuntimePreset],
    seed_base: u64,
    ticks: u32,
    search_seeds: u32,
) -> Result<CandidateResult> {
    let mut best: Option<CandidateResult> = None;

    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let mut summaries = Vec::new();
        for seed_index in 0..search_seeds {
            let mut config = apply_runtime_preset(base_config, candidate, bank_path, true);
            config.seed = Some(seed_base + candidate_index as u64 * 100 + seed_index as u64);
            summaries.push(run_eval(config, ticks).await?);
        }

        let result = summarize_candidate(candidate, &summaries);
        if best.as_ref().map(|current| result.score > current.score).unwrap_or(true) {
            best = Some(result);
        }
    }

    best.ok_or_else(|| anyhow::anyhow!("No runtime candidates evaluated"))
}

fn summarize_candidate(candidate: &RuntimePreset, summaries: &[EvalSummary]) -> CandidateResult {
    let worst_final_population = summaries
        .iter()
        .map(|summary| summary.final_population)
        .min()
        .unwrap_or(0);
    let worst_min_population = summaries
        .iter()
        .map(|summary| summary.min_population)
        .min()
        .unwrap_or(0);
    let avg_final_population = summaries
        .iter()
        .map(|summary| summary.final_population as f32)
        .sum::<f32>()
        / summaries.len().max(1) as f32;
    let avg_max_generation = summaries
        .iter()
        .map(|summary| summary.max_generation as f32)
        .sum::<f32>()
        / summaries.len().max(1) as f32;
    let avg_total_food = summaries
        .iter()
        .map(|summary| summary.total_food)
        .sum::<f32>()
        / summaries.len().max(1) as f32;

    let collapse_penalty = if worst_final_population == 0 || worst_min_population == 0 {
        5_000_000.0
    } else if worst_min_population < 24 {
        (24 - worst_min_population) as f32 * 80_000.0
    } else {
        0.0
    };

    let food_penalty = (avg_total_food - 180_000.0).max(0.0) * 0.05;

    let score = worst_final_population as f32 * 5_000.0
        + worst_min_population as f32 * 4_000.0
        + avg_final_population * 300.0
        + avg_max_generation * 50.0
        - food_penalty
        - collapse_penalty;

    CandidateResult {
        preset: candidate.clone(),
        score,
        worst_final_population,
        worst_min_population,
        avg_final_population,
        avg_max_generation,
        avg_total_food,
    }
}

fn apply_runtime_preset(
    base: &SimulationConfig,
    preset: &RuntimePreset,
    bank_path: &PathBuf,
    load_bank: bool,
) -> SimulationConfig {
    let mut config = base.clone();
    config.population.initial_organisms = preset.initial_organisms.min(config.population.max_organisms);
    config.energy.passive_drain = preset.passive_drain;
    config.reproduction.threshold = preset.reproduction_threshold;
    config.reproduction.min_age = preset.reproduction_min_age;
    config.reproduction.cooldown = preset.reproduction_cooldown;
    config.food.growth_rate = preset.food_growth_rate;
    config.food.effectiveness = preset.food_effectiveness;
    config.food.initial_patches = preset.initial_patches;
    config.bootstrap.enabled = true;
    config.bootstrap.path = bank_path.clone();
    config.bootstrap.founder_count = if load_bank { preset.founder_count } else { 0 };
    config.bootstrap.load_on_start = load_bank;
    config.bootstrap.save_on_exit = false;
    config.system.diagnostic_interval = 10_000;
    config.sanitize();
    config
}

async fn run_eval(config: SimulationConfig, ticks: u32) -> Result<EvalSummary> {
    let mut runner = HeadlessRunner::new(config).await?;
    let mut min_population = runner.simulation().organism_count();
    let mut max_population = min_population;
    let mut max_generation = 0;

    for _ in 0..ticks {
        let metrics = runner.step()?;
        min_population = min_population.min(metrics.population);
        max_population = max_population.max(metrics.population);
        max_generation = max_generation.max(metrics.max_generation);
    }

    let final_metrics = runner.flush()?;
    min_population = min_population.min(final_metrics.population);
    max_population = max_population.max(final_metrics.population);
    max_generation = max_generation.max(final_metrics.max_generation);

    Ok(EvalSummary {
        final_population: final_metrics.population,
        min_population,
        max_population,
        max_generation,
        avg_energy: final_metrics.avg_energy,
        total_food: final_metrics.total_food,
    })
}