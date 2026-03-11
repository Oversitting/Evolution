//! Automated survivor-bank training and evaluation loop.
//!
//! Trains founders over multiple epochs, exports a survivor bank, then compares
//! banked founders against fresh random founders under leaner starting configs.

#[path = "support/headless.rs"]
mod headless;

use anyhow::Result;
use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::SurvivorBank;
use headless::HeadlessRunner;
use std::path::PathBuf;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("Survivor-bank trainer failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Args::parse();
    let bank_path = args.bank_path.clone();

    if bank_path.exists() {
        std::fs::remove_file(&bank_path)?;
    }

    println!(
        "Training survivor bank: epochs={}, train_ticks={}, eval_ticks={}, bank={}",
        args.epochs,
        args.train_ticks,
        args.eval_ticks,
        bank_path.display()
    );

    let validation_preset = evaluation_presets()
        .into_iter()
        .find(|preset| preset.name == "starter")
        .expect("starter preset must exist");
    let validation_ticks = args.eval_ticks.min(1_000);
    let mut best_bank: Option<SurvivorBank> = None;
    let mut best_validation_score = f32::MIN;

    for epoch in 0..args.epochs {
        let train_seed = args.seed + epoch as u64;
        let train_config = training_config(train_seed, &bank_path, epoch > 0);
        let (train_summary, candidate_bank) = run_training_epoch(train_config, args.train_ticks).await?;
        println!(
            "train epoch {:2} | pop {:4} -> {:4} | max_gen {:3} | births {:4} | deaths {:4} | food {:9.1}",
            epoch + 1,
            train_summary.initial_population,
            train_summary.final_population,
            train_summary.max_generation,
            train_summary.total_births,
            train_summary.total_deaths,
            train_summary.final_food,
        );

        if let Some(bank) = candidate_bank {
            bank.save_to_file(&bank_path)?;

            let validation = run_eval(
                eval_config(
                    args.seed + 50_000 + epoch as u64,
                    &bank_path,
                    true,
                    &validation_preset,
                ),
                validation_ticks,
            )
            .await?;
            let validation_score = validation.final_population as f32 * 1_000.0
                + validation.max_generation as f32 * 100.0
                + validation.final_avg_energy;
            println!(
                "  validation {:>7} | final_pop {:4} | max_gen {:3} | avg_energy {:6.1}",
                validation_preset.name,
                validation.final_population,
                validation.max_generation,
                validation.final_avg_energy,
            );

            if validation_score > best_validation_score {
                best_validation_score = validation_score;
                best_bank = Some(bank);
            }
        }
    }

    let Some(best_bank) = best_bank else {
        anyhow::bail!("Training produced no living survivors to store in the bank");
    };
    best_bank.save_to_file(&bank_path)?;

    let bank = SurvivorBank::load_from_file(&bank_path)?;
    println!(
        "\nFinal survivor bank: entries={}, source_tick={}",
        bank.entries.len(),
        bank.source_tick
    );

    println!("\nEvaluation sweep (random vs banked):");
    let mut recommended: Option<(EvalPreset, EvalSummary, EvalSummary)> = None;
    for preset in evaluation_presets() {
        let random_summary = run_eval(eval_config(args.seed + 10_000 + preset.id as u64, &bank_path, false, &preset), args.eval_ticks).await?;
        let banked_summary = run_eval(eval_config(args.seed + 20_000 + preset.id as u64, &bank_path, true, &preset), args.eval_ticks).await?;

        println!(
            "{} | random pop {} -> {} min {} gen {} food {:.0} | banked pop {} -> {} min {} gen {} food {:.0}",
            preset.name,
            random_summary.initial_population,
            random_summary.final_population,
            random_summary.min_population,
            random_summary.max_generation,
            random_summary.final_food,
            banked_summary.initial_population,
            banked_summary.final_population,
            banked_summary.min_population,
            banked_summary.max_generation,
            banked_summary.final_food,
        );

        if is_acceptable_start(&preset, &banked_summary, &random_summary) {
            match &recommended {
                Some((_, best, _)) if banked_summary.final_population <= best.final_population => {}
                _ => recommended = Some((preset.clone(), banked_summary, random_summary)),
            }
        }
    }

    match recommended {
        Some((preset, banked, random)) => {
            println!(
                "\nRecommended starting preset: {}\n  random final pop={} max_gen={}\n  banked final pop={} max_gen={} min_pop={} avg_energy={:.1}",
                preset.name,
                random.final_population,
                random.max_generation,
                banked.final_population,
                banked.max_generation,
                banked.min_population,
                banked.final_avg_energy,
            );
        }
        None => {
            anyhow::bail!("No evaluation preset met the survival thresholds; training config likely still needs adjustment");
        }
    }

    Ok(())
}

#[derive(Clone)]
struct EvalPreset {
    id: u32,
    name: &'static str,
    growth_rate: f32,
    effectiveness: f32,
    initial_patches: u32,
    patch_size: u32,
    passive_drain: f32,
    reproduction_threshold: f32,
}

#[derive(Debug)]
struct EpochSummary {
    initial_population: u32,
    final_population: u32,
    total_births: u32,
    total_deaths: u32,
    max_generation: u32,
    final_food: f32,
}

#[derive(Debug, Clone)]
struct EvalSummary {
    initial_population: u32,
    final_population: u32,
    min_population: u32,
    max_generation: u32,
    final_avg_energy: f32,
    final_food: f32,
}

async fn run_training_epoch(config: SimulationConfig, ticks: u32) -> Result<(EpochSummary, Option<SurvivorBank>)> {
    let survivor_limit = config.bootstrap.survivor_count as usize;
    let mut runner = HeadlessRunner::new(config).await?;
    let initial_population = runner.simulation().organism_count();
    let mut total_births = 0;
    let mut total_deaths = 0;
    let mut max_generation = 0;

    for _ in 0..ticks {
        let metrics = runner.step()?;
        total_births += metrics.births;
        total_deaths += metrics.deaths;
        max_generation = max_generation.max(metrics.max_generation);
    }

    let final_metrics = runner.flush()?;
    total_deaths += final_metrics.deaths;
    max_generation = max_generation.max(final_metrics.max_generation);

    let bank = runner.simulation().to_survivor_bank(runner.tick() as u64, survivor_limit);

    Ok((
        EpochSummary {
            initial_population,
            final_population: final_metrics.population,
            total_births,
            total_deaths,
            max_generation,
            final_food: final_metrics.total_food,
        },
        bank,
    ))
}

async fn run_eval(config: SimulationConfig, ticks: u32) -> Result<EvalSummary> {
    let mut runner = HeadlessRunner::new(config).await?;
    let initial_population = runner.simulation().organism_count();
    let mut min_population = initial_population;
    let mut max_generation = 0;

    for _ in 0..ticks {
        let metrics = runner.step()?;
        min_population = min_population.min(metrics.population);
        max_generation = max_generation.max(metrics.max_generation);
    }

    let final_metrics = runner.flush()?;
    min_population = min_population.min(final_metrics.population);
    max_generation = max_generation.max(final_metrics.max_generation);

    Ok(EvalSummary {
        initial_population,
        final_population: final_metrics.population,
        min_population,
        max_generation,
        final_avg_energy: final_metrics.avg_energy,
        final_food: final_metrics.total_food,
    })
}

fn training_config(seed: u64, bank_path: &PathBuf, load_bank: bool) -> SimulationConfig {
    let mut config = SimulationConfig::default();
    config.seed = Some(seed);
    config.population.max_organisms = 320;
    config.population.initial_organisms = 72;
    config.world.width = 192;
    config.world.height = 192;
    config.energy.starting = 80.0;
    config.energy.passive_drain = 0.085;
    config.reproduction.threshold = 64.0;
    config.reproduction.cost = 36.0;
    config.reproduction.min_age = 70;
    config.reproduction.cooldown = 60;
    config.food.growth_rate = 0.009;
    config.food.effectiveness = 1.0;
    config.food.initial_patches = 42;
    config.food.patch_size = 9;
    config.bootstrap.path = bank_path.clone();
    config.bootstrap.founder_count = if load_bank { 36 } else { 0 };
    config.bootstrap.survivor_count = 192;
    config.bootstrap.load_on_start = load_bank;
    config.bootstrap.save_on_exit = true;
    config.system.diagnostic_interval = 10_000;
    config
}

fn eval_config(seed: u64, bank_path: &PathBuf, load_bank: bool, preset: &EvalPreset) -> SimulationConfig {
    let mut config = SimulationConfig::default();
    config.seed = Some(seed);
    config.population.max_organisms = 320;
    config.population.initial_organisms = 72;
    config.world.width = 192;
    config.world.height = 192;
    config.energy.starting = 70.0;
    config.energy.passive_drain = preset.passive_drain;
    config.reproduction.threshold = preset.reproduction_threshold;
    config.reproduction.cost = 32.0;
    config.reproduction.min_age = 60;
    config.reproduction.cooldown = 60;
    config.food.growth_rate = preset.growth_rate;
    config.food.effectiveness = preset.effectiveness;
    config.food.initial_patches = preset.initial_patches;
    config.food.patch_size = preset.patch_size;
    config.bootstrap.path = bank_path.clone();
    config.bootstrap.founder_count = if load_bank { 36 } else { 0 };
    config.bootstrap.survivor_count = 192;
    config.bootstrap.load_on_start = load_bank;
    config.bootstrap.save_on_exit = false;
    config.system.diagnostic_interval = 10_000;
    config
}

fn evaluation_presets() -> Vec<EvalPreset> {
    vec![
        EvalPreset {
            id: 1,
            name: "starter",
            growth_rate: 0.0030,
            effectiveness: 0.94,
            initial_patches: 26,
            patch_size: 7,
            passive_drain: 0.130,
            reproduction_threshold: 68.0,
        },
        EvalPreset {
            id: 2,
            name: "lean-a",
            growth_rate: 0.0030,
            effectiveness: 0.92,
            initial_patches: 24,
            patch_size: 7,
            passive_drain: 0.132,
            reproduction_threshold: 70.0,
        },
        EvalPreset {
            id: 3,
            name: "lean-b",
            growth_rate: 0.0026,
            effectiveness: 0.89,
            initial_patches: 22,
            patch_size: 7,
            passive_drain: 0.136,
            reproduction_threshold: 72.0,
        },
        EvalPreset {
            id: 4,
            name: "lean-c",
            growth_rate: 0.0023,
            effectiveness: 0.86,
            initial_patches: 20,
            patch_size: 6,
            passive_drain: 0.140,
            reproduction_threshold: 74.0,
        },
    ]
}

fn is_acceptable_start(preset: &EvalPreset, banked: &EvalSummary, random: &EvalSummary) -> bool {
    let _ = preset;
    banked.final_population >= 10
        && banked.max_generation >= 1
        && banked.final_population > random.final_population
}

struct Args {
    epochs: u32,
    train_ticks: u32,
    eval_ticks: u32,
    seed: u64,
    bank_path: PathBuf,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            epochs: 6,
            train_ticks: 1_500,
            eval_ticks: 1_500,
            seed: 42,
            bank_path: PathBuf::from("trained_survivor_bank.bin"),
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--epochs" => parsed.epochs = parse_value(&mut args, "--epochs"),
                "--train-ticks" => parsed.train_ticks = parse_value(&mut args, "--train-ticks"),
                "--eval-ticks" => parsed.eval_ticks = parse_value(&mut args, "--eval-ticks"),
                "--seed" => parsed.seed = parse_value(&mut args, "--seed"),
                "--bank" => {
                    let value = args.next().unwrap_or_else(|| panic!("Missing value for --bank"));
                    parsed.bank_path = PathBuf::from(value);
                }
                other => panic!("Unknown argument: {other}"),
            }
        }

        parsed.epochs = parsed.epochs.max(1);
        parsed.train_ticks = parsed.train_ticks.max(1);
        parsed.eval_ticks = parsed.eval_ticks.max(1);
        parsed
    }
}

fn parse_value<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, flag: &str) -> T {
    let value = args.next().unwrap_or_else(|| panic!("Missing value for {flag}"));
    value.parse::<T>().unwrap_or_else(|_| panic!("Invalid value for {flag}: {value}"))
}