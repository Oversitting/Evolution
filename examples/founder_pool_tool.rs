//! Inspect and manage readable founder pools.

use anyhow::Result;
use clap::{Parser, Subcommand};
use evolution_sim::simulation::{FounderPool, SurvivorBank};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Summary {
        #[arg(long, default_value = "founder_pool.json")]
        path: PathBuf,
    },
    List {
        #[arg(long, default_value = "founder_pool.json")]
        path: PathBuf,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    ConvertBank {
        #[arg(long, default_value = "survivor_bank.bin")]
        input: PathBuf,
        #[arg(long, default_value = "founder_pool.json")]
        output: PathBuf,
    },
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Founder pool tool failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Summary { path } => {
            let pool = load_pool(&path)?;
            println!("path={}", path.display());
            println!("entries={}", pool.entries.len());
            println!("quality_score={:.1}", pool.quality_score());
            println!("description={}", pool.description);
            if let Some(best) = pool.entries.first() {
                println!(
                    "best={} score={:.1} successes={}/{} best_steps={}",
                    best.label,
                    best.score,
                    best.successes,
                    best.evaluations,
                    best.best_steps_to_food,
                );
            }
        }
        Command::List { path, limit } => {
            let pool = load_pool(&path)?;
            for entry in pool.entries.iter().take(limit) {
                println!(
                    "{} | score={:.1} | successes={}/{} | best_steps={} | source={} | notes={}",
                    entry.label,
                    entry.score,
                    entry.successes,
                    entry.evaluations,
                    entry.best_steps_to_food,
                    entry.source,
                    entry.notes,
                );
            }
        }
        Command::ConvertBank { input, output } => {
            let bank = SurvivorBank::load_from_file(&input)?;
            let pool = FounderPool::from_survivor_bank(
                &bank,
                "legacy_bank",
                "Converted from legacy survivor_bank.bin",
            );
            pool.save_to_file(&output)?;
            println!("converted {} -> {}", input.display(), output.display());
        }
    }

    Ok(())
}

fn load_pool(path: &PathBuf) -> Result<FounderPool> {
    let is_json = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    if is_json {
        Ok(FounderPool::load_from_file(path)?)
    } else {
        let bank = SurvivorBank::load_from_file(path)?;
        Ok(FounderPool::from_survivor_bank(
            &bank,
            "legacy_bank",
            "Converted from legacy survivor_bank.bin",
        ))
    }
}