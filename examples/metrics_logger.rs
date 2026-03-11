//! Long-run metrics logger for simulation tuning.
//! 
//! Writes tick/population/birth/death/energy/food metrics to a CSV file.

#[path = "support/headless.rs"]
mod headless;

use anyhow::Result;
use evolution_sim::config::SimulationConfig;
use headless::HeadlessRunner;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("Metrics logger failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Args::parse();
    let mut config = if let Some(path) = &args.config {
        SimulationConfig::from_file(path)?
    } else {
        SimulationConfig::default()
    };
    config.seed = Some(args.seed);
    config.population.max_organisms = args.max_organisms.unwrap_or(config.population.max_organisms);
    config.population.initial_organisms = args
        .initial_organisms
        .unwrap_or(config.population.initial_organisms)
        .min(config.population.max_organisms);
    if let Some(world_size) = args.world_size {
        config.world.width = world_size;
        config.world.height = world_size;
    }
    config.system.diagnostic_interval = 10_000;
    config.sanitize();

    let mut runner = HeadlessRunner::new(config).await?;
    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "tick,population,births,deaths,avg_energy,total_food,max_generation,species_count")?;

    for _ in 0..args.ticks {
        let metrics = runner.step()?;
        if metrics.tick % args.interval == 0 {
            write_metrics(&mut writer, &metrics)?;
            println!(
                "tick {:5} | pop {:4} | births {:3} | deaths {:3} | avg_energy {:6.2} | food {:9.1} | max_gen {:3} | species {:3}",
                metrics.tick,
                metrics.population,
                metrics.births,
                metrics.deaths,
                metrics.avg_energy,
                metrics.total_food,
                metrics.max_generation,
                metrics.species_count,
            );
        }
    }

    let final_metrics = runner.flush()?;
    write_metrics(&mut writer, &final_metrics)?;
    writer.flush()?;

    println!("\nMetrics written to {}", args.output.display());
    println!(
        "Final state: tick={}, population={}, avg_energy={:.2}, total_food={:.1}, max_generation={}, species_count={}",
        final_metrics.tick,
        final_metrics.population,
        final_metrics.avg_energy,
        final_metrics.total_food,
        final_metrics.max_generation,
        final_metrics.species_count,
    );

    Ok(())
}

fn write_metrics(writer: &mut BufWriter<File>, metrics: &headless::TickMetrics) -> Result<()> {
    writeln!(
        writer,
        "{},{},{},{},{:.4},{:.4},{},{}",
        metrics.tick,
        metrics.population,
        metrics.births,
        metrics.deaths,
        metrics.avg_energy,
        metrics.total_food,
        metrics.max_generation,
        metrics.species_count,
    )?;
    Ok(())
}

struct Args {
    ticks: u32,
    interval: u32,
    seed: u64,
    output: PathBuf,
    config: Option<PathBuf>,
    world_size: Option<u32>,
    initial_organisms: Option<u32>,
    max_organisms: Option<u32>,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            ticks: 2_000,
            interval: 20,
            seed: 42,
            output: PathBuf::from("metrics.csv"),
            config: None,
            world_size: None,
            initial_organisms: None,
            max_organisms: None,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--ticks" => parsed.ticks = parse_value(&mut args, "--ticks"),
                "--interval" => parsed.interval = parse_value(&mut args, "--interval"),
                "--seed" => parsed.seed = parse_value(&mut args, "--seed"),
                "--config" => {
                    let value = args.next().unwrap_or_else(|| panic!("Missing value for --config"));
                    parsed.config = Some(PathBuf::from(value));
                }
                "--world-size" => parsed.world_size = Some(parse_value(&mut args, "--world-size")),
                "--initial" => parsed.initial_organisms = Some(parse_value(&mut args, "--initial")),
                "--max" => parsed.max_organisms = Some(parse_value(&mut args, "--max")),
                "--output" => {
                    let value = args.next().unwrap_or_else(|| panic!("Missing value for --output"));
                    parsed.output = PathBuf::from(value);
                }
                other => panic!("Unknown argument: {other}"),
            }
        }

        parsed.interval = parsed.interval.max(1);
        parsed.ticks = parsed.ticks.max(1);
        parsed
    }
}

fn parse_value<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, flag: &str) -> T {
    let value = args.next().unwrap_or_else(|| panic!("Missing value for {flag}"));
    value.parse::<T>().unwrap_or_else(|_| panic!("Invalid value for {flag}: {value}"))
}