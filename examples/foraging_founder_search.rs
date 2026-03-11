//! Search for founders that can turn toward nearby food and reach it.
//!
//! This runs large batches of random genomes against a direct foraging challenge:
//! each organism starts near food, but never directly pointed at it. Any genome
//! that manages to eat the food is recorded in a human-readable founder pool.

use anyhow::Result;
use clap::Parser;
use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::genome::{Genome, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM};
use evolution_sim::simulation::{FounderPool, FounderRecord, SavedGenome};
use glam::Vec2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "founder_pool.json")]
    output: PathBuf,

    #[arg(long, default_value_t = 10_000)]
    batch_size: usize,

    #[arg(long, default_value_t = 40)]
    iterations: u32,

    #[arg(long, default_value_t = 256)]
    target_founders: usize,

    #[arg(long, default_value_t = 90)]
    max_steps: u32,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Clone, Copy)]
struct Challenge {
    food_distance: f32,
    angle_offset_degrees: f32,
}

#[derive(Default)]
struct SearchOutcome {
    evaluations: u32,
    successes: u32,
    best_steps_to_food: u32,
    average_steps_to_food: f32,
    score: f32,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(error) = run() {
        eprintln!("Foraging founder search failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let config = SimulationConfig::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let mut pool = if args.output.exists() {
        FounderPool::load_from_file(&args.output)?
    } else {
        FounderPool {
            version: FounderPool::VERSION,
            source_tick: 0,
            description: "Founders discovered by the direct foraging challenge".to_string(),
            entries: Vec::new(),
        }
    };

    let challenges = build_challenges();
    let mut total_saved = 0usize;

    for iteration in 0..args.iterations {
        let mut saved_this_iteration = 0usize;

        for batch_index in 0..args.batch_size {
            let genome = Genome::new_random(&mut rng, &config.morphology);
            let outcome = evaluate_genome(&genome, &config, &challenges, args.max_steps);
            if outcome.successes == 0 {
                continue;
            }

            let record = FounderRecord {
                enabled: true,
                label: format!("forager-{:04}-{:05}", iteration + 1, batch_index + 1),
                source: "foraging_search".to_string(),
                genome: SavedGenome::from(&genome),
                generation: 0,
                offspring_count: 0,
                age: outcome.best_steps_to_food,
                energy: config.energy.starting,
                species_id: 0,
                score: outcome.score,
                evaluations: outcome.evaluations,
                successes: outcome.successes,
                best_steps_to_food: outcome.best_steps_to_food,
                average_steps_to_food: outcome.average_steps_to_food,
                notes: format!(
                    "Direct foraging challenge: {} / {} scenarios reached food",
                    outcome.successes,
                    outcome.evaluations
                ),
                tags: vec!["foraging".to_string(), "local-food".to_string()],
            };

            if upsert_record(&mut pool, record, args.target_founders) {
                saved_this_iteration += 1;
            }
        }

        total_saved += saved_this_iteration;
        pool.source_tick = iteration as u64 + 1;
        pool.description = format!(
            "Founders discovered by the direct foraging challenge after {} iterations",
            iteration + 1
        );
        pool.save_to_file(&args.output)?;

        let best_score = pool.entries.first().map(|entry| entry.score).unwrap_or(0.0);
        println!(
            "iteration {:3} | saved {:4} | pool {:4} | best_score {:9.1}",
            iteration + 1,
            saved_this_iteration,
            pool.entries.len(),
            best_score,
        );

        if pool.entries.len() >= args.target_founders && saved_this_iteration == 0 {
            break;
        }
    }

    println!(
        "\nFounder search complete: pool_entries={} total_new_saved={} output={}",
        pool.entries.len(),
        total_saved,
        args.output.display()
    );

    Ok(())
}

fn build_challenges() -> Vec<Challenge> {
    vec![
        Challenge { food_distance: 10.0, angle_offset_degrees: -40.0 },
        Challenge { food_distance: 10.0, angle_offset_degrees: -30.0 },
        Challenge { food_distance: 10.0, angle_offset_degrees: -20.0 },
        Challenge { food_distance: 10.0, angle_offset_degrees: 20.0 },
        Challenge { food_distance: 10.0, angle_offset_degrees: 30.0 },
        Challenge { food_distance: 10.0, angle_offset_degrees: 40.0 },
    ]
}

fn evaluate_genome(
    genome: &Genome,
    config: &SimulationConfig,
    challenges: &[Challenge],
    max_steps: u32,
) -> SearchOutcome {
    let mut success_steps = Vec::new();

    for challenge in challenges {
        if let Some(steps) = run_challenge(genome, config, *challenge, max_steps) {
            success_steps.push(steps);
        }
    }

    if success_steps.is_empty() {
        return SearchOutcome {
            evaluations: challenges.len() as u32,
            best_steps_to_food: max_steps,
            ..SearchOutcome::default()
        };
    }

    let best_steps = success_steps.iter().copied().min().unwrap_or(max_steps);
    let average_steps = success_steps.iter().map(|&step| step as f32).sum::<f32>() / success_steps.len() as f32;
    let score = success_steps.len() as f32 * 10_000.0 + (max_steps - best_steps) as f32 * 50.0 + (max_steps as f32 - average_steps) * 10.0;

    SearchOutcome {
        evaluations: challenges.len() as u32,
        successes: success_steps.len() as u32,
        best_steps_to_food: best_steps,
        average_steps_to_food: average_steps,
        score,
    }
}

fn run_challenge(genome: &Genome, config: &SimulationConfig, challenge: Challenge, max_steps: u32) -> Option<u32> {
    let world_size = Vec2::new(64.0, 64.0);
    let mut position = world_size / 2.0;
    let mut velocity = Vec2::ZERO;
    let mut rotation = 0.0f32;
    let mut energy = config.energy.starting;

    let food_direction = Vec2::from_angle(challenge.angle_offset_degrees.to_radians());
    let food_pos = position + food_direction * challenge.food_distance;
    let food_radius = 1.75;

    for step in 0..max_steps {
        let sensory = build_sensory(position, velocity, rotation, energy, config, genome, food_pos, world_size, step);
        let outputs = forward_pass(genome, &sensory);

        rotation += outputs[1] * config.physics.max_rotation;
        if rotation < 0.0 {
            rotation += std::f32::consts::TAU;
        }
        if rotation > std::f32::consts::TAU {
            rotation -= std::f32::consts::TAU;
        }

        let direction = Vec2::new(rotation.cos(), rotation.sin());
        let speed = outputs[0] * config.physics.max_speed * genome.morphology.speed_mult;
        velocity = direction * speed;
        position += velocity;
        position.x = position.x.rem_euclid(world_size.x);
        position.y = position.y.rem_euclid(world_size.y);

        let size_cost_mult = genome.morphology.size * genome.morphology.size;
        let movement_cost =
            (outputs[0].abs() * config.energy.movement_cost_forward + outputs[1].abs() * config.energy.movement_cost_rotate)
                * size_cost_mult;
        let passive_drain = config.energy.passive_drain * (1.0 / genome.morphology.metabolism) * genome.morphology.size;
        energy -= movement_cost + passive_drain;

        if torus_distance(position, food_pos, world_size) <= food_radius {
            return Some(step + 1);
        }

        if energy <= 0.0 {
            return None;
        }
    }

    None
}

fn build_sensory(
    position: Vec2,
    velocity: Vec2,
    rotation: f32,
    energy: f32,
    config: &SimulationConfig,
    genome: &Genome,
    food_pos: Vec2,
    world_size: Vec2,
    step: u32,
) -> [f32; INPUT_DIM] {
    let mut sensory = [0.0; INPUT_DIM];
    let fov = config.vision.fov_degrees.to_radians();
    let half_fov = fov / 2.0;
    let effective_vision_range = config.vision.range * genome.morphology.vision_mult;

    for ray in 0..8usize {
        let t = ray as f32 / 7.0;
        let ray_angle = rotation - half_fov + t * fov;
        let ray_dir = Vec2::new(ray_angle.cos(), ray_angle.sin());
        let mut hit_dist = 0.0;
        let mut hit_type = 0.0;
        let mut distance = 1.0;

        while distance < effective_vision_range {
            let sample = position + ray_dir * distance;
            if torus_distance(sample, food_pos, world_size) <= 1.25 {
                hit_dist = 1.0 - (distance / effective_vision_range);
                hit_type = 0.5;
                break;
            }
            distance += 1.0;
        }

        sensory[ray * 2] = hit_dist;
        sensory[ray * 2 + 1] = hit_type;
    }

    sensory[16] = energy / config.energy.maximum;
    sensory[17] = (step as f32 / 1000.0).min(1.0);
    sensory[18] = velocity.length() / config.physics.max_speed;
    sensory[19] = 1.0;
    sensory
}

fn forward_pass(genome: &Genome, sensory: &[f32; INPUT_DIM]) -> [f32; OUTPUT_DIM] {
    let mut hidden = [0.0; HIDDEN_DIM];
    for hidden_idx in 0..HIDDEN_DIM {
        let mut sum = genome.biases_l1[hidden_idx];
        for input_idx in 0..INPUT_DIM {
            let weight_idx = input_idx * HIDDEN_DIM + hidden_idx;
            sum += sensory[input_idx] * genome.weights_l1[weight_idx];
        }
        hidden[hidden_idx] = sum.max(0.0);
    }

    let mut output = [0.0; OUTPUT_DIM];
    for output_idx in 0..OUTPUT_DIM {
        let mut sum = genome.biases_l2[output_idx];
        for hidden_idx in 0..HIDDEN_DIM {
            let weight_idx = hidden_idx * OUTPUT_DIM + output_idx;
            sum += hidden[hidden_idx] * genome.weights_l2[weight_idx];
        }
        output[output_idx] = sum.tanh();
    }

    output
}

fn upsert_record(pool: &mut FounderPool, record: FounderRecord, limit: usize) -> bool {
    let new_genome = Genome::from(&record.genome);

    for existing in &mut pool.entries {
        let existing_genome = Genome::from(&existing.genome);
        if new_genome.distance_to(&existing_genome) < 1.2 {
            if record.score > existing.score {
                *existing = record;
                sort_pool(pool, limit);
                return true;
            }
            return false;
        }
    }

    pool.entries.push(record);
    sort_pool(pool, limit);
    true
}

fn sort_pool(pool: &mut FounderPool, limit: usize) {
    pool.entries.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(right.successes.cmp(&left.successes))
            .then(left.best_steps_to_food.cmp(&right.best_steps_to_food))
    });
    pool.entries.truncate(limit);
}

fn torus_distance(a: Vec2, b: Vec2, world_size: Vec2) -> f32 {
    let mut delta = b - a;
    if delta.x > world_size.x * 0.5 {
        delta.x -= world_size.x;
    }
    if delta.x < -world_size.x * 0.5 {
        delta.x += world_size.x;
    }
    if delta.y > world_size.y * 0.5 {
        delta.y -= world_size.y;
    }
    if delta.y < -world_size.y * 0.5 {
        delta.y += world_size.y;
    }
    delta.length()
}