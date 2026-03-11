use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::genome::{Genome, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM};
use evolution_sim::simulation::Simulation;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    disable_bootstrap: bool,
}

#[derive(Clone, Copy, Default)]
struct ScalarStats {
    min: f32,
    max: f32,
    mean: f32,
    stddev: f32,
}

impl ScalarStats {
    fn from_values(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / values.len() as f32;

        Self {
            min,
            max,
            mean,
            stddev: variance.sqrt(),
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let mut config = if let Some(path) = args.config.as_deref() {
        SimulationConfig::from_file(path)?
    } else {
        let mut config = SimulationConfig::default();
        config.sanitize();
        config
    };

    if args.disable_bootstrap {
        config.bootstrap.enabled = false;
        config.bootstrap.load_on_start = false;
        config.bootstrap.founder_count = 0;
    }

    let simulation = Simulation::new(&config);
    let alive: Vec<_> = simulation
        .organisms
        .iter()
        .enumerate()
        .filter(|(_, organism)| organism.is_alive())
        .collect();

    let seeded_founders = if config.bootstrap.enabled && config.bootstrap.load_on_start {
        config.bootstrap.founder_count.min(alive.len() as u32)
    } else {
        0
    };

    let mut morph_size = Vec::with_capacity(alive.len());
    let mut morph_speed = Vec::with_capacity(alive.len());
    let mut morph_vision = Vec::with_capacity(alive.len());
    let mut morph_metabolism = Vec::with_capacity(alive.len());
    let mut active_hidden_counts = Vec::with_capacity(alive.len());
    let mut pairwise_distances = Vec::new();
    let mut action_columns = vec![Vec::with_capacity(alive.len()); OUTPUT_DIM];
    let mut sensory_columns = vec![Vec::with_capacity(alive.len()); INPUT_DIM];
    let mut quantized_signatures = HashSet::new();

    let pair_limit = alive.len().min(64);
    for left in 0..pair_limit {
        for right in (left + 1)..pair_limit {
            let left_org = alive[left].1;
            let right_org = alive[right].1;
            let Some(left_genome) = simulation.genomes.get(left_org.genome_id) else {
                continue;
            };
            let Some(right_genome) = simulation.genomes.get(right_org.genome_id) else {
                continue;
            };
            pairwise_distances.push(left_genome.distance_to(right_genome));
        }
    }

    for &(index, organism) in &alive {
        let Some(genome) = simulation.genomes.get(organism.genome_id) else {
            continue;
        };

        morph_size.push(organism.morph_size);
        morph_speed.push(organism.morph_speed_mult);
        morph_vision.push(organism.morph_vision_mult);
        morph_metabolism.push(organism.morph_metabolism);

        let sensory = build_sensory_vector(&simulation, &config, index as u32);
        let (actions, active_hidden) = forward_pass_gpu_style(genome, &sensory);
        active_hidden_counts.push(active_hidden as f32);

        for (column, value) in sensory_columns.iter_mut().zip(sensory.iter()) {
            column.push(*value);
        }
        for (column, value) in action_columns.iter_mut().zip(actions.iter()) {
            column.push(*value);
        }

        let signature = actions.map(|value| (value * 4.0).round() as i32);
        quantized_signatures.insert(signature);
    }

    println!("Neural diversity probe");
    println!("config.initial_organisms={}", config.population.initial_organisms);
    println!("alive_startup_population={}", alive.len());
    println!(
        "bootstrap.enabled={} load_on_start={} founder_count={} assumed_seeded_founders={}",
        config.bootstrap.enabled,
        config.bootstrap.load_on_start,
        config.bootstrap.founder_count,
        seeded_founders
    );
    println!();

    report_scalar("pairwise_genome_distance_sample", &pairwise_distances);
    report_scalar("hidden_active_neurons", &active_hidden_counts);
    println!("quantized_action_signatures={} (0.25 buckets across 6 outputs)", quantized_signatures.len());
    println!();

    report_scalar("morph_size", &morph_size);
    report_scalar("morph_speed_mult", &morph_speed);
    report_scalar("morph_vision_mult", &morph_vision);
    report_scalar("morph_metabolism", &morph_metabolism);
    println!();

    for (idx, column) in sensory_columns.iter().enumerate() {
        let label = format!("sensory[{idx}]");
        report_scalar(&label, column);
    }
    println!();

    let action_names = ["forward", "rotate", "mouth", "reproduce", "attack", "output5"];
    for (idx, column) in action_columns.iter().enumerate() {
        let stats = ScalarStats::from_values(column);
        let near_zero = if column.is_empty() {
            0.0
        } else {
            column.iter().filter(|value| value.abs() < 0.1).count() as f32 / column.len() as f32
        };
        println!(
            "action[{idx}] {:>10}: min={:.3} max={:.3} mean={:.3} std={:.3} near_zero={:.1}%",
            action_names[idx],
            stats.min,
            stats.max,
            stats.mean,
            stats.stddev,
            near_zero * 100.0
        );
    }

    Ok(())
}

fn report_scalar(label: &str, values: &[f32]) {
    let stats = ScalarStats::from_values(values);
    println!(
        "{label}: min={:.3} max={:.3} mean={:.3} std={:.3} count={}",
        stats.min,
        stats.max,
        stats.mean,
        stats.stddev,
        values.len()
    );
}

fn build_sensory_vector(simulation: &Simulation, config: &SimulationConfig, org_idx: u32) -> [f32; INPUT_DIM] {
    let mut sensory = [0.0; INPUT_DIM];
    let Some(org) = simulation.organisms.get(org_idx) else {
        return sensory;
    };

    let fov = config.vision.fov_degrees.to_radians();
    let half_fov = fov / 2.0;
    let effective_vision_range = config.vision.range * org.morph_vision_mult;
    let width = config.world.width as f32;
    let height = config.world.height as f32;
    let ray_count = 8usize;

    for ray in 0..ray_count {
        let t = ray as f32 / (ray_count as f32 - 1.0);
        let angle_offset = -half_fov + t * fov;
        let ray_angle = org.rotation + angle_offset;
        let ray_dir = glam::Vec2::new(ray_angle.cos(), ray_angle.sin());

        let mut hit_dist = 0.0;
        let mut hit_type = 0.0;
        let mut distance = 1.0;
        while distance < effective_vision_range {
            let sample_pos = org.position + ray_dir * distance;
            let wrap_x = sample_pos.x.rem_euclid(width);
            let wrap_y = sample_pos.y.rem_euclid(height);
            let grid_x = wrap_x as u32;
            let grid_y = wrap_y as u32;
            let grid_idx = (grid_y * config.world.width + grid_x) as usize;

            if simulation.world.obstacles[grid_idx] != 0 {
                hit_dist = 1.0 - (distance / effective_vision_range);
                hit_type = 0.25;
                break;
            }

            if simulation.world.food[grid_idx] > 0.5 {
                hit_dist = 1.0 - (distance / effective_vision_range);
                hit_type = 0.5;
                break;
            }

            distance += 1.0;
        }

        sensory[ray * 2] = hit_dist;
        sensory[ray * 2 + 1] = hit_type;
    }

    sensory[16] = org.energy / config.energy.maximum;
    sensory[17] = (org.age as f32 / 1000.0).min(1.0);
    sensory[18] = org.velocity.length() / config.physics.max_speed;

    let mut nearest_dist = config.vision.range;
    let mut nearest_angle = 0.0;
    let organism_count = simulation.organism_count();
    let check_stride = (organism_count / 64).max(1);

    let mut index = 0;
    while index < organism_count {
        if index != org_idx {
            if let Some(other) = simulation.organisms.get(index) {
                if other.is_alive() {
                    let mut delta = other.position - org.position;
                    if delta.x > width * 0.5 {
                        delta.x -= width;
                    }
                    if delta.x < -width * 0.5 {
                        delta.x += width;
                    }
                    if delta.y > height * 0.5 {
                        delta.y -= height;
                    }
                    if delta.y < -height * 0.5 {
                        delta.y += height;
                    }

                    let dist = delta.length();
                    if dist < config.vision.range && dist < nearest_dist {
                        nearest_dist = dist;
                        nearest_angle = delta.y.atan2(delta.x) - org.rotation;
                    }
                }
            }
        }

        index += check_stride;
    }

    if nearest_dist < config.vision.range {
        let mut angle_norm = nearest_angle / std::f32::consts::PI;
        if angle_norm > 1.0 {
            angle_norm -= 2.0;
        }
        if angle_norm < -1.0 {
            angle_norm += 2.0;
        }
        sensory[19] = angle_norm;
    } else {
        sensory[19] = 1.0;
    }

    sensory
}

fn forward_pass_gpu_style(genome: &Genome, sensory: &[f32; INPUT_DIM]) -> ([f32; OUTPUT_DIM], usize) {
    let mut hidden = [0.0; HIDDEN_DIM];
    let mut active_hidden = 0usize;

    for h in 0..HIDDEN_DIM {
        let mut sum = genome.biases_l1[h];
        for i in 0..INPUT_DIM {
            let weight_index = i * HIDDEN_DIM + h;
            sum += sensory[i] * genome.weights_l1[weight_index];
        }
        hidden[h] = sum.max(0.0);
        if hidden[h] > 0.0 {
            active_hidden += 1;
        }
    }

    let mut output = [0.0; OUTPUT_DIM];
    for o in 0..OUTPUT_DIM {
        let mut sum = genome.biases_l2[o];
        for h in 0..HIDDEN_DIM {
            let weight_index = h * OUTPUT_DIM + o;
            sum += hidden[h] * genome.weights_l2[weight_index];
        }
        output[o] = sum.tanh();
    }

    (output, active_hidden)
}