//! Headless simulation smoke test.
//! 
//! Runs a small GPU-backed simulation without opening a window and checks
//! invariants that should hold if the main loop is behaving sanely.

use evolution_sim::compute::ComputePipeline;
use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::Simulation;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    pollster::block_on(run()).unwrap();
}

async fn run() -> anyhow::Result<()> {
    let mut config = SimulationConfig::default();
    config.seed = Some(42);
    config.population.max_organisms = 256;
    config.population.initial_organisms = 64;
    config.world.width = 128;
    config.world.height = 128;
    config.food.initial_patches = 24;
    config.food.patch_size = 8;
    config.system.diagnostic_interval = 30;
    config.sanitize();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!("Failed to acquire GPU adapter for smoke test"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;

    let mut simulation = Simulation::new(&config);
    let mut compute = ComputePipeline::new(&device, &config, &simulation)?;

    let initial_population = simulation.organism_count();
    let ticks = 180u32;

    for tick in 0..ticks {
        let _ = compute.read_gpu_state(&device, &mut simulation, &config, tick);
        assert_invariants(&simulation, &config, tick)?;

        let result = simulation.handle_reproduction(&config);
        for genome_id in &result.new_genome_ids {
            if let Some(weights) = simulation.genomes.get_weights_flat(*genome_id) {
                compute.buffers.update_nn_weights_for_genome(&queue, *genome_id, &weights);
            }
        }
        for (idx, organism) in &result.organism_changes {
            compute.buffers.update_organism_at(&queue, *idx, organism);
        }

        simulation.update_species();
        compute.buffers.set_organism_count(simulation.organism_count());
        let _ = compute.dispatch(&device, &queue, &simulation, &config, tick);
    }

    let _ = compute.read_gpu_state(&device, &mut simulation, &config, ticks);
    assert_invariants(&simulation, &config, ticks)?;

    println!("Simulation smoke test complete:");
    println!("  Initial population: {}", initial_population);
    println!("  Final population:   {}", simulation.organism_count());
    println!("  Max generation:     {}", simulation.max_generation());
    println!("  Species count:      {}", simulation.species_count());
    println!("  Total food:         {:.1}", simulation.total_food());
    println!("  Average energy:     {:.2}", simulation.avg_energy());

    Ok(())
}

fn assert_invariants(simulation: &Simulation, config: &SimulationConfig, tick: u32) -> anyhow::Result<()> {
    let width = config.world.width as f32;
    let height = config.world.height as f32;
    let max_total_food = config.world.width as f32 * config.world.height as f32 * config.food.max_per_cell;

    if simulation.organism_count() > config.population.max_organisms {
        anyhow::bail!(
            "Tick {}: organism count {} exceeded configured max {}",
            tick,
            simulation.organism_count(),
            config.population.max_organisms
        );
    }

    let total_food = simulation.total_food();
    if !total_food.is_finite() || total_food < 0.0 || total_food > max_total_food + 1.0 {
        anyhow::bail!("Tick {}: invalid total food {:.3}", tick, total_food);
    }

    for (idx, organism) in simulation.organisms.iter().enumerate() {
        if !organism.is_alive() {
            continue;
        }

        if organism.genome_id != idx as u32 {
            anyhow::bail!(
                "Tick {}: organism {} has genome_id {} (slot invariant broken)",
                tick,
                idx,
                organism.genome_id
            );
        }

        if !organism.position.x.is_finite() || !organism.position.y.is_finite() {
            anyhow::bail!("Tick {}: organism {} has non-finite position", tick, idx);
        }
        if organism.position.x < 0.0 || organism.position.x >= width || organism.position.y < 0.0 || organism.position.y >= height {
            anyhow::bail!(
                "Tick {}: organism {} escaped world bounds at ({:.3}, {:.3})",
                tick,
                idx,
                organism.position.x,
                organism.position.y
            );
        }

        if !organism.velocity.x.is_finite() || !organism.velocity.y.is_finite() || !organism.rotation.is_finite() || !organism.energy.is_finite() {
            anyhow::bail!("Tick {}: organism {} has non-finite state", tick, idx);
        }
        if organism.energy < 0.0 || organism.energy > config.energy.maximum + 0.001 {
            anyhow::bail!("Tick {}: organism {} has out-of-range energy {:.3}", tick, idx, organism.energy);
        }
        if organism.species_id == 0 {
            anyhow::bail!("Tick {}: organism {} is alive but has no species assignment", tick, idx);
        }

        if config.morphology.enabled {
            if organism.morph_size < config.morphology.min_size || organism.morph_size > config.morphology.max_size {
                anyhow::bail!("Tick {}: organism {} has invalid morph_size {:.3}", tick, idx, organism.morph_size);
            }
            if organism.morph_speed_mult < config.morphology.min_speed_mult || organism.morph_speed_mult > config.morphology.max_speed_mult {
                anyhow::bail!("Tick {}: organism {} has invalid morph_speed_mult {:.3}", tick, idx, organism.morph_speed_mult);
            }
            if organism.morph_vision_mult < config.morphology.min_vision_mult || organism.morph_vision_mult > config.morphology.max_vision_mult {
                anyhow::bail!("Tick {}: organism {} has invalid morph_vision_mult {:.3}", tick, idx, organism.morph_vision_mult);
            }
            if organism.morph_metabolism < config.morphology.min_metabolism || organism.morph_metabolism > config.morphology.max_metabolism {
                anyhow::bail!("Tick {}: organism {} has invalid morph_metabolism {:.3}", tick, idx, organism.morph_metabolism);
            }
        }
    }

    Ok(())
}