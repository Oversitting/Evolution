use anyhow::Result;
use evolution_sim::compute::ComputePipeline;
use evolution_sim::config::SimulationConfig;
use evolution_sim::simulation::Simulation;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TickMetrics {
    pub tick: u32,
    pub population: u32,
    pub births: u32,
    pub deaths: u32,
    pub avg_energy: f32,
    pub total_food: f32,
    pub max_generation: u32,
    pub species_count: usize,
}

pub struct HeadlessRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    simulation: Simulation,
    compute: ComputePipeline,
    config: SimulationConfig,
    tick: u32,
}

impl HeadlessRunner {
    pub async fn new(mut config: SimulationConfig) -> Result<Self> {
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
            .ok_or_else(|| anyhow::anyhow!("Failed to acquire GPU adapter for headless run"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        let simulation = Simulation::new(&config);
        let compute = ComputePipeline::new(&device, &config, &simulation)?;

        Ok(Self {
            device,
            queue,
            simulation,
            compute,
            config,
            tick: 0,
        })
    }

    #[allow(dead_code)]
    pub fn simulation(&self) -> &Simulation {
        &self.simulation
    }

    #[allow(dead_code)]
    pub fn tick(&self) -> u32 {
        self.tick
    }

    pub fn step(&mut self) -> Result<TickMetrics> {
        let pop_before = self.simulation.organism_count();
        let _ = self.compute.read_gpu_state(&self.device, &mut self.simulation, &self.config, self.tick);
        let pop_after_readback = self.simulation.organism_count();
        let deaths = pop_before.saturating_sub(pop_after_readback);

        let result = self.simulation.handle_reproduction(&self.config);
        let births = result.new_genome_ids.len() as u32;

        for genome_id in &result.new_genome_ids {
            if let Some(weights) = self.simulation.genomes.get_weights_flat(*genome_id) {
                self.compute
                    .buffers
                    .update_nn_weights_for_genome(&self.queue, *genome_id, &weights);
            }
        }
        for (idx, organism) in &result.organism_changes {
            self.compute.buffers.update_organism_at(&self.queue, *idx, organism);
        }

        self.simulation.update_species();
        self.compute.buffers.set_organism_count(self.simulation.organism_count());
        let _ = self.compute.dispatch(&self.device, &self.queue, &self.simulation, &self.config, self.tick);

        self.tick += 1;
        Ok(self.snapshot(births, deaths))
    }

    pub fn flush(&mut self) -> Result<TickMetrics> {
        let pop_before = self.simulation.organism_count();
        let _ = self.compute.read_gpu_state(&self.device, &mut self.simulation, &self.config, self.tick);
        let deaths = pop_before.saturating_sub(self.simulation.organism_count());
        Ok(self.snapshot(0, deaths))
    }

    fn snapshot(&self, births: u32, deaths: u32) -> TickMetrics {
        TickMetrics {
            tick: self.tick,
            population: self.simulation.organism_count(),
            births,
            deaths,
            avg_energy: self.simulation.avg_energy(),
            total_food: self.simulation.total_food(),
            max_generation: self.simulation.max_generation(),
            species_count: self.simulation.species_count(),
        }
    }
}