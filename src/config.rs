//! Simulation configuration parameters

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Complete simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Optional fixed seed for reproducibility. 
    /// If provided, the simulation will always start with the same initial conditions.
    #[serde(default)]
    pub seed: Option<u64>,     
    
    /// Population limits and initial generation settings.
    #[serde(default)]
    pub population: PopulationConfig,
    
    /// Energy dynamics, metabolism, and lifespan settings.
    #[serde(default)]
    pub energy: EnergyConfig,
    
    /// Reproduction costs, requirements, and cooldowns.
    #[serde(default)]
    pub reproduction: ReproductionConfig,
    
    /// Genetic mutation rates and strengths.
    #[serde(default)]
    pub mutation: MutationConfig,
    
    /// Sensory capabilities of the organisms.
    #[serde(default)]
    pub vision: VisionConfig,
    
    /// Movement and physical constraints.
    #[serde(default)]
    pub physics: PhysicsConfig,
    
    /// Food generation, growth, and distribution settings.
    #[serde(default)]
    pub food: FoodConfig,
    
    /// World boundaries.
    #[serde(default)]
    pub world: WorldConfig,
    
    /// Predation (attack) system settings.
    #[serde(default)]
    pub predation: PredationConfig,
    
    /// Morphology (evolvable physical traits) settings.
    #[serde(default)]
    pub morphology: MorphologyConfig,
    
    /// Biomes (regional environmental differences) settings.
    #[serde(default)]
    pub biomes: BiomesConfig,

    /// Persistent survivor-bank bootstrap settings.
    #[serde(default)]
    pub bootstrap: BootstrapConfig,
    
    /// System and performance settings.
    #[serde(default)]
    pub system: SystemConfig,
}

impl SimulationConfig {
    /// Load configuration from a TOML file
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let mut config: SimulationConfig = toml::from_str(&contents)?;
        config.sanitize();
        Ok(config)
    }

    /// Enforce invariants that are required for a valid simulation.
    pub fn sanitize(&mut self) {
        if self.population.max_organisms == 0 {
            log::warn!(
                "Configured population.max_organisms=0 would create empty GPU buffers; forcing max_organisms=1"
            );
            self.population.max_organisms = 1;
        }

        if self.population.initial_organisms > self.population.max_organisms {
            log::warn!(
                "Configured population.initial_organisms={} exceeds max_organisms={}; clamping initial_organisms to {}",
                self.population.initial_organisms,
                self.population.max_organisms,
                self.population.max_organisms
            );
            self.population.initial_organisms = self.population.max_organisms;
        }

        if self.world.width == 0 {
            log::warn!(
                "Configured world.width=0 would break world allocation and wrapping; forcing width=1"
            );
            self.world.width = 1;
        }

        if self.world.height == 0 {
            log::warn!(
                "Configured world.height=0 would break world allocation and wrapping; forcing height=1"
            );
            self.world.height = 1;
        }

        if self.vision.rays != 8 {
            log::warn!(
                "Configured vision.rays={} does not match the hardcoded 20-input network; forcing rays=8",
                self.vision.rays
            );
            self.vision.rays = 8;
        }

        clamp_non_negative_f32(&mut self.vision.range, "vision.range");
        clamp_non_negative_f32(&mut self.energy.passive_drain, "energy.passive_drain");
        clamp_non_negative_f32(
            &mut self.energy.movement_cost_forward,
            "energy.movement_cost_forward",
        );
        clamp_non_negative_f32(
            &mut self.energy.movement_cost_rotate,
            "energy.movement_cost_rotate",
        );
        clamp_non_negative_f32(&mut self.energy.crowding_factor, "energy.crowding_factor");
        clamp_non_negative_f32(&mut self.energy.age_drain_factor, "energy.age_drain_factor");

        if self.energy.maximum <= 0.0 {
            log::warn!(
                "Configured energy.maximum={} is invalid; forcing maximum={} from defaults",
                self.energy.maximum,
                EnergyConfig::default().maximum
            );
            self.energy.maximum = EnergyConfig::default().maximum;
        }

        if self.energy.starting < 0.0 {
            log::warn!(
                "Configured energy.starting={} is invalid; forcing starting=0",
                self.energy.starting
            );
            self.energy.starting = 0.0;
        }

        if self.energy.starting > self.energy.maximum {
            log::warn!(
                "Configured energy.starting={} exceeds energy.maximum={}; clamping starting to {}",
                self.energy.starting,
                self.energy.maximum,
                self.energy.maximum
            );
            self.energy.starting = self.energy.maximum;
        }

        clamp_non_negative_f32(&mut self.physics.max_speed, "physics.max_speed");
        clamp_non_negative_f32(&mut self.physics.max_rotation, "physics.max_rotation");

        if self.physics.organism_radius <= 0.0 {
            log::warn!(
                "Configured physics.organism_radius={} is invalid; forcing organism_radius={} from defaults",
                self.physics.organism_radius,
                PhysicsConfig::default().organism_radius
            );
            self.physics.organism_radius = PhysicsConfig::default().organism_radius;
        }

        clamp_non_negative_f32(&mut self.reproduction.threshold, "reproduction.threshold");
        clamp_unit_f32(&mut self.reproduction.signal_min, "reproduction.signal_min");
        clamp_non_negative_f32(&mut self.reproduction.cost, "reproduction.cost");
        clamp_non_negative_f32(&mut self.reproduction.mate_range, "reproduction.mate_range");
        clamp_unit_f32(
            &mut self.reproduction.mate_signal_min,
            "reproduction.mate_signal_min",
        );
        clamp_unit_f32(
            &mut self.reproduction.crossover_ratio,
            "reproduction.crossover_ratio",
        );

        if self.reproduction.threshold > self.energy.maximum {
            log::warn!(
                "Configured reproduction.threshold={} exceeds energy.maximum={}; clamping threshold to {}",
                self.reproduction.threshold,
                self.energy.maximum,
                self.energy.maximum
            );
            self.reproduction.threshold = self.energy.maximum;
        }

        if self.reproduction.cost > self.reproduction.threshold {
            log::warn!(
                "Configured reproduction.cost={} exceeds reproduction.threshold={}; clamping cost to {}",
                self.reproduction.cost,
                self.reproduction.threshold,
                self.reproduction.threshold
            );
            self.reproduction.cost = self.reproduction.threshold;
        }

        if self.energy.max_age != 0 && self.reproduction.min_age > self.energy.max_age {
            log::warn!(
                "Configured reproduction.min_age={} exceeds energy.max_age={}; clamping min_age to {}",
                self.reproduction.min_age,
                self.energy.max_age,
                self.energy.max_age
            );
            self.reproduction.min_age = self.energy.max_age;
        }

        clamp_unit_f32(&mut self.mutation.rate, "mutation.rate");
        clamp_non_negative_f32(&mut self.mutation.strength, "mutation.strength");

        if self.food.max_per_cell <= 0.0 {
            log::warn!(
                "Configured food.max_per_cell={} would cause divide-by-zero in food growth; forcing max_per_cell={} from defaults",
                self.food.max_per_cell,
                FoodConfig::default().max_per_cell
            );
            self.food.max_per_cell = FoodConfig::default().max_per_cell;
        }

        clamp_non_negative_f32(&mut self.food.growth_rate, "food.growth_rate");
        clamp_non_negative_f32(&mut self.food.energy_value, "food.energy_value");
        clamp_non_negative_f32(&mut self.food.effectiveness, "food.effectiveness");

        if self.food.patch_size == 0 {
            log::warn!(
                "Configured food.patch_size=0 would create degenerate patches; forcing patch_size=1"
            );
            self.food.patch_size = 1;
        }

        if self.food.baseline_food < 0.0 {
            log::warn!(
                "Configured food.baseline_food={} is invalid; forcing baseline_food=0",
                self.food.baseline_food
            );
            self.food.baseline_food = 0.0;
        }

        if self.food.baseline_food > self.food.max_per_cell {
            log::warn!(
                "Configured food.baseline_food={} exceeds food.max_per_cell={}; clamping baseline_food to {}",
                self.food.baseline_food,
                self.food.max_per_cell,
                self.food.max_per_cell
            );
            self.food.baseline_food = self.food.max_per_cell;
        }

        clamp_non_negative_f32(&mut self.food.spawn_chance, "food.spawn_chance");
        clamp_non_negative_f32(&mut self.food.spawn_amount, "food.spawn_amount");

        if self.food.spawn_amount > self.food.max_per_cell {
            log::warn!(
                "Configured food.spawn_amount={} exceeds food.max_per_cell={}; clamping spawn_amount to {}",
                self.food.spawn_amount,
                self.food.max_per_cell,
                self.food.max_per_cell
            );
            self.food.spawn_amount = self.food.max_per_cell;
        }

        if self.food.seasonal_period == 0 {
            log::warn!(
                "Configured food.seasonal_period=0 would cause modulo-by-zero in shaders; forcing seasonal_period={} from defaults",
                default_seasonal_period()
            );
            self.food.seasonal_period = default_seasonal_period();
        }

        clamp_unit_f32(
            &mut self.food.seasonal_amplitude,
            "food.seasonal_amplitude",
        );

        if self.food.hotspots_enabled && self.food.hotspot_count == 0 {
            log::warn!(
                "Configured food.hotspot_count=0 disables hotspot generation while hotspots are enabled; forcing hotspot_count=1"
            );
            self.food.hotspot_count = 1;
        }

        clamp_non_negative_f32(&mut self.food.hotspot_radius, "food.hotspot_radius");
        clamp_non_negative_f32(&mut self.food.hotspot_intensity, "food.hotspot_intensity");

        clamp_non_negative_f32(&mut self.predation.attack_range, "predation.attack_range");
        clamp_non_negative_f32(&mut self.predation.attack_damage, "predation.attack_damage");
        clamp_unit_f32(&mut self.predation.energy_transfer, "predation.energy_transfer");
        clamp_non_negative_f32(&mut self.predation.attack_cost, "predation.attack_cost");

        normalize_f32_range(
            &mut self.morphology.min_size,
            &mut self.morphology.max_size,
            "morphology.min_size",
            "morphology.max_size",
        );
        normalize_f32_range(
            &mut self.morphology.min_speed_mult,
            &mut self.morphology.max_speed_mult,
            "morphology.min_speed_mult",
            "morphology.max_speed_mult",
        );
        normalize_f32_range(
            &mut self.morphology.min_vision_mult,
            &mut self.morphology.max_vision_mult,
            "morphology.min_vision_mult",
            "morphology.max_vision_mult",
        );
        normalize_f32_range(
            &mut self.morphology.min_metabolism,
            &mut self.morphology.max_metabolism,
            "morphology.min_metabolism",
            "morphology.max_metabolism",
        );
        clamp_unit_f32(
            &mut self.morphology.mutation_rate,
            "morphology.mutation_rate",
        );
        clamp_non_negative_f32(
            &mut self.morphology.mutation_strength,
            "morphology.mutation_strength",
        );

        if self.biomes.enabled && self.biomes.biome_count == 0 {
            log::warn!(
                "Configured biomes.biome_count=0 disables biome generation while biomes are enabled; forcing biome_count=1"
            );
            self.biomes.biome_count = 1;
        }

        if self.system.readback_interval != 1 {
            log::warn!(
                "Configured system.readback_interval={} is unsafe for reproduction sync; forcing readback_interval=1",
                self.system.readback_interval
            );
            self.system.readback_interval = 1;
        }

        if self.system.food_readback_interval == 0 {
            log::warn!(
                "Configured system.food_readback_interval=0 would panic during modulo checks; forcing food_readback_interval=1"
            );
            self.system.food_readback_interval = 1;
        }

        if self.system.diagnostic_interval == 0 {
            log::warn!(
                "Configured system.diagnostic_interval=0 would panic during modulo checks; forcing diagnostic_interval=1"
            );
            self.system.diagnostic_interval = 1;
        }

        if self.bootstrap.founder_count == 0 && self.bootstrap.enabled && self.bootstrap.load_on_start {
            log::warn!(
                "Configured bootstrap.founder_count=0 disables survivor reuse at startup; no founders will be loaded"
            );
        }

        if self.bootstrap.survivor_count == 0 && self.bootstrap.enabled && self.bootstrap.save_on_exit {
            log::warn!(
                "Configured bootstrap.survivor_count=0 disables survivor-bank export; no survivors will be saved"
            );
        }

        if self.bootstrap.enabled && self.bootstrap.path.as_os_str().is_empty() {
            log::warn!(
                "Configured bootstrap.path is empty; forcing bootstrap.path={} from defaults",
                default_bootstrap_path().display()
            );
            self.bootstrap.path = default_bootstrap_path();
        }
    }
    
    /// Save configuration to a TOML file
    pub fn save_to_file(&self, path: &Path) -> anyhow::Result<()> {
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    /// Load from file or create default if not exists
    pub fn load_or_create_default(path: &Path) -> Self {
        match Self::from_file(path) {
            Ok(mut config) => {
                config.sanitize();
                log::info!("Loaded configuration from {:?}", path);
                config
            }
            Err(e) => {
                log::warn!("Could not load config from {:?}: {}. Using defaults.", path, e);
                let mut config = Self::default();
                config.sanitize();
                // Try to save default config for user reference
                if let Err(e) = config.save_to_file(path) {
                    log::warn!("Could not save default config: {}", e);
                } else {
                    log::info!("Created default config file at {:?}", path);
                }
                config
            }
        }
    }
}

fn clamp_non_negative_f32(value: &mut f32, field: &str) {
    if *value < 0.0 {
        log::warn!("Configured {}={} is invalid; forcing {}=0", field, *value, field);
        *value = 0.0;
    }
}

fn clamp_unit_f32(value: &mut f32, field: &str) {
    if *value < 0.0 || *value > 1.0 {
        let clamped = value.clamp(0.0, 1.0);
        log::warn!(
            "Configured {}={} is outside [0, 1]; clamping {} to {}",
            field,
            *value,
            field,
            clamped
        );
        *value = clamped;
    }
}

fn normalize_f32_range(min_value: &mut f32, max_value: &mut f32, min_field: &str, max_field: &str) {
    if *min_value > *max_value {
        log::warn!(
            "Configured {}={} exceeds {}={}; swapping the bounds",
            min_field,
            *min_value,
            max_field,
            *max_value
        );
        std::mem::swap(min_value, max_value);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PopulationConfig {
    /// Maximum number of organisms allowed in the simulation.
    /// Range: [100, 1,000,000] depends on GPU memory.
    /// Recommended: 2,000 - 20,000 for smooth 60 FPS on typical hardware.
    pub max_organisms: u32,
    
    /// Number of organisms to spawn at the start of the simulation.
    /// Range: [1, max_organisms]
    /// Recommended: 10-20% of max_organisms to allow room for growth.
    pub initial_organisms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnergyConfig {
    /// Energy organisms start with when spawned or born.
    /// Range: [10.0, maximum]
    /// Recommended: 30-50% of maximum.
    pub starting: f32,
    
    /// Maximum energy an organism can store.
    /// Range: [50.0, 1000.0]
    /// Recommended: 100.0 - 200.0. Higher values allow longer survival without food.
    pub maximum: f32,
    
    /// Energy lost per tick just by existing.
    /// Range: [0.01, 1.0]
    /// Recommended: 0.1 - 0.2. High values kill idle organisms.
    pub passive_drain: f32,
    
    /// Energy cost per unit of forward movement.
    /// Range: [0.0, 0.5]
    /// Recommended: 0.01 - 0.05. Makes moving expensive.
    pub movement_cost_forward: f32,
    
    /// Energy cost per radian of rotation.
    /// Range: [0.0, 0.5]
    /// Recommended: 0.005 - 0.02.
    pub movement_cost_rotate: f32,
    
    /// Maximum age in ticks before an organism dies of old age. 0 = no limit.
    /// Range: [500, 100,000]
    /// Recommended: 2,000 - 5,000 (~30-80 seconds at 60 FPS).
    pub max_age: u32,              
    
    /// Multiplier for energy drain when the population is near max_organisms.
    /// Range: [0.0, 10.0]
    /// Recommended: 1.0 (doubles drain at max capacity) to 5.0 (drastic reduction).
    /// Used to enforce the carrying capacity smoothly.
    pub crowding_factor: f32,      
    
    /// Multiplier for energy drain based on age.
    /// Range: [0.0, 5.0]
    /// Recommended: 1.0. Increases drain linearly as organism approaches max_age.
    pub age_drain_factor: f32,     
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReproductionConfig {
    /// Minimum energy required to attempt reproduction.
    /// Range: [cost, energy.maximum]
    /// Recommended: > 50% of maximum energy.
    pub threshold: f32,
    
    /// Minimum output value (0.0-1.0) from the reproduction neuron to trigger valid reproduction.
    /// Range: [0.0, 1.0]
    /// Recommended: 0.3 - 0.7.
    pub signal_min: f32,
    
    /// Number of ticks an organism must wait after reproducing before doing so again.
    /// Range: [10, 1000]
    /// Recommended: 60 - 300 (1-5 seconds). Prevents rapid-fire cloning.
    pub cooldown: u32,
    
    /// Minimum age in ticks required to reproduce.
    /// Range: [0, max_age]
    /// Recommended: 100 - 500. Prevents newborns from reproducing immediately.
    pub min_age: u32,
    
    /// Energy cost to produce offspring. This energy is transferred to the child (partially).
    /// Range: [10.0, threshold]
    /// Recommended: 30.0 - 70.0.
    pub cost: f32,
    
    // === Sexual Reproduction Settings ===
    
    /// Enable sexual reproduction (genome crossover with a mate).
    /// When disabled, organisms reproduce asexually (clone + mutate).
    #[serde(default = "default_sexual_enabled")]
    pub sexual_enabled: bool,
    
    /// Maximum distance to find a mate.
    /// Range: [5.0, 50.0]
    #[serde(default = "default_mate_range")]
    pub mate_range: f32,
    
    /// Minimum mate desire signal from neural output to seek a mate.
    /// Range: [0.0, 1.0]
    #[serde(default = "default_mate_signal_min")]
    pub mate_signal_min: f32,
    
    /// Crossover rate - fraction of genes from parent vs mate.
    /// 0.5 = 50/50 split, 0.7 = 70% from parent, 30% from mate
    /// Range: [0.3, 0.7]
    #[serde(default = "default_crossover_ratio")]
    pub crossover_ratio: f32,
}

fn default_sexual_enabled() -> bool {
    false // Default to asexual for backward compatibility
}

fn default_mate_range() -> f32 {
    15.0
}

fn default_mate_signal_min() -> f32 {
    0.4
}

fn default_crossover_ratio() -> f32 {
    0.5 // 50/50 gene split
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MutationConfig {
    /// Probability that a single genome weight will mutate during reproduction.
    /// Range: [0.0, 1.0]
    /// Recommended: 0.01 - 0.2 (1% to 20%).
    pub rate: f32,
    
    /// Magnitude of mutation. Weights change by random value in [-strength, strength].
    /// Range: [0.01, 1.0]
    /// Recommended: 0.1 - 0.5.
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    /// Number of raycasts used for vision. 
    /// CRITICAL: Must match the neural network input size defined in `genome.rs`.
    /// Do not change via config without recompiling/updating constants.
    /// Default: 8
    pub rays: u32,
    
    /// Field of view in degrees.
    /// Range: [10.0, 360.0]
    /// Recommended: 90.0 - 180.0.
    pub fov_degrees: f32,
    
    /// Maximum distance organisms can see.
    /// Range: [10.0, 500.0]
    /// Recommended: 50.0 - 200.0.
    pub range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsConfig {
    /// Maximum movement speed (units per tick).
    /// Range: [0.1, 10.0]
    /// Recommended: 1.0 - 5.0.
    pub max_speed: f32,
    
    /// Maximum rotation speed (radians per tick).
    /// Range: [0.01, 1.0]
    /// Recommended: 0.1 - 0.4.
    pub max_rotation: f32,
    
    /// Physical radius of the organism for collision/rendering.
    /// Range: [1.0, 10.0]
    /// Recommended: 2.0 - 5.0.
    pub organism_radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoodConfig {
    /// Percentage of maximum food value to grow back per tick in patches.
    /// Range: [0.001, 1.0]
    /// Recommended: 0.01 - 0.1 (1% - 10%).
    pub growth_rate: f32,
    
    /// Maximum energy contained in a single food cell.
    /// Range: [1.0, 100.0]
    /// Recommended: 10.0 - 50.0.
    pub max_per_cell: f32,
    
    /// Energy gained by organism per unit of food consumed.
    /// Range: [1.0, 10.0]
    /// Recommended: 2.0 - 5.0. 
    /// Higher values make food more potent.
    pub energy_value: f32,
    
    /// Number of initial food patches generated.
    /// Range: [1, 500]
    /// Recommended: 50 - 200 depending on world size.
    pub initial_patches: u32,
    
    /// Radius of initial food patches.
    /// Range: [1, 100]
    /// Recommended: 10 - 40.
    pub patch_size: u32,
    
    /// Difficulty multiplier for eating.
    /// Range: [0.0, 1.0]
    /// Recommended: 1.0 (normal). Lower values imply inefficient digestion.
    pub effectiveness: f32,        
    
    /// Minimum food level that exists everywhere (background food).
    /// Range: [0.0, 1.0]
    /// Recommended: 0.0 for patch-only food, >0.0 prevents complete starvation in deserts.
    #[serde(default = "default_baseline_food")]
    pub baseline_food: f32,        

    /// Probability per cell per tick to spontaneously spawn a new food source (1.0 = 100%).
    /// Range: [0.0, 0.001]
    /// Recommended: 0.000001 (1 in a million) for occasional new patches.
    #[serde(default = "default_spawn_chance")]
    pub spawn_chance: f32,         
    
    /// Amount of food deposited when a spontaneous spawn occurs.
    /// Range: [0.1, max_per_cell]
    /// Recommended: 1.0 - 5.0.
    #[serde(default = "default_spawn_amount")]
    pub spawn_amount: f32,
    
    // === Dynamic Environment Features ===
    
    /// Enable seasonal food growth cycles.
    /// When enabled, food growth rate oscillates over time.
    #[serde(default)]
    pub seasonal_enabled: bool,
    
    /// Period of one full seasonal cycle in ticks.
    /// Default: 6000 (~100 seconds at 60 FPS).
    #[serde(default = "default_seasonal_period")]
    pub seasonal_period: u32,
    
    /// Amplitude of seasonal variation (0.0 to 1.0).
    /// 0.0 = no variation, 1.0 = growth drops to near zero in winter.
    #[serde(default = "default_seasonal_amplitude")]
    pub seasonal_amplitude: f32,
    
    /// Enable moving resource hotspots.
    /// Creates high-value food zones that drift slowly across the world.
    #[serde(default)]
    pub hotspots_enabled: bool,
    
    /// Number of resource hotspots (1-5).
    #[serde(default = "default_hotspot_count")]
    pub hotspot_count: u32,
    
    /// Radius of hotspot influence in cells.
    #[serde(default = "default_hotspot_radius")]
    pub hotspot_radius: f32,
    
    /// Extra food growth per tick at hotspot center.
    #[serde(default = "default_hotspot_intensity")]
    pub hotspot_intensity: f32,
}

fn default_baseline_food() -> f32 {
    0.0 // No baseline - food only in patches
}

fn default_spawn_chance() -> f32 {
    0.000001 // 1 in a million chance per cell per tick
}

fn default_spawn_amount() -> f32 {
    2.0 // Enough to start a new patch
}

fn default_seasonal_period() -> u32 {
    6000 // ~100 seconds at 60 ticks/sec
}

fn default_seasonal_amplitude() -> f32 {
    0.7 // Growth varies from 30% to 100%
}

fn default_hotspot_count() -> u32 {
    3
}

fn default_hotspot_radius() -> f32 {
    100.0
}

fn default_hotspot_intensity() -> f32 {
    0.3 // Extra growth bonus at center
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorldConfig {
    /// Width of the simulation grid.
    /// Recommended: Power of 2 (1024, 2048) for texture efficiency.
    pub width: u32,
    
    /// Height of the simulation grid.
    /// Recommended: Power of 2 (1024, 2048) for texture efficiency.
    pub height: u32,
}

/// Predation (attack) system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PredationConfig {
    /// Whether predation is enabled.
    #[serde(default = "default_predation_enabled")]
    pub enabled: bool,
    
    /// Minimum attack signal (output[4]) threshold to trigger attack.
    /// Range: [0.0, 1.0]
    /// Recommended: 0.3 - 0.5.
    #[serde(default = "default_attack_threshold")]
    pub attack_threshold: f32,
    
    /// Maximum distance for attack to connect.
    /// Range: [1.0, 20.0]
    /// Recommended: 5.0 - 10.0 (close range only).
    #[serde(default = "default_attack_range")]
    pub attack_range: f32,
    
    /// Damage dealt per successful attack.
    /// Range: [1.0, 50.0]
    /// Recommended: 10.0 - 30.0.
    #[serde(default = "default_attack_damage")]
    pub attack_damage: f32,
    
    /// Fraction of victim's energy transferred to attacker on kill.
    /// Range: [0.0, 1.0]
    /// Recommended: 0.3 - 0.7.
    #[serde(default = "default_energy_transfer")]
    pub energy_transfer: f32,
    
    /// Energy cost per attack attempt.
    /// Range: [0.0, 10.0]
    /// Recommended: 1.0 - 5.0 (prevents attack spam).
    #[serde(default = "default_attack_cost")]
    pub attack_cost: f32,
}

fn default_predation_enabled() -> bool {
    false // Disabled by default for backward compatibility
}

fn default_attack_threshold() -> f32 {
    0.3
}

fn default_attack_range() -> f32 {
    8.0
}

fn default_attack_damage() -> f32 {
    20.0
}

fn default_energy_transfer() -> f32 {
    0.5
}

fn default_attack_cost() -> f32 {
    2.0
}

impl Default for PredationConfig {
    fn default() -> Self {
        Self {
            enabled: default_predation_enabled(),
            attack_threshold: default_attack_threshold(),
            attack_range: default_attack_range(),
            attack_damage: default_attack_damage(),
            energy_transfer: default_energy_transfer(),
            attack_cost: default_attack_cost(),
        }
    }
}

/// Morphology configuration - evolvable physical traits
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MorphologyConfig {
    /// Whether morphology evolution is enabled.
    /// When disabled, all organisms have default traits.
    #[serde(default = "default_morphology_enabled")]
    pub enabled: bool,
    
    /// Minimum allowed size multiplier (1.0 = base size).
    /// Range: [0.5, 1.0]
    #[serde(default = "default_min_size")]
    pub min_size: f32,
    
    /// Maximum allowed size multiplier.
    /// Range: [1.0, 3.0]
    #[serde(default = "default_max_size")]
    pub max_size: f32,
    
    /// Minimum speed multiplier.
    /// Range: [0.5, 1.0]
    #[serde(default = "default_min_speed_mult")]
    pub min_speed_mult: f32,
    
    /// Maximum speed multiplier.
    /// Range: [1.0, 2.0]
    #[serde(default = "default_max_speed_mult")]
    pub max_speed_mult: f32,
    
    /// Minimum vision range multiplier.
    /// Range: [0.5, 1.0]
    #[serde(default = "default_min_vision_mult")]
    pub min_vision_mult: f32,
    
    /// Maximum vision range multiplier.
    /// Range: [1.0, 2.0]
    #[serde(default = "default_max_vision_mult")]
    pub max_vision_mult: f32,
    
    /// Minimum metabolism efficiency (lower = more passive drain).
    /// Range: [0.5, 1.0]
    #[serde(default = "default_min_metabolism")]
    pub min_metabolism: f32,
    
    /// Maximum metabolism efficiency (higher = less passive drain).
    /// Range: [1.0, 1.5]
    #[serde(default = "default_max_metabolism")]
    pub max_metabolism: f32,
    
    /// Mutation rate for morphology traits (separate from NN weights).
    /// Range: [0.0, 1.0]
    #[serde(default = "default_morph_mutation_rate")]
    pub mutation_rate: f32,
    
    /// Mutation strength for morphology traits (std dev).
    /// Range: [0.01, 0.3]
    #[serde(default = "default_morph_mutation_strength")]
    pub mutation_strength: f32,
}

fn default_morphology_enabled() -> bool {
    true
}

fn default_min_size() -> f32 {
    0.6
}

fn default_max_size() -> f32 {
    2.0
}

fn default_min_speed_mult() -> f32 {
    0.6
}

fn default_max_speed_mult() -> f32 {
    1.5
}

fn default_min_vision_mult() -> f32 {
    0.7
}

fn default_max_vision_mult() -> f32 {
    1.5
}

fn default_min_metabolism() -> f32 {
    0.7
}

fn default_max_metabolism() -> f32 {
    1.3
}

fn default_morph_mutation_rate() -> f32 {
    0.15
}

fn default_morph_mutation_strength() -> f32 {
    0.08
}

impl Default for MorphologyConfig {
    fn default() -> Self {
        Self {
            enabled: default_morphology_enabled(),
            min_size: default_min_size(),
            max_size: default_max_size(),
            min_speed_mult: default_min_speed_mult(),
            max_speed_mult: default_max_speed_mult(),
            min_vision_mult: default_min_vision_mult(),
            max_vision_mult: default_max_vision_mult(),
            min_metabolism: default_min_metabolism(),
            max_metabolism: default_max_metabolism(),
            mutation_rate: default_morph_mutation_rate(),
            mutation_strength: default_morph_mutation_strength(),
        }
    }
}

/// Biomes configuration - regional environmental differences
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BiomesConfig {
    /// Whether biomes are enabled.
    /// When enabled, the world is divided into regions with different properties.
    #[serde(default = "default_biomes_enabled")]
    pub enabled: bool,
    
    /// Number of biome regions (2-8).
    #[serde(default = "default_biome_count")]
    pub biome_count: u32,
    
    /// Food growth rate multiplier for "fertile" biome (1.0 = normal).
    #[serde(default = "default_fertile_growth_mult")]
    pub fertile_growth_mult: f32,
    
    /// Food growth rate multiplier for "barren" biome.
    #[serde(default = "default_barren_growth_mult")]
    pub barren_growth_mult: f32,
    
    /// Movement speed multiplier in "swamp" biome (lower = slower).
    #[serde(default = "default_swamp_speed_mult")]
    pub swamp_speed_mult: f32,
    
    /// Energy drain multiplier in "harsh" biome (higher = more drain).
    #[serde(default = "default_harsh_drain_mult")]
    pub harsh_drain_mult: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BootstrapConfig {
    /// Enables saving and reusing curated founders between runs.
    pub enabled: bool,

    /// Path to the persistent founder store. Use `.json` for a readable founder pool
    /// or `.bin` for the legacy binary survivor bank.
    pub path: PathBuf,

    /// Maximum number of organisms at startup that can reuse stored survivor genomes.
    pub founder_count: u32,

    /// Maximum number of survivors to retain when exporting a bank.
    pub survivor_count: u32,

    /// Whether startup should try to load founders from the survivor bank.
    pub load_on_start: bool,

    /// Whether shutdown should export the best living organisms into the survivor bank.
    pub save_on_exit: bool,
}

fn default_bootstrap_enabled() -> bool {
    true
}

fn default_bootstrap_path() -> PathBuf {
    PathBuf::from("founder_pool.json")
}

fn default_founder_count() -> u32 {
    128
}

fn default_survivor_count() -> u32 {
    256
}

fn default_load_on_start() -> bool {
    true
}

fn default_save_on_exit() -> bool {
    true
}

fn default_biomes_enabled() -> bool {
    false
}

fn default_biome_count() -> u32 {
    4
}

fn default_fertile_growth_mult() -> f32 {
    2.0 // Double food growth
}

fn default_barren_growth_mult() -> f32 {
    0.25 // Quarter food growth
}

fn default_swamp_speed_mult() -> f32 {
    0.5 // Half speed
}

fn default_harsh_drain_mult() -> f32 {
    2.0 // Double energy drain
}

impl Default for BiomesConfig {
    fn default() -> Self {
        Self {
            enabled: default_biomes_enabled(),
            biome_count: default_biome_count(),
            fertile_growth_mult: default_fertile_growth_mult(),
            barren_growth_mult: default_barren_growth_mult(),
            swamp_speed_mult: default_swamp_speed_mult(),
            harsh_drain_mult: default_harsh_drain_mult(),
        }
    }
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            enabled: default_bootstrap_enabled(),
            path: default_bootstrap_path(),
            founder_count: default_founder_count(),
            survivor_count: default_survivor_count(),
            load_on_start: default_load_on_start(),
            save_on_exit: default_save_on_exit(),
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            seed: None,
            population: PopulationConfig::default(),
            energy: EnergyConfig::default(),
            reproduction: ReproductionConfig::default(),
            mutation: MutationConfig::default(),
            vision: VisionConfig::default(),
            physics: PhysicsConfig::default(),
            food: FoodConfig::default(),
            world: WorldConfig::default(),
            predation: PredationConfig::default(),
            morphology: MorphologyConfig::default(),
            biomes: BiomesConfig::default(),
            bootstrap: BootstrapConfig::default(),
            system: SystemConfig::default(),
        }
    }
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            max_organisms: 4000,
            initial_organisms: 600,
        }
    }
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            starting: 70.0,
            maximum: 200.0,
            passive_drain: 0.14,
            movement_cost_forward: 0.02,
            movement_cost_rotate: 0.01,
            max_age: 2000,
            crowding_factor: 1.0,
            age_drain_factor: 1.0,
        }
    }
}

impl Default for ReproductionConfig {
    fn default() -> Self {
        Self {
            threshold: 70.0,
            signal_min: 0.3,
            cooldown: 120,
            min_age: 150,
            cost: 50.0,
            sexual_enabled: default_sexual_enabled(),
            mate_range: default_mate_range(),
            mate_signal_min: default_mate_signal_min(),
            crossover_ratio: default_crossover_ratio(),
        }
    }
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            rate: 0.05,
            strength: 0.2,
        }
    }
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            rays: 8,
            fov_degrees: 90.0,
            range: 50.0,
        }
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            max_speed: 2.5,
            max_rotation: 0.25,
            organism_radius: 3.0,
        }
    }
}

impl Default for FoodConfig {
    fn default() -> Self {
        Self {
            growth_rate: 0.003,
            max_per_cell: 10.0,
            energy_value: 4.0,
            initial_patches: 200,
            patch_size: 10,
            effectiveness: 1.0,
            baseline_food: 0.0,
            spawn_chance: 0.000001,
            spawn_amount: 2.0,
            seasonal_enabled: false,
            seasonal_period: default_seasonal_period(),
            seasonal_amplitude: default_seasonal_amplitude(),
            hotspots_enabled: false,
            hotspot_count: default_hotspot_count(),
            hotspot_radius: default_hotspot_radius(),
            hotspot_intensity: default_hotspot_intensity(),
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemConfig {
    /// Interval for reading back organism state from GPU to CPU (ticks).
    /// Default: 1 (Every tick) - Required for accurate reproduction logic.
    pub readback_interval: u32,

    /// Interval for reading back food state (ticks).
    /// Default: 60 (Every ~1-2 seconds) - Statistics only.
    pub food_readback_interval: u32,
    
    /// Interval for logging diagnostic info (ticks).
    /// Default: 60.
    pub diagnostic_interval: u32,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            readback_interval: 1,
            food_readback_interval: 60,
            diagnostic_interval: 60,
        }
    }
}

/// GPU-compatible simulation uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimUniform {
    // World
    pub world_width: u32,
    pub world_height: u32,
    pub num_organisms: u32,
    pub tick: u32,
    
    // Vision
    pub vision_range: f32,
    pub vision_fov: f32,
    pub vision_rays: u32,
    pub _pad1: u32,
    
    // Energy
    pub max_energy: f32,
    pub passive_drain: f32,
    pub movement_cost_forward: f32,
    pub movement_cost_rotate: f32,
    
    // Age and crowding
    pub max_age: u32,
    pub crowding_factor: f32,
    pub max_organisms: u32,
    pub age_drain_factor: f32,
    
    // Physics
    pub max_speed: f32,
    pub max_rotation: f32,
    pub organism_radius: f32,
    pub _pad2: u32,
    
    // Food
    pub food_growth_rate: f32,
    pub food_max_per_cell: f32,
    pub food_energy_value: f32,
    pub food_effectiveness: f32,
    
    // Food extended
    pub food_baseline: f32,
    pub food_spawn_chance: f32,
    pub food_spawn_amount: f32,
    pub _pad5: f32,
    
    // Reproduction
    pub reproduction_threshold: f32,
    pub reproduction_signal_min: f32,
    pub reproduction_cost: f32,
    pub reproduction_min_age: u32,
    
    // Predation (16-byte aligned block)
    pub predation_enabled: u32,      // 0 = disabled, 1 = enabled
    pub attack_threshold: f32,
    pub attack_range: f32,
    pub attack_damage: f32,
    
    pub energy_transfer: f32,
    pub attack_cost: f32,
    
    // Dynamic environment (16-byte aligned block)
    pub seasonal_enabled: u32,       // 0 = disabled, 1 = enabled
    pub seasonal_period: u32,
    pub seasonal_amplitude: f32,
    pub hotspots_enabled: u32,       // 0 = disabled, 1 = enabled
    
    pub hotspot_count: u32,
    pub hotspot_radius: f32,
    pub hotspot_intensity: f32,
    pub _pad8: f32,
    
    // Biomes (16-byte aligned block)
    pub biomes_enabled: u32,         // 0 = disabled, 1 = enabled
    pub fertile_growth_mult: f32,
    pub barren_growth_mult: f32,
    pub swamp_speed_mult: f32,
    
    pub harsh_drain_mult: f32,
    pub _pad9: [f32; 3],
}

impl SimUniform {
    pub fn from_config(config: &SimulationConfig, num_organisms: u32, tick: u32) -> Self {
        Self {
            world_width: config.world.width,
            world_height: config.world.height,
            num_organisms,
            tick,
            
            vision_range: config.vision.range,
            vision_fov: config.vision.fov_degrees.to_radians(),
            vision_rays: config.vision.rays,
            _pad1: 0,
            
            max_energy: config.energy.maximum,
            passive_drain: config.energy.passive_drain,
            movement_cost_forward: config.energy.movement_cost_forward,
            movement_cost_rotate: config.energy.movement_cost_rotate,
            
            max_age: config.energy.max_age,
            crowding_factor: config.energy.crowding_factor,
            max_organisms: config.population.max_organisms,
            age_drain_factor: config.energy.age_drain_factor,
            
            max_speed: config.physics.max_speed,
            max_rotation: config.physics.max_rotation,
            organism_radius: config.physics.organism_radius,
            _pad2: 0,
            
            food_growth_rate: config.food.growth_rate,
            food_max_per_cell: config.food.max_per_cell,
            food_energy_value: config.food.energy_value,
            food_effectiveness: config.food.effectiveness,
            
            food_baseline: config.food.baseline_food,
            food_spawn_chance: config.food.spawn_chance,
            food_spawn_amount: config.food.spawn_amount,
            _pad5: 0.0,
            
            reproduction_threshold: config.reproduction.threshold,
            reproduction_signal_min: config.reproduction.signal_min,
            reproduction_cost: config.reproduction.cost,
            reproduction_min_age: config.reproduction.min_age,
            
            predation_enabled: if config.predation.enabled { 1 } else { 0 },
            attack_threshold: config.predation.attack_threshold,
            attack_range: config.predation.attack_range,
            attack_damage: config.predation.attack_damage,
            
            energy_transfer: config.predation.energy_transfer,
            attack_cost: config.predation.attack_cost,
            
            seasonal_enabled: if config.food.seasonal_enabled { 1 } else { 0 },
            seasonal_period: config.food.seasonal_period,
            seasonal_amplitude: config.food.seasonal_amplitude,
            hotspots_enabled: if config.food.hotspots_enabled { 1 } else { 0 },
            
            hotspot_count: config.food.hotspot_count,
            hotspot_radius: config.food.hotspot_radius,
            hotspot_intensity: config.food.hotspot_intensity,
            _pad8: 0.0,
            
            biomes_enabled: if config.biomes.enabled { 1 } else { 0 },
            fertile_growth_mult: config.biomes.fertile_growth_mult,
            barren_growth_mult: config.biomes.barren_growth_mult,
            swamp_speed_mult: config.biomes.swamp_speed_mult,
            
            harsh_drain_mult: config.biomes.harsh_drain_mult,
            _pad9: [0.0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;

    #[test]
    fn sanitize_enforces_critical_runtime_invariants() {
        let mut config = SimulationConfig::default();
        config.vision.rays = 12;
        config.system.readback_interval = 4;
        config.system.food_readback_interval = 0;
        config.system.diagnostic_interval = 0;

        config.sanitize();

        assert_eq!(config.vision.rays, 8);
        assert_eq!(config.system.readback_interval, 1);
        assert_eq!(config.system.food_readback_interval, 1);
        assert_eq!(config.system.diagnostic_interval, 1);
    }

    #[test]
    fn partial_toml_uses_defaults_for_omitted_sections_and_fields() {
        let toml = r#"
            seed = 7

            [population]
            initial_organisms = 128

            [food]
            effectiveness = 0.8

            [system]
            diagnostic_interval = 15
        "#;

        let mut config: SimulationConfig = toml::from_str(toml).expect("partial config should deserialize");
        config.sanitize();

        assert_eq!(config.seed, Some(7));
        assert_eq!(config.population.initial_organisms, 128);
        assert_eq!(config.population.max_organisms, 4000);
        assert_eq!(config.food.effectiveness, 0.8);
        assert_eq!(config.food.max_per_cell, 10.0);
        assert_eq!(config.food.growth_rate, 0.003);
        assert_eq!(config.vision.rays, 8);
        assert_eq!(config.system.readback_interval, 1);
        assert_eq!(config.system.diagnostic_interval, 15);
    }

    #[test]
    fn sanitize_clamps_invalid_runtime_parameters() {
        let mut config = SimulationConfig::default();
        config.population.max_organisms = 0;
        config.population.initial_organisms = 10;
        config.world.width = 0;
        config.world.height = 0;
        config.energy.maximum = -5.0;
        config.energy.starting = 500.0;
        config.reproduction.threshold = 600.0;
        config.reproduction.cost = 700.0;
        config.reproduction.min_age = 5000;
        config.food.max_per_cell = 0.0;
        config.food.patch_size = 0;
        config.food.baseline_food = 20.0;
        config.food.spawn_amount = 50.0;
        config.food.seasonal_period = 0;
        config.food.seasonal_amplitude = 2.0;
        config.food.hotspots_enabled = true;
        config.food.hotspot_count = 0;
        config.morphology.min_size = 2.0;
        config.morphology.max_size = 0.5;
        config.morphology.mutation_rate = 1.5;
        config.predation.energy_transfer = 3.0;
        config.bootstrap.path = PathBuf::new();
        config.biomes.enabled = true;
        config.biomes.biome_count = 0;

        config.sanitize();

        assert_eq!(config.population.max_organisms, 1);
        assert_eq!(config.population.initial_organisms, 1);
        assert_eq!(config.world.width, 1);
        assert_eq!(config.world.height, 1);
        assert_eq!(config.energy.maximum, EnergyConfig::default().maximum);
        assert_eq!(config.energy.starting, config.energy.maximum);
        assert_eq!(config.reproduction.threshold, config.energy.maximum);
        assert_eq!(config.reproduction.cost, config.reproduction.threshold);
        assert_eq!(config.reproduction.min_age, config.energy.max_age);
        assert_eq!(config.food.max_per_cell, FoodConfig::default().max_per_cell);
        assert_eq!(config.food.patch_size, 1);
        assert_eq!(config.food.baseline_food, config.food.max_per_cell);
        assert_eq!(config.food.spawn_amount, config.food.max_per_cell);
        assert_eq!(config.food.seasonal_period, default_seasonal_period());
        assert_eq!(config.food.seasonal_amplitude, 1.0);
        assert_eq!(config.food.hotspot_count, 1);
        assert_eq!(config.morphology.min_size, 0.5);
        assert_eq!(config.morphology.max_size, 2.0);
        assert_eq!(config.morphology.mutation_rate, 1.0);
        assert_eq!(config.predation.energy_transfer, 1.0);
        assert_eq!(config.bootstrap.path, default_bootstrap_path());
        assert_eq!(config.biomes.biome_count, 1);
    }

    #[test]
    fn sim_uniform_carries_extended_config_fields() {
        let mut config = SimulationConfig::default();
        config.food.seasonal_enabled = true;
        config.food.seasonal_period = 4321;
        config.food.seasonal_amplitude = 0.65;
        config.food.hotspots_enabled = true;
        config.food.hotspot_count = 4;
        config.food.hotspot_radius = 120.0;
        config.food.hotspot_intensity = 0.45;
        config.food.baseline_food = 0.25;
        config.food.spawn_chance = 0.0025;
        config.food.spawn_amount = 3.5;
        config.predation.enabled = true;
        config.predation.attack_threshold = -0.2;
        config.predation.attack_range = 14.0;
        config.predation.attack_damage = 18.0;
        config.predation.energy_transfer = 0.35;
        config.predation.attack_cost = 1.2;
        config.biomes.enabled = true;
        config.biomes.fertile_growth_mult = 1.8;
        config.biomes.barren_growth_mult = 0.4;
        config.biomes.swamp_speed_mult = 0.7;
        config.biomes.harsh_drain_mult = 1.6;

        let uniform = SimUniform::from_config(&config, 77, 9);

        assert_eq!(uniform.num_organisms, 77);
        assert_eq!(uniform.tick, 9);
        assert_eq!(uniform.food_baseline, 0.25);
        assert_eq!(uniform.food_spawn_chance, 0.0025);
        assert_eq!(uniform.food_spawn_amount, 3.5);
        assert_eq!(uniform.predation_enabled, 1);
        assert_eq!(uniform.attack_threshold, -0.2);
        assert_eq!(uniform.attack_range, 14.0);
        assert_eq!(uniform.attack_damage, 18.0);
        assert_eq!(uniform.energy_transfer, 0.35);
        assert_eq!(uniform.attack_cost, 1.2);
        assert_eq!(uniform.seasonal_enabled, 1);
        assert_eq!(uniform.seasonal_period, 4321);
        assert_eq!(uniform.seasonal_amplitude, 0.65);
        assert_eq!(uniform.hotspots_enabled, 1);
        assert_eq!(uniform.hotspot_count, 4);
        assert_eq!(uniform.hotspot_radius, 120.0);
        assert_eq!(uniform.hotspot_intensity, 0.45);
        assert_eq!(uniform.biomes_enabled, 1);
        assert_eq!(uniform.fertile_growth_mult, 1.8);
        assert_eq!(uniform.barren_growth_mult, 0.4);
        assert_eq!(uniform.swamp_speed_mult, 0.7);
        assert_eq!(uniform.harsh_drain_mult, 1.6);
    }

    #[test]
    fn reproduction_spawns_child_near_parent() {
        let mut config = SimulationConfig::default();
        config.seed = Some(5);
        config.population.max_organisms = 8;
        config.population.initial_organisms = 1;
        config.world.width = 64;
        config.world.height = 64;

        let mut simulation = Simulation::new(&config);
        let parent_id = 0u32;
        let parent_before = simulation.organisms.get(parent_id).unwrap().position;

        {
            let parent = simulation.organisms.get_mut(parent_id).unwrap();
            parent.energy = config.reproduction.threshold + config.reproduction.cost + 10.0;
            parent.age = config.reproduction.min_age;
            parent.cooldown = 0;
            parent.reproduce_signal = config.reproduction.signal_min + 0.1;
        }

        let result = simulation.handle_reproduction(&config);
        assert_eq!(result.new_genome_ids.len(), 1);

        let child_id = result.new_genome_ids[0];
        let child = simulation.organisms.get(child_id).unwrap();

        let width = config.world.width as f32;
        let height = config.world.height as f32;
        let mut dx = child.position.x - parent_before.x;
        let mut dy = child.position.y - parent_before.y;
        if dx > width / 2.0 { dx -= width; }
        if dx < -width / 2.0 { dx += width; }
        if dy > height / 2.0 { dy -= height; }
        if dy < -height / 2.0 { dy += height; }

        assert!(dx.abs() <= 5.0);
        assert!(dy.abs() <= 5.0);
    }

    #[test]
    fn initial_spawn_positions_stay_near_food() {
        let mut config = SimulationConfig::default();
        config.seed = Some(9);
        config.population.max_organisms = 24;
        config.population.initial_organisms = 24;
        config.world.width = 96;
        config.world.height = 96;
        config.food.initial_patches = 12;
        config.food.patch_size = 8;

        let simulation = Simulation::new(&config);
        let baseline = config.food.baseline_food + 1.0;
        let food_positions: Vec<glam::Vec2> = simulation
            .world
            .food
            .iter()
            .enumerate()
            .filter(|(_, food)| **food > baseline)
            .map(|(idx, _)| {
                let x = (idx as u32 % config.world.width) as f32 + 0.5;
                let y = (idx as u32 / config.world.width) as f32 + 0.5;
                glam::Vec2::new(x, y)
            })
            .collect();

        assert!(!food_positions.is_empty());

        let width = config.world.width as f32;
        let height = config.world.height as f32;

        for organism in simulation.organisms.iter().filter(|org| org.is_alive()) {
            let near_food = food_positions.iter().any(|food_pos| {
                let mut dx = organism.position.x - food_pos.x;
                let mut dy = organism.position.y - food_pos.y;
                if dx > width / 2.0 { dx -= width; }
                if dx < -width / 2.0 { dx += width; }
                if dy > height / 2.0 { dy -= height; }
                if dy < -height / 2.0 { dy += height; }
                dx.abs() <= 3.5 && dy.abs() <= 3.5
            });

            assert!(near_food, "initial organism at {:?} was not placed near a food cell", organism.position);
        }
    }
}
