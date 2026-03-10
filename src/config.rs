//! Simulation configuration parameters

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Complete simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Optional fixed seed for reproducibility. 
    /// If provided, the simulation will always start with the same initial conditions.
    #[serde(default)]
    pub seed: Option<u64>,     
    
    /// Population limits and initial generation settings.
    pub population: PopulationConfig,
    
    /// Energy dynamics, metabolism, and lifespan settings.
    pub energy: EnergyConfig,
    
    /// Reproduction costs, requirements, and cooldowns.
    pub reproduction: ReproductionConfig,
    
    /// Genetic mutation rates and strengths.
    pub mutation: MutationConfig,
    
    /// Sensory capabilities of the organisms.
    pub vision: VisionConfig,
    
    /// Movement and physical constraints.
    pub physics: PhysicsConfig,
    
    /// Food generation, growth, and distribution settings.
    pub food: FoodConfig,
    
    /// World boundaries.
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
    
    /// System and performance settings.
    #[serde(default)]
    pub system: SystemConfig,
}

impl SimulationConfig {
    /// Load configuration from a TOML file
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: SimulationConfig = toml::from_str(&contents)?;
        Ok(config)
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
            Ok(config) => {
                log::info!("Loaded configuration from {:?}", path);
                config
            }
            Err(e) => {
                log::warn!("Could not load config from {:?}: {}. Using defaults.", path, e);
                let config = Self::default();
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            seed: None,
            population: PopulationConfig {
                max_organisms: 4000,
                initial_organisms: 600,
            },
            energy: EnergyConfig {
                starting: 70.0,
                maximum: 200.0,
                passive_drain: 0.14,
                movement_cost_forward: 0.02,
                movement_cost_rotate: 0.01,
                max_age: 2000,
                crowding_factor: 1.0,
                age_drain_factor: 1.0,
            },
            reproduction: ReproductionConfig {
                threshold: 70.0,
                signal_min: 0.3,
                cooldown: 120,
                min_age: 150,
                cost: 50.0,
                sexual_enabled: default_sexual_enabled(),
                mate_range: default_mate_range(),
                mate_signal_min: default_mate_signal_min(),
                crossover_ratio: default_crossover_ratio(),
            },
            mutation: MutationConfig {
                rate: 0.05,
                strength: 0.2,
            },
            vision: VisionConfig {
                rays: 8,
                fov_degrees: 90.0,
                range: 50.0,
            },
            physics: PhysicsConfig {
                max_speed: 2.5,
                max_rotation: 0.25,
                organism_radius: 3.0,
            },
            food: FoodConfig {
                growth_rate: 0.05,
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
            },
            world: WorldConfig {
                width: 2048,
                height: 2048,
            },
            predation: PredationConfig::default(),
            morphology: MorphologyConfig::default(),
            biomes: BiomesConfig::default(),
            system: SystemConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
