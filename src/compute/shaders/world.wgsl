// World shader - food growth and world updates

struct SimConfig {
    world_width: u32,
    world_height: u32,
    num_organisms: u32,
    tick: u32,
    
    vision_range: f32,
    vision_fov: f32,
    vision_rays: u32,
    _pad1: u32,
    
    max_energy: f32,
    passive_drain: f32,
    movement_cost_forward: f32,
    movement_cost_rotate: f32,
    
    // Age and crowding
    max_age: u32,
    crowding_factor: f32,
    max_organisms: u32,
    age_drain_factor: f32,
    
    max_speed: f32,
    max_rotation: f32,
    organism_radius: f32,
    _pad2: u32,
    
    food_growth_rate: f32,
    food_max_per_cell: f32,
    food_energy_value: f32,
    food_effectiveness: f32,
    
    // Food extended
    food_baseline: f32,
    food_spawn_chance: f32,
    food_spawn_amount: f32,
    _pad5: f32,
    
    reproduction_threshold: f32,
    reproduction_signal_min: f32,
    reproduction_cost: f32,
    reproduction_min_age: u32,
    
    // Predation
    predation_enabled: u32,
    attack_threshold: f32,
    attack_range: f32,
    attack_damage: f32,
    
    energy_transfer: f32,
    attack_cost: f32,
    
    // Dynamic environment
    seasonal_enabled: u32,
    seasonal_period: u32,
    seasonal_amplitude: f32,
    hotspots_enabled: u32,
    
    hotspot_count: u32,
    hotspot_radius: f32,
    hotspot_intensity: f32,
    _pad8: f32,
    
    // Biomes
    biomes_enabled: u32,
    fertile_growth_mult: f32,
    barren_growth_mult: f32,
    swamp_speed_mult: f32,
    
    harsh_drain_mult: f32,
    _pad9_a: f32,
    _pad9_b: f32,
    _pad9_c: f32,
}

// Biome types
const BIOME_NORMAL: u32 = 0u;
const BIOME_FERTILE: u32 = 1u;
const BIOME_BARREN: u32 = 2u;
const BIOME_SWAMP: u32 = 3u;
const BIOME_HARSH: u32 = 4u;

// Note: We only need food buffer and config for world update
// Other bindings are included for layout compatibility
struct Organism {
    position: vec2<f32>,
    velocity: vec2<f32>,
    rotation: f32,
    energy: f32,
    age: u32,
    flags: u32,
    genome_id: u32,
    generation: u32,
    offspring_count: u32,
    parent_id: u32,
    reproduce_signal: f32,
    species_id: u32,
}

@group(0) @binding(0) var<storage, read_write> organisms: array<Organism>;
@group(0) @binding(1) var<storage, read_write> food: array<f32>;
@group(0) @binding(2) var<storage, read> obstacles: array<u32>;
@group(0) @binding(3) var<storage, read_write> sensory: array<f32>;
@group(0) @binding(4) var<storage, read_write> actions: array<f32>;
@group(0) @binding(5) var<storage, read> nn_weights: array<f32>; // Combined: weights_l1 + biases_l1 + weights_l2 + biases_l2 per genome
@group(0) @binding(6) var<uniform> config: SimConfig;
@group(0) @binding(7) var<storage, read> biomes: array<u32>;

// PCG-style hash function for better randomness than sin()
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Convert hash to 0-1 float
fn hash_to_float(h: u32) -> f32 {
    return f32(h) / 4294967295.0;
}

// Calculate seasonal growth multiplier (0.0 to 1.0)
fn seasonal_multiplier(tick: u32, period: u32, amplitude: f32) -> f32 {
    let phase = (f32(tick % period) / f32(period)) * 6.28318530718; // 2*PI
    let wave = sin(phase) * 0.5 + 0.5; // 0 to 1
    // amplitude controls how much the growth drops during "winter"
    // amplitude=0.7 means growth goes from 30% to 100%
    return 1.0 - amplitude + amplitude * wave;
}

// Calculate hotspot bonus for a given cell position
fn hotspot_bonus(x: u32, y: u32, tick: u32, count: u32, radius: f32, intensity: f32, world_width: u32, world_height: u32) -> f32 {
    var bonus = 0.0;
    let cell_pos = vec2<f32>(f32(x), f32(y));
    
    // Generate deterministic hotspot positions that drift over time
    for (var h = 0u; h < count; h++) {
        // Base position from hash (spread evenly across world)
        let base_seed = pcg_hash(h * 12345u + 9876u);
        let base_x = hash_to_float(base_seed) * f32(world_width);
        let base_y = hash_to_float(pcg_hash(base_seed)) * f32(world_height);
        
        // Slow drift in a circular pattern
        let drift_speed = 0.0003; // How fast hotspots move
        let drift_radius = 150.0; // How far hotspots wander
        let angle = f32(tick) * drift_speed + f32(h) * 2.094; // 120° offset per hotspot
        
        let hotspot_x = (base_x + cos(angle) * drift_radius) % f32(world_width);
        let hotspot_y = (base_y + sin(angle) * drift_radius) % f32(world_height);
        
        let dist = length(cell_pos - vec2<f32>(hotspot_x, hotspot_y));
        if dist < radius {
            // Falloff from center
            let falloff = 1.0 - (dist / radius);
            bonus += intensity * falloff * falloff; // Quadratic falloff
        }
    }
    
    return bonus;
}

fn wrapped_food_at(x: u32, y: u32) -> f32 {
    let wrapped_x = x % config.world_width;
    let wrapped_y = y % config.world_height;
    return food[wrapped_y * config.world_width + wrapped_x];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    
    if x >= config.world_width || y >= config.world_height {
        return;
    }
    
    let idx = y * config.world_width + x;
    
    // Skip obstacles
    if obstacles[idx] != 0u {
        return;
    }
    
    var current_food = food[idx];
    
    // Calculate effective growth rate with seasonal modifier
    var effective_growth_rate = config.food_growth_rate;
    if config.seasonal_enabled != 0u {
        effective_growth_rate *= seasonal_multiplier(
            config.tick, 
            config.seasonal_period, 
            config.seasonal_amplitude
        );
    }
    
    // Apply biome growth modifier
    if config.biomes_enabled != 0u {
        let current_biome = biomes[idx];
        if current_biome == BIOME_FERTILE {
            effective_growth_rate *= config.fertile_growth_mult;
        } else if current_biome == BIOME_BARREN {
            effective_growth_rate *= config.barren_growth_mult;
        }
        // Swamp and Harsh biomes don't affect food growth rate
    }
    
    // Food regrowth is driven by local patch support rather than self-amplifying cell value.
    // This keeps patches sustainable without letting total food inflate too aggressively.
    let left = wrapped_food_at((x + config.world_width - 1u) % config.world_width, y);
    let right = wrapped_food_at((x + 1u) % config.world_width, y);
    let up = wrapped_food_at(x, (y + config.world_height - 1u) % config.world_height);
    let down = wrapped_food_at(x, (y + 1u) % config.world_height);
    let neighborhood_support = (current_food + left + right + up + down) / (5.0 * config.food_max_per_cell);

    if neighborhood_support > 0.01 {
        let growth = effective_growth_rate * neighborhood_support * (1.0 - current_food / config.food_max_per_cell);
        current_food += growth;
    }
    
    // Add hotspot bonus growth
    if config.hotspots_enabled != 0u {
        let bonus = hotspot_bonus(
            x, y, 
            config.tick, 
            config.hotspot_count,
            config.hotspot_radius,
            config.hotspot_intensity,
            config.world_width,
            config.world_height
        );
        if bonus > 0.0 {
            // Hotspots add food even to empty cells (creating new patches)
            current_food += bonus;
        }
    }
    
    // Random food spawning - creates new patches for organisms to discover
    let seed = pcg_hash(x + pcg_hash(y + pcg_hash(config.tick)));
    let rand_val = hash_to_float(seed);
    
    // Rare spontaneous food cluster - rewards exploration
    // Uses configurable spawn chance
    if rand_val < config.food_spawn_chance {
        current_food = max(current_food, config.food_spawn_amount);  // New food source
    }
    
    // Clamp to valid range
    // Ensure we don't go below baseline (if configured)
    current_food = max(current_food, config.food_baseline);
    current_food = min(current_food, config.food_max_per_cell);
    
    food[idx] = current_food;
}
