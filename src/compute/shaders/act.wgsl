// Act shader - apply actions to organisms (movement, eating)

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
    // Morphology traits
    morph_size: f32,
    morph_speed_mult: f32,
    morph_vision_mult: f32,
    morph_metabolism: f32,
}

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

@group(0) @binding(0) var<storage, read_write> organisms: array<Organism>;
@group(0) @binding(1) var<storage, read_write> food: array<f32>;
@group(0) @binding(2) var<storage, read> obstacles: array<u32>;
@group(0) @binding(3) var<storage, read_write> sensory: array<f32>;
@group(0) @binding(4) var<storage, read_write> actions: array<f32>;
@group(0) @binding(5) var<storage, read> nn_weights: array<f32>; // Combined: weights_l1 + biases_l1 + weights_l2 + biases_l2 per genome
@group(0) @binding(6) var<uniform> config: SimConfig;
@group(0) @binding(7) var<storage, read> biomes: array<u32>;

const OUTPUT_DIM: u32 = 6u;
const PI: f32 = 3.14159265359;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let org_idx = id.x;
    if org_idx >= config.num_organisms {
        return;
    }
    
    var org = organisms[org_idx];
    
    // Check if alive
    if (org.flags & 1u) == 0u {
        return;
    }
    
    let action_base = org_idx * OUTPUT_DIM;
    let forward = actions[action_base + 0u];  // [-1, 1]
    let rotate = actions[action_base + 1u];   // [-1, 1]
    // actions[2] = mouth (eating is automatic)
    let reproduce = actions[action_base + 3u]; // [-1, 1], store for CPU reproduction check
    
    // Store reproduce signal in organism for CPU to read back
    org.reproduce_signal = reproduce;
    
    // Apply rotation
    org.rotation += rotate * config.max_rotation;
    // Keep rotation in [0, 2π]
    if org.rotation < 0.0 {
        org.rotation += 2.0 * PI;
    }
    if org.rotation > 2.0 * PI {
        org.rotation -= 2.0 * PI;
    }
    
    // Calculate movement - apply morphology speed multiplier
    let move_dir = vec2<f32>(cos(org.rotation), sin(org.rotation));
    var effective_max_speed = config.max_speed * org.morph_speed_mult;
    
    // Apply biome speed modifier (before calculating velocity)
    if config.biomes_enabled == 1u {
        let grid_x = u32(org.position.x) % config.world_width;
        let grid_y = u32(org.position.y) % config.world_height;
        let biome_idx = grid_y * config.world_width + grid_x;
        let current_biome = biomes[biome_idx];
        
        if current_biome == BIOME_SWAMP {
            effective_max_speed *= config.swamp_speed_mult;
        }
    }
    
    let speed = forward * effective_max_speed;
    org.velocity = move_dir * speed;
    
    // Store old position for collision check
    let old_pos = org.position;
    var new_pos = org.position + org.velocity;
    
    // Wrap around world bounds
    let width = f32(config.world_width);
    let height = f32(config.world_height);
    
    new_pos.x = (new_pos.x % width + width) % width;
    new_pos.y = (new_pos.y % height + height) % height;
    
    // Check for obstacle collision
    let grid_x = u32(new_pos.x);
    let grid_y = u32(new_pos.y);
    if grid_x < config.world_width && grid_y < config.world_height {
        let grid_idx = grid_y * config.world_width + grid_x;
        if obstacles[grid_idx] != 0u {
            // Hit obstacle, stay at old position
            new_pos = old_pos;
            org.velocity = vec2<f32>(0.0, 0.0);
        }
    }
    
    org.position = new_pos;
    
    // Energy cost for movement (larger organisms cost more to move)
    let size_cost_mult = org.morph_size * org.morph_size; // Quadratic scaling with size
    let move_cost = (abs(forward) * config.movement_cost_forward + 
                    abs(rotate) * config.movement_cost_rotate) * size_cost_mult;
    org.energy -= move_cost;
    
    // Passive energy drain with crowding and age factors
    // Crowding increases drain as population approaches max_organisms
    let crowding_ratio = f32(config.num_organisms) / f32(config.max_organisms);
    let crowding_multiplier = 1.0 + config.crowding_factor * crowding_ratio * crowding_ratio;
    
    // Age factor: drain increases as organism approaches max_age
    // Uses quadratic scaling so young organisms are barely affected
    var age_multiplier = 1.0;
    if config.max_age > 0u {
        let age_ratio = f32(org.age) / f32(config.max_age);
        age_multiplier = 1.0 + config.age_drain_factor * age_ratio * age_ratio;
    }
    
    // Metabolism efficiency: higher = less drain, also affected by size (larger = more drain)
    let metabolism_factor = (1.0 / org.morph_metabolism) * org.morph_size;
    
    // Base passive drain
    var passive_drain = config.passive_drain * crowding_multiplier * age_multiplier * metabolism_factor;
    
    // Apply biome drain modifier (harsh biome = extra drain)
    if config.biomes_enabled == 1u {
        let grid_x = u32(org.position.x) % config.world_width;
        let grid_y = u32(org.position.y) % config.world_height;
        let biome_idx = grid_y * config.world_width + grid_x;
        let current_biome = biomes[biome_idx];
        
        if current_biome == BIOME_HARSH {
            passive_drain *= config.harsh_drain_mult;
        }
    }
    
    org.energy -= passive_drain;
    
    // Eating - automatic consumption when on food
    let eat_x = u32(org.position.x);
    let eat_y = u32(org.position.y);
    if eat_x < config.world_width && eat_y < config.world_height {
        let food_idx = eat_y * config.world_width + eat_x;
        let food_available = food[food_idx];
        
        // Simple eating: consume up to 0.5 food per tick
        let max_food = 0.5;
        let food_eaten = min(food_available, max_food);
        
        if food_eaten > 0.0 {
            food[food_idx] = food_available - food_eaten;
            // Apply food effectiveness multiplier
            org.energy += food_eaten * config.food_energy_value * config.food_effectiveness;
            org.energy = min(org.energy, config.max_energy);
        }
    }
    
    // ===== PREDATION SYSTEM =====
    // Attack is triggered by neural output[4] (tanh scaled to [-1, 1])
    // Positive values indicate desire to attack
    if config.predation_enabled == 1u {
        let attack_signal = actions[action_base + 4u];  // [-1, 1]
        
        // Only attack if signal is above threshold
        if attack_signal > config.attack_threshold {
            // Pay attack cost regardless of hit
            org.energy -= config.attack_cost;
            
            // Find nearest organism within attack range
            var nearest_dist = config.attack_range + 1.0;  // Start beyond range
            var nearest_idx: u32 = 0xFFFFFFFFu;  // Invalid index
            
            // Simple O(n) scan - Note: spatial hash would improve this
            for (var i = 0u; i < config.num_organisms; i++) {
                if i == org_idx {
                    continue;  // Skip self
                }
                
                let other = organisms[i];
                
                // Skip dead organisms
                if (other.flags & 1u) == 0u {
                    continue;
                }
                
                // Calculate distance with world wrap
                var dx = other.position.x - org.position.x;
                var dy = other.position.y - org.position.y;
                
                // Handle world wrapping
                if dx > f32(config.world_width) / 2.0 {
                    dx -= f32(config.world_width);
                } else if dx < -f32(config.world_width) / 2.0 {
                    dx += f32(config.world_width);
                }
                if dy > f32(config.world_height) / 2.0 {
                    dy -= f32(config.world_height);
                } else if dy < -f32(config.world_height) / 2.0 {
                    dy += f32(config.world_height);
                }
                
                let dist = sqrt(dx * dx + dy * dy);
                
                // Check if within attack range and is nearest
                if dist < config.attack_range && dist < nearest_dist {
                    nearest_dist = dist;
                    nearest_idx = i;
                }
            }
            
            // Execute attack on nearest target
            if nearest_idx != 0xFFFFFFFFu {
                var victim = organisms[nearest_idx];
                
                // Deal damage (scaled by attack intensity)
                // Cap damage to victim's actual energy to prevent over-transfer
                let raw_damage = config.attack_damage * attack_signal;
                let damage = min(raw_damage, victim.energy);
                let victim_energy_before = victim.energy;
                victim.energy -= damage;
                
                // Transfer energy from victim to attacker (capped to actual damage dealt)
                let energy_transferred = damage * config.energy_transfer;
                org.energy += energy_transferred;
                org.energy = min(org.energy, config.max_energy);
                
                // Check if victim died from attack
                if victim.energy <= 0.0 {
                    victim.flags = victim.flags & ~1u;  // Clear alive bit
                    
                    // Note: No additional kill_bonus since damage was already capped
                    // to victim's energy, so we transferred the appropriate amount
                }
                
                // Write victim back
                organisms[nearest_idx] = victim;
            }
        }
    }
    
    // Check death conditions
    // 1. Out of energy
    if org.energy <= 0.0 {
        org.flags = org.flags & ~1u;  // Clear alive bit
    }
    
    // 2. Old age (if max_age > 0)
    if config.max_age > 0u && org.age >= config.max_age {
        org.flags = org.flags & ~1u;  // Clear alive bit
    }
    
    // Increment age
    org.age += 1u;
    
    // Write back
    organisms[org_idx] = org;
}
