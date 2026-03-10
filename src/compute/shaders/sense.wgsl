// Sense shader - compute vision and internal state for all organisms

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

@group(0) @binding(0) var<storage, read_write> organisms: array<Organism>;
@group(0) @binding(1) var<storage, read_write> food: array<f32>;
@group(0) @binding(2) var<storage, read> obstacles: array<u32>;
@group(0) @binding(3) var<storage, read_write> sensory: array<f32>;
@group(0) @binding(4) var<storage, read_write> actions: array<f32>;
@group(0) @binding(5) var<storage, read> nn_weights: array<f32>; // Combined: weights_l1 + biases_l1 + weights_l2 + biases_l2 per genome
@group(0) @binding(6) var<uniform> config: SimConfig;
@group(0) @binding(7) var<storage, read> biomes: array<u32>;

// Network architecture constants - MUST match genome.rs
const NUM_RAYS: u32 = 8u;    // 8 vision rays
const INPUT_DIM: u32 = 20u;  // 8 rays * 2 (dist, type) + 4 internal state
const PI: f32 = 3.14159265359;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let org_idx = id.x;
    if org_idx >= config.num_organisms {
        return;
    }
    
    let org = organisms[org_idx];
    
    // Check if alive
    if (org.flags & 1u) == 0u {
        return;
    }
    
    let sensory_base = org_idx * INPUT_DIM;
    let half_fov = config.vision_fov / 2.0;
    
    // Apply morphology to vision range
    let effective_vision_range = config.vision_range * org.morph_vision_mult;
    
    // Cast vision rays
    for (var r = 0u; r < NUM_RAYS; r++) {
        let t = f32(r) / f32(NUM_RAYS - 1u);
        let angle_offset = -half_fov + t * config.vision_fov;
        let ray_angle = org.rotation + angle_offset;
        let ray_dir = vec2<f32>(cos(ray_angle), sin(ray_angle));
        
        var hit_dist = 0.0;  // 0 = nothing hit (far away)
        var hit_type = 0.0;  // 0 = empty
        
        // Ray march - use effective vision range
        for (var d = 1.0; d < effective_vision_range; d += 1.0) {
            let sample_pos = org.position + ray_dir * d;
            let width = f32(config.world_width);
            let height = f32(config.world_height);
            let wrap_x = (sample_pos.x % width + width) % width;
            let wrap_y = (sample_pos.y % height + height) % height;
            
            let grid_x = u32(wrap_x);
            let grid_y = u32(wrap_y);
            
            let grid_idx = grid_y * config.world_width + grid_x;
            
            // Check obstacle
            if obstacles[grid_idx] != 0u {
                hit_dist = 1.0 - (d / effective_vision_range);
                hit_type = 0.25;  // Obstacle
                break;
            }
            
            // Check food - lower threshold so organisms can see food patches
            // Food values range from 0 to max_per_cell (e.g., 8.0)
            // Threshold of 0.5 allows seeing most food
            if food[grid_idx] > 0.5 {
                hit_dist = 1.0 - (d / effective_vision_range);
                hit_type = 0.5;  // Food
                break;
            }
        }
        
        sensory[sensory_base + r * 2u] = hit_dist;
        sensory[sensory_base + r * 2u + 1u] = hit_type;
    }
    
    // Internal state (indices 16-18)
    sensory[sensory_base + 16u] = org.energy / config.max_energy;
    sensory[sensory_base + 17u] = min(f32(org.age) / 1000.0, 1.0);
    sensory[sensory_base + 18u] = length(org.velocity) / config.max_speed;
    
    // Organism proximity detection - separate from raycasting for efficiency
    // Instead of O(n²) per-ray checking, do a single proximity scan
    // Check a subset of organisms (stride through the array) to detect nearby organisms
    // This gives approximate awareness without the full O(n²) cost
    var nearest_organism_dist = config.vision_range;
    var nearest_organism_angle = 0.0;
    let check_stride = max(config.num_organisms / 64u, 1u);  // Check ~64 organisms max
    
    for (var i = 0u; i < config.num_organisms; i += check_stride) {
        if i == org_idx {
            continue;
        }
        let other = organisms[i];
        if (other.flags & 1u) == 0u {
            continue;
        }
        
        var delta = other.position - org.position;
        let w = f32(config.world_width);
        let h = f32(config.world_height);
        
        // Wrap delta for shortest path on torus
        if delta.x > w * 0.5 { delta.x -= w; }
        if delta.x < -w * 0.5 { delta.x += w; }
        if delta.y > h * 0.5 { delta.y -= h; }
        if delta.y < -h * 0.5 { delta.y += h; }
        
        let dist = length(delta);
        
        if dist < config.vision_range && dist < nearest_organism_dist {
            nearest_organism_dist = dist;
            nearest_organism_angle = atan2(delta.y, delta.x) - org.rotation;
        }
    }
    
    // Encode nearest organism info in the last sensory slot (bias becomes organism awareness)
    // If organism detected: value encodes direction (-1 to 1 based on angle)
    // If no organism: 1.0 (original bias value)
    if nearest_organism_dist < config.vision_range {
        // Normalize angle to [-1, 1] where 0 = directly ahead
        var angle_norm = nearest_organism_angle / PI;
        if angle_norm > 1.0 { angle_norm -= 2.0; }
        if angle_norm < -1.0 { angle_norm += 2.0; }
        sensory[sensory_base + 19u] = angle_norm;  // Replace bias with organism direction
    } else {
        sensory[sensory_base + 19u] = 1.0;  // Bias (no organism detected)
    }
}
