// Food compute shader for testing
// Simplified version of world.wgsl focusing only on food growth

struct WorldConfig {
    world_width: u32,
    world_height: u32,
    tick: u32,
    food_growth_rate: f32,
    food_max_per_cell: f32,
    spontaneous_enabled: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> food: array<f32>;
@group(0) @binding(1) var<uniform> config: WorldConfig;

// PCG-style hash function for better randomness
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_to_float(h: u32) -> f32 {
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    
    if x >= config.world_width || y >= config.world_height {
        return;
    }
    
    let idx = y * config.world_width + x;
    var current_food = food[idx];
    
    // Logistic growth: dF/dt = r * F * (1 - F/K)
    if current_food > 0.1 {
        let growth = config.food_growth_rate * current_food * (1.0 - current_food / config.food_max_per_cell);
        current_food += growth;
    }
    
    // Spontaneous food generation (can be toggled)
    if config.spontaneous_enabled == 1u && current_food < 0.001 {
        let seed = pcg_hash(x + pcg_hash(y + pcg_hash(config.tick)));
        let rand_val = hash_to_float(seed);
        
        // Very rare spontaneous generation
        if rand_val < 0.00001 {
            current_food = 0.3;
        }
    }
    
    current_food = clamp(current_food, 0.0, config.food_max_per_cell);
    food[idx] = current_food;
}
