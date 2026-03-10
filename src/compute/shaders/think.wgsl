// Think shader - neural network forward pass for all organisms

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

const INPUT_DIM: u32 = 20u;
const HIDDEN_DIM: u32 = 16u;
const OUTPUT_DIM: u32 = 6u;

// Layout per genome in nn_weights buffer:
// - weights_l1: INPUT_DIM * HIDDEN_DIM = 320 floats
// - biases_l1: HIDDEN_DIM = 16 floats  
// - weights_l2: HIDDEN_DIM * OUTPUT_DIM = 96 floats
// - biases_l2: OUTPUT_DIM = 6 floats
// Total: 438 floats per genome
const WEIGHTS_L1_SIZE: u32 = 320u;  // INPUT_DIM * HIDDEN_DIM
const BIASES_L1_SIZE: u32 = 16u;    // HIDDEN_DIM
const WEIGHTS_L2_SIZE: u32 = 96u;   // HIDDEN_DIM * OUTPUT_DIM
const BIASES_L2_SIZE: u32 = 6u;     // OUTPUT_DIM
const GENOME_SIZE: u32 = 438u;      // Total params per genome

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

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
    
    let genome_id = org.genome_id;
    let sensory_base = org_idx * INPUT_DIM;
    let action_base = org_idx * OUTPUT_DIM;
    
    // Base offset for this genome in the combined nn_weights buffer
    let genome_base = genome_id * GENOME_SIZE;
    
    // Offsets within the genome's section of the buffer
    let w1_offset = genome_base;                                    // weights_l1 starts at 0
    let b1_offset = genome_base + WEIGHTS_L1_SIZE;                  // biases_l1 after weights_l1
    let w2_offset = genome_base + WEIGHTS_L1_SIZE + BIASES_L1_SIZE; // weights_l2 after biases_l1
    let b2_offset = genome_base + WEIGHTS_L1_SIZE + BIASES_L1_SIZE + WEIGHTS_L2_SIZE; // biases_l2 after weights_l2
    
    // Layer 1: Input -> Hidden (ReLU activation)
    var hidden: array<f32, 16>;
    for (var h = 0u; h < HIDDEN_DIM; h++) {
        var sum = nn_weights[b1_offset + h];
        for (var i = 0u; i < INPUT_DIM; i++) {
            let w_idx = w1_offset + i * HIDDEN_DIM + h;
            sum += sensory[sensory_base + i] * nn_weights[w_idx];
        }
        hidden[h] = relu(sum);
    }
    
    // Layer 2: Hidden -> Output (tanh activation)
    for (var o = 0u; o < OUTPUT_DIM; o++) {
        var sum = nn_weights[b2_offset + o];
        for (var h = 0u; h < HIDDEN_DIM; h++) {
            let w_idx = w2_offset + h * OUTPUT_DIM + o;
            sum += hidden[h] * nn_weights[w_idx];
        }
        actions[action_base + o] = tanh(sum);
    }
}
