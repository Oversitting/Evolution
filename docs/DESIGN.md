# Evolution Simulator: Technical Design Specification

**Version**: 1.5 (Phase 6 Partial)  
**Created**: January 26, 2026  
**Last Updated**: February 2026  
**Status**: Morphology, Sexual Reproduction, Biomes Implemented

---

## 1. System Overview

### 1.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAME LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│  Input (CPU)     │  winit events → App state                   │
│  Readback (CPU)  │  Read prev frame GPU state (Energy/Pos)     │
│  Simulation (CPU)│  Reproduction logic & Stats tracking        │
│  Sync (CPU→GPU)  │  Upload parent energy & new organisms       │
│  Compute (GPU)   │  Compute shaders: sense → think → act       │
│  Rendering (GPU) │  Render pipeline: organisms + world + UI    │
│  UI (CPU+GPU)    │  egui immediate mode                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Rust | 1.75+ |
| GPU API | wgpu | 0.19 |
| Windowing | winit | 0.29 |
| UI | egui + egui-wgpu | 0.27 |
| Math | glam | 0.27 |
| RNG | rand + rand_xoshiro | 0.8 |
| Serialization | serde + bincode | 1.0 |
| Async | pollster | 0.3 |

### 1.3 Target Hardware

| Tier | GPU Example | VRAM | Organisms | FPS |
|------|-------------|------|-----------|-----|
| Minimum | GTX 1050 Ti | 4 GB | 2,000 | 30 |
| Recommended | GTX 1060 | 6 GB | 2,000 | 60 |
| High | RTX 3060 | 8 GB | 10,000 | 60 |

---

## 2. Data Structures

### 2.1 Organism State

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Organism {
    // Position & Movement (16 bytes)
    pub position: [f32; 2],      // World coordinates
    pub velocity: [f32; 2],      // Current velocity
    
    // Rotation (4 bytes)
    pub rotation: f32,           // Radians, 0 = right, π/2 = up
    
    // State (12 bytes)
    pub energy: f32,             // 0-150 (max configurable)
    pub age: u32,                // Ticks since birth
    pub flags: u32,              // Bit 0: alive
    
    // Genome reference (8 bytes)
    pub genome_id: u32,          // INVARIANT: genome_id == organism slot index
    pub generation: u32,         // Generation number
    
    // Stats (8 bytes)
    pub offspring_count: u32,    // Number of children produced
    pub parent_id: u32,          // Parent organism ID (or u32::MAX if none)
    
    // Neural network output (8 bytes)
    pub reproduce_signal: f32,   // Reproduction willingness from neural network
    pub _pad: u32,               // Padding for 8-byte alignment
}
// Total: 56 bytes per organism
// KEY INVARIANT: genome_id always equals the organism's slot index.
// This bounds genome IDs to max_organisms, preventing buffer overflow.
```

### 2.2 Genome (Neural Network Weights)

```rust
// Network architecture: 20 inputs → 16 hidden → 6 outputs
pub const INPUT_DIM: usize = 20;
pub const HIDDEN_DIM: usize = 16;
pub const OUTPUT_DIM: usize = 6;

// Weights stored in separate buffers for GPU efficiency
pub struct GenomeBuffers {
    // Layer 1: input → hidden
    pub weights_1: Buffer,  // [N_organisms, INPUT_DIM, HIDDEN_DIM] = N × 320 floats
    pub biases_1: Buffer,   // [N_organisms, HIDDEN_DIM] = N × 16 floats
    
    // Layer 2: hidden → output
    pub weights_2: Buffer,  // [N_organisms, HIDDEN_DIM, OUTPUT_DIM] = N × 96 floats
    pub biases_2: Buffer,   // [N_organisms, OUTPUT_DIM] = N × 6 floats
}
// Total: 438 floats × 4 bytes = 1,752 bytes per organism
```

### 2.3 Sensory Input Layout

```rust
// Total: 20 inputs
pub struct SensoryInput {
    // Vision: 8 rays × 2 values = 16 floats
    // Per ray: [distance (0-1), type (0=empty, 0.5=food, 1=organism)]
    pub vision: [f32; 16],   // Indices 0-15
    
    // Internal state: 4 floats
    pub energy: f32,         // Index 16: normalized (current/max)
    pub age: f32,            // Index 17: normalized (current/1000)
    pub speed: f32,          // Index 18: normalized (velocity magnitude/max)
    pub nearest_dir: f32,    // Index 19: nearest-organism direction (-1..1) or 1.0 if none
}
```

### 2.4 Action Output Layout

```rust
// Total: 6 outputs, all in range [-1, 1] via tanh
pub struct ActionOutput {
    pub forward: f32,        // Index 0: forward/backward speed
    pub rotate: f32,         // Index 1: angular velocity
    pub mouth: f32,          // Index 2: unused (eating is automatic)
    pub reproduce: f32,      // Index 3: reproduction willingness (stored for CPU)
    pub reserved_0: f32,     // Index 4: unused
    pub reserved_1: f32,     // Index 5: unused
}
```

### 2.5 World State

```rust
pub struct World {
    pub width: u32,              // 2048 default
    pub height: u32,             // 2048 default
    pub food_grid: Buffer,       // [H, W] f32 - food amount per cell
    pub obstacle_grid: Buffer,   // [H, W] u32 - 0=passable, 1=blocked
}

Note: Obstacles are currently initialized to zero and are not populated elsewhere, so vision and movement are unblocked in the present build.

pub struct WorldConfig {
    pub food_growth_rate: f32,   // 0.05 per tick
    pub food_max_per_cell: f32,  // 10.0 units
    pub food_energy_value: f32,  // 4.0 energy per food unit
}
```

### 2.6 Simulation Parameters

```rust
pub struct SimulationConfig {
    // Population
    pub max_organisms: u32,          // 4000
    pub initial_organisms: u32,      // 600

    // Energy
    pub starting_energy: f32,        // 70.0
    pub max_energy: f32,             // 200.0
    pub passive_drain: f32,          // 0.14 per tick
    pub movement_cost_forward: f32,  // 0.02 per unit speed
    pub movement_cost_rotate: f32,   // 0.01 per unit rotation
    pub max_age: u32,                // 2000 (ticks before death from old age)
    pub age_drain_factor: f32,       // 1.0 (extra drain at max_age, quadratic)
    pub crowding_factor: f32,        // 1.0 (quadratic scaling with population ratio)

    // Reproduction
    pub reproduction_threshold: f32, // 70.0 energy
    pub reproduction_signal_min: f32,// 0.3
    pub reproduction_cooldown: u32,  // 120 ticks
    pub reproduction_min_age: u32,   // 150 ticks
    pub reproduction_cost: f32,      // 50.0 energy (= offspring start)

    // Mutation
    pub mutation_rate: f32,          // 0.05 (5% per weight)
    pub mutation_strength: f32,      // 0.2 (std dev)

    // Vision
    pub vision_rays: u32,            // 8
    pub vision_fov: f32,             // 90.0 degrees (π/2 radians)
    pub vision_range: f32,           // 50.0 pixels

    // Physics
    pub max_speed: f32,              // 2.5 pixels per tick
    pub max_rotation: f32,           // 0.25 radians per tick
    pub organism_radius: f32,        // 3.0 pixels

    // Food
    pub food_growth_rate: f32,       // 0.05 per tick
    pub food_max_per_cell: f32,      // 10.0 units
    pub food_energy_value: f32,      // 4.0 energy per food unit
    pub food_effectiveness: f32,     // 1.0 (multiplier, lower = harder environment)
    pub initial_patches: u32,        // 200 patches at start
    pub patch_size: u32,             // 10 cells radius per patch
    
    // Dynamic Environment (New in 1.4)
    pub seasonal_enabled: bool,      // false (toggle seasonal cycles)
    pub seasonal_period: u32,        // 6000 ticks per full cycle (~100 seconds)
    pub seasonal_amplitude: f32,     // 0.7 (growth varies from 30% to 100%)
    pub hotspots_enabled: bool,      // false (toggle resource hotspots)
    pub hotspot_count: u32,          // 3 hotspots
    pub hotspot_radius: f32,         // 100.0 cells influence radius
    pub hotspot_intensity: f32,      // 0.3 extra growth at center

    // System (New in 1.1)
    pub system_readback_interval: u32,      // 1 (Critical for sync)
    pub system_food_readback_interval: u32, // 60
    pub system_diagnostic_interval: u32,    // 60
}
```

### 2.7 Age-Based Energy Drain

Organisms experience increased energy drain as they age:

$$E_{drain} = E_{base} \times \left(1 + \text{age\_drain\_factor} \times \left(\frac{age}{max\_age}\right)^2\right)$$

This quadratic scaling means:
- Young organisms: Normal energy drain
- Middle-aged: Slight increase
- Near max_age: Significant drain (up to 1 + age_drain_factor at max_age)

### 2.8 Crowding Penalty

Crowding scales drain quadratically with the current population ratio (applies across the whole range):

$$E_{drain} = E_{drain} \times \left(1 + \text{crowding\_factor} \times r^2\right), \quad r = \frac{\text{num\_organisms}}{\text{max\_organisms}}$$

This increases pressure as the population rises and doubles drain when `r = 1` with `crowding_factor = 1.0`.

### 2.9 Dynamic Environment Systems

**Seasonal Cycles**: Food growth rate varies sinusoidally over time:

$$\text{growth\_mult} = 1.0 - \text{amplitude} \times \frac{1 + \cos\left(\frac{2\pi \times \text{tick}}{\text{period}}\right)}{2}$$

This creates feast/famine cycles that reward organisms capable of energy storage.

**Resource Hotspots**: High-value food zones that drift slowly across the world. Each hotspot:
- Has a center that moves in a circular orbit
- Applies bonus food growth that decays with distance
- Uses perlin-like movement based on tick time

$$\text{hotspot\_bonus} = \text{intensity} \times \max\left(0, 1 - \frac{\text{distance}}{\text{radius}}\right)$$

Hotspots encourage organism migration and exploration behaviors.

---

## 3. GPU Compute Pipeline

### 3.1 Pipeline Overview

```
Per Tick (16.67ms budget):
┌────────────────────────────────────────────────────────────────┐
│  1. Sense Pass      │ Vision raycasting for all organisms     │
│  2. Think Pass      │ Neural network forward pass             │
│  3. Act Pass        │ Apply actions (movement, eating)        │
│  4. World Pass      │ Food growth and spawning                │
└────────────────────────────────────────────────────────────────┘

Reproduction runs on the CPU. The flow is:
1. **Readback**: CPU reads GPU organism state from the *previous* frame.
2. **Logic**: CPU checks energy threshold, deducts reproduction cost, and spawns offspring.
3. **Sync**: CPU uploads updated parent (lower energy) and new child to GPU.
4. **Dispatch**: GPU runs the *next* frame's physics/NN with the corrected state.

*Note: readback_interval is set to 1 to ensure energy costs are permanently deducted before the next simulation step, preventing state desynchronization.*
```

### 3.2 Compute Shader: Sense (sense.wgsl)

```wgsl
// Input bindings
@group(0) @binding(0) var<storage, read> organisms: array<Organism>;
@group(0) @binding(1) var<storage, read> food_grid: array<f32>;
@group(0) @binding(2) var<storage, read> obstacle_grid: array<u32>;
@group(0) @binding(3) var<uniform> config: SimConfig;

// Output binding
@group(0) @binding(4) var<storage, read_write> sensory: array<f32>;

const NUM_RAYS: u32 = 8u;
const RAY_VALUES: u32 = 2u;  // distance, type

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let org_idx = id.x;
    if org_idx >= config.num_organisms { return; }
    
    let org = organisms[org_idx];
    if (org.flags & 1u) == 0u { return; }  // Dead
    
    let sensory_base = org_idx * 20u;  // 20 inputs per organism
    
    // Cast rays
    let fov = config.vision_fov;
    let half_fov = fov / 2.0;
    
    for (var r = 0u; r < NUM_RAYS; r++) {
        let angle_offset = -half_fov + (f32(r) / f32(NUM_RAYS - 1u)) * fov;
        let ray_angle = org.rotation + angle_offset;
        let ray_dir = vec2<f32>(cos(ray_angle), sin(ray_angle));
        
        var hit_dist = 1.0;  // Normalized, 1.0 = nothing hit
        var hit_type = 0.0;  // 0 = empty
        
        // Ray march
        for (var d = 1.0; d < config.vision_range; d += 1.0) {
            let sample_pos = org.position + ray_dir * d;
            let grid_x = u32(sample_pos.x);
            let grid_y = u32(sample_pos.y);
            
            // Bounds check
            if grid_x >= config.world_width || grid_y >= config.world_height {
                hit_dist = 1.0 - (d / config.vision_range);
                break;
            }
            
            let grid_idx = grid_y * config.world_width + grid_x;
            
            // Check obstacle
            if obstacle_grid[grid_idx] != 0u {
                hit_dist = 1.0 - (d / config.vision_range);
                hit_type = 0.25;  // Obstacle
                break;
            }
            
            // Check food
            if food_grid[grid_idx] > 0.5 {
                hit_dist = 1.0 - (d / config.vision_range);
                hit_type = 0.5;  // Food
                break;
            }
            
            // Strided nearest-organism scan runs after raymarch (separate loop)
        }
        
        sensory[sensory_base + r * 2u] = hit_dist;
        sensory[sensory_base + r * 2u + 1u] = hit_type;
    }
    
    // Internal state
    sensory[sensory_base + 16u] = org.energy / config.max_energy;
    sensory[sensory_base + 17u] = f32(org.age) / 1000.0;
    sensory[sensory_base + 18u] = length(org.velocity) / config.max_speed;

    // Last slot encodes nearest-organism direction (-1..1 relative to heading).
    // A strided scan (~64 samples) finds the closest alive organism within vision_range;
    // default is 1.0 when none is found.
    var nearest_dir = 1.0;
    // ... update nearest_dir during the scan ...
    sensory[sensory_base + 19u] = nearest_dir;
}
```

In practice, the implementation replaces the former bias input with the nearest-organism direction computed via a strided scan over the organism buffer.

### 3.3 Compute Shader: Think (think.wgsl)

```wgsl
// Genome buffers
@group(0) @binding(0) var<storage, read> weights_1: array<f32>;  // [N, 20, 16]
@group(0) @binding(1) var<storage, read> biases_1: array<f32>;   // [N, 16]
@group(0) @binding(2) var<storage, read> weights_2: array<f32>;  // [N, 16, 6]
@group(0) @binding(3) var<storage, read> biases_2: array<f32>;   // [N, 6]

// IO
@group(0) @binding(4) var<storage, read> sensory: array<f32>;    // [N, 20]
@group(0) @binding(5) var<storage, read_write> actions: array<f32>; // [N, 6]
@group(0) @binding(6) var<storage, read> organisms: array<Organism>;
@group(0) @binding(7) var<uniform> config: SimConfig;

const INPUT_DIM: u32 = 20u;
const HIDDEN_DIM: u32 = 16u;
const OUTPUT_DIM: u32 = 6u;

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let org_idx = id.x;
    if org_idx >= config.num_organisms { return; }
    
    let org = organisms[org_idx];
    if (org.flags & 1u) == 0u { return; }  // Dead
    
    let sensory_base = org_idx * INPUT_DIM;
    let w1_base = org_idx * INPUT_DIM * HIDDEN_DIM;
    let b1_base = org_idx * HIDDEN_DIM;
    let w2_base = org_idx * HIDDEN_DIM * OUTPUT_DIM;
    let b2_base = org_idx * OUTPUT_DIM;
    let action_base = org_idx * OUTPUT_DIM;
    
    // Layer 1: Input → Hidden (ReLU)
    var hidden: array<f32, 16>;
    for (var h = 0u; h < HIDDEN_DIM; h++) {
        var sum = biases_1[b1_base + h];
        for (var i = 0u; i < INPUT_DIM; i++) {
            let w_idx = w1_base + i * HIDDEN_DIM + h;
            sum += sensory[sensory_base + i] * weights_1[w_idx];
        }
        hidden[h] = relu(sum);
    }
    
    // Layer 2: Hidden → Output (tanh)
    for (var o = 0u; o < OUTPUT_DIM; o++) {
        var sum = biases_2[b2_base + o];
        for (var h = 0u; h < HIDDEN_DIM; h++) {
            let w_idx = w2_base + h * OUTPUT_DIM + o;
            sum += hidden[h] * weights_2[w_idx];
        }
        actions[action_base + o] = tanh(sum);
    }
}
```

### 3.4 Compute Shader: Act (act.wgsl)

```wgsl
// Outputs: 0=forward, 1=rotate, 3=reproduce (others unused)
let forward = actions[action_base + 0u];
let rotate = actions[action_base + 1u];
let reproduce = actions[action_base + 3u];

// Store reproduce signal for CPU-side reproduction checks
org.reproduce_signal = reproduce;

// Integrate motion and wrap on a torus
org.rotation = wrap_angle(org.rotation + rotate * config.max_rotation);
org.velocity = dir_from_rotation(org.rotation) * forward * config.max_speed;
org.position = wrap_position(org.position + org.velocity, config.world_width, config.world_height);

// Movement + passive drain with crowding (quadratic on population ratio) and age multiplier
let move_cost = abs(forward) * config.movement_cost_forward + abs(rotate) * config.movement_cost_rotate;
let crowd = f32(config.num_organisms) / f32(config.max_organisms);
let crowd_mult = 1.0 + config.crowding_factor * crowd * crowd;
let age_mult = if config.max_age > 0u {
    let r = f32(org.age) / f32(config.max_age);
    1.0 + config.age_drain_factor * r * r
} else { 1.0 };
org.energy -= move_cost + config.passive_drain * crowd_mult * age_mult;

// Eating is automatic: consume up to 0.5 food at the current cell
let food_idx = u32(org.position.y) * config.world_width + u32(org.position.x);
let food_available = food_grid[food_idx];
let food_eaten = min(food_available, 0.5);
food_grid[food_idx] = food_available - food_eaten;
org.energy = min(config.max_energy, org.energy + food_eaten * config.food_energy_value * config.food_effectiveness);

// Death on zero energy or exceeding max_age
if org.energy <= 0.0 || (config.max_age > 0u && org.age >= config.max_age) {
    org.flags &= ~1u;
}

org.age += 1u;
```

### 3.5 Compute Shader: World Update (world.wgsl)

```wgsl
@group(0) @binding(0) var<storage, read_write> food_grid: array<f32>;
@group(0) @binding(1) var<uniform> config: SimConfig;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    
    if x >= config.world_width || y >= config.world_height { return; }
    
    let idx = y * config.world_width + x;
    var food = food_grid[idx];
    
    // Logistic growth: dF/dt = r * F * (1 - F/K)
    let growth = config.food_growth_rate * food * (1.0 - food / config.food_max_per_cell);
    food += growth;
    food = clamp(food, 0.0, config.food_max_per_cell);
    
    food_grid[idx] = food;
}
```

- Food regrows only in cells that already have food (`> 0.01`), using the logistic term shown above.
- A rare spawn (`food_spawn_chance`, default 1e-6 per cell per tick) can seed `food_spawn_amount` into empty cells to start new patches.
- `food_baseline` enforces a floor across the map (default 0.0, so patches only).

---

## 4. Rendering Pipeline

### 4.1 Organism Rendering

Organisms are rendered as instanced triangles:

```wgsl
// organism.wgsl - Vertex shader

struct VertexInput {
    @location(0) local_pos: vec2<f32>,  // Triangle vertex
}

struct InstanceInput {
    @location(1) position: vec2<f32>,
    @location(2) rotation: f32,
    @location(3) energy: f32,
    @location(4) flags: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Skip dead organisms
    if (instance.flags & 1u) == 0u {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);  // Behind camera
        return out;
    }
    
    // Rotate local position
    let cos_r = cos(instance.rotation);
    let sin_r = sin(instance.rotation);
    let rotated = vec2<f32>(
        vertex.local_pos.x * cos_r - vertex.local_pos.y * sin_r,
        vertex.local_pos.x * sin_r + vertex.local_pos.y * cos_r
    );
    
    // Transform to world space
    let world_pos = instance.position + rotated * 6.0;  // 6 pixel radius
    
    // Apply camera transform
    let view_pos = (world_pos - camera.position) * camera.zoom;
    let ndc = view_pos / camera.viewport_size * 2.0;
    
    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    
    // Color supplied per instance (hashed from genome weights; cyan if selected)
    out.color = instance.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

### 4.2 World Rendering

Food layer rendered as a textured quad with color mapping:

```wgsl
// world.wgsl - Fragment shader

@group(0) @binding(0) var food_texture: texture_2d<f32>;
@group(0) @binding(1) var food_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let food = textureSample(food_texture, food_sampler, uv).r;
    
    // Green gradient based on food amount
    let intensity = food / 10.0;  // Normalize by max food
    let color = vec3<f32>(0.1, 0.2 + intensity * 0.6, 0.1);
    
    return vec4<f32>(color, 1.0);
}
```

---

## 5. Reproduction System

Reproduction is handled on the CPU after GPU→CPU readbacks (every tick by default, configurable via `readback_interval`). The GPU only supplies `reproduce_signal`; the CPU evaluates thresholds, mutates genomes, and syncs parent/child organisms and weights back to the GPU.

### 5.1 Reproduction Check (CPU)

```rust
// Pseudocode for reproduction logic
fn check_reproduction(org: &Organism, config: &SimConfig) -> bool {
    let alive = (org.flags & 1) != 0;
    let enough_energy = org.energy >= config.reproduction_threshold;
    let old_enough = org.age >= config.reproduction_min_age;
    let not_cooling_down = /* check cooldown timer */;
    let wants_to = org.reproduce_signal > config.reproduction_signal_min;
    
    alive && enough_energy && old_enough && not_cooling_down && wants_to
}

fn reproduce(parent: &Organism, organisms: &mut OrganismPool, genomes: &mut GenomePool, config: &SimConfig, rng: &mut Rng) -> Option<u32> {
    // First spawn organism to get its slot index
    let child_idx = organisms.spawn(
        parent.position + random_offset(5.0),
        config.reproduction_cost,  // Child starts with this energy
        0,  // placeholder genome_id
        parent.generation + 1,
    )?;
    
    // Create mutated genome at child's slot index (genome_id = slot index invariant)
    genomes.clone_and_mutate_at(
        child_idx,        // Target slot = child's organism slot
        parent.genome_id, // Source genome (parent's slot)
        config.mutation_rate,
        config.mutation_strength,
        rng,
    );
    
    // Update organism's genome_id to match its slot
    organisms.get_mut(child_idx)?.genome_id = child_idx;
    
    // Deduct parent energy and set cooldown
    let parent_org = organisms.get_mut(parent.id)?;
    parent_org.energy -= config.reproduction_cost;
    parent_org.cooldown = config.reproduction_cooldown;
    parent_org.offspring_count += 1;
    
    Some(child_idx)
}
```

### 5.2 Sexual Reproduction (Phase 6)

When sexual reproduction is enabled, organisms can mate with nearby compatible organisms instead of cloning themselves.

**Configuration (`ReproductionConfig`):**
```toml
sexual_enabled = true      # Enable sexual reproduction
mate_range = 50.0          # Max distance to find mate
mate_signal_min = 0.3      # Both organisms must have reproduce_signal above this
crossover_ratio = 0.5      # Probability of taking gene from parent 2
```

**Mate Finding Algorithm:**
1. Organism signals reproduction desire (reproduce_signal > threshold)
2. CPU scans nearby organisms within `mate_range`
3. Potential mate must also have `reproduce_signal > mate_signal_min`
4. If mate found: uniform crossover of genomes + mutation
5. If no mate found: fall back to asexual cloning

**Crossover:**
- Each neural network weight randomly chosen from parent1 or parent2
- Morphology traits averaged then mutated
- Child spawns near parent1 (the initiator)

---

## 5.3 Morphology System (Phase 6)

Organisms have evolvable physical traits that affect their performance.

**Morphology Traits (`MorphTraits` struct):**
```rust
pub struct MorphTraits {
    pub size: f32,          // Affects rendering size and movement cost
    pub speed_mult: f32,    // Multiplier for max_speed
    pub vision_mult: f32,   // Multiplier for vision_range  
    pub metabolism: f32,    // Efficiency (higher = less drain)
}
```

**Configuration (`MorphologyConfig`):**
```toml
enabled = true
min_size = 0.5
max_size = 2.0
min_speed_mult = 0.5
max_speed_mult = 1.5
min_vision_mult = 0.5
max_vision_mult = 1.5
min_metabolism = 0.5
max_metabolism = 1.5
mutation_rate = 0.3
mutation_strength = 0.1
```

**Physics Effects (in act.wgsl):**
- `effective_max_speed = config.max_speed * morph_speed_mult`
- Movement cost scales with `morph_size²`
- Passive drain scales with `morph_size / morph_metabolism`

**Vision Effects (in sense.wgsl):**
- `effective_vision_range = config.vision_range * morph_vision_mult`

**Rendering Effects (in organism.wgsl):**
- Triangle size scales with `morph_size`

---

## 5.4 Biomes System (Phase 6)

The world is divided into regions with different environmental effects.

**Biome Types:**
| Type | Food Growth | Speed | Energy Drain |
|------|-------------|-------|--------------|
| Normal | 1.0x | 1.0x | 1.0x |
| Fertile | 2.0x | 1.0x | 1.0x |
| Barren | 0.25x | 1.0x | 1.0x |
| Swamp | 1.0x | 0.6x | 1.0x |
| Harsh | 1.0x | 1.0x | 1.5x |

**Configuration (`BiomesConfig`):**
```toml
enabled = true
biome_count = 16              # Number of Voronoi cells
fertile_growth_mult = 2.0
barren_growth_mult = 0.25
swamp_speed_mult = 0.6
harsh_drain_mult = 1.5
```

**Generation Algorithm:**
1. Generate `biome_count` random Voronoi cell centers
2. Assign each center a biome type (weighted: 30% Normal, 20% Fertile, 20% Barren, 15% Swamp, 15% Harsh)
3. Each cell belongs to nearest center (with world wrapping)
4. Stored as `Vec<u8>` in `World` struct, uploaded to GPU as `array<u32>`

**Shader Usage:**
- `world.wgsl`: Applies growth multiplier based on cell biome
- `act.wgsl`: Applies speed multiplier (Swamp) and drain multiplier (Harsh)

---

## 6. Frame Loop

### 6.1 Main Loop Structure

```rust
fn run_frame(&mut self) {
    // 1. Handle input
    self.process_input();
    
    // 2. Simulation step (may run multiple times for speed-up)
    for _ in 0..self.speed_multiplier {
        self.simulation_step();
    }
    
    // 3. Render
    self.render();
    
    // 4. Present
    self.present();
}

fn simulation_step(&mut self) {
    let mut encoder = self.device.create_command_encoder(&Default::default());
    
    // Compute passes
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        
        // Sense pass
        pass.set_pipeline(&self.sense_pipeline);
        pass.set_bind_group(0, &self.sense_bind_group, &[]);
        pass.dispatch_workgroups((self.organism_count + 63) / 64, 1, 1);
        
        // Think pass
        pass.set_pipeline(&self.think_pipeline);
        pass.set_bind_group(0, &self.think_bind_group, &[]);
        pass.dispatch_workgroups((self.organism_count + 63) / 64, 1, 1);
        
        // Act pass
        pass.set_pipeline(&self.act_pipeline);
        pass.set_bind_group(0, &self.act_bind_group, &[]);
        pass.dispatch_workgroups((self.organism_count + 63) / 64, 1, 1);
        
        // World pass
        pass.set_pipeline(&self.world_pipeline);
        pass.set_bind_group(0, &self.world_bind_group, &[]);
        let world_groups = (self.world_size + 7) / 8;
        pass.dispatch_workgroups(world_groups, world_groups, 1);
    }
    
    self.queue.submit(std::iter::once(encoder.finish()));
    
    // CPU-side: reproduction (read back minimal data)
    self.handle_reproduction();
    
    self.tick += 1;
}
```

### 6.2 GPU↔CPU Data Synchronization

The simulation uses GPU as the **source of truth** for organism state. Data flow:

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU (source of truth)                                          │
│  - Organisms: position, velocity, energy, age, reproduce_signal │
│  - Neural network weights (per genome)                          │
│  - Food grid, world state                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                     ┌───────┴───────┐
                     ▼               ▼
              Every tick       On Reproduction
              (async readback)  (sync changes)
                     │               │
                     ▼               ▼
┌────────────────────────────────────────────────────────────────┐
│  CPU (reproduction logic + statistics)                          │
│  - Checks reproduction conditions                               │
│  - Spawns new organisms with mutated genomes                    │
│  - Updates parent energy (cost deduction)                       │
└────────────────────────────────────────────────────────────────┘
```

**Key invariants:**
1. CPU never overwrites GPU organism state except for specific changes:
   - New organisms spawned during reproduction
   - Parent energy deducted after reproduction
2. Genome weights are synced to GPU immediately after mutation
3. Readback happens every tick by default (configurable via `system.readback_interval`)
4. GPU buffers are pre-allocated to `max_organisms` capacity

**Reproduction sync flow:**
```rust
// 1. GPU executes sense→think→act (updates reproduce_signal)
// 2. Async readback copies organism buffer to staging buffer
// 3. CPU reads back organism state (on next frame)
// 4. CPU checks reproduction conditions
// 5. For each reproducing organism:
//    a. Deduct parent energy (CPU)
//    b. Clone and mutate genome (CPU)
//    c. Sync new genome weights to GPU
//    d. Create child organism (CPU)
//    e. Sync parent and child organisms to GPU
// 6. GPU continues with updated data
```

---

## 7. Memory Layout

### 7.1 GPU Buffer Allocation (4,000 organisms, 2048² world)

| Buffer | Size | Total |
|--------|------|-------|
| Organisms | 4,000 × 56 bytes | ~224 KB |
| Sensory | 4,000 × 20 × 4 bytes | ~320 KB |
| Actions | 4,000 × 6 × 4 bytes | ~96 KB |
| Weights L1 | 4,000 × 320 × 4 bytes | ~4.9 MB |
| Biases L1 | 4,000 × 16 × 4 bytes | ~256 KB |
| Weights L2 | 4,000 × 96 × 4 bytes | ~1.5 MB |
| Biases L2 | 4,000 × 6 × 4 bytes | ~96 KB |
| Food Grid | 2048 × 2048 × 4 bytes | ~16.0 MB |
| Obstacle Grid | 2048 × 2048 × 4 bytes | ~16.0 MB |
| **Total** | | **~39 MB** |

---

## 8. Configuration

### 8.1 Runtime Configuration File

Configuration is loaded from `config.toml` at startup. Edit this file to change simulation parameters **without recompiling**.

```toml
# config.toml - Runtime simulation configuration

# Optional fixed seed for reproducibility
# seed = 42

[population]
max_organisms = 4000
initial_organisms = 600

[energy]
starting = 70.0
maximum = 200.0
passive_drain = 0.14        # Energy lost per tick passively
movement_cost_forward = 0.02# Extra cost for forward movement
movement_cost_rotate = 0.01 # Extra cost for rotation
max_age = 2000
age_drain_factor = 1.0
crowding_factor = 1.0

[reproduction]
threshold = 70.0    # Minimum energy to reproduce
signal_min = 0.3    # Minimum neural network signal to trigger reproduction
cooldown = 120      # Ticks between reproduction attempts
min_age = 150       # Minimum age to reproduce
cost = 50.0         # Energy cost (also child's starting energy)

[mutation]
rate = 0.05     # Probability of mutating each weight
strength = 0.2  # Standard deviation of mutation noise

[vision]
rays = 8              # Number of vision rays per organism
fov_degrees = 90.0    # Field of view in degrees
range = 50.0          # Maximum vision distance

[physics]
max_speed = 2.5       # Maximum organism speed
max_rotation = 0.25   # Maximum rotation per tick (radians)
organism_radius = 3.0 # Collision/visual radius

[food]
growth_rate = 0.05    # Food regeneration rate per tick
max_per_cell = 10.0   # Maximum food per grid cell
energy_value = 4.0    # Energy gained from eating 1 unit of food
initial_patches = 200 # Number of initial food patches
patch_size = 10       # Size of each food patch
baseline_food = 0.0   # Minimum food level everywhere (enables exploration)
effectiveness = 1.0   # Food digestion multiplier
spawn_chance = 0.000001
spawn_amount = 2.0

[world]
width = 2048
height = 2048
```

### 8.2 Command Line Options

```bash
# Run with default settings
cargo run --release

# Use a custom configuration file
cargo run --release -- --config my_config.toml

# Auto-exit after 30 seconds (for testing/benchmarking)
cargo run --release -- --auto-exit 30

# Start paused with 2x speed
cargo run --release -- --paused --speed 2
```

---

## 9. Key Implementation Notes

### 9.1 Organism Free List

Dead organisms are tracked for reuse:

```rust
struct OrganismPool {
    free_list: Vec<u32>,  // Indices of dead organisms
    next_new: u32,        // Next index if free_list empty
    count: u32,           // Current living count
}

impl OrganismPool {
    fn allocate(&mut self) -> Option<u32> {
        if let Some(idx) = self.free_list.pop() {
            self.count += 1;
            Some(idx)
        } else if self.next_new < MAX_ORGANISMS {
            let idx = self.next_new;
            self.next_new += 1;
            self.count += 1;
            Some(idx)
        } else {
            None  // At capacity
        }
    }
    
    fn deallocate(&mut self, idx: u32) {
        self.free_list.push(idx);
        self.count -= 1;
    }
}
```

### 9.2 Random Number Generation

Use xoshiro256++ for reproducibility:

```rust
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::SeedableRng;

let mut rng = Xoshiro256PlusPlus::seed_from_u64(simulation_seed);
```

### 9.3 Weight Initialization

Xavier uniform initialization:

```rust
fn init_weights(fan_in: usize, fan_out: usize, rng: &mut impl Rng) -> Vec<f32> {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let dist = Uniform::new(-limit, limit);
    (0..fan_in * fan_out).map(|_| dist.sample(rng)).collect()
}
```

### 9.4 Ecosystem Dynamics

Several mechanisms create a stable, sustainable ecosystem:

**Food Patches:**
Initial food patches use `patch_size=10` and are seeded at 70–100% of `max_per_cell`, providing dense but compact starting resources.

**Logistic Food Regrowth:**
Food regrows in cells that already have food (>0.01) using logistic growth:
```wgsl
let growth = config.food_growth_rate * current_food * (1.0 - current_food / config.food_max_per_cell);
```
This creates self-sustaining patches that naturally reach equilibrium.

**Spontaneous Food Spawning:**
Rare random food spawns (`food_spawn_chance=0.000001`, i.e., 0.0001% per cell per tick) create new patches for organisms to discover, rewarding exploration.

**Distributed Organism Spawning:**
Initial organisms are distributed across different food patches using stride-based position selection, preventing all organisms from starting in the same location and depleting the same food source.

**Age/Crowding Penalties:**
- `age_drain_factor=1.0`: Energy drain scales quadratically with age, doubling at `max_age`.
- `crowding_factor=1.0`: Energy drain scales quadratically with the population ratio; doubles at capacity and increases progressively before that.
These keep population turnover high and create selection pressure.

---

## 10. Success Metrics

### 10.1 Performance Targets

| Metric | Target | Measured By |
|--------|--------|-------------|
| Simulation FPS | 100+ | Ticks per second |
| Render FPS | 60 | Frame time |
| GPU utilization | 50-80% | Profiler |
| VRAM usage | <100 MB | wgpu stats |

### 10.2 Evolution Targets

| Behavior | Expected Generation |
|----------|---------------------|
| Random movement | 0 |
| Move toward visible food | 50-100 |
| Efficient foraging | 200-500 |
| Avoid starvation | 100-200 |

---

*This specification provides all details needed to implement Phase 1 MVP.*
