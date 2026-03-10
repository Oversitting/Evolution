# Evolution Simulator: GPU-Accelerated Digital Life

## Executive Summary

This document outlines the design for a GPU-accelerated evolution simulator where digital organisms live, interact, reproduce, and evolve in a 2D world. By representing organism genomes and neural architectures as tensor operations, we can leverage the same hardware acceleration used in machine learning to simulate thousands to millions of organisms simultaneously.

**Target Platform**: Desktop game for general gaming computers (Windows/Mac/Linux)

---

## 0. Game-Ready Design Considerations

### 0.1 Target Hardware Specifications

| Tier | GPU | VRAM | RAM | Target Organisms | Target FPS |
|------|-----|------|-----|------------------|------------|
| **Minimum** | GTX 1060 / RX 580 | 4 GB | 8 GB | 2,000 - 5,000 | 30 |
| **Recommended** | RTX 3060 / RX 6700 | 8 GB | 16 GB | 10,000 - 25,000 | 60 |
| **High-End** | RTX 4080 / RX 7900 | 16 GB | 32 GB | 50,000 - 100,000 | 60+ |

### 0.2 Key Changes for Game Distribution

| Research Prototype | Game-Ready Version |
|--------------------|-------------------|
| PyTorch/JAX (Python) | **Rust + wgpu** or **Godot + compute shaders** |
| CUDA-only | **Cross-platform GPU** (Vulkan/Metal/DX12) |
| CLI/scripts | **Polished GUI** with menus, settings |
| Raw tensors | **Memory-pooled, pre-allocated buffers** |
| Dev dependencies | **Single executable** or installer |

### 0.3 Technology Stack Decision

**Option A: Rust + wgpu + egui (Recommended)**
```
┌──────────────────────────────────────────────────────────────┐
│                         GAME                                 │
├──────────────────────────────────────────────────────────────┤
│  UI Layer          │  egui (immediate mode GUI)              │
│  Rendering         │  wgpu (WebGPU) → Vulkan/Metal/DX12     │
│  Compute           │  wgpu compute shaders (WGSL)            │
│  Game Logic        │  Rust (ECS with bevy_ecs or hecs)       │
│  Audio             │  rodio or kira                          │
│  Distribution      │  Single binary, ~50-100 MB              │
└──────────────────────────────────────────────────────────────┘
```

**Why Rust + wgpu:**
- ✅ Cross-platform GPU compute (Vulkan, Metal, DX12, WebGPU)
- ✅ No runtime dependencies (single .exe)
- ✅ Excellent performance, no GC pauses
- ✅ Can compile to WebAssembly for browser demo
- ✅ Strong ecosystem (bevy, winit, egui)

**Option B: Godot 4 + GDExtension**
```
┌──────────────────────────────────────────────────────────────┐
│  Godot 4 Engine                                              │
├──────────────────────────────────────────────────────────────┤
│  UI / Menus        │  Built-in Control nodes                 │
│  Rendering         │  Godot's Vulkan renderer                │
│  Compute           │  RenderingDevice compute shaders        │
│  Game Logic        │  GDScript + Rust GDExtension for perf   │
│  Audio             │  Built-in                               │
│  Distribution      │  Export templates (~80 MB)              │
└──────────────────────────────────────────────────────────────┘
```

**Why Godot:**
- ✅ Faster UI/menu development
- ✅ Built-in 2D rendering optimized
- ✅ Easier for adding game features
- ⚠️ Compute shader API is newer, less documented
- ⚠️ Godot overhead may limit max organism count

**Recommendation**: Start with **Rust + wgpu** for maximum control and performance. The simulation is the core product; UI can be minimal initially.

### 0.4 Performance Budget per Frame (60 FPS = 16.67ms)

```
┌────────────────────────────────────────────────────────────┐
│  FRAME BUDGET: 16.67ms                                     │
├────────────────────────────────────────────────────────────┤
│  GPU Compute (simulation)          │  8-10 ms   (60%)     │
│  ├── Sensory computation           │  3-4 ms              │
│  ├── Neural network forward pass   │  2-3 ms              │
│  ├── Physics + interactions        │  2-3 ms              │
│  └── Reproduction + death          │  1 ms                │
│                                                            │
│  GPU Render                         │  3-4 ms   (25%)     │
│  ├── Draw organisms                 │  2 ms                │
│  ├── Draw world (food, terrain)    │  1 ms                │
│  └── UI overlay                     │  1 ms                │
│                                                            │
│  CPU (orchestration)               │  2-3 ms   (15%)     │
│  ├── Input handling                │  0.5 ms              │
│  ├── GPU dispatch                  │  1 ms                │
│  └── Statistics/UI updates         │  1 ms                │
└────────────────────────────────────────────────────────────┘
```

---

## 1. Core Concept: Tensor-Based Life

### 1.1 The Key Insight

Modern GPUs are optimized for parallel matrix operations. Evolution and neural networks share a fundamental property: **both transform inputs through parameterized functions to produce outputs**. 

- In ML: weights transform inputs → outputs
- In biology: genes encode proteins that transform sensory inputs → behavioral outputs

By encoding an organism's genome as a neural network's weights, we can:
1. Run all organisms' "brains" in a single batched forward pass
2. Simulate physics interactions using spatial tensor operations
3. Apply genetic operations (mutation, crossover) as tensor manipulations

### 1.2 What This Enables

| Traditional Simulation | Tensor-Accelerated |
|------------------------|-------------------|
| Loop over each organism | Batch all organisms in parallel |
| CPU-bound, ~1,000 organisms | GPU-bound, ~100,000+ organisms |
| Simple behavior rules | Complex neural controllers |
| Hours for evolution | Minutes for same generations |

---

## 2. Organism Architecture

### 2.1 The Genome as a Neural Network

Each organism's DNA is encoded as the weights of a small neural network:

```
┌─────────────────────────────────────────────────────────────┐
│                     ORGANISM GENOME                         │
├─────────────────────────────────────────────────────────────┤
│  Sensory Layer (Input)                                      │
│  ├── Vision rays (distances, colors)      [N_rays × 3]      │
│  ├── Internal state (energy, age, health) [8]               │
│  ├── Chemical sensors (pheromones)        [4]               │
│  └── Touch/proximity                      [8]               │
│                                                             │
│  Hidden Layers (Genome-Encoded Weights)                     │
│  ├── Layer 1: [input_dim × 64] + bias                       │
│  ├── Layer 2: [64 × 32] + bias                              │
│  └── Layer 3: [32 × 16] + bias                              │
│                                                             │
│  Action Layer (Output)                                      │
│  ├── Movement (dx, dy, rotation)          [3]               │
│  ├── Eat/attack intensity                 [2]               │
│  ├── Reproduce signal                     [1]               │
│  └── Pheromone emission                   [2]               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Genome Encoding Details

```python
# Conceptual structure
class GenomeTensor:
    # Core neural weights (the "DNA")
    weights: List[Tensor]  # Shape: [(in, h1), (h1, h2), (h2, out)]
    biases: List[Tensor]   # Shape: [(h1,), (h2,), (out,)]
    
    # Morphology genes (body plan)
    body_size: float       # 0.5 - 2.0 scale factor
    speed_factor: float    # metabolism/speed tradeoff
    vision_range: float    # how far can see
    color_genes: Tensor    # [R, G, B] for display/recognition
    
    # Metabolic genes
    energy_efficiency: float
    reproduction_threshold: float
    diet_preference: Tensor  # herbivore ↔ carnivore spectrum
```

### 2.3 Batched Organism Computation

The key to GPU acceleration is batching all organisms into single tensor operations:

```python
# All organisms processed simultaneously
class PopulationState:
    # Positions and physics - Shape: [N_organisms, ...]
    positions: Tensor      # [N, 2] - x, y coordinates
    velocities: Tensor     # [N, 2] - dx, dy
    rotations: Tensor      # [N, 1] - facing angle
    
    # Internal state
    energy: Tensor         # [N, 1]
    age: Tensor           # [N, 1]
    health: Tensor        # [N, 1]
    
    # Genome weights for ALL organisms (batched)
    # This is the magic - all neural nets in one tensor
    genome_weights: List[Tensor]  # [N, in, h1], [N, h1, h2], etc.

def step_all_organisms(state: PopulationState, world: WorldState):
    # 1. Gather sensory inputs for ALL organisms (parallel)
    sensory_inputs = compute_all_sensors(state, world)  # [N, input_dim]
    
    # 2. Run ALL organism brains in ONE batched forward pass
    actions = batched_forward(sensory_inputs, state.genome_weights)  # [N, action_dim]
    
    # 3. Apply physics for ALL organisms (parallel)
    new_state = apply_physics(state, actions, world)
    
    return new_state
```

---

## 3. World Simulation

### 3.1 Grid-Based World with Continuous Positions

The world uses a hybrid approach:
- **Organisms**: Continuous floating-point positions
- **Resources/Environment**: Discrete grid for efficient spatial queries

```
┌────────────────────────────────────────────┐
│  World Grid (e.g., 1024 × 1024)            │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                │
│  │  │░░│  │  │▓▓│  │  │  │  ░░ = Food     │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤  ▓▓ = Obstacle │
│  │  │  │●→│  │  │  │○ │  │  ● = Organism  │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤  ○ = Other org │
│  │░░│  │  │  │  │░░│  │  │                │
│  └──┴──┴──┴──┴──┴──┴──┴──┘                │
└────────────────────────────────────────────┘
```

### 3.2 Spatial Hashing for Interactions

To efficiently compute organism-to-organism interactions:

```python
# Assign organisms to grid cells
cell_indices = (positions / cell_size).floor().long()  # [N, 2]

# Build spatial hash map (GPU-friendly)
# For each cell, list which organisms are in it
spatial_hash = scatter_to_grid(organism_ids, cell_indices)

# Query neighbors - only check adjacent cells
def get_nearby_organisms(org_idx):
    cell = cell_indices[org_idx]
    neighbor_cells = get_adjacent_cells(cell)  # 9 cells (3x3)
    return spatial_hash[neighbor_cells]
```

### 3.3 Environment Dynamics

```python
class WorldState:
    # Food/resource layer
    food_grid: Tensor          # [H, W] - food amount per cell
    food_growth_rate: Tensor   # [H, W] - regrowth speed
    
    # Pheromone layers (diffuse over time)
    pheromone_layers: Tensor   # [N_types, H, W]
    
    # Obstacles/terrain
    terrain: Tensor            # [H, W] - 0=passable, 1=blocked
    
    # Climate/seasonal effects
    temperature: Tensor        # [H, W] - affects metabolism
    light_level: float         # day/night cycle
```

---

## 4. Evolutionary Mechanics

### 4.1 Reproduction with Genetic Operations

```python
def reproduce(parent_genomes: Tensor, fitness_scores: Tensor):
    # Selection: higher fitness = higher reproduction chance
    selection_probs = softmax(fitness_scores / temperature)
    parent_indices = multinomial(selection_probs, n_offspring)
    
    # Get parent genomes
    offspring_genomes = parent_genomes[parent_indices].clone()
    
    # Sexual reproduction (crossover)
    if sexual_reproduction:
        mate_indices = select_mates(parent_indices, positions)
        offspring_genomes = crossover(
            offspring_genomes, 
            parent_genomes[mate_indices]
        )
    
    # Mutation (add noise to weights)
    mutation_mask = torch.rand_like(offspring_genomes) < mutation_rate
    mutations = torch.randn_like(offspring_genomes) * mutation_strength
    offspring_genomes += mutation_mask * mutations
    
    return offspring_genomes
```

### 4.2 Fitness is Implicit

Unlike traditional genetic algorithms, **fitness is not explicitly calculated**. Instead, it emerges from survival:

- Organisms that gather food survive longer
- Organisms that avoid predators live to reproduce
- Organisms that find mates pass on genes
- **Natural selection happens automatically**

### 4.3 Speciation

Track species by genome similarity:

```python
def compute_species(genomes: Tensor, threshold: float):
    # Flatten all genome weights
    flat_genomes = flatten_genome(genomes)  # [N, genome_dim]
    
    # Compute pairwise distances
    distances = cdist(flat_genomes, flat_genomes)  # [N, N]
    
    # Cluster into species (DBSCAN-like on GPU)
    species_ids = gpu_clustering(distances, threshold)
    
    return species_ids
```

---

## 5. Sensory System

### 5.1 Vision via Raycasting

Each organism has vision rays computed in parallel:

```python
def compute_vision(positions, rotations, world, n_rays=16, fov=120):
    """
    Cast rays for all organisms simultaneously.
    Returns: [N_organisms, N_rays, 4] - (distance, R, G, B) per ray
    """
    N = positions.shape[0]
    
    # Generate ray directions for each organism
    angles = torch.linspace(-fov/2, fov/2, n_rays)  # [n_rays]
    ray_dirs = angle_to_vector(rotations.unsqueeze(1) + angles)  # [N, n_rays, 2]
    
    # March rays through world (parallel)
    vision_data = raycast_batch(
        origins=positions,
        directions=ray_dirs,
        world=world,
        max_distance=vision_range
    )
    
    return vision_data  # [N, n_rays, 4]
```

### 5.2 Other Senses

```python
def compute_all_senses(state: PopulationState, world: WorldState):
    # Vision (most expensive - raycasting)
    vision = compute_vision(state.positions, state.rotations, world)
    
    # Smell (sample pheromone grid at position)
    smell = sample_grid(world.pheromone_layers, state.positions)
    
    # Touch (check immediate neighbors)
    touch = compute_proximity(state.positions, radius=body_size)
    
    # Internal state (already have this)
    internal = torch.stack([state.energy, state.age, state.health], dim=1)
    
    # Concatenate all senses
    return torch.cat([vision.flatten(1), smell, touch, internal], dim=1)
```

---

## 6. Game Implementation Architecture

### 6.1 Technology Stack (Game-Ready)

```
┌─────────────────────────────────────────────────────────────┐
│                      GAME APPLICATION                       │
├─────────────────────────────────────────────────────────────┤
│  Window/Input    │  winit (cross-platform windowing)       │
│  UI/Menus        │  egui (immediate mode GUI)              │
│  2D Rendering    │  wgpu render pipeline (instanced)       │
│  GPU Compute     │  wgpu compute shaders (WGSL)            │
│  Game Logic      │  Rust + ECS (hecs or bevy_ecs)          │
│  Audio           │  kira (ambient, events)                 │
│  Serialization   │  serde + bincode (save/load)            │
├─────────────────────────────────────────────────────────────┤
│  GPU Backend     │  Vulkan (Win/Linux) / Metal (Mac) /     │
│                  │  DX12 (Windows) / WebGPU (Browser)      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Project Structure

```
evolution-game/
├── Cargo.toml                 # Rust dependencies
├── src/
│   ├── main.rs                # Entry point, game loop
│   ├── app.rs                 # Application state machine
│   ├── simulation/
│   │   ├── mod.rs
│   │   ├── world.rs           # World grid, food, terrain
│   │   ├── organism.rs        # Organism data structures
│   │   ├── genome.rs          # Neural network genome
│   │   ├── physics.rs         # Movement, collision
│   │   └── evolution.rs       # Reproduction, mutation
│   ├── compute/
│   │   ├── mod.rs
│   │   ├── pipeline.rs        # GPU compute pipeline setup
│   │   ├── buffers.rs         # GPU buffer management
│   │   └── shaders/
│   │       ├── sense.wgsl     # Sensory computation
│   │       ├── think.wgsl     # Neural network forward pass
│   │       ├── act.wgsl       # Physics update
│   │       ├── interact.wgsl  # Organism interactions
│   │       └── world.wgsl     # Food regrowth, pheromones
│   ├── render/
│   │   ├── mod.rs
│   │   ├── organisms.rs       # Instanced organism rendering
│   │   ├── world.rs           # Terrain, food visualization
│   │   ├── camera.rs          # Pan, zoom controls
│   │   └── shaders/
│   │       ├── organism.wgsl  # Organism vertex/fragment
│   │       └── world.wgsl     # World tile rendering
│   ├── ui/
│   │   ├── mod.rs
│   │   ├── main_menu.rs       # Title screen
│   │   ├── hud.rs             # In-game overlay
│   │   ├── settings.rs        # Graphics, simulation settings
│   │   ├── inspector.rs       # Selected organism details
│   │   └── graphs.rs          # Population charts
│   └── audio/
│       ├── mod.rs
│       └── ambient.rs         # Background soundscape
├── assets/
│   ├── fonts/
│   ├── sounds/
│   └── textures/
└── docs/
    └── DESIGN.md              # This document
```

### 6.3 Core Game Loop (Rust)

```rust
fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Evolution Simulator")
        .with_inner_size(LogicalSize::new(1920, 1080))
        .build(&event_loop)?;
    
    let mut game = Game::new(&window).await;
    
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => game.handle_input(event),
            Event::MainEventsCleared => {
                // Fixed timestep simulation
                while game.accumulator >= SIMULATION_DT {
                    game.simulation_step();  // GPU compute
                    game.accumulator -= SIMULATION_DT;
                }
                
                // Render at display refresh rate
                game.render();  // GPU render
                window.request_redraw();
            }
            _ => {}
        }
    });
}

impl Game {
    fn simulation_step(&mut self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        
        // All simulation on GPU - no CPU round-trips
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            
            // 1. Sensory pass
            compute_pass.set_pipeline(&self.sense_pipeline);
            compute_pass.set_bind_group(0, &self.simulation_bind_group, &[]);
            compute_pass.dispatch_workgroups(self.organism_count / 64, 1, 1);
            
            // 2. Neural network pass
            compute_pass.set_pipeline(&self.think_pipeline);
            compute_pass.dispatch_workgroups(self.organism_count / 64, 1, 1);
            
            // 3. Physics pass
            compute_pass.set_pipeline(&self.act_pipeline);
            compute_pass.dispatch_workgroups(self.organism_count / 64, 1, 1);
            
            // 4. Interaction pass (spatial hash + resolve)
            compute_pass.set_pipeline(&self.interact_pipeline);
            compute_pass.dispatch_workgroups(self.grid_size / 8, self.grid_size / 8, 1);
            
            // 5. World update pass
            compute_pass.set_pipeline(&self.world_pipeline);
            compute_pass.dispatch_workgroups(self.grid_size / 8, self.grid_size / 8, 1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        self.frame_count += 1;
    }
}
```

### 6.4 WGSL Compute Shader Example (Neural Forward Pass)

```wgsl
// think.wgsl - Batched neural network for all organisms

struct Organism {
    position: vec2<f32>,
    velocity: vec2<f32>,
    rotation: f32,
    energy: f32,
    age: f32,
    alive: u32,
    genome_offset: u32,  // Index into genome buffer
}

@group(0) @binding(0) var<storage, read> organisms: array<Organism>;
@group(0) @binding(1) var<storage, read> sensory_input: array<f32>;  // [N * INPUT_DIM]
@group(0) @binding(2) var<storage, read> genomes: array<f32>;         // All genome weights
@group(0) @binding(3) var<storage, read_write> actions: array<f32>;   // [N * ACTION_DIM]

const INPUT_DIM: u32 = 48u;   // Vision + internal + smell + touch
const HIDDEN1: u32 = 32u;
const HIDDEN2: u32 = 16u;
const OUTPUT_DIM: u32 = 8u;

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let org_idx = id.x;
    if org_idx >= arrayLength(&organisms) { return; }
    
    let org = organisms[org_idx];
    if org.alive == 0u { return; }
    
    // Get this organism's genome weights
    let genome_base = org.genome_offset;
    let input_base = org_idx * INPUT_DIM;
    
    // Layer 1: Input -> Hidden1
    var hidden1: array<f32, 32>;
    for (var h = 0u; h < HIDDEN1; h++) {
        var sum = 0.0;
        for (var i = 0u; i < INPUT_DIM; i++) {
            let weight_idx = genome_base + h * INPUT_DIM + i;
            sum += sensory_input[input_base + i] * genomes[weight_idx];
        }
        hidden1[h] = relu(sum);
    }
    
    // Layer 2: Hidden1 -> Hidden2
    let layer2_offset = genome_base + INPUT_DIM * HIDDEN1;
    var hidden2: array<f32, 16>;
    for (var h = 0u; h < HIDDEN2; h++) {
        var sum = 0.0;
        for (var i = 0u; i < HIDDEN1; i++) {
            let weight_idx = layer2_offset + h * HIDDEN1 + i;
            sum += hidden1[i] * genomes[weight_idx];
        }
        hidden2[h] = relu(sum);
    }
    
    // Layer 3: Hidden2 -> Output
    let layer3_offset = layer2_offset + HIDDEN1 * HIDDEN2;
    let output_base = org_idx * OUTPUT_DIM;
    for (var o = 0u; o < OUTPUT_DIM; o++) {
        var sum = 0.0;
        for (var i = 0u; i < HIDDEN2; i++) {
            let weight_idx = layer3_offset + o * HIDDEN2 + i;
            sum += hidden2[i] * genomes[weight_idx];
        }
        actions[output_base + o] = tanh(sum);  // Output in [-1, 1]
    }
}
```

---

## 7. Performance Estimates (Game Hardware)

### 7.1 Scaling by Hardware Tier

| GPU Tier | Example | Organisms | Neural Net | VRAM Used | Target FPS |
|----------|---------|-----------|------------|-----------|------------|
| **Entry** | GTX 1060 | 2,000 | 32-16-8 | 1.5 GB | 30-60 |
| **Mid** | RTX 3060 | 10,000 | 48-32-16 | 3 GB | 60 |
| **High** | RTX 4070 | 25,000 | 48-32-16 | 5 GB | 60 |
| **Ultra** | RTX 4090 | 100,000 | 64-32-16 | 12 GB | 60 |

### 7.2 Memory Budget (8 GB VRAM Target)

```
┌────────────────────────────────────────────────────────────┐
│  VRAM BUDGET: 8 GB (Recommended tier)                     │
├────────────────────────────────────────────────────────────┤
│  Organism State Buffers                                    │
│  ├── Positions, velocities (25K × 16 bytes)    = 400 KB   │
│  ├── Energy, age, health (25K × 12 bytes)      = 300 KB   │
│  ├── Genome weights (25K × 3K floats × 4)      = 300 MB   │
│  └── Sensory buffers (25K × 64 floats × 4)     = 6 MB     │
│  Subtotal: ~310 MB                                         │
│                                                            │
│  World State                                               │
│  ├── Food grid (2048² × 4 bytes)               = 16 MB    │
│  ├── Pheromone layers (4 × 1024² × 4)          = 16 MB    │
│  ├── Terrain (2048² × 1 byte)                  = 4 MB     │
│  └── Spatial hash (2048² × 16 bytes)           = 64 MB    │
│  Subtotal: ~100 MB                                         │
│                                                            │
│  Rendering                                                 │
│  ├── Organism instance buffer (25K × 32 bytes) = 800 KB   │
│  ├── World textures                            = 32 MB    │
│  └── UI textures                               = 16 MB    │
│  Subtotal: ~50 MB                                          │
│                                                            │
│  GPU Working Memory / Headroom                 = 500 MB   │
├────────────────────────────────────────────────────────────┤
│  TOTAL: ~1 GB (plenty of headroom)                        │
└────────────────────────────────────────────────────────────┘
```

### 7.3 Bottlenecks and Mitigations

| Bottleneck | Impact | Game-Friendly Mitigation |
|------------|--------|-------------------------|
| **Raycasting** | 40% of compute | Reduce to 8 rays, 64px max range |
| **Spatial hash rebuild** | 15% of compute | Rebuild every 4 frames |
| **Neural forward pass** | 25% of compute | Use FP16, smaller nets on low settings |
| **Rendering** | Separate from sim | Instanced rendering, LOD for distant |
| **CPU-GPU sync** | Stalls pipeline | Async readback for stats only |

### 7.4 Quality Presets

| Setting | Low | Medium | High | Ultra |
|---------|-----|--------|------|-------|
| Max Organisms | 2,000 | 8,000 | 25,000 | 100,000 |
| World Size | 512² | 1024² | 2048² | 4096² |
| Vision Rays | 4 | 8 | 12 | 16 |
| Vision Range | 32px | 64px | 96px | 128px |
| Neural Net | 24-12-8 | 32-16-8 | 48-32-16 | 64-32-16 |
| Food Resolution | 256² | 512² | 1024² | 2048² |
| Pheromone Layers | 2 | 3 | 4 | 6 |

---

## 8. Emergent Behaviors to Expect

Based on similar simulations, we should observe:

### 8.1 Individual Level
- Foraging strategies
- Predator avoidance
- Navigation and exploration

### 8.2 Population Level
- Speciation into niches
- Predator-prey dynamics
- Resource competition

### 8.3 Ecosystem Level
- Population cycles (boom/bust)
- Food web formation
- Territorial behavior

### 8.4 Novel Possibilities (with enough scale)
- Communication/signaling via pheromones
- Cooperative hunting
- Symbiotic relationships
- "Cultural" behaviors (learned from parents)

---

## 9. Limitations and Challenges

### 9.1 Fundamental Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Discrete time steps** | Misses fast dynamics | Smaller dt, physics substeps |
| **Simplified physics** | Not real biomechanics | Good enough for evolution |
| **Fixed neural architecture** | Limits evolvability | Use NEAT-like topology evolution |
| **2D only** | Misses 3D behaviors | Start 2D, extend to 3D later |
| **No development** | Organisms don't "grow" | Could add developmental genes |

### 9.2 Technical Challenges

1. **Dynamic population size**
   - GPUs prefer fixed batch sizes
   - Solution: Over-allocate, use masks for dead/unborn

2. **Variable-length genomes**
   - For topology evolution (NEAT)
   - Solution: Pad to max size, or use separate species batches

3. **Reproduction synchronization**
   - When to spawn new organisms?
   - Solution: Batch reproduction at intervals, or continuous with masks

4. **Determinism for reproducibility**
   - GPU random numbers can vary
   - Solution: Fixed seeds, record/replay capability

### 9.3 Biological Realism Trade-offs

```
More Realistic ←───────────────────────────→ More Computable

Cell-level simulation    Organ systems    Neural abstraction    Simple rules
(impossible at scale)    (very slow)      (OUR TARGET)          (limited emergence)
```

We're targeting the **neural abstraction** level - complex enough for interesting emergence, simple enough for massive parallelism.

---

## 10. Implementation Roadmap (Game Development)

### Phase 1: Engine Foundation (3-4 weeks)
- [ ] Rust project setup with wgpu, winit, egui
- [ ] Basic window, input handling, camera (pan/zoom)
- [ ] GPU buffer management system
- [ ] Simple compute shader test (moving particles)
- [ ] Basic instanced 2D rendering
- [ ] **Deliverable**: Window with 10K moving dots, 60 FPS

### Phase 2: Core Simulation (4-5 weeks)
- [ ] Organism data structures on GPU
- [ ] World grid (food layer)
- [ ] Neural network forward pass shader
- [ ] Movement physics shader
- [ ] Eating mechanics (organism ↔ food)
- [ ] Energy, death, despawn system
- [ ] Asexual reproduction with mutation
- [ ] **Deliverable**: Self-sustaining ecosystem, organisms evolve foraging

### Phase 3: Richer Simulation (3-4 weeks)
- [ ] Vision system (raycasting shader)
- [ ] Spatial hashing for organism-organism detection
- [ ] Predation mechanics
- [ ] Sexual reproduction with mate finding
- [ ] Pheromone system (deposit, diffuse, sense)
- [ ] Species tracking and coloring
- [ ] **Deliverable**: Predator-prey dynamics, visible speciation

### Phase 4: Game Polish (3-4 weeks)
- [ ] Main menu, pause menu, settings
- [ ] HUD with population stats, generation counter
- [ ] Organism inspector (click to view brain, stats)
- [ ] Population graphs over time
- [ ] Save/load simulation state
- [ ] Quality presets (Low/Medium/High/Ultra)
- [ ] **Deliverable**: Playable game experience

### Phase 5: Content & Features (ongoing)
- [ ] Multiple biomes/terrain types
- [ ] Environmental events (drought, flood, seasons)
- [ ] User tools (place food, walls, disasters)
- [ ] Time controls (pause, speed up, slow-mo)
- [ ] Phylogenetic tree visualization
- [ ] Achievements/milestones
- [ ] Steam/itch.io release preparation
- [ ] **Deliverable**: Feature-complete game

### Phase 6: Advanced Evolution (stretch goals)
- [ ] Topology evolution (NEAT-like, variable networks)
- [ ] Morphology evolution (body shape/size genes)
- [ ] Learning within lifetime (Hebbian plasticity)
- [ ] Social behaviors (signaling, cooperation)
- [ ] 3D mode (optional future)

### Timeline Summary

```
Week:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
       ├──────────┼────────────────┼────────────┼────────────┤
       │ Phase 1  │    Phase 2     │  Phase 3   │  Phase 4   │
       │ Engine   │ Core Sim       │ Rich Sim   │  Polish    │
       └──────────┴────────────────┴────────────┴────────────┘
                                                        ↓
                                              Alpha Release (~4 months)
```

---

## 11. Feasibility Assessment

### ✅ YES, This is Feasible as a Game

**Evidence from similar shipped games/projects:**
1. **The Bibites** (2021) - Commercial neural organism simulator (Unity)
2. **Biosim4** - Open source C++ evolution sim with neural nets
3. **Species: ALRE** (2018) - Steam game, evolution sandbox
4. **Equilinox** (2018) - Ecosystem simulation game
5. **Lenia** (2020) - GPU-accelerated continuous life (WebGL)

**Why the game approach works:**
- Rust + wgpu provides cross-platform GPU compute without CUDA lock-in
- Similar games have proven market interest (The Bibites has 100K+ downloads)
- Single executable distribution, no Python/ML dependencies
- Can scale simulation complexity to match hardware

### ⚠️ Game-Specific Challenges

1. **Balancing emergence vs. entertainment**
   - Pure emergence can be slow/boring to watch
   - Add speed controls, highlights, notifications for interesting events

2. **Onboarding new players**
   - Evolution concepts need gentle introduction
   - Tutorial mode, tooltips, "what's happening" explanations

3. **Long-term engagement**
   - Evolution takes time - need ways to make waiting fun
   - Achievements, milestones, time acceleration

4. **Cross-platform testing**
   - Must test on AMD, Intel, NVIDIA GPUs
   - Metal on Mac, Vulkan on Linux
   - Use wgpu's abstraction layer carefully

5. **Save file compatibility**
   - Genome format may evolve during development
   - Version save files, provide migration

---

## 12. Technology Alternatives Considered

### 12.1 Unity + Compute Shaders
- ✅ Rich editor, easy UI
- ✅ Cross-platform out of the box
- ⚠️ C# performance overhead
- ⚠️ Licensing costs at scale
- **Verdict**: Good option if you know Unity

### 12.2 Godot 4 + Compute Shaders
- ✅ Open source, no licensing
- ✅ Built-in 2D rendering
- ⚠️ Compute shader API is new
- ⚠️ Less documentation for GPU compute
- **Verdict**: Consider for faster UI development

### 12.3 Python + PyTorch + Pygame
- ✅ Fast to prototype
- ✅ Familiar if you know ML
- ❌ Hard to distribute (Python runtime)
- ❌ Pygame rendering is slow
- **Verdict**: Use for research prototype only

### 12.4 C++ + Vulkan
- ✅ Maximum performance
- ❌ Massive development effort
- ❌ Vulkan is very low-level
- **Verdict**: Not worth the complexity

### 12.5 Bevy Engine (Rust)
- ✅ Full ECS game engine
- ✅ Built-in rendering, audio, input
- ⚠️ Still maturing, API changes
- ⚠️ May be overkill for 2D
- **Verdict**: Consider if you want more engine features

**Final Choice**: **Rust + wgpu + egui** - best balance of performance, cross-platform, and control.

---

## 13. Success Criteria (Game Metrics)

### Technical Success
| Metric | Target |
|--------|--------|
| FPS on recommended hardware | 60 FPS stable |
| Organism count (medium settings) | 10,000+ |
| Startup time | < 5 seconds |
| Memory usage | < 4 GB RAM, < 4 GB VRAM |
| Binary size | < 100 MB |

### Gameplay Success
| Metric | Target |
|--------|--------|
| Time to first speciation | < 10 minutes |
| Ecosystem stability | Runs 1+ hour without extinction |
| Visible behavior evolution | Within 30 minutes |
| Player engagement | "Just one more generation" feeling |

### Emergence Success
| Behavior | Expected Timeline |
|----------|------------------|
| Efficient foraging | 5-10 generations |
| Predator avoidance | 20-50 generations |
| Predator specialization | 50-100 generations |
| Niche differentiation | 100-200 generations |
| Pheromone-based coordination | 200+ generations |

---

## 14. Game Features Specification

### 14.1 Main Menu
```
┌────────────────────────────────────────────────────────┐
│                                                        │
│              E V O L U T I O N                         │
│                 SIMULATOR                              │
│                                                        │
│              [ New Simulation ]                        │
│              [ Load Simulation ]                       │
│              [ Settings ]                              │
│              [ Quit ]                                  │
│                                                        │
│                                    v0.1.0              │
└────────────────────────────────────────────────────────┘
```

### 14.2 New Simulation Options
- **World Size**: Small (512²) / Medium (1024²) / Large (2048²)
- **Starting Organisms**: 100 / 500 / 1000
- **Food Abundance**: Scarce / Normal / Abundant
- **Mutation Rate**: Low / Medium / High / Custom
- **Seed**: Random or enter specific seed
- **Preset**: "Beginner" / "Standard" / "Harsh" / "Custom"

### 14.3 In-Game HUD
```
┌────────────────────────────────────────────────────────────────────┐
│ Gen: 1,247  Pop: 8,432  Species: 12       [▶ 1x] [⏸] [⏩ 4x] [⏭]  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│                     (World View - Zoomable/Pannable)               │
│                                                                    │
│                             ● ●   ○                                │
│                          ●     ░░░    ●                            │
│                        ○   ░░░░░░░░░    ○                          │
│                              ░░░░░                                 │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│ 📊 Population │ 🧬 Genetics │ 🌍 World │ ⚙ Settings                │
└────────────────────────────────────────────────────────────────────┘
```

### 14.4 Organism Inspector (Click to Select)
```
┌─────────────────────────────────┐
│ ORGANISM #4,721                 │
│ Species: Green Forager          │
│ Age: 847 ticks                  │
│ Energy: ███████░░░ 72%          │
│ Children: 4                     │
│                                 │
│ [Brain Viewer]                  │
│  I1 ─┬─ H1 ─┬─ O1 (move)       │
│  I2 ─┤      ├─ O2 (turn)       │
│  ... │      └─ O3 (eat)        │
│                                 │
│ [Ancestry] [Follow] [Track]     │
└─────────────────────────────────┘
```

### 14.5 Population Graphs Panel
- Total population over time
- Species breakdown (stacked area)
- Average energy levels
- Birth/death rates
- Genome diversity metrics

### 14.6 Keyboard Controls
| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| 1-4 | Speed (1x, 2x, 4x, 8x) |
| WASD / Arrows | Pan camera |
| Scroll | Zoom in/out |
| Click | Select organism |
| F | Follow selected |
| G | Toggle grid |
| H | Toggle HUD |
| Esc | Menu / Deselect |
| F5 | Quick save |
| F9 | Quick load |

---

## Appendix A: Reference Games & Projects

### Commercial/Released Games
- **The Bibites** - https://leocaussan.itch.io/the-bibites (closest reference)
- **Species: ALRE** - https://store.steampowered.com/app/774541/Species_Artificial_Life_Real_Evolution/
- **Equilinox** - https://store.steampowered.com/app/853550/Equilinox/
- **Thrive** - https://revolutionarygamesstudio.com/ (open source evolution game)

### Open Source References
- **Biosim4** - https://github.com/davidrmiller/biosim4 (C++, neural organisms)
- **Primer's Evolution** - https://github.com/Primer-Learning/PrimerLearning (educational)
- **Lenia** - https://github.com/Chakazul/Lenia (GPU life, beautiful visuals)
- **Neural Slime Volleyball** - Example of batched neural agents in games

### Rust/wgpu References
- **wgpu examples** - https://github.com/gfx-rs/wgpu/tree/trunk/examples
- **Learn wgpu** - https://sotrh.github.io/learn-wgpu/
- **egui examples** - https://github.com/emilk/egui
- **Bevy** - https://bevyengine.org/ (if you want a full engine later)

## Appendix B: Key Resources

### Papers (for algorithm understanding)
1. Stanley & Miikkulainen (2002) - "NEAT: Evolving Neural Networks through Augmenting Topologies"
2. Sims (1994) - "Evolving 3D Morphology and Behavior by Competition"
3. Yaeger (1994) - "Polyworld: Life in a New Context"

### Tutorials (for implementation)
1. **wgpu compute shaders** - https://sotrh.github.io/learn-wgpu/
2. **WGSL specification** - https://www.w3.org/TR/WGSL/
3. **Spatial hashing in games** - Various game dev tutorials
4. **Instanced rendering** - wgpu examples/instancing

## Appendix C: Rust Dependencies (Cargo.toml)

```toml
[package]
name = "evolution-sim"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core
wgpu = "0.19"              # GPU compute & rendering
winit = "0.29"             # Windowing
pollster = "0.3"           # Async runtime for wgpu
bytemuck = { version = "1.14", features = ["derive"] }  # Safe transmutes

# UI
egui = "0.27"              # Immediate mode GUI
egui-wgpu = "0.27"         # egui wgpu backend
egui-winit = "0.27"        # egui winit integration

# Math
glam = "0.27"              # Vector/matrix math
rand = "0.8"               # Random number generation
rand_xoshiro = "0.6"       # Fast, seedable RNG

# Serialization
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"            # Fast binary serialization

# Audio (optional initially)
kira = "0.8"               # Game audio

# Utilities
log = "0.4"
env_logger = "0.11"
anyhow = "1.0"             # Error handling

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

---

*Document Version: 0.2 - Game-Ready Design*
*Created: 2026-01-24*
*Updated: 2026-01-26*
