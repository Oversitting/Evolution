# Evolution Simulator: Design Decisions

**Purpose**: Answers to critical design questions, organized by priority for iterative implementation.  
**Philosophy**: Start small, run on low-end GPUs, scale up progressively, maximize performance while preserving interesting interactions.

**Created**: January 26, 2026  
**Based on**: DESIGN_QUESTIONS.md

---

## 🔴 CRITICAL PRIORITY - Phase 1 (MVP Foundation)

### 1. Technology Stack

**Q1.1.1 - Primary GPU Backend**
- **Answer**: Vulkan via wgpu
- **Rationale**: Best performance on Windows/Linux, wgpu handles Metal/DX12 fallback automatically

**Q1.1.2 - CPU Fallback**
- **Answer**: No CPU fallback initially
- **Rationale**: GPU is core to concept; warning message on incompatible hardware acceptable for MVP

**Q1.1.3 - wgpu Version**
- **Answer**: Start with wgpu 0.19, lock minor version
- **Rationale**: Stable API, update only when needed for features

**Q1.1.6 - Required wgpu Limits**
- **Answer**: Query and require:
  - Max buffer size: 256 MB minimum
  - Max compute workgroup size: 256x1x1 minimum
  - Max storage buffers: 8 minimum
- **Rationale**: Covers low-end GPUs (GTX 1050 Ti and up)

**Q1.2.1 - ECS Library**
- **Answer**: Use hecs (lightweight ECS)
- **Rationale**: Simple, minimal overhead, sufficient for our needs

**Q1.2.2 - Bevy Engine?**
- **Answer**: No, use raw wgpu + egui
- **Rationale**: Maximum control over GPU compute, lighter weight

**Q1.2.3 - Async Runtime**
- **Answer**: Stick with pollster for now
- **Rationale**: Minimal async needs, avoid tokio complexity

**Q1.2.5 - Audio Priority**
- **Answer**: Defer to Phase 4 (Polish)
- **Rationale**: Not critical for MVP evolution mechanics

**Q1.3.1 - CI/CD**
- **Answer**: GitHub Actions for automated builds
- **Rationale**: Free, well-integrated, sufficient

**Q1.3.2 - Platform Testing**
- **Answer**: Windows primary, Linux secondary, Mac stretch goal
- **Rationale**: Resource constraints, Windows covers majority

**Q1.3.4 - Shader Hot-Reload**
- **Answer**: Yes, implement for development
- **Rationale**: Massive time saver for GPU development

---

### 2. Neural Network Architecture (MVP)

**Q2.1.1 - Fixed or Variable Topology?**
- **Answer**: Fixed topology for MVP, all organisms identical structure
- **Rationale**: Much simpler GPU batching, add NEAT-style topology evolution in Phase 6

**Q2.1.2 - Layer Sizes per Preset**
- **Answer**: Start with single "Low" preset for MVP:
  - **MVP (Low)**: Input[20] → Hidden[16] → Output[6]
  - Phase 3: Medium: Input[32] → Hidden[24,12] → Output[8]
  - Phase 5: High: Input[48] → Hidden[32,16] → Output[8]
- **Rationale**: Smallest viable network for MVP, proves concept

**Q2.1.3 - Hidden Layer Activation**
- **Answer**: ReLU only for MVP
- **Rationale**: Fastest GPU implementation, works well

**Q2.1.4 - Output Layer Activation**
- **Answer**: tanh (outputs in [-1, 1])
- **Rationale**: Natural for movement/action outputs

**Q2.1.5 - Bias Terms**
- **Answer**: Yes, include biases
- **Rationale**: Small memory cost, significant expressiveness gain

**Q2.1.6 - Weight Initialization**
- **Answer**: Xavier uniform initialization (sqrt(6/(fan_in + fan_out)))
- **Rationale**: Standard practice, prevents vanishing/exploding gradients

---

### 3. Sensory System (MVP)

**Q2.2.1 - Vision Rays per Preset**
- **Answer**: MVP: 8 rays, 90° FOV
- **Rationale**: Enough for basic navigation, low GPU cost

**Q2.2.2 - Vision Data Format**
- **Answer**: MVP: Distance + Type (2 values per ray)
  - Distance: normalized [0,1]
  - Type: 0=empty, 0.5=food, 1.0=organism
- **Rationale**: Minimal but sufficient for predator/prey

**Q2.2.3 - Distance Encoding**
- **Answer**: Normalized inverse distance: `1.0 - (dist / max_range)`
- **Rationale**: Closer objects = stronger signal, natural for networks

**Q2.2.4 - Internal State Inputs**
- **Answer**: MVP: 4 values
  - Energy: normalized [0,1] by max energy
  - Age: normalized by expected lifespan (1000 ticks)
  - Speed: current velocity magnitude / max speed
  - Constant 1.0 (bias input)
- **Rationale**: Core survival info, simple

**Q2.2.5 - Pheromone Channels**
- **Answer**: MVP: 0 channels (defer to Phase 3)
- **Rationale**: Complex system, not needed for basic evolution

**Q2.2.7 - Touch/Proximity**
- **Answer**: MVP: 0 (implicit in vision)
- **Rationale**: Vision covers proximity needs initially

**Total MVP Input Dimension**: 8 rays × 2 + 4 internal = **20 inputs** ✓

---

### 4. Action Outputs (MVP)

**Q2.3.1 - Movement Interpretation**
- **Answer**: Direct velocity control
  - Output[0]: forward/backward speed [-1, 1]
  - Output[1]: rotation (angular velocity) [-1, 1]
- **Rationale**: Simplest to implement, no momentum physics initially

**Q2.3.2 - Maximum Speed**
- **Answer**: Capped at 2.0 pixels/tick
- **Rationale**: Prevents instability, reasonable for 60 FPS

**Q2.3.4 - Eat/Attack**
- **Answer**: Single unified output[2]: "mouth" [-1, 1]
  - Positive = eat food
  - Negative = attack organism
- **Rationale**: Forces specialization, simpler than separate channels

**Q2.3.7 - Reproduce Signal**
- **Answer**: Output[3]: reproduce willingness [0, 1] (only positive values used)
- **Rationale**: Must actively choose to reproduce

**Q2.3.8 - Pheromones**
- **Answer**: MVP: No pheromone outputs (defer to Phase 3)

**Remaining Outputs**: Output[4-5]: Reserved for future use

**Total MVP Output Dimension**: **6 outputs** (2 movement, 1 eat/attack, 1 reproduce, 2 reserved)

---

### 5. Genome Storage (MVP)

**Q2.4.1 - GPU Memory Organization**
- **Answer**: Struct of Arrays (SoA)
  - Buffer: weights_layer1[N, 20, 16]
  - Buffer: biases_layer1[N, 16]
  - Buffer: weights_layer2[N, 16, 6]
  - Buffer: biases_layer2[N, 6]
- **Rationale**: Better GPU memory coalescing than AoS

**Q2.4.2 - Max Genome Size**
- **Answer**: MVP: (20×16 + 16 + 16×6 + 6) = 438 floats × 4 bytes = 1,752 bytes per organism
- **Rationale**: Tiny, can fit 10K organisms in ~17 MB

**Q2.4.3 - Dead Organism Slots**
- **Answer**: Maintain free list, reuse immediately
- **Rationale**: No fragmentation, constant memory

**Q2.4.4 - FP16 vs FP32**
- **Answer**: FP32 for MVP
- **Rationale**: Simpler, plenty of VRAM, FP16 optimization later if needed

**Q2.4.5 - Weight Clamping**
- **Answer**: No clamping during evolution
- **Rationale**: Let evolution explore, natural selection handles extremes

---

### 6. Energy & Metabolism (MVP)

**Q5.1.1 - Starting Energy**
- **Answer**: Newborns: 50 energy units (50% of parent's max)
- **Rationale**: Parent invests significantly, offspring viable

**Q5.1.2 - Maximum Energy**
- **Answer**: 100 energy units (fixed for MVP)
- **Rationale**: Simple, consistent, scales with body size later

**Q5.1.3 - Passive Drain**
- **Answer**: 0.1 energy/tick
- **Rationale**: ~1000 tick lifespan if doing nothing

**Q5.1.4 - Movement Cost**
- **Answer**: 0.05 × abs(forward_speed) + 0.02 × abs(rotation)
- **Rationale**: Movement expensive, forces efficiency

**Q5.1.5 - Reproduction Cost**
- **Answer**: Parent loses 50 energy (= offspring starting energy)
- **Rationale**: Direct transfer, zero-sum

**Q5.2.1 - Food Energy Value**
- **Answer**: 20 energy per food unit consumed
- **Rationale**: Enough to justify seeking food

**Q5.2.2 - Eating Rate**
- **Answer**: Max 1.0 food unit/tick when mouth output > 0.5
- **Rationale**: Rate limited to create competition

**Q5.2.3 - Food Contention**
- **Answer**: First-come first-served per tick
- **Rationale**: Simple, creates scramble competition

**Q5.2.4 - Predation Energy**
- **Answer**: Attacker gains 50% of prey's current energy
- **Rationale**: Inefficiency makes predation challenging

**Q5.3.1 - Death Threshold**
- **Answer**: Die when energy ≤ 0
- **Rationale**: Clean, simple

**Q5.3.2 - Maximum Age**
- **Answer**: No age limit for MVP
- **Rationale**: Energy is limiting factor, adds later if needed

**Q5.3.4 - Dead Organism Handling**
- **Answer**: Immediate despawn, no corpses initially
- **Rationale**: Simplest, add carrion in Phase 3 if interesting

---

### 7. Reproduction Mechanics (MVP)

**Q7.1.1 - Reproduction Trigger**
- **Answer**: Asexual reproduction when:
  - Energy ≥ 60 (threshold) AND
  - Reproduce signal > 0.8 AND
  - Age > 100 ticks (prevents immediate re-reproduction)
- **Rationale**: Must survive AND choose to reproduce

**Q7.1.2 - Synchronous or Async**
- **Answer**: Asynchronous (any tick)
- **Rationale**: More realistic, no artificial generations

**Q7.1.3 - Multiple Reproductions**
- **Answer**: Yes, unlimited if energy permits
- **Rationale**: Successful organisms should proliferate

**Q7.1.4 - Reproduction Cooldown**
- **Answer**: 50 ticks after reproduction
- **Rationale**: Prevents spam, forces strategy

**Q7.2.1 - Offspring Position**
- **Answer**: Random offset within 5 pixels of parent
- **Rationale**: Prevents instant overlap

**Q7.2.2 - Genome Cloning**
- **Answer**: Yes, exact copy before mutation
- **Rationale**: Asexual = clone

**Q7.4.1 - Mutation Rate**
- **Answer**: 5% per weight (each weight has 5% chance to mutate)
- **Rationale**: Balanced exploration

**Q7.4.2 - Mutation Strength**
- **Answer**: Gaussian noise, σ = 0.2
- **Rationale**: Small tweaks, not complete scrambling

**Q7.4.3 - Mutation Distribution**
- **Answer**: Gaussian (normal distribution)
- **Rationale**: Natural, most mutations small

**Q7.4.4 - Morphology Mutation**
- **Answer**: MVP: No morphology genes yet
- **Rationale**: Fixed bodies, brains-only evolution for MVP

**Q7.5.1 - Population Cap**
- **Answer**: Soft cap at target population
  - MVP: 2,000 organisms on low-end GPU
  - When reached: stop reproduction until population < 90% of cap
  - implement a crowding mechanism in future
- **Rationale**: Memory efficient, prevents overflow

---

### 8. World Structure (MVP)

**Q6.1.1 - Grid Size**
- **Answer**: MVP: 512×512 grid
- **Rationale**: Low VRAM (~1 MB), runs on weak GPUs

**Q6.1.2 - Grid to Pixel Relationship**
- **Answer**: 1 grid cell = 1 pixel, organisms can be sub-pixel (float positions)
- **Rationale**: Simple 1:1 mapping

**Q6.2.1 - Food Growth Model**
- **Answer**: Logistic growth per cell
  - `growth = rate × food × (1 - food/max)`
  - rate = 0.01/tick
  - max = 10 units per cell
- **Rationale**: Self-limiting, creates stable food distribution

**Q6.2.6 - Initial Food Distribution**
- **Answer**: Random clusters
  - 20 food patches, each 10×10 cells
  - 8 units per cell initial value
- **Rationale**: Creates foraging challenges

**Q6.4.1 - Terrain Representation**
- **Answer**: MVP: Binary (passable=0, blocked=1)
- **Rationale**: Simplest collision

**Q6.4.5 - Vision Through Obstacles**
- **Answer**: Obstacles block vision rays
- **Rationale**: Creates navigation challenges

**Q6.4.6 - Obstacle Placement**
- **Answer**: MVP: 5-10 random rectangular obstacles
- **Rationale**: Adds environmental complexity

---

### 9. Physics & Movement (MVP)

**Q4.1.1 - Movement Model**
- **Answer**: Direct velocity (no momentum/inertia)
- **Rationale**: Simpler, responsive control

**Q4.1.2 - Physics Timestep**
- **Answer**: Fixed 1:1 with simulation tick
- **Rationale**: Deterministic, no sub-stepping needed at this scale

**Q4.1.5 - Friction/Drag**
- **Answer**: No friction, organisms stop when output = 0
- **Rationale**: Direct neural control, energy cost handles efficiency

**Q4.2.1 - Organism-Organism Collision**
- **Answer**: Elastic push-apart (spring force)
- **Rationale**: Prevents stacking, looks organic

**Q4.2.2 - Organism-Obstacle Collision**
- **Answer**: Hard stop (velocity = 0)
- **Rationale**: Simple, clear boundary

**Q4.3.1 - World Edges**
- **Answer**: Hard walls (stop movement)
- **Rationale**: Simplest, creates boundaries

---

### 10. Spatial Interactions (MVP)

**Q8.2.1 - Spatial Hash Cell Size**
- **Answer**: 32×32 pixels per cell (16×16 grid for 512² world)
- **Rationale**: Balance between granularity and overhead

**Q8.2.2 - Hash Rebuild Frequency**
- **Answer**: Every tick
- **Rationale**: Low organism count (2K), negligible cost

**Q8.2.5 - Interaction Resolution**
- **Answer**: Check all organisms in same + adjacent 8 cells (3×3 region)
- **Rationale**: Catches all nearby interactions

**Q8.1.1 - Predation Mechanics**
- **Answer**: 
  - Attacker within 3 pixels of prey AND
  - Attacker's mouth output < -0.5 (attack mode) AND
  - Sustained contact for 3 ticks to kill
- **Rationale**: Requires commitment, prey has escape chance

**Q8.1.2 - Attack Damage**
- **Answer**: 10 energy/tick drained from prey
- **Rationale**: 3 ticks = 30 damage minimum to kill

---

## 🟡 HIGH PRIORITY - Phase 2-3 (Core Features)

### 11. Rendering (Phase 2)

**Q10.1.1 - Organism Rendering**
- **Answer**: Colored triangles pointing in facing direction
  - Base: equilateral triangle, 6 pixel base
  - Color: RGB from genome hash (deterministic species coloring)
- **Rationale**: Clear directionality, low draw call count

**Q10.1.4 - State Indicators**
- **Answer**: 
  - Brightness scales with energy (0.3 to 1.0)
  - Red flash when taking damage
  - Green glow when ready to reproduce
- **Rationale**: Visual feedback without clutter

**Q10.2.1 - Food Visualization**
- **Answer**: Green color gradient (darker = less food, brighter = more)
- **Rationale**: Clear, intuitive

**Q10.3.1 - Zoom Limits**
- **Answer**: 
  - Max zoom out: entire world visible
  - Max zoom in: 4x (individual organisms large)
- **Rationale**: Covers all useful viewing ranges

**Q10.3.2 - Pan Controls**
- **Answer**: 
  - WASD keyboard
  - Middle-mouse drag
  - Arrow keys
  - Auto-follow selected organism (F key)
- **Rationale**: Multiple input methods

---

### 12. User Interface (Phase 2-3)

**Q11.2.1 - HUD Statistics**
- **Answer**: Top bar shows:
  - Generation count (oldest organism's generation)
  - Population count / cap
  - Simulation tick
  - FPS (compute + render)
  - Speed multiplier
- **Rationale**: Essential info, minimal space

**Q11.3.1 - Inspector Details**
- **Answer**: Right panel when organism selected:
  - ID, age, energy (bar)
  - Parent ID, offspring count
  - Generation number
  - Current action outputs (bars)
  - Brain visualization (simple node diagram)
  - "Follow" and "Kill" buttons
- **Rationale**: Deep inspection for observation

**Q11.4.1 - Available Graphs**
- **Answer**: Phase 3, bottom panel:
  - Population over time (line)
  - Birth/death rate (line)
  - Average energy (line)
  - Window: last 5000 ticks
- **Rationale**: Track evolutionary progress

**Q11.5.1 - Exposed Settings**
- **Answer**: Pause menu:
  - Simulation speed (1x, 2x, 4x, 8x, uncapped)
  - Mutation rate (0.5x to 2x multiplier)
  - Food growth rate (0.5x to 2x)
  - Add/remove obstacles (click to paint)
  - Save/Load
- **Rationale**: Experiment-friendly

---

### 13. Performance Optimization (Phase 2-3)

**Q12.1.1 - Compute Pipeline Order**
- **Answer**: Single compute pass per tick:
  1. Spatial hash build (one kernel)
  2. Sense + Think + Act (fused kernel)
  3. Interact (organism-organism, organism-food)
  4. World update (food growth, cleanup)
- **Rationale**: Minimize dispatches, maximize GPU occupancy

**Q12.1.2 - FP32 or FP16**
- **Answer**: FP32 for all MVP
- **Rationale**: Sufficient VRAM budget, FP16 later if needed

**Q12.1.3 - Workgroup Size**
- **Answer**: 64 threads per workgroup
- **Rationale**: Balances occupancy and divergence

**Q12.2.2 - Dead Organism Slots**
- **Answer**: Free list in GPU buffer
  - Atomic counter for next free slot
  - Reuse immediately on death
- **Rationale**: No compaction needed, O(1) allocation

**Q12.3.1 - Out of VRAM Handling**
- **Answer**: 
  - Query limits at startup
  - Allocate conservatively (2K organisms × buffers = ~50 MB)
  - Error if can't allocate minimum
- **Rationale**: Fail fast, clear message

---

### 14. Save/Load System (Phase 3)

**Q13.1.1 - Save Data**
- **Answer**: Binary format with:
  - All organism states (pos, vel, energy, age, genome)
  - World food grid
  - RNG state
  - Simulation parameters (mutation rate, etc.)
  - Statistics history
- **Rationale**: Complete reproducibility

**Q13.1.2 - File Format**
- **Answer**: bincode (Rust binary serialization)
- **Rationale**: Fast, compact, versioned

**Q13.1.3 - Expected Size**
- **Answer**: 2K organisms ≈ 5 MB save file
- **Rationale**: Acceptable for disk

**Q13.2.2 - Autosave**
- **Answer**: Every 10,000 ticks (auto_save.bin)
- **Rationale**: Safety net, not intrusive

**Q13.3.1 - Reproducibility**
- **Answer**: Yes, if same seed and no user intervention
- **Rationale**: Valuable for debugging and science

---

### 15. Species Tracking (Phase 3)

**Q9.1.1 - Genome Similarity Metric**
- **Answer**: Euclidean distance in normalized weight space
  - Flatten all weights to vector
  - Normalize by weight count
  - Threshold = 5.0 (tunable)
- **Rationale**: Simple, effective

**Q9.1.3 - Speciation Frequency**
- **Answer**: Recompute every 500 ticks
- **Rationale**: Not critical for gameplay, expensive

**Q9.2.1 - Species ID Assignment**
- **Answer**: Incremental counter, never reuse
- **Rationale**: Clean history tracking

**Q9.2.4 - Species Metadata**
- **Answer**: Track per species:
  - Population count
  - First seen tick
  - Representative genome (founder)
  - Average energy
- **Rationale**: For visualization and stats

---

## 🟢 MEDIUM PRIORITY - Phase 4-5 (Enhancement)

### 16. Advanced Sensing (Phase 4)

**Q2.2.5 - Pheromone System**
- **Answer**: Phase 4, add 2 pheromone channels
  - Channel 0: arbitrary meaning (evolved)
  - Channel 1: arbitrary meaning (evolved)
  - Each organism can emit [0,1] on each channel (cost: 0.1 energy/unit)
- **Rationale**: Enables communication evolution

**Q6.3.3 - Pheromone Diffusion**
- **Answer**: Gaussian blur (3×3 kernel) every 5 ticks
  - Decay rate: 0.95 per tick
- **Rationale**: Smooth spread, trails fade

**Extended Sensory System (Phase 5)**:
- Vision: 12 rays, 120° FOV
- Add 2 pheromone inputs (sampled at position)
- Input dimension: 12×2 + 4 + 2 = **30 inputs**

---

### 17. Morphology Evolution (Phase 5)

**Q3.1.1 - Body Size Effects**
- **Answer**: Phase 5, add body_size gene [0.5, 2.0]
  - Affects: collision radius, visual size, food capacity (max energy)
  - Larger = more storage, but higher passive drain (0.1 × size)
- **Rationale**: Size-strategy tradeoff

**Q3.1.3 - Speed Factor**
- **Answer**: Add speed_gene [0.5, 2.0]
  - Max speed = 2.0 × speed_gene
  - Movement cost = base_cost × speed_gene²
- **Rationale**: Speed-efficiency tradeoff

**Q3.1.5 - Vision Range**
- **Answer**: Add vision_range gene [20, 80] pixels
- **Rationale**: Awareness-energy tradeoff

**Q3.3.3 - Diet Preference**
- **Answer**: Add diet gene [-1 herbivore, +1 carnivore]
  - Herbivore: 1.5× food energy, 0.5× predation energy
  - Carnivore: 0.5× food energy, 1.5× predation energy
  - Omnivore (0): 1.0× both
- **Rationale**: Niche specialization

---

### 18. Sexual Reproduction (Phase 5)

**Q7.3.1 - Mate Selection**
- **Answer**: Nearest organism within 10 pixels with:
  - Reproduce signal > 0.8
  - Genome similarity < threshold (prevents self-mating if clones nearby)
- **Rationale**: Simple proximity-based

**Q7.3.4 - Crossover Method**
- **Answer**: Uniform crossover (50/50 random per weight)
- **Rationale**: Preserves good combinations

**Q7.3.3 - Mutual Willingness**
- **Answer**: Both must signal willingness
- **Rationale**: Prevents forced reproduction

**Q7.3.5 - Energy Cost**
- **Answer**: Both parents pay 30 energy, offspring gets 50
- **Rationale**: Collaborative investment, some efficiency

---

### 19. Environmental Dynamics (Phase 5)

**Q6.5.1 - Temperature Effects**
- **Answer**: Phase 5, add temperature field [0, 1]
  - High temp (>0.7): 1.5× passive drain
  - Low temp (<0.3): 0.5× food growth
- **Rationale**: Creates climate niches

**Q6.5.2 - Day/Night Cycle**
- **Answer**: Phase 5, 2000-tick cycle
  - Night: 0.5× vision range, 0.5× food growth (plants rest)
- **Rationale**: Temporal niches (nocturnal/diurnal)

**Q6.5.4 - Disasters**
- **Answer**: Phase 5, user-triggerable:
  - Meteor: kills all in radius
  - Drought: sets food to 0 in region
  - Blessing: adds food in region
- **Rationale**: Test adaptation, fun experimentation

---

## 🔵 LOW PRIORITY - Phase 6+ (Polish & Advanced)

### 20. Topology Evolution (Phase 6)

**Q16.1.1 - NEAT-Style Evolution**
- **Answer**: Phase 6, optional advanced mode
  - Allow add/remove neuron mutations (1% chance)
  - Allow add/remove connection mutations (2% chance)
  - Variable genome sizes, pad to max in GPU buffer
- **Rationale**: Interesting but complex, not MVP

---

### 21. Learning Within Lifetime (Phase 6)

**Q16.2.1 - Hebbian Plasticity**
- **Answer**: Phase 6, experimental feature
  - Simple Hebbian: Δw = η × pre_activation × post_activation
  - Apply during organism lifetime
  - Learned weights saved in genome (Lamarckian)
- **Rationale**: Academic interest, may speed evolution

---

### 22. Audio (Phase 4)

**Q14.1.1 - Sound Design**
- **Answer**: Phase 4, minimal audio:
  - Ambient background (gentle synth)
  - Soft click on organism birth
  - UI sounds
  - No spatial audio (too complex)
- **Rationale**: Atmosphere without distraction

---

### 23. Accessibility (Phase 6)

**Q20.3.1 - Colorblind Mode**
- **Answer**: Phase 6, add species symbol shapes (not just color)
- **Rationale**: Inclusive but not critical

**Q20.3.2 - UI Scaling**
- **Answer**: Phase 4, support high DPI displays
- **Rationale**: Modern monitors, not hard to implement

---

### 24. Distribution (Phase 5)

**Q18.1.2 - Asset Embedding**
- **Answer**: Embed all assets in binary (include_bytes!)
- **Rationale**: Single-file distribution, no data folder

**Q18.2.1 - Platform Support**
- **Answer**: 
  - Phase 4: Windows x64
  - Phase 5: Linux x64
  - Phase 6: macOS (if resources permit)
- **Rationale**: Incremental platform expansion

**Q18.3.1 - Release Strategy**
- **Answer**: 
  - Phase 4: Closed alpha (friends, testers)
  - Phase 5: Public beta on itch.io (free)
  - Phase 6: Steam release (paid or free, TBD)
- **Rationale**: Iterate with feedback

---

## Performance Budget Summary (MVP Target)

### Hardware Target: GTX 1060 (6GB VRAM, ~4.4 TFLOPS)

| Component | Time Budget | Notes |
|-----------|-------------|-------|
| Sensory (raycasting) | 3 ms | 2K organisms × 8 rays |
| Neural forward pass | 2 ms | 2K × (20→16→6) network |
| Physics & movement | 1 ms | Simple velocity integration |
| Spatial interactions | 3 ms | Collision, eating, fighting |
| World update | 1 ms | Food growth |
| **Total Compute** | **10 ms** | **100 FPS sim rate** |
| Rendering | 6 ms | Instanced triangles |
| **Frame Time** | **16 ms** | **60 FPS target** |

### Memory Budget (2K Organisms)

| Buffer | Size | Notes |
|--------|------|-------|
| Organism states | 200 KB | Pos, vel, energy, etc. |
| Genomes | 3.5 MB | Neural weights |
| World grid (512²) | 1 MB | Food layer |
| Spatial hash | 256 KB | 16×16 cells |
| Rendering | 100 KB | Instance buffer |
| **Total** | **~5 MB** | Leaves ample headroom |

---

## Key Principles for Implementation

1. **Start Minimal**: Prove core loop (sense→think→act→evolve) with simplest systems
2. **Profile Early**: Measure GPU timings from day one
3. **Deterministic**: Fixed seed + no floating point errors = reproducible runs
4. **Observable**: Rich inspection tools reveal emergent behaviors
5. **Tunable**: Expose parameters for experimentation
6. **Scalable**: Architecture supports 2K → 100K organisms with hardware upgrades

---

## Phase 1 MVP Checklist (Minimal Viable Evolution)

- [ ] wgpu compute pipeline running
- [ ] 2K organisms with [20→16→6] brains
- [ ] 8-ray vision system
- [ ] Food growth and eating
- [ ] Movement and collision
- [ ] Asexual reproduction with mutation
- [ ] Energy-based survival
- [ ] Basic rendering (colored triangles)
- [ ] Camera pan/zoom
- [ ] Simple HUD (pop, FPS)
- [ ] **Success criterion**: Organisms evolve to seek food within 1 hour of sim time

**Estimated MVP Development Time**: 6-8 weeks (Phase 1 + Phase 2)

---

*This document provides concrete answers to enable immediate implementation. Adjust based on profiling and emergent behavior observations.*
