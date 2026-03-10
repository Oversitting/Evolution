# Evolution Simulator: Project Plan

**Project**: GPU-Accelerated Evolution Simulator  
**Version**: 1.7  
**Last Updated**: February 2026  
**Status**: Phase 6 Partial Complete - Morphology, Sexual Reproduction, Biomes Implemented  

---

## 1. Project Overview

A GPU-accelerated 2D evolution simulator where digital organisms with neural network brains live, compete, reproduce, and evolve. The simulation runs entirely on the GPU using wgpu compute shaders for maximum parallelism.

### Core Philosophy
- **Performance first**: Maximize simulation throughput on the GPU.
- **Preserve emergence**: Design rules that allow complex behaviors to evolve naturally.
- **Scalable**: Support 2K → 100K organisms.
- **Interactive**: Allow users to observe, inspect, and influence the simulation.

---

## 2. Roadmap Overview

| Phase | Name | Status | Duration | Key Deliverable |
|-------|------|--------|----------|-----------------|
| **1** | MVP Foundation | ✅ Complete | 4 weeks | Working evolution, GPU pipeline, basic rendering |
| **2** | Core Polish | ✅ Complete | 3 weeks | UI, inspection, save/load, time controls |
| **3** | Rich Simulation | ✅ Complete | 3-4 weeks | Analytics, interaction tools, obstacles, coloring |
| **4** | **Predation & Species** | ✅ Complete | 3-4 weeks | Attack/defense, species tracking, dynamic environments |
| **5** | Game Features | 📅 Planned | 2-3 weeks | Audio, main menu, user tools |
| **6** | Content & Scale | � Partial | 3-4 weeks | Morphology, sexual reproduction, biomes (✅), pheromones (⏳) |
| **7** | Advanced | 📅 Planned | Ongoing | Topology evolution, learning, 3D |

---

## 3. Completed: Phase 3 (Rich Simulation)

**Goal**: Add analytics, user interaction, and visual polish to make the simulation more engaging and informative.

### 3.1 Statistics & Analytics ✅ COMPLETE
- [x] **Stats Panel**: Toggle with button in HUD, shows collapsible graphs
- [x] **Population Graph**: Rolling history chart of organism count over time
- [x] **Generation Graph**: Track max generation progression
- [x] **Birth/Death Counters**: Track events per tick, displayed as dual-line graph
- [x] **Energy Graph**: Average energy over time
- [x] **Food Graph**: Total food in world over time

### 3.2 User Interaction Tools ✅ COMPLETE
- [x] **Kill Organism**: Right-click on organism OR press K with selection
- [x] **Feed Organism**: Press E to add 20 energy to selected organism
- [x] **Spawn Food**: Right-click on empty space to add food at cursor
- [x] **HUD Help**: Updated controls display with new keybinds

### 3.3 Visual Enhancements ✅ COMPLETE
- [x] **Generation Coloring**: Higher generations have more saturated, warmer colors
- [x] **Energy Visualization**: Brightness modulated by energy level (in shader)
- [x] **Selection Highlight**: Bright cyan color for selected organism

### 3.4 Obstacles (Medium Priority) - DEFERRED
- [ ] **Static Obstacles**: Place blocking regions in the world
- [ ] **Vision Blocking**: Rays stop at obstacles (already partially implemented)
- [ ] **Obstacle Brushes**: Paint obstacles in settings/editor mode

*Obstacles deferred to Phase 5 (Game Features) as they require more design work.*

### 3.5 Deferred to Phase 4
The following complex features are moved to Phase 4 for proper implementation:
- **Spatial Hash Grid**: GPU acceleration for O(1) neighbor queries
- **Predation System**: Attack neural output, damage, energy transfer
- **Species Clustering**: Genome similarity analysis and grouping

---

## 4. Upcoming Phases

### Phase 4: Predation & Species ✅ COMPLETE
- [~] **Spatial Hash Grid**: Deferred - current brute force acceptable at 2K scale
- [x] **Attack Output**: Add neural network output for "attack"
- [x] **Damage Logic**: Deal damage based on attack + proximity
- [x] **Energy Transfer**: Killer gains % of victim's energy (capped to victim's actual energy)
- [x] **Species Detection**: Genetic distance algorithm (Euclidean on sampled NN weights)
- [x] **Species Coloring**: Color by species cluster ID (golden ratio hue spread)
- [x] **Species Inspector**: Show species ID in organism inspector panel
- [x] **Death Tracking**: Species member counts updated when organisms die

### Phase 4.5: UI Polish ✅ COMPLETE
- [x] **Theme System**: Consistent styling across all panels (ui/theme.rs)
- [x] **Toolbar**: Quick-access buttons in top-right corner
- [x] **Help Overlay**: Press H to show all keybinds
- [x] **Inspector Toggle**: Press I to show/hide inspector panel
- [x] **Graph Improvements**: Hover values on stat graphs
- [x] **Cleaner HUD**: Simplified HUD with tooltips on metrics

### Phase 4.6: Dynamic Environments ✅ COMPLETE
- [x] **Seasonal Cycles**: Food growth oscillates over time (configurable period/amplitude)
- [x] **Resource Hotspots**: High-value food zones that drift across the world
- [x] **Config Integration**: All dynamic features toggleable in Settings panel

### Phase 5: Game Features (Game Loop Polish)
- [ ] **Main Menu**: Start/Load/Settings screens
- [ ] **Audio**: Ambient music, procedural sound effects (birth/eat/die)
- [ ] **Tutorial/Guide**: Interactive introduction to the simulation
- [ ] **God Mode Tools**: Brush (paint food/obstacles), Smite, Spawn

### Phase 6: Content & Scale (Complexity) 🔄 IN PROGRESS
- [x] **Morphology**: Evolvable traits for size, speed multiplier, vision range, metabolism
  - Affects physics (movement speed, energy costs)
  - Affects rendering (organism size)
  - Inheritable with mutation
- [x] **Sexual Reproduction**: Crossover of genomes, mate finding
  - Uniform crossover with configurable ratio
  - Mate range and signal requirements
  - Optional (can be disabled for asexual-only)
- [x] **Biomes**: Voronoi-based regional environments
  - 5 biome types: Normal, Fertile, Barren, Swamp, Harsh
  - Fertile: 2x food growth rate
  - Barren: 0.25x food growth rate
  - Swamp: 0.6x movement speed
  - Harsh: 1.5x energy drain
- [ ] **Pheromones**: Scent trails for social insects behavior (ants/bees)
- [~] **Spatial Hash Grid**: Deferred - brute force acceptable at 2K scale, complex GPU implementation
- [ ] **Scale**: Optimizations to reach 10,000+ organisms at 60 FPS

### Phase 7: Experimental / Advanced
- [ ] **NEAT / Topology Evolution**: Add/remove neurons and connections dynamically
- [ ] **Lifetime Learning**: Hebbian learning (brain changes during life)
- [ ] **3D Visualization**: Render the 2D world in 3D perspective
- [ ] **Web Export**: Compile to WASM/WebGPU for browser play

---

## 5. Feature Wishlist & Ideas
*Potential features to consider for future phases.*

### Simulation Depth
- **Day/Night Cycle**: Varying visibility and temperature.
- **Fluids**: Water currents that push organisms.
- **Disease**: Viral agents that spread on contact.
- **Carry Items**: Ability to pick up and move food or objects.
- **Communication**: Ability to signal other organisms (color change, sound).

### User Experience
- **Organism Library**: Save favorite genomes to a persistent library to spawn in other worlds.
- **Tournament Mode**: Place two saved species in an arena to battle.
- **Map Editor**: Draw custom maps with walls and food zones.
- **Video Export**: Record high-quality video of the simulation.
- **Headless Training**: Run simulation at max speed without rendering to "train" a species, then watch the replay.

### Analytics
- **Phylogenetic Tree**: Visual tree of all species ancestry.
- **Heatmaps**: Death locations, movement paths, food hotspots.
- **Genome Diff**: Visual comparison of two neural networks.

---

## 6. Completed Milestones (Archive)

### Phase 1: MVP Foundation (Completed Jan 2026)
- **Core Loop**: Sense → Think → Act pipeline working on GPU.
- **Evolution**: Natural selection functioning; food seeking behavior evolves.
- **Performance**: 60+ FPS with 2,000 organisms.
- **Infrastructure**: Config system, build profiles, debugging tools.

### Phase 2: Core Polish (Completed Jan 2026)
- **Inspection**: Click to select, brain visualization, stats panel.
- **Time Control**: Pause, fast forward (up to 64x), step-by-step.
- **Save/Load**: Full simulation state serialization.
- **Settings**: Runtime configuration of all simulation parameters.
- **Stability Fix**: Solved reproduction race condition via strictly ordered GPU readback/dispatch cycle.
- **Determinism**: Full reproducibility with seed-based RNG (see [DETERMINISM.md](DETERMINISM.md)).

---

## 7. Technical Reference

### File Structure
```
src/
├── main.rs              # Entry point
├── app.rs               # App state machine
├── config.rs            # Configuration structs
├── simulation/          # CPU-side logic (Organism, World, Genome)
├── compute/             # GPU compute (Pipeline, Buffers, Shaders)
├── render/              # GPU rendering (Camera, Instance rendering)
└── ui/                  # Egui interface (Inspector, HUD, Menu)
```

### Key Documentation
- [DESIGN.md](DESIGN.md): Technical architecture and data structures.
- [DETERMINISM.md](DETERMINISM.md): Reproducibility guarantees and testing.
- [DEBUGGING.md](DEBUGGING.md): Troubleshooting guide.
- [BUILD_OPTIMIZATION.md](BUILD_OPTIMIZATION.md): Performance tips.

### Risk Log
- **GPU Compatibility**: Compute shader support varies by hardware. Use `wgpu` limits carefully.
- **Simulation Stability**: Population explosion or extinction. Mitigated by `crowding_factor`, `food.effectiveness`, and logistic growth.
- **Complexity**: Spaghetti code in shaders. Mitigated by strict module boundaries and documentation.

### Known Issues (Phase 4)
- **Predation Race Condition**: Multiple attackers targeting the same victim in a single tick may cause minor energy accounting inconsistencies. Acceptable at current scale; documented in DESIGN.md.
- **No Attack Cooldown**: Organisms can attack every tick. May add in future if balance issues arise.
- **Representative Genome Invalidation**: When a species representative organism dies, the genome reference may become stale until next recalculation (every 60 ticks).
- **RNG State Not Serialized**: Save/load doesn't preserve exact RNG state, so loaded simulations may diverge from original trajectory.
