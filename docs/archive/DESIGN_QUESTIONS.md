# Evolution Simulator: Critical Design Questions

**Purpose**: This document lists all design questions that must be answered before implementation begins. These questions identify ambiguities, missing specifications, and design decisions that will significantly impact development.

**Created**: January 26, 2026  
**Based on**: DESIGN.md v0.2

---

## 1. Technology Stack & Platform Questions

### 1.1 GPU Backend Selection
- **Q1.1.1**: Which GPU backend will be the primary development target (Vulkan, Metal, or DX12)?
- **Q1.1.2**: Will fallback to CPU be supported if GPU is unavailable or incompatible?
- **Q1.1.3**: What is the minimum wgpu version requirement, and will we track upstream updates?
- **Q1.1.4**: How will we handle GPU driver compatibility issues across different vendors (NVIDIA, AMD, Intel)?
- **Q1.1.5**: Will WebGPU/WASM browser builds be a release target, or stretch goal only?
- **Q1.1.6**: What specific wgpu features and limits must be queried at startup (max buffer size, max compute workgroup size, etc.)?

### 1.2 Rust Ecosystem Choices
- **Q1.2.1**: Will we use an ECS (Entity Component System) library, and if so, which one (bevy_ecs, hecs, legion)?
- **Q1.2.2**: Should we consider using Bevy engine instead of raw wgpu, despite the overhead concerns?
- **Q1.2.3**: For async operations, will we use tokio, async-std, or stick with pollster for simplicity?
- **Q1.2.4**: What is the versioning and update policy for dependencies? Lock to specific versions or allow minor updates?
- **Q1.2.5**: How will we handle the audio system priority - implement from day one or defer to later phases?

### 1.3 Development Environment
- **Q1.3.1**: What CI/CD pipeline should be established (GitHub Actions, other)?
- **Q1.3.2**: What platforms will receive automated build testing (Windows, Mac, Linux)?
- **Q1.3.3**: What is the strategy for shader compilation and validation during development?
- **Q1.3.4**: Will we use hot-reloading for shaders during development?
- **Q1.3.5**: What debugging tools will be used for GPU compute debugging (RenderDoc, Nsight, etc.)?

---

## 2. Organism Neural Architecture Questions

### 2.1 Network Topology
- **Q2.1.1**: Is the neural network architecture fixed for all organisms, or can it vary per species/individual?
- **Q2.1.2**: If topology is fixed, what exactly are the layer sizes for each quality preset?
  - Low: 24-12-8 or different dimensions?
  - Medium: 32-16-8 or different?
  - High: 48-32-16 or different?
  - Ultra: 64-32-16 or different?
- **Q2.1.3**: What activation functions will be used for hidden layers (ReLU only, or also tanh, sigmoid, leaky ReLU)?
- **Q2.1.4**: What activation function for output layer (tanh for [-1,1], sigmoid for [0,1], or mixed)?
- **Q2.1.5**: Will bias terms be included in the genome, or weights only?
- **Q2.1.6**: How will weights be initialized for the first generation (Xavier/He initialization, random uniform, random normal)?

### 2.2 Sensory Inputs
- **Q2.2.1**: Exactly how many vision rays per organism in each quality preset?
- **Q2.2.2**: What is the precise format of vision data per ray?
  - Distance only?
  - Distance + RGB color?
  - Distance + entity type ID?
  - Distance + RGB + additional channels (entity type, food value)?
- **Q2.2.3**: How is distance encoded in vision rays?
  - Raw pixel distance?
  - Normalized [0,1] by max vision range?
  - Inverse distance (closer = stronger signal)?
- **Q2.2.4**: What internal state variables are exposed to the network?
  - Energy (normalized how?)
  - Age (absolute ticks, normalized by expected lifespan?)
  - Health (separate from energy, or same?)
  - Speed/velocity magnitude?
  - Recent damage taken?
  - Offspring count?
- **Q2.2.5**: How many pheromone channels will exist, and what do they represent?
- **Q2.2.6**: How is pheromone concentration sampled (single point at position, 3x3 grid average, max in radius)?
- **Q2.2.7**: What exactly is "touch/proximity" detecting?
  - Number of nearby organisms?
  - Distance to nearest organism?
  - Vector to nearest organism?
  - Separate channels for different species?

### 2.3 Action Outputs
- **Q2.3.1**: How exactly are movement outputs interpreted?
  - dx, dy as force/acceleration vs. direct velocity?
  - Movement in world coordinates or relative to organism rotation?
  - Is rotation a separate output, or derived from dx/dy?
- **Q2.3.2**: What is the maximum movement speed per tick (capped or unbounded)?
- **Q2.3.3**: How does the "rotation" output work?
  - Absolute angle in [0, 2π]?
  - Delta rotation in radians?
  - Angular velocity?
- **Q2.3.4**: Is "eat/attack intensity" a single unified output or two separate channels?
- **Q2.3.5**: How does eat intensity affect food consumption rate (linear, quadratic, capped)?
- **Q2.3.6**: How does attack intensity affect damage dealt to other organisms?
- **Q2.3.7**: What does "reproduce signal" do exactly?
  - Trigger immediate reproduction if energy threshold met?
  - Signal willingness to mate (for sexual reproduction)?
  - Both?
- **Q2.3.8**: How are pheromone emissions handled?
  - Two separate pheromone types the organism can emit?
  - Emission rate per output value?
  - Cost in energy per pheromone unit emitted?

### 2.4 Genome Storage
- **Q2.4.1**: How is the genome buffer organized in GPU memory?
  - One contiguous buffer with all weights interleaved?
  - Separate buffers per layer?
  - Struct of arrays vs. array of structs?
- **Q2.4.2**: What is the maximum genome size in floats/bytes?
- **Q2.4.3**: How are dead organism genome slots reclaimed?
- **Q2.4.4**: Will FP16 be used for genome weights to save memory, or FP32 only?
- **Q2.4.5**: Is there a minimum/maximum weight value enforced (weight clamping)?

---

## 3. Morphology and Physical Traits Questions

### 3.1 Body Properties
- **Q3.1.1**: How exactly does "body_size" affect the organism?
  - Collision radius?
  - Rendering size?
  - Food storage capacity?
  - Movement speed penalty?
  - All of the above?
- **Q3.1.2**: What is the range for body_size (0.5-2.0 as stated, but what's the starting value)?
- **Q3.1.3**: How does "speed_factor" interact with actual movement?
  - Multiplier on velocity outputs?
  - Affects energy cost of movement?
  - Affects maximum speed cap?
- **Q3.1.4**: Is there an energy cost for maintaining a larger body size (passive drain)?
- **Q3.1.5**: How does "vision_range" gene affect raycasting?
  - Maximum distance rays can travel?
  - Field of view angle?
  - Both?
- **Q3.1.6**: How many genes control morphology (fixed set or evolvable count)?

### 3.2 Color Genes
- **Q3.2.1**: Are color genes [R, G, B] purely cosmetic or functional?
- **Q3.2.2**: Can organisms see their own color vs. others' colors through vision rays?
- **Q3.2.3**: Should similar-colored organisms be treated as kin/species-mates (automatic recognition)?
- **Q3.2.4**: What color should food have, and is it distinguishable from organisms?
- **Q3.2.5**: How are obstacles/terrain rendered differently from organisms in vision?

### 3.3 Metabolic Genes
- **Q3.3.1**: How exactly does "energy_efficiency" work?
  - Multiplier on energy consumption rates?
  - Affects movement cost, passive drain, or both?
  - Linear or non-linear scaling?
- **Q3.3.2**: What is "reproduction_threshold"?
  - Minimum energy required to reproduce?
  - Energy fraction at which organism becomes willing to reproduce?
  - Different for asexual vs. sexual reproduction?
- **Q3.3.3**: What does "diet_preference" control?
  - Efficiency of digesting plants vs. meat?
  - Attraction/repulsion to food types?
  - Damage bonus when attacking?
  - How is it encoded (scalar [-1 herbivore, +1 carnivore], or vector)?
- **Q3.3.4**: Can organisms be omnivores, or is there a forced specialization?

---

## 4. Physics and Movement Questions

### 4.1 Movement Mechanics
- **Q4.1.1**: Is movement acceleration-based or velocity-based?
  - Do organisms have momentum/inertia?
  - Or does velocity directly set from neural outputs each frame?
- **Q4.1.2**: What is the physics timestep (same as simulation timestep, or sub-stepped)?
- **Q4.1.3**: Is rotation instant or does it have angular velocity/inertia?
- **Q4.1.4**: What is the maximum speed cap (if any)?
- **Q4.1.5**: How does friction/drag work?
  - Velocity decay per frame?
  - Energy cost per distance traveled?
  - No friction at all?
- **Q4.1.6**: Can organisms move backwards or only forward relative to rotation?

### 4.2 Collision System
- **Q4.2.1**: How are organism-organism collisions resolved?
  - Elastic bounce?
  - Inelastic (stop on contact)?
  - Push-apart with spring force?
  - No collision, allow overlap?
- **Q4.2.2**: How are organism-obstacle collisions handled?
  - Hard stop?
  - Slide along walls?
  - Bounce?
- **Q4.2.3**: What happens when organisms overlap?
  - Position correction (teleport apart)?
  - Damage from crushing?
  - Nothing (ignore small overlaps)?
- **Q4.2.4**: Is there a collision detection phase in the compute pipeline, or is it merged with interaction?
- **Q4.2.5**: Do larger organisms push smaller ones more easily (mass-based physics)?

### 4.3 Boundary Conditions
- **Q4.3.1**: What happens at world edges?
  - Hard walls (bounce/stop)?
  - Toroidal wrap-around?
  - Reflect?
- **Q4.3.2**: Can the world be infinite/unbounded, or must it have fixed size?
- **Q4.3.3**: How are world coordinates represented (integer grid indices, floating point, normalized [0,1])?

---

## 5. Energy and Metabolism Questions

### 5.1 Energy Mechanics
- **Q5.1.1**: What is the starting energy for newborn organisms?
  - Fixed value?
  - Inherited from parent (what percentage)?
  - Based on parent's reproduction threshold gene?
- **Q5.1.2**: What is the maximum energy an organism can store?
  - Fixed cap?
  - Scales with body size?
  - Unbounded?
- **Q5.1.3**: What is the passive energy drain per tick?
  - Fixed value?
  - Scales with body size?
  - Affected by energy_efficiency gene?
- **Q5.1.4**: What is the energy cost of movement?
  - Proportional to distance moved?
  - Proportional to velocity magnitude?
  - Proportional to acceleration (change in velocity)?
- **Q5.1.5**: What is the energy cost of reproduction?
  - Fixed value?
  - Percentage of parent's current energy?
  - Equals energy given to offspring?
- **Q5.1.6**: Is there an energy cost for neural computation (thinking)?
- **Q5.1.7**: Is there an energy cost for emitting pheromones?

### 5.2 Energy Gain
- **Q5.2.1**: How much energy does food provide?
  - Fixed value per food tile?
  - Variable based on food tile's current amount?
  - Affected by organism's diet_preference gene?
- **Q5.2.2**: How fast can organisms eat food?
  - Instant consumption?
  - Rate limited by eat intensity output?
  - Rate limited by body size?
- **Q5.2.3**: Can multiple organisms eat from the same food tile simultaneously (contention)?
- **Q5.2.4**: How much energy does an organism gain from eating another organism (predation)?
  - Full energy of prey?
  - Percentage (inefficiency)?
  - Affected by diet_preference?
- **Q5.2.5**: Do dead organisms become food (carrion)?

### 5.3 Death Conditions
- **Q5.3.1**: At what energy level does an organism die (0, negative, threshold)?
- **Q5.3.2**: Is there a maximum age before death (aging)?
- **Q5.3.3**: Can organisms die from damage/combat, separate from energy?
- **Q5.3.4**: What happens to dead organisms?
  - Immediate despawn?
  - Become food corpse?
  - Removed at end of tick vs. immediately?
- **Q5.3.5**: Is there a "minimum viable population" check to prevent total extinction?

---

## 6. World and Environment Questions

### 6.1 World Grid Structure
- **Q6.1.1**: What is the exact grid size for each quality preset (512², 1024², 2048², 4096²)?
- **Q6.1.2**: What is the relationship between grid cells and organism positions?
  - One grid cell = how many pixels in rendering?
  - Can organisms be smaller than one cell?
- **Q6.1.3**: How is the food grid updated?
  - Every frame?
  - Every N frames?
  - Asynchronously?
- **Q6.1.4**: Is food stored as integer count or floating-point amount?

### 6.2 Food System
- **Q6.2.1**: What is the food growth model?
  - Linear regrowth?
  - Logistic growth (carrying capacity)?
  - Diffusion from high-food areas?
  - Spawn at random locations?
- **Q6.2.2**: What is the food growth rate (units per tick per cell)?
- **Q6.2.3**: Is food growth uniform, or are there biomes with different rates?
- **Q6.2.4**: What is the maximum food per cell (saturation value)?
- **Q6.2.5**: Can food ever decay or disappear (besides being eaten)?
- **Q6.2.6**: How is initial food distributed?
  - Uniform random?
  - Clustered patches?
  - Biome-dependent?

### 6.3 Pheromones
- **Q6.3.1**: How many pheromone layers exist (2, 4, or configurable)?
- **Q6.3.2**: What do different pheromone channels represent?
  - Semantically meaningful (alarm, food trail, mating signal)?
  - Or generic (any meaning learned by evolution)?
- **Q6.3.3**: How do pheromones diffuse?
  - Gaussian blur?
  - Simple neighbor averaging?
  - Advection (flow with velocity field)?
  - Diffusion rate constant?
- **Q6.3.4**: What is the pheromone decay rate per tick?
- **Q6.3.5**: Is pheromone diffusion computed every frame or every N frames?
- **Q6.3.6**: What is the maximum pheromone concentration per cell?
- **Q6.3.7**: Can organisms distinguish pheromones by source (self vs. kin vs. other species)?

### 6.4 Terrain and Obstacles
- **Q6.4.1**: How is terrain represented?
  - Binary (passable/blocked)?
  - Height/elevation values?
  - Material types (water, rock, grass)?
- **Q6.4.2**: Can terrain change dynamically during simulation?
- **Q6.4.3**: Does terrain affect food growth (e.g., fertile vs. barren)?
- **Q6.4.4**: Does terrain affect organism movement speed?
- **Q6.4.5**: Can organisms see through obstacles, or do obstacles block vision rays?
- **Q6.4.6**: How are obstacles initially placed (user-defined, procedural generation)?

### 6.5 Environmental Effects
- **Q6.5.1**: What does "temperature" do?
  - Affects metabolism rate?
  - Required for reproduction?
  - Lethal outside certain range?
- **Q6.5.2**: How does "light_level" (day/night) affect organisms?
  - Affects vision range?
  - Affects food growth?
  - Affects energy consumption?
- **Q6.5.3**: Will there be seasonal cycles?
  - If yes, on what timescale (real-time, ticks, generations)?
  - What changes seasonally (food growth, temperature, light)?
- **Q6.5.4**: Will there be disasters/events (floods, droughts, meteors)?
  - If yes, are they random or user-triggered?

---

## 7. Reproduction and Genetics Questions

### 7.1 Reproduction Triggers
- **Q7.1.1**: When exactly does reproduction occur?
  - When organism's energy exceeds threshold AND reproduce signal is active?
  - Automatically when energy exceeds threshold?
  - Only when reproduce signal is high?
- **Q7.1.2**: Is reproduction synchronous (all at specific intervals) or asynchronous (any time)?
- **Q7.1.3**: Can organisms reproduce multiple times per lifetime?
- **Q7.1.4**: Is there a cooldown period after reproduction?
- **Q7.1.5**: Is there a minimum age requirement before reproduction?

### 7.2 Asexual Reproduction
- **Q7.2.1**: How is the offspring's position determined?
  - Same as parent?
  - Random offset from parent?
  - Directional (behind or in front of parent)?
- **Q7.2.2**: Does asexual reproduction clone the genome exactly (before mutation)?
- **Q7.2.3**: What is the energy split between parent and offspring?
  - 50/50?
  - Parent keeps most (90/10)?
  - Configurable?

### 7.3 Sexual Reproduction
- **Q7.3.1**: How is a mate selected?
  - Nearest organism of same species?
  - Nearest organism with compatible genome similarity?
  - Based on color similarity?
  - First organism within range with high reproduce signal?
- **Q7.3.2**: What is the maximum distance for mate selection?
- **Q7.3.3**: Must both organisms signal willingness to reproduce, or just one?
- **Q7.3.4**: How is crossover performed?
  - Single-point crossover on flattened genome?
  - Uniform crossover (random per weight)?
  - Layer-wise crossover?
  - Arithmetic mean of parent weights?
- **Q7.3.5**: Do both parents pay energy cost, or just one?
- **Q7.3.6**: Does offspring appear at midpoint between parents, or at one parent's location?
- **Q7.3.7**: Can an organism mate with multiple partners per reproduction cycle?

### 7.4 Mutation
- **Q7.4.1**: What is the mutation rate?
  - Probability per weight?
  - Probability per organism (all-or-nothing)?
  - Configurable per simulation?
- **Q7.4.2**: What is the mutation strength (standard deviation of noise added)?
- **Q7.4.3**: Is mutation Gaussian noise or uniform noise?
- **Q7.4.4**: Do morphology genes mutate at the same rate as neural weights?
- **Q7.4.5**: Should there be a minimum/maximum for morphology genes after mutation (clamping)?
- **Q7.4.6**: Can mutation add/remove neurons or connections (topology evolution), or only weight mutation?
- **Q7.4.7**: Is mutation rate itself evolvable?

### 7.5 Population Management
- **Q7.5.1**: What is the maximum population cap?
  - Hard limit (no reproduction when reached)?
  - Soft limit (oldest/weakest die)?
  - Dynamic based on available VRAM?
- **Q7.5.2**: When population hits max, which organisms are culled?
  - Oldest?
  - Lowest energy?
  - Random?
  - None (stop reproduction)?
- **Q7.5.3**: What is the minimum starting population for a simulation?
- **Q7.5.4**: How are offspring inserted into GPU buffers?
  - Reuse slots of recently dead organisms?
  - Append to end (defragment later)?
  - Maintain free list?
- **Q7.5.5**: Is there a target steady-state population the simulation should converge to?

---

## 8. Interactions and Combat Questions

### 8.1 Predation Mechanics
- **Q8.1.1**: How does predation work?
  - Attacker must be within certain distance AND activate attack output?
  - Automatic if carnivore touches herbivore?
  - Requires sustained contact over multiple ticks?
- **Q8.1.2**: How much damage does an attack deal per tick?
  - Fixed value?
  - Based on attacker's attack intensity output?
  - Based on attacker's body size?
  - Based on attacker's diet_preference (carnivore bonus)?
- **Q8.1.3**: Where does damage go?
  - Directly reduces energy?
  - Reduces separate "health" stat?
  - Both?
- **Q8.1.4**: Can prey defend itself or fight back?
- **Q8.1.5**: Is there a concept of "hunting" vs. "scavenging" (dead organisms)?
- **Q8.1.6**: Can herbivores harm/eat carnivores, or is it one-directional?

### 8.2 Spatial Interactions
- **Q8.2.1**: What is the spatial hash cell size?
  - Equal to organism vision range?
  - Fixed (e.g., 32px)?
  - Dynamic based on organism density?
- **Q8.2.2**: How are organisms assigned to spatial hash cells?
  - Every frame?
  - Every N frames?
  - Only when organism moves significantly?
- **Q8.2.3**: How many organisms per spatial cell on average (expected density)?
- **Q8.2.4**: What is the maximum number of organisms per cell (for fixed buffer sizes)?
- **Q8.2.5**: How are interactions between nearby organisms resolved?
  - All pairs in 3x3 cell region?
  - Only organisms within body size + interaction radius?
  - Limit to K nearest neighbors?

### 8.3 Food Gathering
- **Q8.3.1**: Does an organism need to be stopped to eat, or can it eat while moving?
- **Q8.3.2**: Can organisms eat food from adjacent cells, or only the cell they're in?
- **Q8.3.3**: What happens if multiple organisms eat from the same cell?
  - First-come-first-served?
  - Split food proportionally?
  - Largest organism gets priority?
- **Q8.3.4**: Is there a biting/eating animation delay, or is it instant?

---

## 9. Species and Lineage Questions

### 9.1 Species Definition
- **Q9.1.1**: How is genome similarity computed for speciation?
  - Euclidean distance in flattened weight space?
  - Cosine similarity?
  - Only compare network structure (if topology varies)?
- **Q9.1.2**: What is the speciation threshold distance?
  - Fixed value?
  - Relative to population diversity?
  - User-configurable?
- **Q9.1.3**: How often is speciation recomputed?
  - Every generation?
  - Every N ticks?
  - Only when user requests?
- **Q9.1.4**: Can species merge back together if genomes converge?
- **Q9.1.5**: Is speciation displayed in real-time or computed on-demand?

### 9.2 Species Tracking
- **Q9.2.1**: How are species assigned unique IDs?
  - Incremental counter?
  - Hash of representative genome?
  - Based on common ancestor?
- **Q9.2.2**: What happens to species ID when all members die (extinction)?
  - ID retired?
  - Can be reused?
  - Marked as extinct but kept in history?
- **Q9.2.3**: Should species be color-coded automatically, or can user assign colors?
- **Q9.2.4**: What metadata is tracked per species?
  - Population count?
  - Age (time since divergence)?
  - Average fitness?
  - Representative genome?
  - Ancestral species?

### 9.3 Lineage Tracking
- **Q9.3.1**: Will each organism store a reference to its parent(s)?
- **Q9.3.2**: Will there be a global ancestry tree structure?
  - If yes, in CPU memory or GPU?
  - How deep (max generations back)?
- **Q9.3.3**: Can the user view an organism's family tree?
- **Q9.3.4**: Should there be a "phylogenetic tree" visualization?
- **Q9.3.5**: Is lineage data saved with simulation state?

---

## 10. Rendering and Visualization Questions

### 10.1 Organism Rendering
- **Q10.1.1**: How are organisms rendered?
  - Simple colored circles?
  - Triangles/arrows showing direction?
  - Sprites with animation?
  - Procedurally generated shapes based on genome?
- **Q10.1.2**: Does organism rendering size match physical collision size exactly?
- **Q10.1.3**: Are organisms rendered with transparency/alpha blending?
- **Q10.1.4**: Should there be a visual indicator for organism state?
  - Energy level (brightness, glow)?
  - Health (cracks, damage)?
  - Reproductive readiness (pulsing)?
  - Species (color)?
- **Q10.1.5**: Should there be a visual indicator when an organism is eating or attacking?
- **Q10.1.6**: What level of detail (LOD) system is needed?
  - Full detail always?
  - Distant organisms as dots?
  - Cull offscreen organisms?

### 10.2 World Rendering
- **Q10.2.1**: How is the food layer visualized?
  - Color gradient (sparse=dark, abundant=bright)?
  - Discrete food items?
  - Not visible (organisms see it, player doesn't)?
- **Q10.2.2**: How are pheromones visualized?
  - Overlay with transparency?
  - Heat map?
  - Toggleable layers?
  - Not visible?
- **Q10.2.3**: How is terrain rendered?
  - Solid color per cell?
  - Textured?
  - Height-based shading?
- **Q10.2.4**: Should there be a grid overlay option?
- **Q10.2.5**: What is the background color/texture?
- **Q10.2.6**: Should environment temperature/light be visualized?

### 10.3 Camera System
- **Q10.3.1**: What are the zoom limits (min/max)?
  - Can zoom out to see entire world?
  - Can zoom in to individual organism detail?
- **Q10.3.2**: How does pan work?
  - WASD keys?
  - Click and drag?
  - Edge-of-screen scrolling?
  - Follow selected organism automatically?
- **Q10.3.3**: Is the camera projection orthographic (2D) or perspective (2.5D)?
- **Q10.3.4**: Should there be camera presets or bookmarks?
- **Q10.3.5**: Should the camera follow interesting events (e.g., first predator kill)?

### 10.4 Performance and Quality
- **Q10.4.1**: What is the rendering resolution (independent of window size)?
- **Q10.4.2**: Should rendering support high DPI displays?
- **Q10.4.3**: Is VSync enforced, optional, or disabled?
- **Q10.4.4**: What anti-aliasing method, if any (MSAA, FXAA, none)?
- **Q10.4.5**: Should there be separate quality settings for simulation vs. rendering?

---

## 11. User Interface Questions

### 11.1 Main Menu
- **Q11.1.1**: What exactly is on the "New Simulation" screen?
  - World size dropdown?
  - Sliders for all parameters?
  - Preset buttons (easy/medium/hard)?
  - Advanced options collapsed by default?
- **Q11.1.2**: Can the user load simulation from command line (e.g., `evolution.exe --load save.bin`)?
- **Q11.1.3**: Will there be a "recent simulations" list?
- **Q11.1.4**: Should the main menu show a live background simulation?

### 11.2 HUD Design
- **Q11.2.1**: What statistics are shown in the HUD?
  - Current generation number?
  - Simulation time (ticks, real time)?
  - FPS (compute and render)?
  - Total population?
  - Species count?
  - Average energy?
  - Birth/death rate?
- **Q11.2.2**: Are HUD elements draggable/rearrangeable?
- **Q11.2.3**: Can the HUD be hidden completely (for screenshots/videos)?
- **Q11.2.4**: Should the HUD have a minimap?

### 11.3 Inspector Panel
- **Q11.3.1**: What details are shown for a selected organism?
  - Full neural network weights?
  - Visual brain diagram (nodes and connections)?
  - Sensory inputs in real-time?
  - Action outputs in real-time?
  - Morphology genes?
  - Ancestry (parents, offspring count)?
  - Statistics (kills, distance traveled, energy gathered)?
- **Q11.3.2**: Can the user view multiple organisms simultaneously (compare)?
- **Q11.3.3**: Can the user manually kill/remove a selected organism?
- **Q11.3.4**: Can the user clone/spawn organisms with specific genomes?
- **Q11.3.5**: Can the user edit an organism's genome in real-time?

### 11.4 Graph/Statistics Panel
- **Q11.4.1**: What graphs are available?
  - Population over time (line chart)?
  - Species breakdown (stacked area)?
  - Energy distribution histogram?
  - Genome diversity metric over time?
  - Birth/death rates?
- **Q11.4.2**: What is the time window for graphs (last 1000 ticks, all history, configurable)?
- **Q11.4.3**: Can graphs be exported (CSV, PNG)?
- **Q11.4.4**: Should statistics be updated every frame, or on a slower cadence for performance?

### 11.5 Settings Menu
- **Q11.5.1**: What settings are exposed to the user?
  - Simulation speed multiplier?
  - Mutation rate?
  - Reproduction threshold?
  - Food growth rate?
  - Max population?
  - GPU compute workgroup size?
  - Rendering quality?
  - Audio volume?
- **Q11.5.2**: Can settings be changed mid-simulation, or only at startup?
- **Q11.5.3**: Should there be separate presets (beginner/standard/harsh)?
- **Q11.5.4**: Are settings saved per simulation or globally?

---

## 12. Performance and Optimization Questions

### 12.1 Compute Pipeline
- **Q12.1.1**: What is the exact order of compute shader dispatches per frame?
  - Sense → Think → Act → Interact → World?
  - Can any be merged or parallelized further?
- **Q12.1.2**: Should compute shaders use FP32 or FP16 for computations?
- **Q12.1.3**: What is the workgroup size for each shader (64, 128, 256)?
  - Tuned per GPU family?
  - Fixed for compatibility?
- **Q12.1.4**: Are there memory barriers needed between shader passes?
- **Q12.1.5**: Can multiple frames of simulation be batched in one GPU submission?
- **Q12.1.6**: Should there be a fixed timestep with interpolation, or variable timestep?

### 12.2 Memory Management
- **Q12.2.1**: How are GPU buffers allocated?
  - Pre-allocated to max organism count?
  - Grown dynamically?
  - Memory pools?
- **Q12.2.2**: When organisms die, are their slots immediately reused or defragmented later?
- **Q12.2.3**: Is there a compaction pass to defragment GPU buffers?
  - If yes, how often?
  - Is the simulation paused during compaction?
- **Q12.2.4**: What CPU-side memory is needed (statistics, UI state)?
- **Q12.2.5**: How often is data read back from GPU to CPU?
  - Every frame (for display)?
  - Only when user requests stats?
  - Async readback?

### 12.3 Scalability
- **Q12.3.1**: What happens if the user's GPU runs out of VRAM?
  - Reduce organism count?
  - Reduce world size?
  - Show error and refuse to start?
  - Fallback to CPU?
- **Q12.3.2**: Can the simulation adapt quality settings automatically based on FPS?
- **Q12.3.3**: Should there be a profiler to identify bottlenecks (e.g., raycasting vs. neural pass)?
- **Q12.3.4**: What is the priority for optimization (latency vs. throughput)?
- **Q12.3.5**: Will there be support for multi-GPU systems?

### 12.4 Frame Budgets
- **Q12.4.1**: What happens if a frame takes longer than 16.67ms?
  - Drop frames?
  - Slow down simulation?
  - Skip rendering but continue simulation?
- **Q12.4.2**: Should simulation and rendering be on separate threads?
- **Q12.4.3**: Is there a max frame time cap to prevent "spiral of death"?

---

## 13. Save/Load System Questions

### 13.1 Save Format
- **Q13.1.1**: What data is saved?
  - All organism states (positions, energy, genomes)?
  - World state (food, pheromones, terrain)?
  - Statistics history?
  - RNG seed/state?
  - Simulation parameters?
- **Q13.1.2**: What file format?
  - Binary (bincode, custom)?
  - JSON (for human readability)?
  - Compressed?
- **Q13.1.3**: What is the expected save file size for 10K organisms?
- **Q13.1.4**: Can save files be loaded on different platforms (cross-platform compatibility)?
- **Q13.1.5**: How is version compatibility handled (format changes)?

### 13.2 Save/Load UI
- **Q13.2.1**: Can the user save at any time, or only when paused?
- **Q13.2.2**: Are there autosaves?
  - If yes, how often?
  - How many autosave slots?
- **Q13.2.3**: Can the user name saves, or are they timestamp-based?
- **Q13.2.4**: Is there a "checkpoint" system for reverting to earlier states?
- **Q13.2.5**: Can the user export specific organisms or genomes separately?

### 13.3 Reproducibility
- **Q13.3.1**: Can simulations be exactly replayed from a seed?
- **Q13.3.2**: Is GPU randomness deterministic across runs?
- **Q13.3.3**: Should there be a record/playback feature for demonstrations?

---

## 14. Audio and Feedback Questions

### 14.1 Sound Design
- **Q14.1.1**: What sounds will the game have?
  - Ambient background music?
  - Organism birth/death sounds?
  - Eating sounds?
  - Attack sounds?
  - UI clicks?
  - Notification for interesting events?
- **Q14.1.2**: Should audio be spatial (positional) based on camera location?
- **Q14.1.3**: Should audio volume scale with organism density (busy areas are louder)?
- **Q14.1.4**: Is audio essential for MVP, or can it be deferred?

### 14.2 Visual Feedback
- **Q14.2.1**: Should there be visual effects for events?
  - Particle effects for reproduction?
  - Blood/impact for predation?
  - Glow for high-energy organisms?
  - Trails for movement (fading line)?
- **Q14.2.2**: Should there be notifications for milestones?
  - "First predator evolved!"
  - "New species detected!"
  - "Population milestone reached!"
- **Q14.2.3**: Should the camera shake or flash for major events?

---

## 15. Evolutionary Dynamics Questions

### 15.1 Selection Pressure
- **Q15.1.1**: Is fitness implicit (survival-based), or should there be explicit fitness scoring?
- **Q15.1.2**: What behaviors should be advantageous?
  - Efficient movement (less energy waste)?
  - Fast reproduction?
  - Avoidance of predators?
  - Cooperative behavior?
- **Q15.1.3**: Should there be artificial selection options (user breeds organisms)?

### 15.2 Expected Timescales
- **Q15.2.1**: How many ticks per generation (average organism lifespan)?
- **Q15.2.2**: How many generations to expect meaningful evolution?
  - Basic foraging: X generations?
  - Predator-prey: Y generations?
  - Specialization: Z generations?
- **Q15.2.3**: What is considered a "successful" run (does not collapse or stagnate)?

### 15.3 Evolvability Concerns
- **Q15.3.1**: Is the neural architecture expressive enough for complex behaviors?
- **Q15.3.2**: Are mutation rates tuned to allow exploration without destroying good genomes?
- **Q15.3.3**: Can evolution get "stuck" in local optima?
  - If yes, are there mechanisms to escape (mutation bursts, catastrophes)?
- **Q15.3.4**: Should there be multiple isolated populations (islands) to encourage diversity?

---

## 16. Advanced Features Questions

### 16.1 Topology Evolution (NEAT-like)
- **Q16.1.1**: Will neurons and connections be added/removed dynamically?
- **Q16.1.2**: If yes, how is this represented in GPU memory (variable-sized genomes)?
- **Q16.1.3**: How is compatibility distance computed with different topologies?
- **Q16.1.4**: Is topology evolution essential for MVP or a stretch goal?

### 16.2 Learning Within Lifetime
- **Q16.2.1**: Should organisms learn via Hebbian plasticity or similar?
- **Q16.2.2**: Are weight updates applied during an organism's lifetime (not just via evolution)?
- **Q16.2.3**: Is learned knowledge passed to offspring (Lamarckian evolution)?
- **Q16.2.4**: Is this feature planned for MVP or later phases?

### 16.3 Social Behaviors
- **Q16.3.1**: Can organisms form groups or swarms?
- **Q16.3.2**: Is there kin recognition (prefer mates/cooperation with relatives)?
- **Q16.3.3**: Can organisms communicate beyond pheromones (direct signals)?
- **Q16.3.4**: Are these emergent or should there be explicit mechanics?

### 16.4 User Intervention
- **Q16.4.1**: Can the user place/remove organisms manually?
- **Q16.4.2**: Can the user add/remove food manually?
- **Q16.4.3**: Can the user trigger disasters or environmental changes?
- **Q16.4.4**: Can the user "paint" terrain or walls?
- **Q16.4.5**: Should there be a "scenario editor" for custom challenges?

---

## 17. Testing and Validation Questions

### 17.1 Correctness Testing
- **Q17.1.1**: How will neural network forward pass correctness be validated?
  - Compare CPU reference implementation?
  - Unit tests for each shader?
- **Q17.1.2**: How will physics correctness be tested?
  - Known initial conditions with expected outcomes?
  - Energy conservation checks?
- **Q17.1.3**: How will reproduction/mutation correctness be verified?
- **Q17.1.4**: Should there be a deterministic test mode (fixed seed, reproducible)?

### 17.2 Performance Testing
- **Q17.2.1**: What benchmarks will be used?
  - Time to simulate X ticks with Y organisms?
  - FPS under different quality settings?
- **Q17.2.2**: What hardware will be tested (GPU models)?
- **Q17.2.3**: Should there be automated performance regression tests?

### 17.3 Emergent Behavior Testing
- **Q17.3.1**: How will we verify that evolution is actually happening?
  - Track fitness proxies (average energy, lifespan)?
  - Compare early vs. late generation genomes?
  - Observe behavior changes?
- **Q17.3.2**: What constitutes a "failed" simulation?
  - Total extinction?
  - Stagnation (no evolution)?
  - Collapse to degenerate behavior?
- **Q17.3.3**: Should there be automated checks for interesting behaviors?

---

## 18. Distribution and Release Questions

### 18.1 Packaging
- **Q18.1.1**: What is the target binary size for the executable?
- **Q18.1.2**: Will assets be embedded in the binary or external files?
- **Q18.1.3**: Should there be an installer or just a zip archive?
- **Q18.1.4**: What license for the game (proprietary, open source, MIT, GPL)?

### 18.2 Platform Support
- **Q18.2.1**: Which platforms will be supported at launch?
  - Windows (x64)?
  - macOS (Intel and/or Apple Silicon)?
  - Linux (which distros)?
- **Q18.2.2**: Will there be a browser build (WebGPU/WASM)?
  - If yes, what are the performance expectations?
- **Q18.2.3**: Are there any platform-specific features or limitations?

### 18.3 Release Strategy
- **Q18.3.1**: Will there be an alpha/beta testing phase?
- **Q18.3.2**: What platforms for distribution (Steam, itch.io, GitHub releases)?
- **Q18.3.3**: Will the game be free, paid, or freemium?
- **Q18.3.4**: Will source code be open or closed?
- **Q18.3.5**: What kind of user feedback collection (analytics, surveys)?

---

## 19. Documentation and Onboarding Questions

### 19.1 User Documentation
- **Q19.1.1**: Will there be a tutorial mode for new players?
  - Interactive guided experience?
  - Video tutorials?
  - Text tooltips?
- **Q19.1.2**: What concepts must be explained to users?
  - How evolution works?
  - What each UI element does?
  - How to interpret statistics?
- **Q19.1.3**: Should there be a user manual or wiki?
- **Q19.1.4**: Will the game explain emergent behaviors when detected?

### 19.2 Developer Documentation
- **Q19.2.1**: What documentation is needed for contributors?
  - Architecture overview?
  - Shader documentation?
  - How to add new features?
- **Q19.2.2**: Should there be inline code comments or external docs?
- **Q19.2.3**: Will there be a modding API or extension system?

---

## 20. Miscellaneous Implementation Questions

### 20.1 Random Number Generation
- **Q20.1.1**: How are random numbers generated on the GPU?
  - PCG, Xorshift, other?
  - Shared seed per frame or per organism?
- **Q20.1.2**: Is there a CPU-side RNG for non-GPU operations?
- **Q20.1.3**: How is the global RNG state managed and updated?

### 20.2 Debugging and Development Tools
- **Q20.2.1**: Should there be a debug mode with extra visualizations?
  - Vision rays drawn?
  - Spatial hash grid overlay?
  - Neural activations displayed?
- **Q20.2.2**: Should there be a command console for dev commands?
- **Q20.2.3**: Will there be logging (if yes, to file or console)?
- **Q20.2.4**: Should shader hot-reloading be supported during development?

### 20.3 Accessibility
- **Q20.3.1**: Should there be colorblind-friendly modes?
- **Q20.3.2**: Should text be scalable for readability?
- **Q20.3.3**: Are there keyboard-only controls for accessibility?

### 20.4 Internationalization
- **Q20.4.1**: Will the UI support multiple languages?
- **Q20.4.2**: If yes, which languages are priority?
- **Q20.4.3**: How are translations managed (external files, embedded)?

---

## 21. Risk and Dependency Questions

### 21.1 Technical Risks
- **Q21.1.1**: What if wgpu has breaking API changes during development?
- **Q21.1.2**: What if a target platform (e.g., macOS) has GPU compatibility issues?
- **Q21.1.3**: What if performance targets cannot be met on recommended hardware?
- **Q21.1.4**: What if emergent behaviors do not appear (evolution "fails")?

### 21.2 External Dependencies
- **Q21.2.1**: What is the update policy for Rust crate dependencies?
  - Lock to specific versions?
  - Update regularly?
- **Q21.2.2**: Are there critical dependencies that could block development if unmaintained?
- **Q21.2.3**: Should there be fallback implementations for key dependencies?

### 21.3 Scope Management
- **Q21.3.1**: What is the MVP (Minimum Viable Product) scope?
- **Q21.3.2**: What features can be cut if timeline is at risk?
- **Q21.3.3**: What is the definition of "done" for each development phase?
- **Q21.3.4**: How will scope creep be managed?

---

## 22. Meta Questions (About This Document)

- **Q22.1**: Are there any topics missing from this question list?
- **Q22.2**: Which questions are highest priority to answer first?
- **Q22.3**: Should these questions be answered in a separate document or inline in the design doc?
- **Q22.4**: Who is responsible for answering each category of questions?
- **Q22.5**: What is the deadline for answering these questions before implementation begins?

---

## Summary: Question Categories by Priority

### 🔴 Critical (Must Answer Before Any Code)
- Neural network architecture exact dimensions (Section 2)
- Energy mechanics (Section 5)
- Reproduction triggers and mechanics (Section 7)
- Rust + wgpu tech stack validation (Section 1)
- MVP scope definition (Section 21.3)

### 🟡 High Priority (Must Answer Before Implementation Phase)
- Physics and collision system (Section 4)
- World grid structure and food system (Section 6)
- Rendering approach (Section 10)
- Spatial interaction system (Section 8)
- Performance budgets and scalability (Section 12)

### 🟢 Medium Priority (Can Decide During Development)
- UI/UX details (Section 11)
- Advanced features (topology evolution, learning) (Section 16)
- Save/load format (Section 13)
- Audio design (Section 14)
- Testing strategy (Section 17)

### 🔵 Low Priority (Can Defer to Polish Phase)
- Accessibility features (Section 20.3)
- Internationalization (Section 20.4)
- Platform-specific optimizations (Section 18)
- Documentation and tutorials (Section 19)

---

**Total Question Count**: 250+ distinct design questions identified

**Next Steps**:
1. Review this document and identify any missing question categories
2. Prioritize which questions must be answered immediately vs. iteratively
3. Assign each question to be answered by design lead, technical lead, or team decision
4. Create a "Design Decisions" document with answers to these questions
5. Use answered questions as the specification for implementation

