//! Application state and main game loop

use std::sync::Arc;
use std::path::PathBuf;
use anyhow::Result;
use glam::Vec2;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::config::SimulationConfig;
use crate::simulation::{Simulation, SaveState};
use crate::simulation::genome::{Genome, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM};
use crate::compute::{ComputePipeline, ExecutionTiming};
use crate::render::Renderer;
use crate::ui::Ui;

/// Application state
#[allow(dead_code)]
pub struct App {
    // GPU state
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    
    // Window reference
    window: Arc<Window>,
    
    // Subsystems
    simulation: Simulation,
    compute: ComputePipeline,
    renderer: Renderer,
    ui: Ui,
    
    // State
    config: SimulationConfig,
    paused: bool,
    speed_multiplier: u32,
    tick: u64,
    
    // Camera
    camera_pos: Vec2,
    camera_zoom: f32,
    
    // Selection
    selected_organism: Option<u32>,
    follow_selected: bool,
    
    // Input state
    cursor_pos: Vec2,
    dragging: bool,
    drag_start: Vec2,
    
    // Timing
    last_frame_time: std::time::Instant,
    frame_times: Vec<f32>,
    last_timing: ExecutionTiming,
    start_time: std::time::Instant,
    
    // Debug: auto-exit after N seconds (0 = disabled)
    auto_exit_seconds: u64,
    
    // Statistics tracking
    births_this_tick: u32,
    deaths_this_tick: u32,
}

#[allow(dead_code)]
impl App {
    /// Create App with configuration and CLI options
    pub async fn new_with_config(
        window: Arc<Window>,
        mut config: SimulationConfig,
        auto_exit_seconds: u64,
        start_paused: bool,
        speed_multiplier: u32,
    ) -> Result<Self> {
        config.sanitize();
        let size = window.inner_size();
        
        // Initialize wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(window.clone())?;
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable GPU adapter"))?;
        
        log::info!("Using GPU: {}", adapter.get_info().name);
        
        // Request device with required features
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Main Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;
        
        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);
        
        // Initialize subsystems
        let simulation = Simulation::new(&config);
        let compute = ComputePipeline::new(&device, &config, &simulation)?;
        let renderer = Renderer::new(&device, &queue, surface_format, &config)?;
        let ui = Ui::new(&device, surface_format, window.clone());
        
        log::info!(
            "Initialized with {} organisms in {}x{} world",
            config.population.initial_organisms,
            config.world.width,
            config.world.height
        );
        
        Ok(Self {
            surface,
            device,
            queue,
            surface_config,
            
            window,
            
            simulation,
            compute,
            renderer,
            ui,
            
            camera_pos: Vec2::new(
                config.world.width as f32 / 2.0,
                config.world.height as f32 / 2.0,
            ),
            camera_zoom: 1.0,
            
            selected_organism: None,
            follow_selected: false,
            
            config,
            paused: start_paused,
            speed_multiplier: speed_multiplier.max(1),
            tick: 0,
            
            cursor_pos: Vec2::ZERO,
            dragging: false,
            drag_start: Vec2::ZERO,
            
            last_frame_time: std::time::Instant::now(),
            frame_times: Vec::with_capacity(60),
            last_timing: ExecutionTiming::default(),
            start_time: std::time::Instant::now(),
            
            auto_exit_seconds,
            
            births_this_tick: 0,
            deaths_this_tick: 0,
        })
    }
    
    /// Create App with default configuration (legacy constructor)
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        Self::new_with_config(window, SimulationConfig::default(), 0, false, 1).await
    }
    
    /// Check if app should auto-exit (returns true if should exit)
    pub fn should_exit(&self) -> bool {
        if self.auto_exit_seconds > 0 {
            self.start_time.elapsed().as_secs() >= self.auto_exit_seconds
        } else {
            false
        }
    }
    
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.renderer.resize(&self.device, new_size);
        }
    }
    
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        self.ui.handle_event(event)
    }
    
    pub fn handle_keyboard(&mut self, event: &KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        
        match event.physical_key {
            PhysicalKey::Code(KeyCode::Space) => {
                self.paused = !self.paused;
                log::info!("Simulation {}", if self.paused { "paused" } else { "resumed" });
            }
            PhysicalKey::Code(KeyCode::Period) => {
                // Single step (when paused)
                if self.paused {
                    self.simulation_step();
                    log::info!("Single step: tick {}", self.tick);
                }
            }
            PhysicalKey::Code(KeyCode::Escape) => {
                // Toggle settings menu (if open, close it; if closed, open it)
                // Deselect organism if settings menu is not open
                if self.ui.settings_open() {
                    self.ui.toggle_settings();
                } else if self.selected_organism.is_some() {
                    self.selected_organism = None;
                    self.follow_selected = false;
                } else {
                    self.ui.toggle_settings();
                    if !self.paused {
                        self.paused = true;
                        log::info!("Simulation paused (settings opened)");
                    }
                }
            }
            PhysicalKey::Code(KeyCode::Digit1) => self.speed_multiplier = 1,
            PhysicalKey::Code(KeyCode::Digit2) => self.speed_multiplier = 2,
            PhysicalKey::Code(KeyCode::Digit3) => self.speed_multiplier = 4,
            PhysicalKey::Code(KeyCode::Digit4) => self.speed_multiplier = 8,
            PhysicalKey::Code(KeyCode::Digit5) => self.speed_multiplier = 16,
            PhysicalKey::Code(KeyCode::Digit6) => self.speed_multiplier = 32,
            PhysicalKey::Code(KeyCode::Digit7) => self.speed_multiplier = 64,
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.camera_pos.y -= 20.0 / self.camera_zoom;
                self.follow_selected = false;
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.camera_pos.y += 20.0 / self.camera_zoom;
                self.follow_selected = false;
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.camera_pos.x -= 20.0 / self.camera_zoom;
                self.follow_selected = false;
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.camera_pos.x += 20.0 / self.camera_zoom;
                self.follow_selected = false;
            }
            PhysicalKey::Code(KeyCode::KeyR) => {
                self.reset_camera();
                self.follow_selected = false;
            }
            PhysicalKey::Code(KeyCode::KeyF) => {
                // Toggle follow mode for selected organism
                if self.selected_organism.is_some() {
                    self.follow_selected = !self.follow_selected;
                    log::info!("Follow mode: {}", if self.follow_selected { "ON" } else { "OFF" });
                } else {
                    // No selection - follow first alive organism (legacy behavior)
                    if let Some(org) = self.simulation.organisms.iter().find(|o| o.is_alive()) {
                        self.camera_pos = org.position;
                    }
                }
            }
            PhysicalKey::Code(KeyCode::KeyE) => {
                // Feed selected organism
                self.feed_selected_organism();
            }
            PhysicalKey::Code(KeyCode::KeyK) => {
                // Kill selected organism
                if let Some(org_id) = self.selected_organism {
                    self.kill_organism(org_id);
                }
            }
            PhysicalKey::Code(KeyCode::F5) => {
                // Quick save
                self.quick_save();
            }
            PhysicalKey::Code(KeyCode::F6) => {
                self.export_survivor_bank(false);
            }
            PhysicalKey::Code(KeyCode::F9) => {
                // Quick load
                self.quick_load();
            }
            PhysicalKey::Code(KeyCode::KeyI) => {
                // Toggle inspector panel
                self.ui.toggle_inspector();
            }
            PhysicalKey::Code(KeyCode::KeyH) => {
                // Toggle help overlay
                self.ui.toggle_help();
            }
            PhysicalKey::Code(KeyCode::KeyO) => {
                self.ui.toggle_founders();
            }
            _ => {}
        }
    }
    
    pub fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
        };
        
        // Scroll up = zoom in (larger zoom value)
        let zoom_factor = 1.1_f32.powf(scroll);
        self.camera_zoom = (self.camera_zoom * zoom_factor).clamp(0.1, 10.0);
    }
    
    pub fn handle_key_release(&mut self, _key: PhysicalKey) {
        // Reserved for future key release handling
    }
    
    pub fn reset_camera(&mut self) {
        self.camera_pos = Vec2::new(
            self.config.world.width as f32 / 2.0,
            self.config.world.height as f32 / 2.0,
        );
        self.camera_zoom = 1.0;
    }
    
    /// Quick save to default path
    fn quick_save(&self) {
        let path = self.get_save_path("quicksave.bin");
        let state = self.simulation.to_save_state(self.tick, &self.config);
        
        match state.save_to_file(&path) {
            Ok(_) => log::info!("Quick saved to {:?}", path),
            Err(e) => log::error!("Failed to quick save: {}", e),
        }
    }
    
    /// Quick load from default path
    fn quick_load(&mut self) {
        let path = self.get_save_path("quicksave.bin");
        
        match SaveState::load_from_file(&path) {
            Ok(state) => {
                // Restore simulation state
                self.tick = state.tick;
                self.config = state.config.clone();
                self.simulation = Simulation::from_save_state(&state);
                
                // Resync GPU buffers with loaded state
                self.sync_gpu_after_load();
                
                log::info!("Quick loaded from {:?} (tick {})", path, self.tick);
            }
            Err(e) => log::error!("Failed to quick load: {}", e),
        }
    }

    pub fn persist_survivor_bank_on_exit(&self) {
        self.export_survivor_bank(true);
    }

    fn export_survivor_bank(&self, shutdown: bool) {
        if !self.config.bootstrap.enabled || (shutdown && !self.config.bootstrap.save_on_exit) {
            return;
        }

        let Some(bank) = self
            .simulation
            .to_survivor_bank(self.tick, self.config.bootstrap.survivor_count as usize)
        else {
            if !shutdown {
                log::warn!("No living organisms available for survivor-bank export");
            }
            return;
        };

        let path = self.config.bootstrap.path.clone();
        let should_write = match crate::simulation::save_load::load_bootstrap_quality_score(&path) {
            Ok(existing_score) => {
                if bank.quality_score() > existing_score {
                    true
                } else {
                    log::info!(
                        "Skipped bootstrap export to {:?}: existing store is stronger (existing_score={:.1}, new_score={:.1})",
                        path,
                        existing_score,
                        bank.quality_score()
                    );
                    false
                }
            }
            Err(_) => true,
        };

        if should_write {
            match crate::simulation::save_load::save_bootstrap_bank(
                &path,
                &bank,
                "runtime_export",
                "Runtime-exported founders selected from living organisms",
            ) {
                Ok(_) => {
                    if shutdown {
                        log::info!("Persisted bootstrap founders to {:?}", path);
                    } else {
                        log::info!("Exported bootstrap founders to {:?}", path);
                    }
                }
                Err(error) => log::error!("Failed to save survivor bank to {:?}: {}", path, error),
            }
        }
    }
    
    /// Get path for save files (in current directory)
    fn get_save_path(&self, filename: &str) -> PathBuf {
        PathBuf::from(filename)
    }
    
    /// Sync all GPU buffers after loading a save
    fn sync_gpu_after_load(&mut self) {
        // Update organism count
        self.compute.buffers.set_organism_count(self.simulation.organisms.count());
        
        // Update all organisms to GPU
        let gpu_organisms = self.simulation.organisms.to_gpu_buffer();
        self.queue.write_buffer(&self.compute.buffers.organisms, 0, bytemuck::cast_slice(&gpu_organisms));
        
        // Update all genome weights to GPU
        let nn_weights = self.simulation.genomes.nn_weights_buffer();
        self.compute.buffers.update_nn_weights(&self.queue, &nn_weights);
        
        // Update food grid to GPU
        self.compute.buffers.update_food(&self.queue, &self.simulation.world.food);
        
        // Deselect organism (may no longer exist)
        self.selected_organism = None;
        self.follow_selected = false;
        
        log::info!("Synced GPU buffers after load");
    }
    
    /// Convert screen coordinates to world coordinates
    fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        let viewport = Vec2::new(
            self.surface_config.width as f32,
            self.surface_config.height as f32,
        );
        let center = viewport / 2.0;
        
        // Offset from screen center
        let offset = Vec2::new(
            (screen_pos.x - center.x) / self.camera_zoom,
            (center.y - screen_pos.y) / self.camera_zoom, // Flip Y
        );
        
        self.camera_pos + offset
    }
    
    /// Find organism at world position (within selection radius)
    fn find_organism_at(&self, world_pos: Vec2) -> Option<u32> {
        let selection_radius = self.config.physics.organism_radius * 2.0;
        let radius_sq = selection_radius * selection_radius;
        
        // Find closest organism within radius
        let mut closest: Option<(u32, f32)> = None;
        
        for (idx, org) in self.simulation.organisms.iter().enumerate() {
            if !org.is_alive() {
                continue;
            }
            
            let dist_sq = (org.position - world_pos).length_squared();
            if dist_sq < radius_sq {
                if closest.is_none() || dist_sq < closest.unwrap().1 {
                    closest = Some((idx as u32, dist_sq));
                }
            }
        }
        
        closest.map(|(idx, _)| idx)
    }
    
    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Middle => {
                self.dragging = state == ElementState::Pressed;
                if self.dragging {
                    self.drag_start = self.cursor_pos;
                    self.follow_selected = false; // Stop following when dragging
                }
            }
            MouseButton::Left if state == ElementState::Pressed => {
                // Select organism at cursor position
                let world_pos = self.screen_to_world(self.cursor_pos);
                if let Some(org_id) = self.find_organism_at(world_pos) {
                    self.selected_organism = Some(org_id);
                    log::info!("Selected organism #{}", org_id);
                }
            }
            MouseButton::Right if state == ElementState::Pressed => {
                // Right-click: Kill organism at cursor or spawn food
                let world_pos = self.screen_to_world(self.cursor_pos);
                if let Some(org_id) = self.find_organism_at(world_pos) {
                    // Kill organism
                    self.kill_organism(org_id);
                } else {
                    // Spawn food at cursor
                    self.spawn_food_at(world_pos);
                }
            }
            _ => {}
        }
    }
    
    /// Kill an organism by ID
    fn kill_organism(&mut self, org_id: u32) {
        if let Some(org) = self.simulation.organisms.get_mut(org_id) {
            if org.is_alive() {
                let species_id = org.species_id; // Capture before death
                org.energy = 0.0;
                org.alive = false;
                log::info!("Killed organism #{}", org_id);
                
                // Track death in species manager
                self.simulation.species_manager.on_organism_death(species_id);
                
                // Sync death to GPU
                let org_gpu = org.to_gpu();
                self.compute.buffers.update_organism_at(&self.queue, org_id, &org_gpu);
                
                // Clear selection if killed organism was selected
                if self.selected_organism == Some(org_id) {
                    self.selected_organism = None;
                    self.follow_selected = false;
                }
            }
        }
    }
    
    /// Spawn food at world position
    fn spawn_food_at(&mut self, world_pos: Vec2) {
        // Calculate cell coordinates
        let cell_x = (world_pos.x as i32).clamp(0, self.config.world.width as i32 - 1) as usize;
        let cell_y = (world_pos.y as i32).clamp(0, self.config.world.height as i32 - 1) as usize;
        let idx = cell_y * self.config.world.width as usize + cell_x;
        
        // Add food (up to max)
        let max_food = self.config.food.max_per_cell as f32;
        let current = self.simulation.world.food[idx];
        let new_amount = (current + 5.0).min(max_food); // Add 5 food per click
        self.simulation.world.food[idx] = new_amount;
        
        log::info!("Spawned food at ({}, {}): {:.1} -> {:.1}", cell_x, cell_y, current, new_amount);
        
        // Sync food to GPU
        self.compute.buffers.update_food(&self.queue, &self.simulation.world.food);
    }
    
    /// Feed the selected organism
    fn feed_selected_organism(&mut self) {
        if let Some(org_id) = self.selected_organism {
            if let Some(org) = self.simulation.organisms.get_mut(org_id) {
                if org.is_alive() {
                    let old_energy = org.energy;
                    org.energy = (org.energy + 20.0).min(self.config.energy.maximum);
                    log::info!("Fed organism #{}: {:.1} -> {:.1}", org_id, old_energy, org.energy);
                    
                    // Sync to GPU
                    let org_gpu = org.to_gpu();
                    self.compute.buffers.update_organism_at(&self.queue, org_id, &org_gpu);
                }
            }
        }
    }
    
    pub fn handle_cursor_move(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let new_pos = Vec2::new(position.x as f32, position.y as f32);
        
        if self.dragging {
            let delta = (new_pos - self.cursor_pos) / self.camera_zoom;
            // Move the camera opposite to the drag on both axes so the world tracks the cursor consistently.
            self.camera_pos.x -= delta.x;
            self.camera_pos.y -= delta.y;
        }
        
        self.cursor_pos = new_pos;
    }
    
    pub fn update(&mut self) {
        // Calculate frame time
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        
        // Track FPS
        self.frame_times.push(dt);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        
        // Run simulation steps
        if !self.paused {
            for _ in 0..self.speed_multiplier {
                self.simulation_step();
            }
        }
        
        // Handle follow mode - update camera to track selected organism
        if self.follow_selected {
            if let Some(org_id) = self.selected_organism {
                if let Some(org) = self.simulation.organisms.get(org_id) {
                    if org.is_alive() {
                        self.camera_pos = org.position;
                    } else {
                        // Selected organism died - stop following
                        self.follow_selected = false;
                    }
                }
            }
        }
        
        // Check if selected organism is still valid
        if let Some(org_id) = self.selected_organism {
            if let Some(org) = self.simulation.organisms.get(org_id) {
                if !org.is_alive() {
                    // Keep selection but mark as dead (inspector will show "dead" status)
                }
            }
        }
    }
    
    fn simulation_step(&mut self) {
        // 1. Readback results from PREVIOUS tick
        // This updates the CPU state (simulation.organisms) with data from the GPU.
        // Track population before readback to count deaths
        let pop_before = self.simulation.organism_count();
        
        let readback_ms = self.compute.read_gpu_state(
            &self.device, 
            &mut self.simulation, 
            &self.config,
            self.tick as u32
        );
        
        // Count deaths from GPU readback (energy depleted on GPU)
        let pop_after_readback = self.simulation.organism_count();
        self.deaths_this_tick = pop_before.saturating_sub(pop_after_readback);

        // 2. Handle reproduction on CPU (using the fresh CPU state)
        // This modifies simulation.organisms (CPU) and generates GPU updates.
        // Returns changes that need to be synced to GPU
        let result = self.simulation.handle_reproduction(&self.config);
        
        // Track births
        self.births_this_tick = result.new_genome_ids.len() as u32;
        
        // Sync new genome weights to GPU (CRITICAL: must be done before organisms can use them)
        for genome_id in &result.new_genome_ids {
            if let Some(weights) = self.simulation.genomes.get_weights_flat(*genome_id) {
                self.compute.buffers.update_nn_weights_for_genome(&self.queue, *genome_id, &weights);
            }
        }
        
        // Sync changed organisms to GPU (parents + new spawns)
        for (idx, org_gpu) in &result.organism_changes {
            self.compute.buffers.update_organism_at(&self.queue, *idx, org_gpu);
        }
        
        // 2b. Update species assignments periodically
        // This recalculates species clusters based on genetic distance
        self.simulation.update_species();
        
        // Update GPU organism count if organisms were spawned or died
        let current_count = self.simulation.organism_count();
        self.compute.buffers.set_organism_count(current_count);
        
        // 3. Dispatch GPU compute for CURRENT tick
        // This will snapshot the state (including reproduction changes) for the NEXT readback.
        let mut timing = self.compute.dispatch(
            &self.device,
            &self.queue,
            &self.simulation,
            &self.config,
            self.tick as u32,
        );

        // Update timing structs
        timing.readback_ms = readback_ms;
        self.last_timing = timing;
        
        self.tick += 1;
    }
    
    /// Build SelectedOrganism data for inspector
    fn get_selected_organism_data(&self) -> Option<crate::ui::SelectedOrganism> {
        let org_id = self.selected_organism?;
        let org = self.simulation.organisms.get(org_id)?;
        let (nn_inputs, nn_outputs) = self.compute_neural_snapshot(org_id, org.genome_id);
        
        Some(crate::ui::SelectedOrganism {
            id: org_id,
            alive: org.is_alive(),
            position: org.position.into(),
            rotation: org.rotation,
            energy: org.energy,
            age: org.age,
            generation: org.generation,
            offspring_count: org.offspring_count,
            parent_id: org.parent_id,
            reproduce_signal: org.reproduce_signal,
            genome_id: org.genome_id,
            species_id: org.species_id,
            nn_inputs,
            nn_outputs,
        })
    }

    fn compute_neural_snapshot(&self, org_id: u32, genome_id: u32) -> ([f32; INPUT_DIM], [f32; OUTPUT_DIM]) {
        let Some(genome) = self.simulation.genomes.get(genome_id) else {
            return ([0.0; INPUT_DIM], [0.0; OUTPUT_DIM]);
        };

        let inputs = self.compute_sensory_snapshot(org_id);
        let outputs = self.forward_pass_snapshot(genome, &inputs);
        (inputs, outputs)
    }

    fn compute_sensory_snapshot(&self, org_id: u32) -> [f32; INPUT_DIM] {
        let mut sensory = [0.0; INPUT_DIM];
        let Some(org) = self.simulation.organisms.get(org_id) else {
            return sensory;
        };

        let fov = self.config.vision.fov_degrees.to_radians();
        let half_fov = fov / 2.0;
        let effective_vision_range = self.config.vision.range * org.morph_vision_mult;
        let width = self.config.world.width as f32;
        let height = self.config.world.height as f32;

        for ray in 0..8usize {
            let t = ray as f32 / 7.0;
            let angle_offset = -half_fov + t * fov;
            let ray_angle = org.rotation + angle_offset;
            let ray_dir = Vec2::new(ray_angle.cos(), ray_angle.sin());

            let mut hit_dist = 0.0;
            let mut hit_type = 0.0;
            let mut distance = 1.0;
            while distance < effective_vision_range {
                let sample_pos = org.position + ray_dir * distance;
                let wrap_x = sample_pos.x.rem_euclid(width);
                let wrap_y = sample_pos.y.rem_euclid(height);
                let grid_x = wrap_x as u32;
                let grid_y = wrap_y as u32;
                let grid_idx = (grid_y * self.config.world.width + grid_x) as usize;

                if self.simulation.world.obstacles[grid_idx] != 0 {
                    hit_dist = 1.0 - (distance / effective_vision_range);
                    hit_type = 0.25;
                    break;
                }

                if self.simulation.world.food[grid_idx] > 0.5 {
                    hit_dist = 1.0 - (distance / effective_vision_range);
                    hit_type = 0.5;
                    break;
                }

                distance += 1.0;
            }

            sensory[ray * 2] = hit_dist;
            sensory[ray * 2 + 1] = hit_type;
        }

        sensory[16] = org.energy / self.config.energy.maximum;
        sensory[17] = (org.age as f32 / 1000.0).min(1.0);
        sensory[18] = org.velocity.length() / self.config.physics.max_speed;

        let organism_count = self.simulation.organism_count();
        let check_stride = (organism_count / 64).max(1);
        let mut nearest_dist = self.config.vision.range;
        let mut nearest_angle = 0.0;
        let mut other_idx = 0;

        while other_idx < organism_count {
            if other_idx != org_id {
                if let Some(other) = self.simulation.organisms.get(other_idx) {
                    if other.is_alive() {
                        let mut delta = other.position - org.position;
                        if delta.x > width * 0.5 {
                            delta.x -= width;
                        }
                        if delta.x < -width * 0.5 {
                            delta.x += width;
                        }
                        if delta.y > height * 0.5 {
                            delta.y -= height;
                        }
                        if delta.y < -height * 0.5 {
                            delta.y += height;
                        }

                        let dist = delta.length();
                        if dist < self.config.vision.range && dist < nearest_dist {
                            nearest_dist = dist;
                            nearest_angle = delta.y.atan2(delta.x) - org.rotation;
                        }
                    }
                }
            }

            other_idx += check_stride;
        }

        if nearest_dist < self.config.vision.range {
            let mut angle_norm = nearest_angle / std::f32::consts::PI;
            if angle_norm > 1.0 {
                angle_norm -= 2.0;
            }
            if angle_norm < -1.0 {
                angle_norm += 2.0;
            }
            sensory[19] = angle_norm;
        } else {
            sensory[19] = 1.0;
        }

        sensory
    }

    fn forward_pass_snapshot(&self, genome: &Genome, sensory: &[f32; INPUT_DIM]) -> [f32; OUTPUT_DIM] {
        let mut hidden = [0.0; HIDDEN_DIM];
        for hidden_idx in 0..HIDDEN_DIM {
            let mut sum = genome.biases_l1[hidden_idx];
            for input_idx in 0..INPUT_DIM {
                let weight_idx = input_idx * HIDDEN_DIM + hidden_idx;
                sum += sensory[input_idx] * genome.weights_l1[weight_idx];
            }
            hidden[hidden_idx] = sum.max(0.0);
        }

        let mut output = [0.0; OUTPUT_DIM];
        for output_idx in 0..OUTPUT_DIM {
            let mut sum = genome.biases_l2[output_idx];
            for hidden_idx in 0..HIDDEN_DIM {
                let weight_idx = hidden_idx * OUTPUT_DIM + output_idx;
                sum += hidden[hidden_idx] * genome.weights_l2[weight_idx];
            }
            output[output_idx] = sum.tanh();
        }

        output
    }
    
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Calculate FPS
        let avg_frame_time: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
        let fps = if avg_frame_time > 0.0 { 1.0 / avg_frame_time } else { 0.0 };
        
        // Prepare UI
        let ui_data = crate::ui::UiData {
            population: self.simulation.organism_count(),
            max_population: self.config.population.max_organisms,
            tick: self.tick,
            fps,
            paused: self.paused,
            speed: self.speed_multiplier,
            avg_energy: self.simulation.avg_energy(),
            max_generation: self.simulation.max_generation(),
            total_food: self.simulation.total_food(),
            species_count: self.simulation.species_count(),
            readback_ms: self.last_timing.readback_ms,
            upload_ms: self.last_timing.upload_ms,
            submit_ms: self.last_timing.submit_ms,
            compute_ms: self.last_timing.total_ms,
            births: self.births_this_tick,
            deaths: self.deaths_this_tick,
        };
        
        // Get selected organism data for inspector
        let selected_data = self.get_selected_organism_data();
        let selected_genome = self.selected_organism
            .and_then(|id| self.simulation.organisms.get(id))
            .map(|org| org.genome_id)
            .and_then(|genome_id| self.simulation.genomes.get(genome_id));
        
        // Update renderer buffers
        self.renderer.update_camera(
            &self.queue,
            self.camera_pos,
            self.camera_zoom,
            (self.surface_config.width, self.surface_config.height),
            (self.config.world.width, self.config.world.height),
            self.config.food.max_per_cell,
        );
        self.renderer.update_organisms(&self.queue, &self.simulation);
        
        // Update selected organism highlight
        self.renderer.set_selected_organism(self.selected_organism);
        
        // Copy food buffer from GPU compute to render texture
        self.renderer.copy_food_from_buffer(
            &mut encoder,
            &self.compute.buffers.food,
            (self.config.world.width, self.config.world.height),
        );
        
        // Render world and organisms
        self.renderer.render(
            &mut encoder,
            &view,
            &self.simulation,
            self.camera_pos,
            self.camera_zoom,
            (self.surface_config.width, self.surface_config.height),
        );
        
        // Render UI
        self.ui.render(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            &ui_data,
            &mut self.config,
            selected_data.as_ref(),
            selected_genome,
            (self.surface_config.width, self.surface_config.height),
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }
    
    pub fn fps(&self) -> f32 {
        let avg: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
        if avg > 0.0 { 1.0 / avg } else { 0.0 }
    }
}
