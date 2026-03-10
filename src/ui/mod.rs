//! UI module using egui

mod inspector;
mod stats;
pub mod theme;

pub use inspector::SelectedOrganism;
pub use stats::StatsHistory;

use egui::Pos2;
use std::sync::Arc;

/// Data displayed in the UI
#[derive(Clone, Debug, Default)]
pub struct UiData {
    /// Current simulation tick
    pub tick: u64,
    
    /// Frames per second
    pub fps: f32,
    
    /// Current population count
    pub population: u32,
    
    /// Maximum population
    pub max_population: u32,
    
    /// Is simulation paused?
    pub paused: bool,
    
    /// Simulation speed multiplier
    pub speed: u32,
    
    /// Average energy of alive organisms
    pub avg_energy: f32,
    
    /// Maximum generation reached
    pub max_generation: u32,
    
    /// Total food in world
    pub total_food: f32,
    
    /// Number of distinct species
    pub species_count: usize,
    
    /// Births this tick (for stats tracking)
    pub births: u32,
    
    /// Deaths this tick (for stats tracking)  
    pub deaths: u32,
    
    /// Profiling: readback time in ms
    pub readback_ms: f32,
    
    /// Profiling: upload time in ms
    pub upload_ms: f32,
    
    /// Profiling: submit time in ms
    pub submit_ms: f32,
    
    /// Profiling: total compute time in ms
    pub compute_ms: f32,
}

/// UI state and rendering using egui
pub struct Ui {
    /// egui state
    egui_state: egui_winit::State,
    /// egui renderer
    egui_renderer: egui_wgpu::Renderer,
    /// Show the HUD overlay
    show_hud: bool,
    /// Show inspector panel
    show_inspector: bool,
    /// Show brain visualization
    show_brain: bool,
    /// Show settings menu
    show_settings: bool,
    /// Show statistics graphs
    show_stats: bool,
    /// Show help overlay  
    show_help: bool,
    /// Statistics history for graphing
    stats_history: StatsHistory,
    /// Store window reference
    window: Arc<winit::window::Window>,
}

impl Ui {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat, window: Arc<winit::window::Window>) -> Self {
        let egui_ctx = egui::Context::default();
        
        // Apply theme on creation
        theme::apply_theme(&egui_ctx);
        
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );
        
        let egui_renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            None,
            1,
        );
        
        Self {
            egui_state,
            egui_renderer,
            show_hud: true,
            show_inspector: true,
            show_brain: false,
            show_settings: false,
            show_stats: false,
            show_help: false,
            stats_history: StatsHistory::new(),
            window,
        }
    }
    
    /// Toggle inspector panel visibility
    pub fn toggle_inspector(&mut self) {
        self.show_inspector = !self.show_inspector;
    }
    
    /// Toggle help overlay visibility
    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }
    
    /// Toggle settings menu visibility
    pub fn toggle_settings(&mut self) {
        self.show_settings = !self.show_settings;
    }
    
    /// Check if settings menu is open
    pub fn settings_open(&self) -> bool {
        self.show_settings
    }
    
    /// Handle window events for egui
    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.egui_state.on_window_event(&self.window, event).consumed
    }
    
    /// Render UI elements
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        data: &UiData,
        config: &mut crate::config::SimulationConfig,
        selected: Option<&SelectedOrganism>,
        genome: Option<&crate::simulation::genome::Genome>,
        screen_size: (u32, u32),
    ) {
        // Record stats (when simulation is running or when single-stepping)
        // Track stats if unpaused, or if births/deaths occurred (single-step happened)
        let tick_occurred = !data.paused || data.births > 0 || data.deaths > 0;
        if tick_occurred {
            self.stats_history.record(
                data.population,
                data.avg_energy,
                data.max_generation,
                data.total_food,
                data.births,
                data.deaths,
            );
        }
        
        // Get accumulated input from winit events (this is the key fix!)
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();
        
        ctx.begin_frame(raw_input);
        
        // Render toolbar (always visible unless settings open)
        if !self.show_settings {
            self.render_toolbar(&ctx);
        }
        
        // Render HUD
        if self.show_hud && !self.show_settings {
            self.render_hud(&ctx, data, config);
        }
        
        // Render statistics graphs
        if self.show_stats && !self.show_settings {
            self.render_stats_panel(&ctx);
        }
        
        // Render inspector panel
        if self.show_inspector && !self.show_settings {
            inspector::render_inspector(&ctx, selected, genome, &mut self.show_brain, &mut self.show_inspector);
        }
        
        // Render help overlay
        if self.show_help && !self.show_settings {
            self.render_help(&ctx);
        }
        
        // Render settings menu
        if self.show_settings {
            self.render_settings(&ctx, config, data);
        }
        
        // End frame
        let full_output = ctx.end_frame();
        
        // Handle textures
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, delta);
        }
        
        // Tessellate
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        
        // Create screen descriptor
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [screen_size.0, screen_size.1],
            pixels_per_point: full_output.pixels_per_point,
        };
        
        // Update buffers
        self.egui_renderer.update_buffers(
            device,
            queue,
            encoder,
            &paint_jobs,
            &screen_descriptor,
        );
        
        // Render
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear, render on top
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            self.egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }
        
        // Free textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
    
    /// Render the toolbar at the top-right
    fn render_toolbar(&mut self, ctx: &egui::Context) {
        egui::Area::new(egui::Id::new("toolbar"))
            .fixed_pos(Pos2::new(ctx.screen_rect().right() - 160.0, 10.0))
            .show(ctx, |ui| {
                theme::panel_frame().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Stats toggle
                        if ui.selectable_label(self.show_stats, "📊").on_hover_text("Statistics (S)").clicked() {
                            self.show_stats = !self.show_stats;
                        }
                        // Inspector toggle
                        if ui.selectable_label(self.show_inspector, "🔍").on_hover_text("Inspector (I)").clicked() {
                            self.show_inspector = !self.show_inspector;
                        }
                        // Settings
                        if ui.selectable_label(self.show_settings, "⚙").on_hover_text("Settings (Esc)").clicked() {
                            self.show_settings = !self.show_settings;
                        }
                        // Help
                        if ui.selectable_label(self.show_help, "❓").on_hover_text("Help (H)").clicked() {
                            self.show_help = !self.show_help;
                        }
                    });
                });
            });
    }
    
    fn render_hud(&mut self, ctx: &egui::Context, data: &UiData, _config: &mut crate::config::SimulationConfig) {
        egui::Area::new(egui::Id::new("hud"))
            .fixed_pos(Pos2::new(10.0, 10.0))
            .show(ctx, |ui| {
                theme::panel_frame().show(ui, |ui| {
                    ui.vertical(|ui| {
                        // Status line
                        let status = if data.paused {
                            format!("⏸ Paused")
                        } else {
                            format!("▶ {}x", data.speed)
                        };
                        ui.horizontal(|ui| {
                            ui.label(format!("Tick: {}", data.tick));
                            ui.separator();
                            ui.label(status);
                            ui.separator();
                            ui.label(format!("{:.0} FPS", data.fps));
                        });
                        
                        ui.separator();
                        
                        // Population stats with tooltips
                        ui.horizontal(|ui| {
                            ui.label("Population:").on_hover_text("Current number of alive organisms");
                            ui.colored_label(
                                theme::value_color_positive(data.population as f32, 0.0, data.max_population as f32),
                                format!("{} / {}", data.population, data.max_population)
                            );
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Avg Energy:").on_hover_text("Average energy across all organisms");
                            ui.colored_label(
                                theme::energy_color(data.avg_energy, 200.0),
                                format!("{:.1}", data.avg_energy)
                            );
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Max Gen:").on_hover_text("Highest generation number reached");
                            ui.colored_label(theme::ACCENT_COLOR, format!("{}", data.max_generation));
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Species:").on_hover_text("Number of genetically distinct populations");
                            ui.label(format!("{}", data.species_count));
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Food:").on_hover_text("Total food available in the world");
                            ui.colored_label(theme::POSITIVE_COLOR, format!("{:.0}", data.total_food));
                        });
                        
                        // Profiling (collapsed by default)
                        ui.collapsing("⏱ Profiling", |ui| {
                            ui.label(format!("Readback: {:.2}ms", data.readback_ms));
                            ui.label(format!("Upload: {:.2}ms", data.upload_ms));
                            ui.label(format!("Submit: {:.2}ms", data.submit_ms));
                            ui.label(format!("Total: {:.2}ms", data.compute_ms));
                        });
                    });
                });
            });
    }
    
    /// Render help overlay with keyboard shortcuts
    fn render_help(&mut self, ctx: &egui::Context) {
        egui::Window::new("❓ Keyboard Shortcuts")
            .open(&mut self.show_help)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .collapsible(false)
            .resizable(false)
            .frame(theme::window_frame())
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                
                // Simulation controls
                ui.heading("Simulation");
                egui::Grid::new("help_sim").num_columns(2).spacing([20.0, 4.0]).show(ui, |ui| {
                    ui.label("Space"); ui.label("Pause / Resume"); ui.end_row();
                    ui.label("."); ui.label("Step one tick (when paused)"); ui.end_row();
                    ui.label("1-7"); ui.label("Set speed (1x to 64x)"); ui.end_row();
                    ui.label("F5"); ui.label("Quick save"); ui.end_row();
                    ui.label("F9"); ui.label("Quick load"); ui.end_row();
                });
                
                ui.add_space(8.0);
                ui.heading("Camera");
                egui::Grid::new("help_cam").num_columns(2).spacing([20.0, 4.0]).show(ui, |ui| {
                    ui.label("WASD / Arrows"); ui.label("Pan camera"); ui.end_row();
                    ui.label("Mouse wheel"); ui.label("Zoom in/out"); ui.end_row();
                    ui.label("R"); ui.label("Reset camera"); ui.end_row();
                    ui.label("F"); ui.label("Follow selected organism"); ui.end_row();
                });
                
                ui.add_space(8.0);
                ui.heading("Selection");
                egui::Grid::new("help_sel").num_columns(2).spacing([20.0, 4.0]).show(ui, |ui| {
                    ui.label("Left click"); ui.label("Select organism"); ui.end_row();
                    ui.label("E"); ui.label("Feed selected (+20 energy)"); ui.end_row();
                    ui.label("K"); ui.label("Kill selected organism"); ui.end_row();
                    ui.label("Right click"); ui.label("Kill organism / spawn food"); ui.end_row();
                });
                
                ui.add_space(8.0);
                ui.heading("UI");
                egui::Grid::new("help_ui").num_columns(2).spacing([20.0, 4.0]).show(ui, |ui| {
                    ui.label("Escape"); ui.label("Toggle settings menu"); ui.end_row();
                    ui.label("I"); ui.label("Toggle inspector panel"); ui.end_row();
                    ui.label("H"); ui.label("Toggle this help"); ui.end_row();
                });
            });
    }
    
    fn render_settings(&mut self, ctx: &egui::Context, config: &mut crate::config::SimulationConfig, data: &UiData) {
        // Semi-transparent overlay
        egui::Area::new(egui::Id::new("settings_overlay"))
            .fixed_pos(Pos2::new(0.0, 0.0))
            .order(egui::Order::Background)
            .show(ctx, |ui| {
                let screen = ui.ctx().screen_rect();
                ui.painter().rect_filled(
                    screen,
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
                );
            });
        
        egui::Window::new("⚙ Settings")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(350.0);
                
                // Status header
                ui.horizontal(|ui| {
                    ui.label(format!("Tick: {} | Pop: {}/{}", data.tick, data.population, data.max_population));
                });
                ui.separator();
                
                // Mutation settings
                ui.heading("🧬 Mutation");
                ui.horizontal(|ui| {
                    ui.label("Rate:");
                    ui.add(egui::Slider::new(&mut config.mutation.rate, 0.0..=1.0)
                        .suffix("")
                        .fixed_decimals(2));
                });
                ui.horizontal(|ui| {
                    ui.label("Strength:");
                    ui.add(egui::Slider::new(&mut config.mutation.strength, 0.0..=1.0)
                        .suffix("")
                        .fixed_decimals(2));
                });
                ui.add_space(8.0);
                
                // Food settings
                ui.heading("🌱 Food");
                ui.horizontal(|ui| {
                    ui.label("Growth Rate:");
                    ui.add(egui::Slider::new(&mut config.food.growth_rate, 0.0..=1.0)
                        .suffix("")
                        .fixed_decimals(3));
                });
                ui.horizontal(|ui| {
                    ui.label("Effectiveness:");
                    ui.add(egui::Slider::new(&mut config.food.effectiveness, 0.0..=2.0)
                        .suffix("")
                        .fixed_decimals(2));
                });
                ui.horizontal(|ui| {
                    ui.label("Energy Value:");
                    ui.add(egui::Slider::new(&mut config.food.energy_value, 1.0..=100.0)
                        .suffix("")
                        .fixed_decimals(0));
                });
                ui.add_space(8.0);
                
                // Energy settings
                ui.heading("⚡ Energy");
                ui.horizontal(|ui| {
                    ui.label("Passive Drain:");
                    ui.add(egui::Slider::new(&mut config.energy.passive_drain, 0.0..=1.0)
                        .suffix("")
                        .fixed_decimals(3));
                });
                ui.horizontal(|ui| {
                    ui.label("Move Cost (Fwd):");
                    ui.add(egui::Slider::new(&mut config.energy.movement_cost_forward, 0.0..=0.5)
                        .suffix("")
                        .fixed_decimals(3));
                });
                ui.horizontal(|ui| {
                    ui.label("Age Drain Factor:");
                    ui.add(egui::Slider::new(&mut config.energy.age_drain_factor, 0.0..=2.0)
                        .suffix("")
                        .fixed_decimals(2));
                });
                ui.add_space(8.0);
                
                // Reproduction settings
                ui.heading("🔄 Reproduction");
                ui.horizontal(|ui| {
                    ui.label("Energy Threshold:");
                    ui.add(egui::Slider::new(&mut config.reproduction.threshold, 10.0..=150.0)
                        .suffix("")
                        .fixed_decimals(0));
                });
                ui.horizontal(|ui| {
                    ui.label("Energy Cost:");
                    ui.add(egui::Slider::new(&mut config.reproduction.cost, 5.0..=100.0)
                        .suffix("")
                        .fixed_decimals(0));
                });
                ui.add_space(8.0);
                
                // Dynamic Environment settings
                ui.heading("🌍 Dynamic Environment");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut config.food.seasonal_enabled, "Seasonal Cycles")
                        .on_hover_text("Food growth rate oscillates over time, creating boom/bust cycles");
                });
                if config.food.seasonal_enabled {
                    ui.horizontal(|ui| {
                        ui.label("Period:");
                        ui.add(egui::Slider::new(&mut config.food.seasonal_period, 1000..=20000)
                            .suffix(" ticks")
                            .fixed_decimals(0))
                            .on_hover_text("Duration of one full seasonal cycle");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Amplitude:");
                        ui.add(egui::Slider::new(&mut config.food.seasonal_amplitude, 0.0..=1.0)
                            .fixed_decimals(2))
                            .on_hover_text("How extreme the seasonal swing is (1.0 = winter has near-zero growth)");
                    });
                }
                
                ui.horizontal(|ui| {
                    ui.checkbox(&mut config.food.hotspots_enabled, "Resource Hotspots")
                        .on_hover_text("High-value food zones that drift slowly across the world");
                });
                if config.food.hotspots_enabled {
                    ui.horizontal(|ui| {
                        ui.label("Count:");
                        ui.add(egui::Slider::new(&mut config.food.hotspot_count, 1..=5)
                            .fixed_decimals(0))
                            .on_hover_text("Number of hotspots");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Radius:");
                        ui.add(egui::Slider::new(&mut config.food.hotspot_radius, 20.0..=200.0)
                            .fixed_decimals(0))
                            .on_hover_text("Size of hotspot influence zone");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Intensity:");
                        ui.add(egui::Slider::new(&mut config.food.hotspot_intensity, 0.1..=1.0)
                            .fixed_decimals(2))
                            .on_hover_text("Food growth bonus at hotspot center");
                    });
                }
                ui.add_space(8.0);
                
                // Quality presets
                ui.heading("🎮 Presets");
                ui.horizontal(|ui| {
                    if ui.button("Easy").clicked() {
                        config.food.effectiveness = 1.5;
                        config.food.growth_rate = 0.02;
                        config.energy.passive_drain = 0.05;
                        config.mutation.rate = 0.3;
                    }
                    if ui.button("Normal").clicked() {
                        config.food.effectiveness = 1.0;
                        config.food.growth_rate = 0.01;
                        config.energy.passive_drain = 0.1;
                        config.mutation.rate = 0.2;
                    }
                    if ui.button("Hard").clicked() {
                        config.food.effectiveness = 0.7;
                        config.food.growth_rate = 0.005;
                        config.energy.passive_drain = 0.15;
                        config.mutation.rate = 0.15;
                    }
                    if ui.button("Extreme").clicked() {
                        config.food.effectiveness = 0.5;
                        config.food.growth_rate = 0.003;
                        config.energy.passive_drain = 0.2;
                        config.mutation.rate = 0.1;
                    }
                });
                
                ui.add_space(16.0);
                ui.separator();
                
                // Close button
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Resume (Esc)").clicked() {
                            self.show_settings = false;
                        }
                    });
                });
            });
    }
    
    /// Render statistics graphs panel
    fn render_stats_panel(&self, ctx: &egui::Context) {
        egui::Window::new("📊 Statistics")
            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(10.0, -10.0))
            .default_width(350.0)
            .default_height(300.0)
            .resizable(true)
            .collapsible(true)
            .frame(theme::window_frame())
            .show(ctx, |ui| {
                // Population graph
                ui.collapsing("Population", |ui| {
                    let data = self.stats_history.population_slice();
                    Self::render_line_graph(ui, &data, theme::POSITIVE_COLOR, "organisms");
                });
                
                // Generation graph
                ui.collapsing("Max Generation", |ui| {
                    let data = self.stats_history.max_generation_slice();
                    Self::render_line_graph(ui, &data, theme::WARNING_COLOR, "gen");
                });
                
                // Energy graph
                ui.collapsing("Average Energy", |ui| {
                    let data = self.stats_history.avg_energy_slice();
                    Self::render_line_graph(ui, &data, theme::ACCENT_COLOR, "energy");
                });
                
                // Food graph
                ui.collapsing("Total Food", |ui| {
                    let data = self.stats_history.total_food_slice();
                    Self::render_line_graph(ui, &data, egui::Color32::from_rgb(50, 200, 50), "food");
                });
                
                // Birth/Death rates
                ui.collapsing("Birth/Death Rates", |ui| {
                    let births = self.stats_history.births_slice();
                    let deaths = self.stats_history.deaths_slice();
                    Self::render_dual_line_graph(
                        ui, 
                        &births, 
                        &deaths,
                        egui::Color32::from_rgb(100, 200, 255),
                        theme::NEGATIVE_COLOR,
                        "births",
                        "deaths",
                    );
                });
            });
    }
    
    /// Render a simple line graph with hover interaction
    fn render_line_graph(ui: &mut egui::Ui, data: &[f32], color: egui::Color32, label: &str) {
        if data.is_empty() {
            ui.label("No data yet...");
            return;
        }
        
        let (min, max) = StatsHistory::min_max(data);
        let height = theme::GRAPH_HEIGHT;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        
        // Draw background
        painter.rect_filled(rect, 4.0, theme::GRAPH_BG);
        
        // Draw line
        let points: Vec<egui::Pos2> = data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = rect.left() + (i as f32 / data.len().max(1) as f32) * rect.width();
                let y = rect.bottom() - ((v - min) / (max - min).max(0.001)) * rect.height();
                egui::pos2(x, y)
            })
            .collect();
        
        if points.len() >= 2 {
            painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
        }
        
        // Hover interaction - show value at cursor position
        if response.hovered() {
            if let Some(hover_pos) = response.hover_pos() {
                let x_ratio = (hover_pos.x - rect.left()) / rect.width();
                let idx = (x_ratio * data.len() as f32) as usize;
                if idx < data.len() {
                    let val = data[idx];
                    // Draw vertical line
                    let line_x = rect.left() + x_ratio * rect.width();
                    painter.line_segment(
                        [egui::pos2(line_x, rect.top()), egui::pos2(line_x, rect.bottom())],
                        egui::Stroke::new(1.0, egui::Color32::from_gray(100)),
                    );
                    // Draw value tooltip
                    painter.text(
                        egui::pos2(line_x + 5.0, rect.top() + 12.0),
                        egui::Align2::LEFT_CENTER,
                        format!("{:.0}", val),
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                }
            }
        }
        
        // Draw labels
        ui.horizontal(|ui| {
            ui.colored_label(color, format!("{}: {:.0}", label, data.last().unwrap_or(&0.0)));
            ui.label(format!("(min: {:.0}, max: {:.0})", min, max));
        });
    }
    
    /// Render a dual line graph (for births/deaths) with hover interaction
    fn render_dual_line_graph(
        ui: &mut egui::Ui, 
        data1: &[f32], 
        data2: &[f32], 
        color1: egui::Color32,
        color2: egui::Color32,
        label1: &str,
        label2: &str,
    ) {
        if data1.is_empty() && data2.is_empty() {
            ui.label("No data yet...");
            return;
        }
        
        // Find global min/max
        let (min1, max1) = StatsHistory::min_max(data1);
        let (min2, max2) = StatsHistory::min_max(data2);
        let min = min1.min(min2);
        let max = max1.max(max2);
        
        let height = theme::GRAPH_HEIGHT;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        
        // Draw background
        painter.rect_filled(rect, 4.0, theme::GRAPH_BG);
        
        // Helper to draw a line
        let draw_line = |data: &[f32], color: egui::Color32| {
            let points: Vec<egui::Pos2> = data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let x = rect.left() + (i as f32 / data.len().max(1) as f32) * rect.width();
                    let y = rect.bottom() - ((v - min) / (max - min).max(0.001)) * rect.height();
                    egui::pos2(x, y)
                })
                .collect();
            
            if points.len() >= 2 {
                painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
            }
        };
        
        draw_line(data1, color1);
        draw_line(data2, color2);
        
        // Draw labels
        ui.horizontal(|ui| {
            ui.colored_label(color1, format!("{}: {:.0}", label1, data1.last().unwrap_or(&0.0)));
            ui.colored_label(color2, format!("{}: {:.0}", label2, data2.last().unwrap_or(&0.0)));
        });
    }
}
