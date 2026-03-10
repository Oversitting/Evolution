//! Organism inspector panel
//! 
//! Displays detailed information about a selected organism including:
//! - Stats (energy, age, generation, offspring)
//! - Neural network visualization
//! - Real-time action outputs

use egui::{Color32, Pos2, Stroke};
use crate::simulation::genome::{Genome, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM};

/// Labels for neural network inputs
pub const INPUT_LABELS: [&str; 20] = [
    // Vision rays (8 rays × 2 values = food distance + organism distance)
    "Ray0 Food", "Ray0 Org",
    "Ray1 Food", "Ray1 Org",
    "Ray2 Food", "Ray2 Org",
    "Ray3 Food", "Ray3 Org",
    "Ray4 Food", "Ray4 Org",
    "Ray5 Food", "Ray5 Org",
    "Ray6 Food", "Ray6 Org",
    "Ray7 Food", "Ray7 Org",
    // Internal state (4 values)
    "Energy", "Velocity", "Angular", "Bias",
];

/// Labels for neural network outputs
pub const OUTPUT_LABELS: [&str; 6] = [
    "Forward",    // Movement forward/backward
    "Turn",       // Rotation left/right  
    "Eat",        // Eating intention
    "Reproduce",  // Reproduction signal
    "Out4",       // Reserved
    "Out5",       // Reserved
];

/// Data about a selected organism for display
#[derive(Clone, Debug, Default)]
pub struct SelectedOrganism {
    /// Organism slot index
    pub id: u32,
    /// Is this organism currently alive?
    pub alive: bool,
    /// World position
    pub position: [f32; 2],
    /// Current rotation in radians
    pub rotation: f32,
    /// Current energy level
    pub energy: f32,
    /// Age in ticks
    pub age: u32,
    /// Generation number
    pub generation: u32,
    /// Number of offspring produced
    pub offspring_count: u32,
    /// Parent organism ID
    pub parent_id: u32,
    /// Current reproduction output from neural network
    #[allow(dead_code)]
    pub reproduce_signal: f32,
    /// Genome ID (same as organism ID in current impl)
    #[allow(dead_code)]
    pub genome_id: u32,
    /// Species cluster ID
    pub species_id: u32,
    /// Neural network input values (last tick)
    pub nn_inputs: [f32; INPUT_DIM],
    /// Neural network output values (last tick)
    pub nn_outputs: [f32; OUTPUT_DIM],
}

/// Render the organism inspector panel
pub fn render_inspector(
    ctx: &egui::Context,
    selected: Option<&SelectedOrganism>,
    genome: Option<&Genome>,
    show_brain: &mut bool,
    show_inspector: &mut bool,
) {
    egui::Window::new("🔍 Inspector")
        .open(show_inspector)
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 50.0))
        .resizable(true)
        .collapsible(true)
        .default_width(280.0)
        .frame(crate::ui::theme::window_frame())
        .show(ctx, |ui| {
            match selected {
                Some(org) if org.alive => {
                    render_organism_stats(ui, org);
                    ui.separator();
                    
                    ui.checkbox(show_brain, "Show Brain Visualization");
                    
                    ui.separator();
                    render_action_outputs(ui, org);
                }
                Some(_) => {
                    ui.label("Selected organism is dead.");
                    ui.label("Click on another organism to select.");
                }
                None => {
                    ui.label("No organism selected.");
                    ui.label("Left-click on an organism to select it.");
                    ui.separator();
                    ui.label("Controls:");
                    ui.label("• Left click: Select organism");
                    ui.label("• F: Follow selected");
                    ui.label("• Escape: Deselect");
                }
            }
        });
    
    // Show brain visualization in a separate window if enabled
    if *show_brain {
        if let (Some(org), Some(genome)) = (selected, genome) {
            if org.alive {
                render_brain_window(ctx, org, genome, show_brain);
            }
        }
    }
}

fn render_organism_stats(ui: &mut egui::Ui, org: &SelectedOrganism) {
    ui.heading(format!("Organism #{}", org.id));
    
    egui::Grid::new("stats_grid")
        .num_columns(2)
        .spacing([20.0, 4.0])
        .show(ui, |ui| {
            ui.label("Energy:");
            let energy_color = if org.energy > 100.0 {
                Color32::GREEN
            } else if org.energy > 50.0 {
                Color32::YELLOW
            } else {
                Color32::RED
            };
            ui.colored_label(energy_color, format!("{:.1}", org.energy));
            ui.end_row();
            
            ui.label("Age:");
            ui.label(format!("{} ticks", org.age));
            ui.end_row();
            
            ui.label("Generation:");
            ui.colored_label(Color32::LIGHT_BLUE, format!("{}", org.generation));
            ui.end_row();
            
            ui.label("Species:");
            // Use golden ratio hue spread to match render coloring
            let golden_ratio = 0.618033988749895_f32;
            let hue = (org.species_id as f32 * golden_ratio).fract();
            let h = (hue * 360.0) as u8;
            // Simple HSV to RGB approximation for display
            let species_color = egui::Color32::from_rgb(
                128 + ((h as f32 / 360.0 * 6.0).sin() * 127.0) as u8,
                128 + (((h as f32 / 360.0 * 6.0) + 2.0).sin() * 127.0) as u8,
                128 + (((h as f32 / 360.0 * 6.0) + 4.0).sin() * 127.0) as u8,
            );
            ui.colored_label(species_color, format!("#{}", org.species_id));
            ui.end_row();
            
            ui.label("Offspring:");
            ui.label(format!("{}", org.offspring_count));
            ui.end_row();
            
            ui.label("Position:");
            ui.label(format!("({:.1}, {:.1})", org.position[0], org.position[1]));
            ui.end_row();
            
            ui.label("Rotation:");
            ui.label(format!("{:.1}°", org.rotation.to_degrees()));
            ui.end_row();
            
            if org.parent_id != u32::MAX {
                ui.label("Parent:");
                ui.label(format!("#{}", org.parent_id));
                ui.end_row();
            }
        });
}

fn render_action_outputs(ui: &mut egui::Ui, org: &SelectedOrganism) {
    ui.heading("Neural Outputs");
    
    egui::Grid::new("outputs_grid")
        .num_columns(2)
        .spacing([20.0, 4.0])
        .show(ui, |ui| {
            for (i, label) in OUTPUT_LABELS.iter().enumerate() {
                let value = org.nn_outputs.get(i).copied().unwrap_or(0.0);
                ui.label(*label);
                
                // Color code by activation level
                let color = activation_color(value);
                ui.colored_label(color, format!("{:+.3}", value));
                ui.end_row();
            }
        });
}

fn render_brain_window(
    ctx: &egui::Context,
    org: &SelectedOrganism,
    genome: &Genome,
    show: &mut bool,
) {
    egui::Window::new("🧠 Neural Network")
        .open(show)
        .resizable(true)
        .default_size([500.0, 400.0])
        .show(ctx, |ui| {
            render_neural_network(ui, org, genome);
        });
}

fn render_neural_network(ui: &mut egui::Ui, org: &SelectedOrganism, genome: &Genome) {
    let available_size = ui.available_size();
    let (response, painter) = ui.allocate_painter(available_size, egui::Sense::hover());
    let rect = response.rect;
    
    // Layout parameters
    let padding = 30.0;
    let layer_width = (rect.width() - padding * 2.0) / 3.0;
    
    let input_x = rect.left() + padding + 40.0; // Extra space for labels
    let hidden_x = rect.left() + padding + layer_width + 20.0;
    let output_x = rect.right() - padding - 40.0;
    
    // Calculate node positions
    let input_nodes = calculate_node_positions(input_x, rect.top() + padding, rect.height() - padding * 2.0, INPUT_DIM);
    let hidden_nodes = calculate_node_positions(hidden_x, rect.top() + padding, rect.height() - padding * 2.0, HIDDEN_DIM);
    let output_nodes = calculate_node_positions(output_x, rect.top() + padding, rect.height() - padding * 2.0, OUTPUT_DIM);
    
    // Draw connections (input -> hidden)
    for (i, &input_pos) in input_nodes.iter().enumerate() {
        for (j, &hidden_pos) in hidden_nodes.iter().enumerate() {
            let weight_idx = i * HIDDEN_DIM + j;
            let weight = genome.weights_l1.get(weight_idx).copied().unwrap_or(0.0);
            draw_connection(&painter, input_pos, hidden_pos, weight);
        }
    }
    
    // Draw connections (hidden -> output)
    for (i, &hidden_pos) in hidden_nodes.iter().enumerate() {
        for (j, &output_pos) in output_nodes.iter().enumerate() {
            let weight_idx = i * OUTPUT_DIM + j;
            let weight = genome.weights_l2.get(weight_idx).copied().unwrap_or(0.0);
            draw_connection(&painter, hidden_pos, output_pos, weight);
        }
    }
    
    // Draw input nodes
    for (i, &pos) in input_nodes.iter().enumerate() {
        let activation = org.nn_inputs.get(i).copied().unwrap_or(0.0);
        draw_node(&painter, pos, activation, true);
        
        // Draw label (abbreviated)
        let label = INPUT_LABELS.get(i).map(|s| {
            if s.len() > 6 { &s[..6] } else { s }
        }).unwrap_or("");
        painter.text(
            Pos2::new(pos.x - 45.0, pos.y),
            egui::Align2::RIGHT_CENTER,
            label,
            egui::FontId::proportional(9.0),
            Color32::GRAY,
        );
    }
    
    // Draw hidden nodes
    for (i, &pos) in hidden_nodes.iter().enumerate() {
        // Compute hidden activation (simplified - actual is on GPU)
        let mut activation = genome.biases_l1.get(i).copied().unwrap_or(0.0);
        for (j, &input_val) in org.nn_inputs.iter().enumerate() {
            let weight_idx = j * HIDDEN_DIM + i;
            let weight = genome.weights_l1.get(weight_idx).copied().unwrap_or(0.0);
            activation += input_val * weight;
        }
        activation = activation.tanh();
        draw_node(&painter, pos, activation, false);
    }
    
    // Draw output nodes
    for (i, &pos) in output_nodes.iter().enumerate() {
        let activation = org.nn_outputs.get(i).copied().unwrap_or(0.0);
        draw_node(&painter, pos, activation, true);
        
        // Draw label
        let label = OUTPUT_LABELS.get(i).unwrap_or(&"");
        painter.text(
            Pos2::new(pos.x + 15.0, pos.y),
            egui::Align2::LEFT_CENTER,
            *label,
            egui::FontId::proportional(10.0),
            Color32::GRAY,
        );
    }
    
    // Draw layer labels
    painter.text(
        Pos2::new(input_x, rect.top() + 15.0),
        egui::Align2::CENTER_CENTER,
        "Input (20)",
        egui::FontId::proportional(11.0),
        Color32::WHITE,
    );
    painter.text(
        Pos2::new(hidden_x, rect.top() + 15.0),
        egui::Align2::CENTER_CENTER,
        "Hidden (16)",
        egui::FontId::proportional(11.0),
        Color32::WHITE,
    );
    painter.text(
        Pos2::new(output_x, rect.top() + 15.0),
        egui::Align2::CENTER_CENTER,
        "Output (6)",
        egui::FontId::proportional(11.0),
        Color32::WHITE,
    );
}

fn calculate_node_positions(x: f32, top: f32, height: f32, count: usize) -> Vec<Pos2> {
    let spacing = height / (count as f32 + 1.0);
    (0..count)
        .map(|i| Pos2::new(x, top + spacing * (i as f32 + 1.0)))
        .collect()
}

fn draw_connection(painter: &egui::Painter, from: Pos2, to: Pos2, weight: f32) {
    let color = if weight > 0.0 {
        Color32::from_rgba_unmultiplied(100, 200, 100, (weight.abs().min(1.0) * 80.0) as u8 + 20)
    } else {
        Color32::from_rgba_unmultiplied(200, 100, 100, (weight.abs().min(1.0) * 80.0) as u8 + 20)
    };
    
    let stroke = Stroke::new(1.0 + weight.abs().min(1.0) * 1.5, color);
    painter.line_segment([from, to], stroke);
}

fn draw_node(painter: &egui::Painter, center: Pos2, activation: f32, show_value: bool) {
    let radius = 6.0;
    let color = activation_color(activation);
    
    painter.circle_filled(center, radius, color);
    painter.circle_stroke(center, radius, Stroke::new(1.0, Color32::DARK_GRAY));
    
    if show_value && activation.abs() > 0.01 {
        // Show value on hover (simplified - always show for now)
    }
}

fn activation_color(value: f32) -> Color32 {
    let clamped = value.clamp(-1.0, 1.0);
    if clamped >= 0.0 {
        let intensity = (clamped * 200.0) as u8;
        Color32::from_rgb(50, 100 + intensity / 2, 50 + intensity)
    } else {
        let intensity = (-clamped * 200.0) as u8;
        Color32::from_rgb(100 + intensity / 2, 50, 50)
    }
}
