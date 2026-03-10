//! UI Theme - Consistent styling across all panels
//!
//! Provides color constants, spacing values, and helper functions
//! for creating uniformly styled UI windows.

#![allow(dead_code)] // Theme constants are API for future use

use egui::{Color32, Frame, Margin, Rounding, Stroke};

// ============================================================================
// Colors
// ============================================================================

/// Window background color (dark with slight transparency)
pub const WINDOW_BG: Color32 = Color32::from_rgba_premultiplied(18, 18, 22, 245);

/// Panel background color (slightly lighter)
pub const PANEL_BG: Color32 = Color32::from_rgba_premultiplied(25, 25, 30, 240);

/// Section header text color
pub const HEADER_COLOR: Color32 = Color32::from_rgb(140, 180, 220);

/// Body text color
pub const TEXT_COLOR: Color32 = Color32::from_gray(200);

/// Muted/secondary text color
pub const TEXT_MUTED: Color32 = Color32::from_gray(120);

/// Positive value color (energy high, population up)
pub const POSITIVE_COLOR: Color32 = Color32::from_rgb(100, 200, 100);

/// Warning value color (energy medium)
pub const WARNING_COLOR: Color32 = Color32::from_rgb(220, 180, 60);

/// Negative/danger color (energy low, deaths)
pub const NEGATIVE_COLOR: Color32 = Color32::from_rgb(200, 100, 100);

/// Accent/highlight color (selection, links)
pub const ACCENT_COLOR: Color32 = Color32::from_rgb(80, 160, 220);

/// Graph background color
pub const GRAPH_BG: Color32 = Color32::from_gray(25);

// ============================================================================
// Spacing & Sizing
// ============================================================================

/// Window inner margin
pub const WINDOW_MARGIN: f32 = 12.0;

/// Section vertical spacing
pub const SECTION_SPACING: f32 = 8.0;

/// Item vertical spacing
pub const ITEM_SPACING: f32 = 4.0;

/// Window corner rounding
pub const WINDOW_ROUNDING: f32 = 6.0;

/// Button corner rounding
pub const BUTTON_ROUNDING: f32 = 4.0;

/// Graph default height
pub const GRAPH_HEIGHT: f32 = 80.0;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a themed window frame
pub fn window_frame() -> Frame {
    Frame {
        inner_margin: Margin::same(WINDOW_MARGIN),
        outer_margin: Margin::ZERO,
        rounding: Rounding::same(WINDOW_ROUNDING),
        shadow: egui::epaint::Shadow {
            offset: egui::vec2(2.0, 2.0),
            blur: 8.0,
            spread: 0.0,
            color: Color32::from_black_alpha(60),
        },
        fill: WINDOW_BG,
        stroke: Stroke::new(1.0, Color32::from_gray(40)),
    }
}

/// Create a themed panel frame (for HUD)
pub fn panel_frame() -> Frame {
    Frame {
        inner_margin: Margin::same(10.0),
        outer_margin: Margin::ZERO,
        rounding: Rounding::same(WINDOW_ROUNDING),
        shadow: egui::epaint::Shadow::NONE,
        fill: PANEL_BG,
        stroke: Stroke::new(1.0, Color32::from_gray(50)),
    }
}

/// Color for energy value based on level
pub fn energy_color(energy: f32, max_energy: f32) -> Color32 {
    let ratio = energy / max_energy;
    if ratio > 0.6 {
        POSITIVE_COLOR
    } else if ratio > 0.3 {
        WARNING_COLOR
    } else {
        NEGATIVE_COLOR
    }
}

/// Color for a value where higher is better
pub fn value_color_positive(value: f32, min: f32, max: f32) -> Color32 {
    let ratio = (value - min) / (max - min).max(0.001);
    if ratio > 0.6 {
        POSITIVE_COLOR
    } else if ratio > 0.3 {
        WARNING_COLOR
    } else {
        NEGATIVE_COLOR
    }
}

/// Apply consistent styling to the egui context
pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    
    // Spacing
    style.spacing.item_spacing = egui::vec2(8.0, ITEM_SPACING);
    style.spacing.window_margin = Margin::same(WINDOW_MARGIN);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    
    // Visuals
    style.visuals.window_rounding = Rounding::same(WINDOW_ROUNDING);
    style.visuals.widgets.noninteractive.bg_fill = PANEL_BG;
    style.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_COLOR);
    style.visuals.widgets.inactive.bg_fill = Color32::from_gray(40);
    style.visuals.widgets.hovered.bg_fill = Color32::from_gray(50);
    style.visuals.widgets.active.bg_fill = ACCENT_COLOR;
    style.visuals.selection.bg_fill = ACCENT_COLOR.linear_multiply(0.5);
    
    ctx.set_style(style);
}

/// Format a tooltip string with consistent styling
pub fn tooltip(short_desc: &str, details: Option<&str>) -> String {
    match details {
        Some(d) => format!("{}\n\n{}", short_desc, d),
        None => short_desc.to_string(),
    }
}
