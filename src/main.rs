//! Evolution Simulator - GPU-accelerated digital life
//!
//! This simulator uses GPU compute shaders to simulate thousands of organisms
//! with neural network brains evolving in real-time.

mod app;
mod config;
mod simulation;
mod compute;
mod render;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result;
use clap::Parser;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    dpi::LogicalSize,
};

/// Evolution Simulator - GPU-accelerated digital life
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file (TOML format)
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
    
    /// Auto-exit after N seconds (0 = disabled, useful for testing/benchmarking)
    #[arg(long, default_value = "0")]
    auto_exit: u64,
    
    /// Start paused
    #[arg(long)]
    paused: bool,
    
    /// Initial simulation speed multiplier
    #[arg(long, default_value = "1")]
    speed: u32,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("Evolution Simulator starting...");

    // Parse command line arguments
    let args = Args::parse();
    
    // Load configuration
    let sim_config = config::SimulationConfig::load_or_create_default(&args.config);
    
    // Create event loop and window
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(WindowBuilder::new()
        .with_title("Evolution Simulator")
        .with_inner_size(LogicalSize::new(1280, 720))
        .with_min_inner_size(LogicalSize::new(800, 600))
        .build(&event_loop)?);

    // Initialize application with config and CLI options
    let mut app = pollster::block_on(app::App::new_with_config(
        window.clone(),
        sim_config,
        args.auto_exit,
        args.paused,
        args.speed,
    ))?;
    log::info!("Application initialized");

    // Run event loop
    let mut exit_logged = false;
    let mut survivor_bank_persisted = false;
    event_loop.run(move |event, elwt| {
        // Check for auto-exit
        if app.should_exit() {
            if !exit_logged {
                if !survivor_bank_persisted {
                    app.persist_survivor_bank_on_exit();
                    survivor_bank_persisted = true;
                }
                log::info!("Auto-exit timeout reached, shutting down");
                exit_logged = true;
            }
            elwt.exit();
            return;
        }
        
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                // Let egui handle events first
                let egui_consumed = app.handle_event(&event);
                
                // Always handle RedrawRequested, regardless of egui consumption
                if let WindowEvent::RedrawRequested = &event {
                    app.update();
                    match app.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => app.resize(window.inner_size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            if !survivor_bank_persisted {
                                app.persist_survivor_bank_on_exit();
                                survivor_bank_persisted = true;
                            }
                            elwt.exit();
                        }
                        Err(e) => log::error!("Render error: {:?}", e),
                    }
                } else if !egui_consumed {
                    match event {
                        WindowEvent::CloseRequested => {
                            if !survivor_bank_persisted {
                                app.persist_survivor_bank_on_exit();
                                survivor_bank_persisted = true;
                            }
                            log::info!("Close requested, shutting down");
                            elwt.exit();
                        }
                        WindowEvent::Resized(size) => {
                            app.resize(size);
                        }
                        WindowEvent::KeyboardInput { event, .. } => {
                            app.handle_keyboard(&event);
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            app.handle_scroll(delta);
                        }
                        WindowEvent::MouseInput { state, button, .. } => {
                            app.handle_mouse_button(button, state);
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            app.handle_cursor_move(position);
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
