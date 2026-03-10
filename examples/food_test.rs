//! Food Grid Test Demo
//!
//! This demo tests food generation and rendering in isolation, without organisms.
//! Useful for debugging food growth, spontaneous generation, and visualization issues.
//!
//! Run with: cargo run --example food_test
//!
//! Controls:
//! - Space: Pause/resume simulation
//! - R: Reset food to initial patches
//! - G: Toggle spontaneous generation (on/off)
//! - 1-4: Speed multiplier
//! - Mouse scroll: Zoom
//! - Middle mouse drag: Pan

use std::sync::Arc;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent, ElementState, MouseButton, MouseScrollDelta, KeyEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
    dpi::LogicalSize,
};

/// Camera uniform for shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub position: [f32; 2],
    pub zoom: f32,
    pub _pad1: f32,
    pub viewport_size: [f32; 2],
    pub world_size: [f32; 2],
    pub food_max_per_cell: f32,
    pub _pad2: f32,
}

/// Config uniform for compute shader
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WorldConfig {
    world_width: u32,
    world_height: u32,
    tick: u32,
    food_growth_rate: f32,
    food_max_per_cell: f32,
    spontaneous_enabled: u32,  // 0 or 1
    _pad: [u32; 2],
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("Food Grid Test Demo starting...");
    
    let world_width: u32 = 256;  // Smaller for testing
    let world_height: u32 = 256;
    let food_max: f32 = 5.0;
    let food_growth_rate: f32 = 0.02;
    
    // Create event loop and window
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(WindowBuilder::new()
        .with_title("Food Grid Test Demo")
        .with_inner_size(LogicalSize::new(800, 800))
        .build(&event_loop)?);

    // Initialize wgpu
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    let surface = instance.create_surface(window.clone())?;
    
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    })).expect("Failed to find suitable GPU adapter");
    
    log::info!("Using GPU: {}", adapter.get_info().name);
    
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Main Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))?;
    
    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);
    
    let mut surface_config = wgpu::SurfaceConfiguration {
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
    
    // Initialize food grid with test pattern
    let grid_size = (world_width * world_height) as usize;
    let mut food_data: Vec<f32> = vec![0.0; grid_size];
    
    // Create test food patches at known locations
    create_test_patches(&mut food_data, world_width, world_height, food_max);
    
    // Create food buffer
    let food_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Food Buffer"),
        contents: bytemuck::cast_slice(&food_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    
    // Create config uniform buffer
    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Config Buffer"),
        size: std::mem::size_of::<WorldConfig>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Create compute shader for food growth (simplified version)
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Food Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("food_compute.wgsl").into()),
    });
    
    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    
    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &compute_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: food_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: config_buffer.as_entire_binding() },
        ],
    });
    
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Food Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "main",
    });
    
    // Create food texture for rendering
    let food_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Food Texture"),
        size: wgpu::Extent3d {
            width: world_width,
            height: world_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    
    let food_texture_view = food_texture.create_view(&wgpu::TextureViewDescriptor::default());
    
    let food_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Food Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    
    // Create camera buffer
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Camera Buffer"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // Create render shader
    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Food Render Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("food_render.wgsl").into()),
    });
    
    let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Render Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    
    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Render Bind Group"),
        layout: &render_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&food_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&food_sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: camera_buffer.as_entire_binding() },
        ],
    });
    
    // Create vertex buffer for fullscreen quad
    let quad_vertices: &[f32] = &[
        -1.0, -1.0, 0.0, 1.0,
         1.0, -1.0, 1.0, 1.0,
         1.0,  1.0, 1.0, 0.0,
        -1.0, -1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 0.0,
    ];
    
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(quad_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&render_bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                    wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });
    
    // State
    let mut tick: u32 = 0;
    let mut paused = false;
    let mut speed = 1;
    let mut spontaneous_enabled = true;
    let mut camera_pos = Vec2::new(world_width as f32 / 2.0, world_height as f32 / 2.0);
    let mut camera_zoom = 1.0f32;
    let mut dragging = false;
    let mut cursor_pos = Vec2::ZERO;
    
    log::info!("Demo ready. Controls: Space=pause, R=reset, G=toggle spontaneous, 1-4=speed");
    
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(new_size) => {
                        if new_size.width > 0 && new_size.height > 0 {
                            surface_config.width = new_size.width;
                            surface_config.height = new_size.height;
                            surface.configure(&device, &surface_config);
                        }
                    }
                    WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state: ElementState::Pressed, .. }, .. } => {
                        match physical_key {
                            PhysicalKey::Code(KeyCode::Space) => {
                                paused = !paused;
                                log::info!("Simulation {}", if paused { "paused" } else { "resumed" });
                            }
                            PhysicalKey::Code(KeyCode::KeyR) => {
                                // Reset food to initial patches
                                food_data.fill(0.0);
                                create_test_patches(&mut food_data, world_width, world_height, food_max);
                                queue.write_buffer(&food_buffer, 0, bytemuck::cast_slice(&food_data));
                                tick = 0;
                                log::info!("Food grid reset");
                            }
                            PhysicalKey::Code(KeyCode::KeyG) => {
                                spontaneous_enabled = !spontaneous_enabled;
                                log::info!("Spontaneous generation: {}", if spontaneous_enabled { "ON" } else { "OFF" });
                            }
                            PhysicalKey::Code(KeyCode::Digit1) => speed = 1,
                            PhysicalKey::Code(KeyCode::Digit2) => speed = 2,
                            PhysicalKey::Code(KeyCode::Digit3) => speed = 4,
                            PhysicalKey::Code(KeyCode::Digit4) => speed = 8,
                            PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                            _ => {}
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let scroll = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                        };
                        camera_zoom = (camera_zoom * 1.1_f32.powf(scroll)).clamp(0.1, 10.0);
                    }
                    WindowEvent::MouseInput { button: MouseButton::Middle, state, .. } => {
                        dragging = state == ElementState::Pressed;
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let new_pos = Vec2::new(position.x as f32, position.y as f32);
                        if dragging {
                            let delta = (new_pos - cursor_pos) / camera_zoom;
                            camera_pos.x -= delta.x;
                            camera_pos.y += delta.y;
                        }
                        cursor_pos = new_pos;
                    }
                    WindowEvent::RedrawRequested => {
                        // Update simulation
                        if !paused {
                            for _ in 0..speed {
                                // Update config
                                let config = WorldConfig {
                                    world_width,
                                    world_height,
                                    tick,
                                    food_growth_rate,
                                    food_max_per_cell: food_max,
                                    spontaneous_enabled: if spontaneous_enabled { 1 } else { 0 },
                                    _pad: [0; 2],
                                };
                                queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));
                                
                                // Run compute shader
                                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Compute Encoder"),
                                });
                                
                                {
                                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: Some("Food Compute Pass"),
                                        timestamp_writes: None,
                                    });
                                    pass.set_pipeline(&compute_pipeline);
                                    pass.set_bind_group(0, &compute_bind_group, &[]);
                                    pass.dispatch_workgroups((world_width + 7) / 8, (world_height + 7) / 8, 1);
                                }
                                
                                queue.submit(std::iter::once(encoder.finish()));
                                tick += 1;
                            }
                        }
                        
                        // Render
                        let output = surface.get_current_texture().expect("Failed to get surface texture");
                        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Render Encoder"),
                        });
                        
                        // Copy food buffer to texture
                        encoder.copy_buffer_to_texture(
                            wgpu::ImageCopyBuffer {
                                buffer: &food_buffer,
                                layout: wgpu::ImageDataLayout {
                                    offset: 0,
                                    bytes_per_row: Some(world_width * 4),
                                    rows_per_image: Some(world_height),
                                },
                            },
                            wgpu::ImageCopyTexture {
                                texture: &food_texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: world_width,
                                height: world_height,
                                depth_or_array_layers: 1,
                            },
                        );
                        
                        // Update camera
                        let camera = CameraUniform {
                            position: camera_pos.into(),
                            zoom: camera_zoom,
                            _pad1: 0.0,
                            viewport_size: [surface_config.width as f32, surface_config.height as f32],
                            world_size: [world_width as f32, world_height as f32],
                            food_max_per_cell: food_max,
                            _pad2: 0.0,
                        };
                        queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera));
                        
                        // Render pass
                        {
                            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Render Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            
                            render_pass.set_pipeline(&render_pipeline);
                            render_pass.set_bind_group(0, &render_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                            render_pass.draw(0..6, 0..1);
                        }
                        
                        queue.submit(std::iter::once(encoder.finish()));
                        output.present();
                        
                        // Log stats periodically
                        if tick % 60 == 0 && !paused {
                            log::info!("Tick {}: spontaneous={}", tick, spontaneous_enabled);
                        }
                    }
                    _ => {}
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

fn create_test_patches(food: &mut [f32], width: u32, height: u32, max_food: f32) {
    // Create 4 patches in known locations for visual verification
    let patches = [
        (width / 4, height / 4),      // Top-left quadrant
        (3 * width / 4, height / 4),  // Top-right quadrant
        (width / 4, 3 * height / 4),  // Bottom-left quadrant
        (3 * width / 4, 3 * height / 4), // Bottom-right quadrant
    ];
    
    for (cx, cy) in patches {
        let patch_size = 10i32;
        for dy in -patch_size..=patch_size {
            for dx in -patch_size..=patch_size {
                let x = (cx as i32 + dx).clamp(0, width as i32 - 1) as u32;
                let y = (cy as i32 + dy).clamp(0, height as i32 - 1) as u32;
                let idx = (y * width + x) as usize;
                food[idx] = max_food * 0.8;
            }
        }
    }
    
    log::info!("Created {} test food patches", patches.len());
}
