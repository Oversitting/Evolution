//! Rendering module

mod camera;

use anyhow::Result;
use wgpu::util::DeviceExt;
use glam::Vec2;
use crate::config::SimulationConfig;
use crate::simulation::Simulation;

#[allow(unused_imports)]
pub use camera::Camera;

/// Camera uniform for shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub position: [f32; 2],
    pub zoom: f32,
    pub _pad1: f32,
    pub viewport_size: [f32; 2],
    pub world_size: [f32; 2],
    pub food_max_per_cell: f32,  // For normalizing food display
    pub _pad2: f32,
}

/// Instance data for organism rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OrganismInstance {
    pub position: [f32; 2],
    pub rotation: f32,
    pub energy: f32,
    pub flags: u32,
    pub color: [f32; 3],
    pub morph_size: f32,  // Morphology size multiplier for rendering
}

/// Main renderer
#[allow(dead_code)]
pub struct Renderer {
    // Organism rendering
    organism_pipeline: wgpu::RenderPipeline,
    organism_vertex_buffer: wgpu::Buffer,
    organism_instance_buffer: wgpu::Buffer,
    
    // World rendering
    world_pipeline: wgpu::RenderPipeline,
    world_vertex_buffer: wgpu::Buffer,
    food_texture: wgpu::Texture,
    food_texture_view: wgpu::TextureView,
    food_sampler: wgpu::Sampler,
    
    // Camera
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    
    // World bind group (for food texture)
    world_bind_group: wgpu::BindGroup,
    
    // Selection state
    selected_organism: Option<u32>,
    
    // Config
    max_organisms: u32,
    world_width: u32,
    world_height: u32,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        config: &SimulationConfig,
    ) -> Result<Self> {
        // Create triangle vertices for organisms (pointing right)
        let organism_vertices: &[f32] = &[
            // Triangle pointing in +X direction (forward)
            1.0, 0.0,     // Tip (front)
            -0.5, 0.5,    // Back left
            -0.5, -0.5,   // Back right
        ];
        
        let organism_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Vertex Buffer"),
            contents: bytemuck::cast_slice(organism_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        // Instance buffer for organisms
        let instance_size = config.population.max_organisms as usize * std::mem::size_of::<OrganismInstance>();
        let organism_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Instance Buffer"),
            size: instance_size as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // World quad vertices (full screen)
        let world_vertices: &[f32] = &[
            // Position (x, y), UV (u, v)
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
        ];
        
        let world_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Vertex Buffer"),
            contents: bytemuck::cast_slice(world_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        // Food texture
        let food_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Food Texture"),
            size: wgpu::Extent3d {
                width: config.world.width,
                height: config.world.height,
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
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Camera uniform buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Camera bind group layout
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        
        // World bind group layout (for food texture)
        let world_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("World Bind Group Layout"),
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let world_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World Bind Group"),
            layout: &world_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&food_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&food_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create shaders
        let organism_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/organism.wgsl").into()),
        });
        
        let world_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("World Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/world_render.wgsl").into()),
        });
        
        // Organism pipeline
        let organism_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Organism Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let organism_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Organism Pipeline"),
            layout: Some(&organism_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &organism_shader,
                entry_point: "vs_main",
                buffers: &[
                    // Vertex buffer
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Instance buffer
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<OrganismInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2, // position
                            },
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32, // rotation
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32, // energy
                            },
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Uint32, // flags
                            },
                            wgpu::VertexAttribute {
                                offset: 20,
                                shader_location: 5,
                                format: wgpu::VertexFormat::Float32x3, // color
                            },
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 6,
                                format: wgpu::VertexFormat::Float32, // morph_size
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &organism_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        // World pipeline
        let world_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("World Pipeline Layout"),
            bind_group_layouts: &[&world_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let world_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("World Pipeline"),
            layout: Some(&world_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &world_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2, // position
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2, // uv
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &world_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        Ok(Self {
            organism_pipeline,
            organism_vertex_buffer,
            organism_instance_buffer,
            world_pipeline,
            world_vertex_buffer,
            food_texture,
            food_texture_view,
            food_sampler,
            camera_buffer,
            camera_bind_group,
            world_bind_group,
            selected_organism: None,
            max_organisms: config.population.max_organisms,
            world_width: config.world.width,
            world_height: config.world.height,
        })
    }
    
    /// Set the currently selected organism for highlighting
    pub fn set_selected_organism(&mut self, selected: Option<u32>) {
        self.selected_organism = selected;
    }
    
    pub fn resize(&mut self, _device: &wgpu::Device, _size: winit::dpi::PhysicalSize<u32>) {
        // Nothing to resize currently
    }
    
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        simulation: &Simulation,
        _camera_pos: Vec2,
        _camera_zoom: f32,
        _viewport_size: (u32, u32),
    ) {
        // Update camera uniform
        // Note: This should use queue.write_buffer but we don't have queue here
        // For now, camera update is handled in App
        
        // Begin render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Main Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        // Draw world (food layer)
        render_pass.set_pipeline(&self.world_pipeline);
        render_pass.set_bind_group(0, &self.world_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.world_vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
        
        // Draw organisms
        render_pass.set_pipeline(&self.organism_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.organism_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.organism_instance_buffer.slice(..));
        render_pass.draw(0..3, 0..simulation.organism_count());
    }
    
    #[allow(dead_code)]
    pub fn update_food_texture(&self, queue: &wgpu::Queue, food_data: &[f32]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.food_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(food_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.world_width * 4),
                rows_per_image: Some(self.world_height),
            },
            wgpu::Extent3d {
                width: self.world_width,
                height: self.world_height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Copy food data directly from GPU compute buffer to render texture
    /// This avoids costly GPU->CPU->GPU transfers
    pub fn copy_food_from_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        food_buffer: &wgpu::Buffer,
        world_size: (u32, u32),
    ) {
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: food_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(world_size.0 * 4), // 4 bytes per f32
                    rows_per_image: Some(world_size.1),
                },
            },
            wgpu::ImageCopyTexture {
                texture: &self.food_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: world_size.0,
                height: world_size.1,
                depth_or_array_layers: 1,
            },
        );
    }
    
    pub fn update_organisms(&self, queue: &wgpu::Queue, simulation: &Simulation) {
        let selected = self.selected_organism;
        
        // Find max generation for normalization
        let max_gen = simulation.organisms.iter()
            .filter(|o| o.is_alive())
            .map(|o| o.generation)
            .max()
            .unwrap_or(1)
            .max(1);
        
        let instances: Vec<OrganismInstance> = simulation
            .organisms
            .iter()
            .enumerate()
            .map(|(idx, org)| {
                // Determine color based on species_id (distinct hue per species)
                let mut color = [0.8, 0.8, 0.8]; // Default grey for unknown species
                
                // Check if this organism is selected - give it a bright white/cyan highlight
                let is_selected = selected == Some(idx as u32);
                
                if is_selected {
                    // Bright cyan/white for selected organism
                    color = [0.2, 1.0, 1.0];
                } else if org.species_id > 0 {
                    // Use species_id to generate a distinct hue via golden ratio
                    // This ensures adjacent species IDs get maximally different colors
                    let golden_ratio = 0.618033988749895;
                    let hue = (org.species_id as f32 * golden_ratio).fract();
                    
                    // HSV to RGB conversion with high saturation and medium-high value
                    let saturation = 0.7;
                    let value = 0.9;
                    
                    let h_i = (hue * 6.0) as u32;
                    let f = hue * 6.0 - h_i as f32;
                    let p = value * (1.0 - saturation);
                    let q = value * (1.0 - f * saturation);
                    let t = value * (1.0 - (1.0 - f) * saturation);
                    
                    color = match h_i % 6 {
                        0 => [value, t, p],
                        1 => [q, value, p],
                        2 => [p, value, t],
                        3 => [p, q, value],
                        4 => [t, p, value],
                        _ => [value, p, q],
                    };
                    
                    // Blend in generation-based brightness
                    // Higher generations get slightly brighter
                    let gen_factor = (org.generation as f32 / max_gen as f32).min(1.0);
                    let brightness_boost = 0.1 * gen_factor;
                    color[0] = (color[0] + brightness_boost).clamp(0.1, 1.0);
                    color[1] = (color[1] + brightness_boost).clamp(0.1, 1.0);
                    color[2] = (color[2] + brightness_boost).clamp(0.1, 1.0);
                } else if let Some(genome) = simulation.genomes.get(org.genome_id) {
                    // Fallback: hash weights to color if species not assigned
                    let mut r = 0.0;
                    let mut g = 0.0;
                    let mut b = 0.0;
                    
                    for (i, w) in genome.weights_l1.iter().enumerate() {
                        match i % 3 {
                            0 => r += w,
                            1 => g += w,
                            2 => b += w,
                            _ => {}
                        }
                    }
                    
                    let gain = 0.1;
                    color = [
                        1.0 / (1.0 + (-r * gain).exp()),
                        1.0 / (1.0 + (-g * gain).exp()),
                        1.0 / (1.0 + (-b * gain).exp()),
                    ];
                }

                OrganismInstance {
                    position: org.position.into(),
                    rotation: org.rotation,
                    energy: org.energy,
                    flags: if org.is_alive() { 1 } else { 0 },
                    color,
                    morph_size: org.morph_size,
                }
            })
            .collect();
        
        queue.write_buffer(&self.organism_instance_buffer, 0, bytemuck::cast_slice(&instances));
    }
    
    pub fn update_camera(&self, queue: &wgpu::Queue, pos: Vec2, zoom: f32, viewport: (u32, u32), world_size: (u32, u32), food_max_per_cell: f32) {
        let uniform = CameraUniform {
            position: pos.into(),
            zoom,
            _pad1: 0.0,
            viewport_size: [viewport.0 as f32, viewport.1 as f32],
            world_size: [world_size.0 as f32, world_size.1 as f32],
            food_max_per_cell,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}
