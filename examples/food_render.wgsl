// Food rendering shader for testing

struct CameraUniform {
    position: vec2<f32>,
    zoom: f32,
    _pad1: f32,
    viewport_size: vec2<f32>,
    world_size: vec2<f32>,
    food_max_per_cell: f32,
    _pad2: f32,
}

@group(0) @binding(0) var food_texture: texture_2d<f32>;
@group(0) @binding(1) var food_sampler: sampler;
@group(0) @binding(2) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    let half_viewport = camera.viewport_size * 0.5 / camera.zoom;
    
    output.world_pos.x = camera.position.x + input.position.x * half_viewport.x;
    output.world_pos.y = camera.position.y - input.position.y * half_viewport.y;
    
    output.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Check bounds
    if input.world_pos.x < 0.0 || input.world_pos.x >= camera.world_size.x ||
       input.world_pos.y < 0.0 || input.world_pos.y >= camera.world_size.y {
        return vec4<f32>(0.02, 0.02, 0.05, 1.0);
    }
    
    // Sample food texture
    let uv = input.world_pos / camera.world_size;
    let food_raw = textureSample(food_texture, food_sampler, uv).r;
    
    // Normalize
    let food = food_raw / max(camera.food_max_per_cell, 1.0);
    
    // Colors
    let base_color = vec3<f32>(0.05, 0.08, 0.05);
    let food_color = vec3<f32>(0.1, 0.8, 0.2);
    
    // Blend
    let color = mix(base_color, food_color, clamp(food, 0.0, 1.0));
    
    // Grid pattern for debugging
    let grid = step(0.95, fract(input.world_pos.x)) + step(0.95, fract(input.world_pos.y));
    let grid_color = color + vec3<f32>(0.03) * grid;
    
    return vec4<f32>(grid_color, 1.0);
}
