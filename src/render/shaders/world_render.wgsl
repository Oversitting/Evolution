// World rendering shader (food layer)

struct CameraUniform {
    position: vec2<f32>,
    zoom: f32,
    _pad1: f32,
    viewport_size: vec2<f32>,
    world_size: vec2<f32>,
    food_max_per_cell: f32,  // For normalizing food display
    _pad2: f32,
}

@group(0) @binding(0)
var food_texture: texture_2d<f32>;
@group(0) @binding(1)
var food_sampler: sampler;
@group(0) @binding(2)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) world_size: vec2<f32>,
    @location(2) food_max: f32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Input position is in clip space (-1 to 1)
    // Calculate which part of the world this covers based on camera
    let half_viewport = camera.viewport_size * 0.5 / camera.zoom;
    
    // Map clip space to world coordinates
    // Flip Y: Screen Top (+1) corresponds to World Top (Lower Y)
    // input.position.y is +1 at top. We want lower Y.
    // So we SUBTRACT the offset for Y.
    output.world_pos.x = camera.position.x + input.position.x * half_viewport.x;
    output.world_pos.y = camera.position.y - input.position.y * half_viewport.y;
    
    output.world_size = camera.world_size;
    output.food_max = camera.food_max_per_cell;
    
    output.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_size = input.world_size;
    
    // Check if within world bounds
    if input.world_pos.x < 0.0 || input.world_pos.x >= world_size.x ||
       input.world_pos.y < 0.0 || input.world_pos.y >= world_size.y {
        // Outside world - dark background
        return vec4<f32>(0.02, 0.02, 0.05, 1.0);
    }
    
    // Sample food texture
    let uv = input.world_pos / world_size;
    let food_raw = textureSample(food_texture, food_sampler, uv).r;
    
    // Normalize food value by max_per_cell so it displays correctly
    let food = food_raw / max(input.food_max, 1.0);
    
    // Base color (dark brown ground)
    let base_color = vec3<f32>(0.08, 0.06, 0.04);
    
    // Food color (yellow-orange to distinguish from organisms)
    let food_color = vec3<f32>(0.9, 0.7, 0.1);
    
    // Blend based on food amount (normalized to 0-1)
    let color = mix(base_color, food_color, clamp(food, 0.0, 1.0));
    
    // Add slight grid pattern for visual interest
    let grid = step(0.98, fract(input.world_pos.x)) + step(0.98, fract(input.world_pos.y));
    let grid_color = color + vec3<f32>(0.02) * grid;
    
    return vec4<f32>(grid_color, 1.0);
}
