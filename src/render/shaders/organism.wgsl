// Organism rendering shader

struct CameraUniform {
    position: vec2<f32>,
    zoom: f32,
    _pad1: f32,
    viewport_size: vec2<f32>,
    _pad2: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) vertex_pos: vec2<f32>,
}

struct InstanceInput {
    @location(1) position: vec2<f32>,
    @location(2) rotation: f32,
    @location(3) energy: f32,
    @location(4) flags: u32,
    @location(5) color: vec3<f32>,
    @location(6) morph_size: f32,  // Morphology size multiplier
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) energy: f32,
    @location(1) is_alive: f32,
    @location(2) color: vec3<f32>,
}

const BASE_ORGANISM_SIZE: f32 = 5.0;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Check if alive
    let is_alive = f32(instance.flags & 1u);
    output.is_alive = is_alive;
    output.energy = instance.energy;
    output.color = instance.color;
    
    // Skip dead organisms by placing them off-screen
    if is_alive < 0.5 {
        output.clip_position = vec4<f32>(0.0, 0.0, -10.0, 1.0);
        return output;
    }
    
    // Rotate vertex
    let cos_r = cos(instance.rotation);
    let sin_r = sin(instance.rotation);
    let rotated = vec2<f32>(
        vertex.vertex_pos.x * cos_r - vertex.vertex_pos.y * sin_r,
        vertex.vertex_pos.x * sin_r + vertex.vertex_pos.y * cos_r
    );
    
    // Scale by organism size (base size * morphology multiplier)
    let organism_size = BASE_ORGANISM_SIZE * instance.morph_size;
    let scaled = rotated * organism_size;
    
    // Add world position
    let world_pos = instance.position + scaled;
    
    // Transform to screen space
    let view_offset = world_pos - camera.position;
    let screen_pos = view_offset * camera.zoom;
    
    // Normalize to clip space (-1 to 1)
    // Flip Y: World Y+ (Down) should map to Screen Bottom (Clip -1)
    // If screen_pos.y is positive, clip_pos.y should be negative.
    let clip_pos = screen_pos / (camera.viewport_size * 0.5);
    
    output.clip_position = vec4<f32>(clip_pos.x, -clip_pos.y, 0.0, 1.0);
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Skip dead organisms
    if input.is_alive < 0.5 {
        discard;
    }
    
    // Color based on genome (passed from CPU)
    // Modulate brightness slightly by energy (dimmer when running out)
    let energy_factor = 0.5 + 0.5 * clamp(input.energy / 100.0, 0.0, 1.0);
    let color = input.color * energy_factor;
    
    return vec4<f32>(color, 1.0);
}
