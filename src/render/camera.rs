//! Camera module for 2D viewport control

use glam::Vec2;

/// 2D camera for viewing the simulation world
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Camera {
    /// Camera position in world space (center of view)
    pub position: Vec2,
    
    /// Zoom level (pixels per world unit)
    pub zoom: f32,
    
    /// Minimum zoom level
    pub min_zoom: f32,
    
    /// Maximum zoom level
    pub max_zoom: f32,
    
    /// Viewport size in pixels
    pub viewport_size: Vec2,
    
    /// World bounds for clamping
    pub world_size: Vec2,
    
    /// Is the camera being dragged?
    pub is_dragging: bool,
    
    /// Last mouse position during drag
    pub drag_start: Vec2,
}

#[allow(dead_code)]
impl Camera {
    pub fn new(world_width: f32, world_height: f32) -> Self {
        Self {
            position: Vec2::new(world_width / 2.0, world_height / 2.0),
            zoom: 1.0,
            min_zoom: 0.1,
            max_zoom: 10.0,
            viewport_size: Vec2::new(1920.0, 1080.0),
            world_size: Vec2::new(world_width, world_height),
            is_dragging: false,
            drag_start: Vec2::ZERO,
        }
    }
    
    /// Set viewport size
    pub fn set_viewport(&mut self, width: f32, height: f32) {
        self.viewport_size = Vec2::new(width, height);
        // Adjust zoom to fit world if needed
        self.clamp();
    }
    
    /// Pan the camera by a delta in screen pixels
    pub fn pan(&mut self, delta_screen: Vec2) {
        // Convert screen delta to world delta (inverted because we move camera opposite to drag)
        let world_delta = delta_screen / self.zoom;
        self.position -= world_delta;
        self.clamp();
    }
    
    /// Zoom the camera, centered on a screen position
    pub fn zoom_at(&mut self, delta: f32, screen_pos: Vec2) {
        // Convert screen position to world position before zoom
        let world_pos = self.screen_to_world(screen_pos);
        
        // Apply zoom
        let old_zoom = self.zoom;
        self.zoom *= 1.0 + delta * 0.1;
        self.zoom = self.zoom.clamp(self.min_zoom, self.max_zoom);
        
        // Adjust position so the world point under cursor stays in place
        if (self.zoom - old_zoom).abs() > 0.001 {
            let new_screen_pos = self.world_to_screen(world_pos);
            let screen_delta = screen_pos - new_screen_pos;
            self.position -= screen_delta / self.zoom;
        }
        
        self.clamp();
    }
    
    /// Set zoom level directly
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.clamp(self.min_zoom, self.max_zoom);
        self.clamp();
    }
    
    /// Center on a world position
    pub fn center_on(&mut self, world_pos: Vec2) {
        self.position = world_pos;
        self.clamp();
    }
    
    /// Reset to view entire world
    pub fn reset(&mut self) {
        self.position = self.world_size / 2.0;
        // Calculate zoom to fit world in viewport
        let zoom_x = self.viewport_size.x / self.world_size.x;
        let zoom_y = self.viewport_size.y / self.world_size.y;
        self.zoom = zoom_x.min(zoom_y).clamp(self.min_zoom, self.max_zoom);
    }
    
    /// Convert screen coordinates to world coordinates
    pub fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        let center = self.viewport_size / 2.0;
        let offset = (screen_pos - center) / self.zoom;
        self.position + offset
    }
    
    /// Convert world coordinates to screen coordinates
    pub fn world_to_screen(&self, world_pos: Vec2) -> Vec2 {
        let center = self.viewport_size / 2.0;
        let offset = (world_pos - self.position) * self.zoom;
        center + offset
    }
    
    /// Get the visible world bounds (min, max)
    pub fn visible_bounds(&self) -> (Vec2, Vec2) {
        let half_size = self.viewport_size / (2.0 * self.zoom);
        let min = self.position - half_size;
        let max = self.position + half_size;
        (min, max)
    }
    
    /// Clamp camera position to keep view within world bounds (with some margin)
    fn clamp(&mut self) {
        let margin = 0.1; // Allow 10% outside world
        let half_view = self.viewport_size / (2.0 * self.zoom);
        
        let min_pos = half_view - self.world_size * margin;
        let max_pos = self.world_size * (1.0 + margin) - half_view;
        
        // Only clamp if the world is smaller than the view
        if max_pos.x > min_pos.x {
            self.position.x = self.position.x.clamp(min_pos.x, max_pos.x);
        } else {
            self.position.x = self.world_size.x / 2.0;
        }
        
        if max_pos.y > min_pos.y {
            self.position.y = self.position.y.clamp(min_pos.y, max_pos.y);
        } else {
            self.position.y = self.world_size.y / 2.0;
        }
    }
    
    /// Start dragging from a screen position
    pub fn start_drag(&mut self, screen_pos: Vec2) {
        self.is_dragging = true;
        self.drag_start = screen_pos;
    }
    
    /// Update drag with new screen position
    pub fn update_drag(&mut self, screen_pos: Vec2) {
        if self.is_dragging {
            let delta = screen_pos - self.drag_start;
            self.pan(delta);
            self.drag_start = screen_pos;
        }
    }
    
    /// End dragging
    pub fn end_drag(&mut self) {
        self.is_dragging = false;
    }
}
