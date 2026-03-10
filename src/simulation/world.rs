//! World state - food grid and obstacles

use crate::config::SimulationConfig;
use rand::Rng;

/// Biome types - each has different environmental effects
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiomeType {
    Normal = 0,    // Standard environment
    Fertile = 1,   // Double food growth
    Barren = 2,    // Reduced food growth
    Swamp = 3,     // Slower movement
    Harsh = 4,     // Higher energy drain
}

impl BiomeType {
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => BiomeType::Fertile,
            2 => BiomeType::Barren,
            3 => BiomeType::Swamp,
            4 => BiomeType::Harsh,
            _ => BiomeType::Normal,
        }
    }
}

/// World grid state
#[allow(dead_code)]
pub struct World {
    pub width: u32,
    pub height: u32,
    pub food: Vec<f32>,
    pub obstacles: Vec<u32>,
    pub biomes: Vec<u8>, // Biome type per cell (0-4)
}

impl World {
    /// Create a new world with the given RNG for deterministic initialization
    pub fn new_with_rng<R: Rng>(config: &SimulationConfig, rng: &mut R) -> Self {
        let size = (config.world.width * config.world.height) as usize;
        
        // Start with baseline food level
        let mut food: Vec<f32> = vec![config.food.baseline_food; size];
        let obstacles = vec![0; size];
        
        // Generate biome map using Voronoi cells
        let biomes = Self::generate_biomes(
            config.world.width,
            config.world.height,
            config.biomes.biome_count,
            config.biomes.enabled,
            rng,
        );
        
        // Store patch centers for organism spawning distribution
        let mut patch_centers = Vec::new();
        
        // Initialize food patches spread across the map
        for _ in 0..config.food.initial_patches {
            let cx = rng.gen_range(0..config.world.width);
            let cy = rng.gen_range(0..config.world.height);
            patch_centers.push((cx, cy));
            
            // Create patch with gradient falloff (more food in center)
            let half_size = config.food.patch_size as i32 / 2;
            for dy in -half_size..=half_size {
                for dx in -half_size..=half_size {
                    let x = (cx as i32 + dx).clamp(0, config.world.width as i32 - 1) as u32;
                    let y = (cy as i32 + dy).clamp(0, config.world.height as i32 - 1) as u32;
                    let idx = (y * config.world.width + x) as usize;
                    
                    // Gradient: max food in center, less at edges
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    let max_dist = half_size as f32;
                    let falloff = 1.0 - (dist / max_dist).min(1.0);
                    // Fill patches with substantial food (70-100% of max)
                    let food_amount = config.food.max_per_cell * (0.7 + 0.3 * falloff);
                    food[idx] = food[idx].max(food_amount as f32);
                }
            }
        }
        
        let total_food: f32 = food.iter().sum();
        log::info!(
            "Created world {}x{} with {} food patches (size={}), total_food={:.0}, biomes={}",
            config.world.width,
            config.world.height,
            config.food.initial_patches,
            config.food.patch_size,
            total_food,
            if config.biomes.enabled { "enabled" } else { "disabled" }
        );
        
        Self {
            width: config.world.width,
            height: config.world.height,
            food,
            obstacles,
            biomes,
        }
    }
    
    #[allow(dead_code)]
    pub fn get_food(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.food[(y * self.width + x) as usize]
    }
    
    #[allow(dead_code)]
    pub fn set_food(&mut self, x: u32, y: u32, value: f32) {
        if x < self.width && y < self.height {
            self.food[(y * self.width + x) as usize] = value;
        }
    }
    
    #[allow(dead_code)]
    pub fn is_obstacle(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return true; // Out of bounds is obstacle
        }
        self.obstacles[(y * self.width + x) as usize] != 0
    }
    
    #[allow(dead_code)]
    pub fn set_obstacle(&mut self, x: u32, y: u32, is_obstacle: bool) {
        if x < self.width && y < self.height {
            self.obstacles[(y * self.width + x) as usize] = if is_obstacle { 1 } else { 0 };
        }
    }
    
    #[allow(dead_code)]
    pub fn total_food(&self) -> f32 {
        self.food.iter().sum()
    }
    
    /// Generate biome map using Voronoi-like cells (public static version)
    pub fn generate_biomes_static(
        width: u32,
        height: u32,
        biome_count: u32,
        enabled: bool,
        rng: &mut impl Rng,
    ) -> Vec<u8> {
        Self::generate_biomes(width, height, biome_count, enabled, rng)
    }
    
    /// Generate biome map using Voronoi-like cells
    fn generate_biomes(
        width: u32,
        height: u32,
        biome_count: u32,
        enabled: bool,
        rng: &mut impl Rng,
    ) -> Vec<u8> {
        let size = (width * height) as usize;
        
        if !enabled || biome_count == 0 {
            // All normal biomes when disabled
            return vec![0u8; size];
        }
        
        // Generate random Voronoi cell centers with assigned biome types
        let num_cells = biome_count as usize;
        let mut centers: Vec<(f32, f32, u8)> = Vec::with_capacity(num_cells);
        
        for _ in 0..num_cells {
            let x = rng.gen_range(0.0..width as f32);
            let y = rng.gen_range(0.0..height as f32);
            // Assign biome type with weighted distribution
            // Normal: 30%, Fertile: 20%, Barren: 20%, Swamp: 15%, Harsh: 15%
            let biome = match rng.gen_range(0..100) {
                0..30 => BiomeType::Normal as u8,
                30..50 => BiomeType::Fertile as u8,
                50..70 => BiomeType::Barren as u8,
                70..85 => BiomeType::Swamp as u8,
                _ => BiomeType::Harsh as u8,
            };
            centers.push((x, y, biome));
        }
        
        // Assign each cell to nearest Voronoi center
        let mut biomes = vec![0u8; size];
        for y in 0..height {
            for x in 0..width {
                let px = x as f32;
                let py = y as f32;
                
                // Find closest center (with world wrapping for seamless borders)
                let mut min_dist = f32::MAX;
                let mut closest_biome = 0u8;
                
                for &(cx, cy, biome) in &centers {
                    // Calculate wrapped distance
                    let dx = (px - cx).abs().min(width as f32 - (px - cx).abs());
                    let dy = (py - cy).abs().min(height as f32 - (py - cy).abs());
                    let dist_sq = dx * dx + dy * dy;
                    
                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        closest_biome = biome;
                    }
                }
                
                biomes[(y * width + x) as usize] = closest_biome;
            }
        }
        
        biomes
    }
    
    /// Get biome type at a world position
    #[allow(dead_code)]
    pub fn get_biome(&self, x: u32, y: u32) -> BiomeType {
        if x >= self.width || y >= self.height {
            return BiomeType::Normal;
        }
        BiomeType::from_u8(self.biomes[(y * self.width + x) as usize])
    }
    
    /// Get biome type at a float position (handles world wrapping)
    #[allow(dead_code)]
    pub fn get_biome_at(&self, x: f32, y: f32) -> BiomeType {
        // Wrap coordinates
        let wx = ((x % self.width as f32) + self.width as f32) % self.width as f32;
        let wy = ((y % self.height as f32) + self.height as f32) % self.height as f32;
        self.get_biome(wx as u32, wy as u32)
    }
}
