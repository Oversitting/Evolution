//! Statistics tracking and visualization
//!
//! Tracks simulation metrics over time for graphing and analysis.

use std::collections::VecDeque;

/// Maximum history length for rolling statistics
pub const MAX_HISTORY: usize = 600; // ~10 seconds at 60 FPS

/// Rolling statistics history
#[derive(Clone, Debug)]
pub struct StatsHistory {
    /// Population count over time
    pub population: VecDeque<u32>,
    /// Average energy over time
    pub avg_energy: VecDeque<f32>,
    /// Maximum generation over time
    pub max_generation: VecDeque<u32>,
    /// Total food over time  
    pub total_food: VecDeque<f32>,
    /// Births in the last interval
    pub births: VecDeque<u32>,
    /// Deaths in the last interval
    pub deaths: VecDeque<u32>,
    /// Tick counter for sparse sampling
    sample_counter: u32,
    /// Births since last sample
    births_accumulator: u32,
    /// Deaths since last sample
    deaths_accumulator: u32,
}

impl Default for StatsHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsHistory {
    pub fn new() -> Self {
        Self {
            population: VecDeque::with_capacity(MAX_HISTORY),
            avg_energy: VecDeque::with_capacity(MAX_HISTORY),
            max_generation: VecDeque::with_capacity(MAX_HISTORY),
            total_food: VecDeque::with_capacity(MAX_HISTORY),
            births: VecDeque::with_capacity(MAX_HISTORY),
            deaths: VecDeque::with_capacity(MAX_HISTORY),
            sample_counter: 0,
            births_accumulator: 0,
            deaths_accumulator: 0,
        }
    }
    
    /// Record a single tick's data
    /// Data is sampled every N ticks to reduce memory usage
    pub fn record(
        &mut self,
        population: u32,
        avg_energy: f32,
        max_generation: u32,
        total_food: f32,
        births_this_tick: u32,
        deaths_this_tick: u32,
    ) {
        // Accumulate births/deaths
        self.births_accumulator += births_this_tick;
        self.deaths_accumulator += deaths_this_tick;
        self.sample_counter += 1;
        
        // Sample every 10 ticks for smooth graphs without excessive memory
        const SAMPLE_INTERVAL: u32 = 10;
        if self.sample_counter >= SAMPLE_INTERVAL {
            self.push_sample(population, avg_energy, max_generation, total_food);
            self.sample_counter = 0;
        }
    }
    
    fn push_sample(&mut self, population: u32, avg_energy: f32, max_generation: u32, total_food: f32) {
        // Push to deques, removing old data if over capacity
        if self.population.len() >= MAX_HISTORY {
            self.population.pop_front();
        }
        self.population.push_back(population);
        
        if self.avg_energy.len() >= MAX_HISTORY {
            self.avg_energy.pop_front();
        }
        self.avg_energy.push_back(avg_energy);
        
        if self.max_generation.len() >= MAX_HISTORY {
            self.max_generation.pop_front();
        }
        self.max_generation.push_back(max_generation);
        
        if self.total_food.len() >= MAX_HISTORY {
            self.total_food.pop_front();
        }
        self.total_food.push_back(total_food);
        
        if self.births.len() >= MAX_HISTORY {
            self.births.pop_front();
        }
        self.births.push_back(self.births_accumulator);
        self.births_accumulator = 0;
        
        if self.deaths.len() >= MAX_HISTORY {
            self.deaths.pop_front();
        }
        self.deaths.push_back(self.deaths_accumulator);
        self.deaths_accumulator = 0;
    }
    
    /// Get population data as a slice for plotting
    pub fn population_slice(&self) -> Vec<f32> {
        self.population.iter().map(|&x| x as f32).collect()
    }
    
    /// Get average energy data for plotting
    pub fn avg_energy_slice(&self) -> Vec<f32> {
        self.avg_energy.iter().copied().collect()
    }
    
    /// Get max generation data for plotting
    pub fn max_generation_slice(&self) -> Vec<f32> {
        self.max_generation.iter().map(|&x| x as f32).collect()
    }
    
    /// Get total food data for plotting
    pub fn total_food_slice(&self) -> Vec<f32> {
        self.total_food.iter().copied().collect()
    }
    
    /// Get birth rate (births per sample interval)
    pub fn births_slice(&self) -> Vec<f32> {
        self.births.iter().map(|&x| x as f32).collect()
    }
    
    /// Get death rate (deaths per sample interval)
    pub fn deaths_slice(&self) -> Vec<f32> {
        self.deaths.iter().map(|&x| x as f32).collect()
    }
    
    /// Get min/max for a dataset
    pub fn min_max(data: &[f32]) -> (f32, f32) {
        if data.is_empty() {
            return (0.0, 1.0);
        }
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Ensure range is at least 1 to avoid division by zero
        if (max - min).abs() < 0.001 {
            (min - 0.5, max + 0.5)
        } else {
            (min, max)
        }
    }
}

/// Event tracking for births and deaths
/// (Currently unused - births/deaths tracked in App and passed via UiData)
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct TickEvents {
    /// Number of organisms born this tick
    pub births: u32,
    /// Number of organisms that died this tick
    pub deaths: u32,
}

#[allow(dead_code)]
impl TickEvents {
    pub fn new() -> Self {
        Self { births: 0, deaths: 0 }
    }
    
    pub fn reset(&mut self) {
        self.births = 0;
        self.deaths = 0;
    }
    
    pub fn record_birth(&mut self) {
        self.births += 1;
    }
    
    pub fn record_death(&mut self) {
        self.deaths += 1;
    }
}
