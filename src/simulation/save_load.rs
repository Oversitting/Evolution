//! Save/Load system for simulation state

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::config::SimulationConfig;
use super::organism::Organism;
use super::genome::{Genome, MorphTraits};
use glam::Vec2;

/// Serializable organism data
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SavedOrganism {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub rotation: f32,
    pub energy: f32,
    pub age: u32,
    pub alive: bool,
    pub genome_id: u32,
    pub generation: u32,
    pub offspring_count: u32,
    pub parent_id: u32,
    pub cooldown: u32,
    pub reproduce_signal: f32,
    #[serde(default)]
    pub species_id: u32,
    // Morphology traits
    #[serde(default = "default_morph_size")]
    pub morph_size: f32,
    #[serde(default = "default_morph_speed_mult")]
    pub morph_speed_mult: f32,
    #[serde(default = "default_morph_vision_mult")]
    pub morph_vision_mult: f32,
    #[serde(default = "default_morph_metabolism")]
    pub morph_metabolism: f32,
}

fn default_morph_size() -> f32 { 1.0 }
fn default_morph_speed_mult() -> f32 { 1.0 }
fn default_morph_vision_mult() -> f32 { 1.0 }
fn default_morph_metabolism() -> f32 { 1.0 }

impl From<&Organism> for SavedOrganism {
    fn from(org: &Organism) -> Self {
        Self {
            position: org.position.into(),
            velocity: org.velocity.into(),
            rotation: org.rotation,
            energy: org.energy,
            age: org.age,
            alive: org.alive,
            genome_id: org.genome_id,
            generation: org.generation,
            offspring_count: org.offspring_count,
            parent_id: org.parent_id,
            cooldown: org.cooldown,
            reproduce_signal: org.reproduce_signal,
            species_id: org.species_id,
            morph_size: org.morph_size,
            morph_speed_mult: org.morph_speed_mult,
            morph_vision_mult: org.morph_vision_mult,
            morph_metabolism: org.morph_metabolism,
        }
    }
}

impl From<&SavedOrganism> for Organism {
    fn from(saved: &SavedOrganism) -> Self {
        Self {
            position: Vec2::from(saved.position),
            velocity: Vec2::from(saved.velocity),
            rotation: saved.rotation,
            energy: saved.energy,
            age: saved.age,
            alive: saved.alive,
            genome_id: saved.genome_id,
            generation: saved.generation,
            offspring_count: saved.offspring_count,
            parent_id: saved.parent_id,
            cooldown: saved.cooldown,
            reproduce_signal: saved.reproduce_signal,
            species_id: saved.species_id,
            morph_size: saved.morph_size,
            morph_speed_mult: saved.morph_speed_mult,
            morph_vision_mult: saved.morph_vision_mult,
            morph_metabolism: saved.morph_metabolism,
        }
    }
}

/// Serializable genome data
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SavedGenome {
    pub weights_l1: Vec<f32>,
    pub biases_l1: Vec<f32>,
    pub weights_l2: Vec<f32>,
    pub biases_l2: Vec<f32>,
    pub alive: bool,
    // Morphology traits
    #[serde(default = "default_morph_size")]
    pub morph_size: f32,
    #[serde(default = "default_morph_speed_mult")]
    pub morph_speed_mult: f32,
    #[serde(default = "default_morph_vision_mult")]
    pub morph_vision_mult: f32,
    #[serde(default = "default_morph_metabolism")]
    pub morph_metabolism: f32,
}

impl From<&Genome> for SavedGenome {
    fn from(g: &Genome) -> Self {
        Self {
            weights_l1: g.weights_l1.clone(),
            biases_l1: g.biases_l1.clone(),
            weights_l2: g.weights_l2.clone(),
            biases_l2: g.biases_l2.clone(),
            alive: g.alive,
            morph_size: g.morphology.size,
            morph_speed_mult: g.morphology.speed_mult,
            morph_vision_mult: g.morphology.vision_mult,
            morph_metabolism: g.morphology.metabolism,
        }
    }
}

impl From<&SavedGenome> for Genome {
    fn from(saved: &SavedGenome) -> Self {
        Self {
            weights_l1: saved.weights_l1.clone(),
            biases_l1: saved.biases_l1.clone(),
            weights_l2: saved.weights_l2.clone(),
            biases_l2: saved.biases_l2.clone(),
            morphology: MorphTraits {
                size: saved.morph_size,
                speed_mult: saved.morph_speed_mult,
                vision_mult: saved.morph_vision_mult,
                metabolism: saved.morph_metabolism,
            },
            alive: saved.alive,
        }
    }
}

/// Complete simulation save state
#[derive(Serialize, Deserialize)]
pub struct SaveState {
    /// Version for compatibility checking
    pub version: u32,
    /// Simulation tick
    pub tick: u64,
    /// Configuration
    pub config: SimulationConfig,
    /// All organisms
    pub organisms: Vec<SavedOrganism>,
    /// All genomes
    pub genomes: Vec<SavedGenome>,
    /// World food grid
    pub food: Vec<f32>,
    /// World dimensions
    pub world_width: u32,
    pub world_height: u32,
}

impl SaveState {
    /// Current save format version
    pub const VERSION: u32 = 1;
    
    /// Save to a binary file
    pub fn save_to_file(&self, path: &Path) -> Result<(), SaveError> {
        let file = File::create(path)
            .map_err(|e| SaveError::Io(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| SaveError::Serialize(e.to_string()))?;
        log::info!("Saved simulation to {:?} (tick {})", path, self.tick);
        Ok(())
    }
    
    /// Load from a binary file
    pub fn load_from_file(path: &Path) -> Result<Self, SaveError> {
        let file = File::open(path)
            .map_err(|e| SaveError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        let state: SaveState = bincode::deserialize_from(reader)
            .map_err(|e| SaveError::Deserialize(e.to_string()))?;
        
        if state.version != Self::VERSION {
            return Err(SaveError::VersionMismatch {
                expected: Self::VERSION,
                found: state.version,
            });
        }
        
        log::info!("Loaded simulation from {:?} (tick {})", path, state.tick);
        Ok(state)
    }
}

/// Errors that can occur during save/load
#[derive(Debug)]
pub enum SaveError {
    Io(String),
    Serialize(String),
    Deserialize(String),
    VersionMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaveError::Io(e) => write!(f, "IO error: {}", e),
            SaveError::Serialize(e) => write!(f, "Serialization error: {}", e),
            SaveError::Deserialize(e) => write!(f, "Deserialization error: {}", e),
            SaveError::VersionMismatch { expected, found } => {
                write!(f, "Save version mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for SaveError {}
