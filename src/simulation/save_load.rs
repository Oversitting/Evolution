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

/// Curated genome entry persisted across runs for smarter future founders.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SurvivorEntry {
    pub genome: SavedGenome,
    pub generation: u32,
    pub offspring_count: u32,
    pub age: u32,
    pub energy: f32,
    pub species_id: u32,
    pub score: f32,
}

/// Persistent survivor bank used to seed future runs with proven founders.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SurvivorBank {
    pub version: u32,
    pub source_tick: u64,
    pub entries: Vec<SurvivorEntry>,
}

/// Human-readable founder record with evaluation metadata.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FounderRecord {
    #[serde(default = "default_founder_enabled")]
    pub enabled: bool,
    pub label: String,
    pub source: String,
    pub genome: SavedGenome,
    pub generation: u32,
    pub offspring_count: u32,
    pub age: u32,
    pub energy: f32,
    pub species_id: u32,
    pub score: f32,
    #[serde(default)]
    pub evaluations: u32,
    #[serde(default)]
    pub successes: u32,
    #[serde(default)]
    pub best_steps_to_food: u32,
    #[serde(default)]
    pub average_steps_to_food: f32,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Human-readable founder pool for explicit inspection and curation.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FounderPool {
    pub version: u32,
    pub source_tick: u64,
    pub description: String,
    pub entries: Vec<FounderRecord>,
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

impl SurvivorBank {
    pub const VERSION: u32 = 1;

    pub fn quality_score(&self) -> f32 {
        self.entries
            .iter()
            .take(64)
            .map(|entry| entry.score)
            .sum()
    }

    #[allow(dead_code)]
    pub fn is_stronger_than(&self, other: &Self) -> bool {
        self.quality_score() > other.quality_score()
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), SaveError> {
        let file = File::create(path)
            .map_err(|e| SaveError::Io(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| SaveError::Serialize(e.to_string()))?;
        log::info!(
            "Saved survivor bank to {:?} (tick {}, entries={})",
            path,
            self.source_tick,
            self.entries.len()
        );
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self, SaveError> {
        let file = File::open(path)
            .map_err(|e| SaveError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        let bank: SurvivorBank = bincode::deserialize_from(reader)
            .map_err(|e| SaveError::Deserialize(e.to_string()))?;

        if bank.version != Self::VERSION {
            return Err(SaveError::VersionMismatch {
                expected: Self::VERSION,
                found: bank.version,
            });
        }

        log::info!(
            "Loaded survivor bank from {:?} (tick {}, entries={})",
            path,
            bank.source_tick,
            bank.entries.len()
        );
        Ok(bank)
    }
}

impl FounderPool {
    pub const VERSION: u32 = 1;

    pub fn quality_score(&self) -> f32 {
        self.entries
            .iter()
            .filter(|entry| entry.enabled)
            .take(64)
            .map(|entry| entry.score)
            .sum()
    }

    #[allow(dead_code)]
    pub fn is_stronger_than(&self, other: &Self) -> bool {
        self.quality_score() > other.quality_score()
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), SaveError> {
        let file = File::create(path)
            .map_err(|e| SaveError::Io(e.to_string()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| SaveError::Serialize(e.to_string()))?;
        log::info!(
            "Saved founder pool to {:?} (tick {}, entries={})",
            path,
            self.source_tick,
            self.entries.len()
        );
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self, SaveError> {
        let file = File::open(path)
            .map_err(|e| SaveError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        let pool: FounderPool = serde_json::from_reader(reader)
            .map_err(|e| SaveError::Deserialize(e.to_string()))?;

        if pool.version != Self::VERSION {
            return Err(SaveError::VersionMismatch {
                expected: Self::VERSION,
                found: pool.version,
            });
        }

        log::info!(
            "Loaded founder pool from {:?} (tick {}, entries={})",
            path,
            pool.source_tick,
            pool.entries.len()
        );
        Ok(pool)
    }

    pub fn from_survivor_bank(bank: &SurvivorBank, source: &str, description: &str) -> Self {
        let entries = bank
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| FounderRecord {
                enabled: true,
                label: format!("{}-{:03}", source, index + 1),
                source: source.to_string(),
                genome: entry.genome.clone(),
                generation: entry.generation,
                offspring_count: entry.offspring_count,
                age: entry.age,
                energy: entry.energy,
                species_id: entry.species_id,
                score: entry.score,
                evaluations: 0,
                successes: 0,
                best_steps_to_food: 0,
                average_steps_to_food: 0.0,
                notes: String::new(),
                tags: vec![source.to_string()],
            })
            .collect();

        Self {
            version: Self::VERSION,
            source_tick: bank.source_tick,
            description: description.to_string(),
            entries,
        }
    }

    pub fn to_survivor_entries(&self, limit: usize) -> Vec<SurvivorEntry> {
        let mut entries: Vec<SurvivorEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.enabled)
            .map(|entry| SurvivorEntry {
                genome: entry.genome.clone(),
                generation: entry.generation,
                offspring_count: entry.offspring_count,
                age: entry.age,
                energy: entry.energy,
                species_id: entry.species_id,
                score: entry.score,
            })
            .collect();
        entries.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(limit);
        entries
    }
}

fn default_founder_enabled() -> bool { true }

fn path_uses_founder_pool(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

pub fn load_bootstrap_entries(path: &Path, limit: usize) -> Result<Vec<SurvivorEntry>, SaveError> {
    if path_uses_founder_pool(path) {
        Ok(FounderPool::load_from_file(path)?.to_survivor_entries(limit))
    } else {
        Ok(SurvivorBank::load_from_file(path)?
            .entries
            .into_iter()
            .take(limit)
            .collect())
    }
}

pub fn load_bootstrap_quality_score(path: &Path) -> Result<f32, SaveError> {
    if path_uses_founder_pool(path) {
        Ok(FounderPool::load_from_file(path)?.quality_score())
    } else {
        Ok(SurvivorBank::load_from_file(path)?.quality_score())
    }
}

pub fn save_bootstrap_bank(path: &Path, bank: &SurvivorBank, source: &str, description: &str) -> Result<(), SaveError> {
    if path_uses_founder_pool(path) {
        FounderPool::from_survivor_bank(bank, source, description).save_to_file(path)
    } else {
        bank.save_to_file(path)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_genome() -> SavedGenome {
        SavedGenome {
            weights_l1: vec![0.1, 0.2],
            biases_l1: vec![0.3],
            weights_l2: vec![0.4, 0.5],
            biases_l2: vec![0.6],
            alive: true,
            morph_size: 1.0,
            morph_speed_mult: 1.0,
            morph_vision_mult: 1.0,
            morph_metabolism: 1.0,
        }
    }

    fn founder(label: &str, score: f32, enabled: bool) -> FounderRecord {
        FounderRecord {
            enabled,
            label: label.to_string(),
            source: "test".to_string(),
            genome: sample_genome(),
            generation: 1,
            offspring_count: 0,
            age: 10,
            energy: 50.0,
            species_id: 7,
            score,
            evaluations: 10,
            successes: 5,
            best_steps_to_food: 12,
            average_steps_to_food: 18.0,
            notes: String::new(),
            tags: vec!["tag".to_string()],
        }
    }

    #[test]
    fn founder_pool_quality_score_ignores_disabled_entries() {
        let pool = FounderPool {
            version: FounderPool::VERSION,
            source_tick: 42,
            description: "test".to_string(),
            entries: vec![
                founder("enabled-a", 100.0, true),
                founder("disabled", 5000.0, false),
                founder("enabled-b", 80.0, true),
            ],
        };

        assert_eq!(pool.quality_score(), 180.0);
    }

    #[test]
    fn founder_pool_bootstrap_entries_only_include_enabled_founders() {
        let pool = FounderPool {
            version: FounderPool::VERSION,
            source_tick: 42,
            description: "test".to_string(),
            entries: vec![
                founder("enabled-low", 10.0, true),
                founder("disabled-high", 1000.0, false),
                founder("enabled-high", 50.0, true),
            ],
        };

        let entries = pool.to_survivor_entries(8);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].score, 50.0);
        assert_eq!(entries[1].score, 10.0);
        assert!(entries.iter().all(|entry| entry.score != 1000.0));
    }

    #[test]
    fn founder_pool_bootstrap_limit_applies_after_enabled_filtering() {
        let pool = FounderPool {
            version: FounderPool::VERSION,
            source_tick: 42,
            description: "test".to_string(),
            entries: vec![
                founder("one", 10.0, true),
                founder("two", 40.0, true),
                founder("three", 20.0, true),
                founder("disabled", 999.0, false),
            ],
        };

        let entries = pool.to_survivor_entries(2);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].score, 40.0);
        assert_eq!(entries[1].score, 20.0);
    }
}
