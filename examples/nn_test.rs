//! Neural Network Test Demo
//!
//! Tests the neural network computation in isolation.
//! Creates organisms with known weight patterns and verifies output.
//!
//! Run with: cargo run --example nn_test

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("Neural Network Test Demo");
    
    // Test neural network computation on CPU
    // This validates the same math that runs on GPU
    
    const INPUT_DIM: usize = 20;
    const HIDDEN_DIM: usize = 16;
    const OUTPUT_DIM: usize = 6;
    
    // Create test weights (identity-like for verification)
    let mut weights_l1 = [[0.0f32; INPUT_DIM]; HIDDEN_DIM];
    let mut biases_l1 = [0.0f32; HIDDEN_DIM];
    let mut weights_l2 = [[0.0f32; HIDDEN_DIM]; OUTPUT_DIM];
    let mut biases_l2 = [0.0f32; OUTPUT_DIM];
    
    // Test case 1: All zeros -> should produce biases through tanh
    log::info!("\n=== Test 1: Zero input ===");
    let input_zeros = [0.0f32; INPUT_DIM];
    let output = forward_pass(&input_zeros, &weights_l1, &biases_l1, &weights_l2, &biases_l2);
    log::info!("Input: all zeros");
    log::info!("Output: {:?}", output);
    log::info!("Expected: all zeros (tanh(0) = 0)");
    
    // Test case 2: Identity-ish weights
    log::info!("\n=== Test 2: Positive input with positive weights ===");
    // Set first weight to 1.0
    weights_l1[0][0] = 1.0;
    weights_l2[0][0] = 1.0;
    let mut input_one = [0.0f32; INPUT_DIM];
    input_one[0] = 1.0;
    let output = forward_pass(&input_one, &weights_l1, &biases_l1, &weights_l2, &biases_l2);
    log::info!("Input: [1.0, 0, 0, ...]");
    log::info!("Output: {:?}", output);
    log::info!("Expected: output[0] = tanh(tanh(1.0)) ≈ 0.63");
    
    // Test case 3: Negative weights
    log::info!("\n=== Test 3: Negative weights ===");
    weights_l1[0][0] = -1.0;
    let output = forward_pass(&input_one, &weights_l1, &biases_l1, &weights_l2, &biases_l2);
    log::info!("Input: [1.0, 0, 0, ...]");
    log::info!("Output: {:?}", output);
    log::info!("Expected: output[0] negative");
    
    // Test case 4: Large weights (saturation)
    log::info!("\n=== Test 4: Saturation ===");
    weights_l1[0][0] = 10.0;
    weights_l2[0][0] = 10.0;
    let output = forward_pass(&input_one, &weights_l1, &biases_l1, &weights_l2, &biases_l2);
    log::info!("Input: [1.0, 0, 0, ...]");
    log::info!("Output: {:?}", output);
    log::info!("Expected: output[0] ≈ 1.0 (saturated)");
    
    // Test case 5: Random weights
    log::info!("\n=== Test 5: Random weights ===");
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in 0..HIDDEN_DIM {
        for j in 0..INPUT_DIM {
            weights_l1[i][j] = rng.gen_range(-1.0..1.0);
        }
        biases_l1[i] = rng.gen_range(-0.5..0.5);
    }
    for i in 0..OUTPUT_DIM {
        for j in 0..HIDDEN_DIM {
            weights_l2[i][j] = rng.gen_range(-1.0..1.0);
        }
        biases_l2[i] = rng.gen_range(-0.5..0.5);
    }
    let mut random_input = [0.0f32; INPUT_DIM];
    for i in 0..INPUT_DIM {
        random_input[i] = rng.gen_range(-1.0..1.0);
    }
    let output = forward_pass(&random_input, &weights_l1, &biases_l1, &weights_l2, &biases_l2);
    log::info!("Input: {:?}", &random_input[..5]);
    log::info!("Output: {:?}", output);
    log::info!("Expected: all outputs in range (-1, 1)");
    
    // Verify output range
    let all_valid = output.iter().all(|&v| v > -1.0 && v < 1.0);
    if all_valid {
        log::info!("✓ All outputs in valid range");
    } else {
        log::error!("✗ Some outputs out of range!");
    }
    
    log::info!("\n=== Neural Network Tests Complete ===");
    
    Ok(())
}

fn forward_pass(
    input: &[f32; 20],
    weights_l1: &[[f32; 20]; 16],
    biases_l1: &[f32; 16],
    weights_l2: &[[f32; 16]; 6],
    biases_l2: &[f32; 6],
) -> [f32; 6] {
    // Layer 1: input -> hidden
    let mut hidden = [0.0f32; 16];
    for i in 0..16 {
        let mut sum = biases_l1[i];
        for j in 0..20 {
            sum += weights_l1[i][j] * input[j];
        }
        hidden[i] = sum.tanh();
    }
    
    // Layer 2: hidden -> output
    let mut output = [0.0f32; 6];
    for i in 0..6 {
        let mut sum = biases_l2[i];
        for j in 0..16 {
            sum += weights_l2[i][j] * hidden[j];
        }
        output[i] = sum.tanh();
    }
    
    output
}
