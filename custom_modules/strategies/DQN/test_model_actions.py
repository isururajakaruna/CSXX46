"""
Test DQN Model Action Distribution

This script tests a trained DQN model to see:
1. Does it output all 3 actions (BUY, SELL, HOLD)?
2. What's the distribution of actions?
3. Are the Q-values reasonable?
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from custom_modules.strategies.DQN.dqn_model import DQN
from custom_modules.strategies.DQN.dqn_agent import DQNAgent

def test_model_outputs(model_path, num_samples=1000):
    """
    Test what actions a trained model outputs.
    
    Args:
        model_path: Path to trained .pth file
        num_samples: Number of random states to test
    """
    print("="*60)
    print("DQN Model Action Distribution Test")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Testing with {num_samples} random states...")
    
    # Load the model
    state_dim = 13  # Match your training config
    action_dim = 3  # BUY, SELL, HOLD
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64],
        device="cpu"
    )
    
    # Load trained weights
    try:
        agent.load(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Generate random states and collect actions
    action_counts = {0: 0, 1: 0, 2: 0}  # BUY, SELL, HOLD
    q_value_stats = []
    
    print(f"\nTesting {num_samples} random states...")
    
    for i in range(num_samples):
        # Generate random state (normalized between -1 and 1)
        state = np.random.randn(state_dim)
        
        # Get action from model (greedy, epsilon=0)
        action = agent.select_action(state, training=False)
        action_counts[action] += 1
        
        # Get Q-values for analysis
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.policy_net(state_tensor).squeeze().numpy()
            q_value_stats.append(q_values)
    
    # Calculate statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    action_names = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
    
    print(f"\n📊 Action Distribution:")
    for action_id, count in action_counts.items():
        percentage = (count / num_samples) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {action_names[action_id]:5s}: {count:4d} ({percentage:5.1f}%)  {bar}")
    
    # Q-value statistics
    q_value_stats = np.array(q_value_stats)
    avg_q_values = q_value_stats.mean(axis=0)
    
    print(f"\n📈 Average Q-values (over {num_samples} states):")
    for i, action_name in action_names.items():
        print(f"  {action_name:5s}: {avg_q_values[i]:+.4f}")
    
    # Check if model is biased
    print(f"\n🔍 Analysis:")
    
    if action_counts[2] == 0:
        print(f"  ⚠️  WARNING: Model NEVER selects HOLD!")
        print(f"      This suggests the model hasn't learned to avoid trading.")
    elif action_counts[2] < num_samples * 0.05:
        print(f"  ⚠️  CAUTION: Model rarely selects HOLD ({action_counts[2]/num_samples*100:.1f}%)")
        print(f"      Model may be over-trading.")
    elif action_counts[2] > num_samples * 0.5:
        print(f"  ⚠️  CAUTION: Model mostly selects HOLD ({action_counts[2]/num_samples*100:.1f}%)")
        print(f"      Model may be too passive.")
    else:
        print(f"  ✓ Model shows balanced action selection")
    
    # Check Q-value dominance
    max_q_idx = np.argmax(avg_q_values)
    if max_q_idx == 2:
        print(f"  ✓ HOLD has highest average Q-value → Model prefers not trading")
    else:
        print(f"  ⚠️  {action_names[max_q_idx]} has highest average Q-value")
    
    print("\n" + "="*60)
    
    return action_counts, q_value_stats


def test_specific_scenarios(model_path):
    """
    Test model on specific market scenarios.
    """
    print("\n" + "="*60)
    print("Testing Specific Scenarios")
    print("="*60)
    
    state_dim = 13
    action_dim = 3
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64],
        device="cpu"
    )
    
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    action_names = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
    
    # Define scenarios
    scenarios = [
        ("Neutral market", np.array([0.0] * state_dim)),
        ("Strong uptrend", np.array([1.0, 0.5, 0.3, 0.8, 0.5, 0.0, 1.0] + [0.0] * 6)),
        ("Strong downtrend", np.array([-1.0, 0.5, -0.3, 0.2, 0.5, 0.0, 1.0] + [0.0] * 6)),
    ]
    
    print("\nModel decisions for specific scenarios:")
    for scenario_name, state in scenarios:
        action = agent.select_action(state, training=False)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.policy_net(state_tensor).squeeze().numpy()
        
        print(f"\n  {scenario_name}:")
        print(f"    Action: {action_names[action]}")
        print(f"    Q-values: BUY={q_values[0]:+.3f}, SELL={q_values[1]:+.3f}, HOLD={q_values[2]:+.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DQN model action distribution")
    parser.add_argument("--model", type=str, help="Path to model .pth file")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to test")
    
    args = parser.parse_args()
    
    if args.model:
        model_path = args.model
    else:
        # Try to find the latest model
        models_dir = Path(__file__).parent / "saved_models"
        pth_files = list(models_dir.glob("*/dqn_final.pth"))
        
        if not pth_files:
            print("No models found. Please train a model first or specify --model path")
            sys.exit(1)
        
        # Use most recent
        model_path = str(sorted(pth_files, key=lambda x: x.stat().st_mtime)[-1])
    
    print(f"\nUsing model: {model_path}\n")
    
    # Run tests
    test_model_outputs(model_path, num_samples=args.samples)
    test_specific_scenarios(model_path)
    
    print("\n✅ Testing complete!\n")

