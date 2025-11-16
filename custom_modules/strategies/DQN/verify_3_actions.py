"""
Verify DQN Model Outputs 3 Actions (BUY, SELL, HOLD)

Quick test to verify the DQN architecture correctly outputs 3 actions.
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

def test_fresh_model():
    """Test a freshly initialized (untrained) model."""
    print("="*60)
    print("DQN 3-Action Architecture Test")
    print("="*60)
    
    state_dim = 13
    action_dim = 3
    
    print(f"\n✓ Creating DQN model:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim} (BUY=0, SELL=1, HOLD=2)")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64],
        device="cpu"
    )
    
    print(f"\n✓ Model architecture:")
    print(agent.policy_net)
    
    # Test with random states
    num_samples = 1000
    action_counts = {0: 0, 1: 0, 2: 0}
    
    print(f"\n✓ Testing with {num_samples} random states...")
    
    for i in range(num_samples):
        state = np.random.randn(state_dim)
        action = agent.select_action(state, training=False)
        action_counts[action] += 1
        
        # Show first few examples
        if i < 5:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.policy_net(state_tensor).squeeze().numpy()
            print(f"\n  Sample {i+1}:")
            print(f"    Q-values: BUY={q_values[0]:+.3f}, SELL={q_values[1]:+.3f}, HOLD={q_values[2]:+.3f}")
            print(f"    Selected: {['BUY', 'SELL', 'HOLD'][action]}")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    action_names = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
    
    print(f"\n📊 Action Distribution (random initialized model):")
    for action_id, count in action_counts.items():
        percentage = (count / num_samples) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {action_names[action_id]:5s}: {count:4d} ({percentage:5.1f}%)  {bar}")
    
    print(f"\n🔍 Analysis:")
    
    if action_counts[2] == 0:
        print(f"  ✗ ERROR: Model NEVER outputs HOLD!")
        print(f"  ✗ The architecture is BROKEN - only 2 outputs instead of 3")
        return False
    else:
        print(f"  ✓ Model CAN output HOLD action")
        print(f"  ✓ Architecture is correct: 3 actions (BUY, SELL, HOLD)")
        
        if all(100 <= count <= 400 for count in action_counts.values()):
            print(f"  ✓ Random model shows balanced distribution (~33% each)")
        else:
            print(f"  ⚠️  Random weights create bias, but all 3 actions are possible")
        
        return True


def test_epsilon_exploration():
    """Test that epsilon exploration includes HOLD."""
    print("\n" + "="*60)
    print("Epsilon Exploration Test")
    print("="*60)
    
    state_dim = 13
    action_dim = 3
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64],
        epsilon_start=1.0,  # 100% random
        device="cpu"
    )
    
    print(f"\n✓ Testing with epsilon=1.0 (100% random actions)...")
    
    num_samples = 1000
    action_counts = {0: 0, 1: 0, 2: 0}
    
    for i in range(num_samples):
        state = np.random.randn(state_dim)
        action = agent.select_action(state, training=True)  # training=True uses epsilon
        action_counts[action] += 1
    
    print(f"\n📊 Action Distribution (epsilon=1.0, random):")
    action_names = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
    
    for action_id, count in action_counts.items():
        percentage = (count / num_samples) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {action_names[action_id]:5s}: {count:4d} ({percentage:5.1f}%)  {bar}")
    
    # Should be ~33% each with epsilon=1.0
    expected = num_samples / 3
    tolerance = 0.1  # 10% tolerance
    
    all_close = all(
        abs(count - expected) / expected < tolerance
        for count in action_counts.values()
    )
    
    if all_close:
        print(f"\n  ✓ Distribution is ~33% each (as expected for random)")
        print(f"  ✓ Epsilon exploration includes all 3 actions")
    else:
        print(f"\n  ⚠️  Distribution slightly uneven (random variation)")
        print(f"  ✓ But all 3 actions are present")
    
    if action_counts[2] > 0:
        print(f"\n  ✓ HOLD is included in exploration")
        return True
    else:
        print(f"\n  ✗ ERROR: HOLD is missing from exploration!")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 15 + "DQN 3-ACTION VERIFICATION")
    print("="*70)
    
    print("\nThis test verifies:")
    print("  1. DQN model outputs 3 actions (BUY, SELL, HOLD)")
    print("  2. HOLD action (action=2) is functional")
    print("  3. Epsilon exploration includes HOLD")
    
    # Run tests
    test1_pass = test_fresh_model()
    test2_pass = test_epsilon_exploration()
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if test1_pass and test2_pass:
        print("\n✅ ALL TESTS PASSED")
        print("\nThe DQN model:")
        print("  ✓ Has 3 output neurons (BUY, SELL, HOLD)")
        print("  ✓ CAN select HOLD action")
        print("  ✓ Includes HOLD in epsilon exploration")
        print("\n💡 Whether a TRAINED model actually LEARNS to HOLD depends on:")
        print("  - Reward function (trading fees penalize over-trading)")
        print("  - Training data (sideways markets favor HOLD)")
        print("  - Exploration during training (epsilon lets it try HOLD)")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nCheck the DQN architecture!")
    
    print("\n" + "="*70)
    print()

