#!/usr/bin/env python3
"""
Create and save a pre-trained RL orchestrator model
"""

import asyncio
import sys
from pathlib import Path

sys.path.append('/app')

from inbox_sentinel.orchestration.rl_orchestrator import RLEnhancedOrchestrator
from inbox_sentinel.utils.email_parser import parse_gmail_forward

async def create_and_save_rl_model():
    print("🚀 Creating and Saving RL Orchestrator Model")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = RLEnhancedOrchestrator()
    await orchestrator.initialize()
    
    # Load training samples (small set for demo)
    training_data = []
    
    # Load 20 legitimate samples
    legitimate_dir = Path("./legitimate")
    for i, email_file in enumerate(legitimate_dir.glob("*.txt")):
        if i >= 20:
            break
        with open(email_file, 'r', encoding='utf-8') as f:
            email = parse_gmail_forward(f.read())
            training_data.append((email, False))  # False = not spam
    
    # Load 20 spam samples
    spam_dir = Path("./spam")
    for i, email_file in enumerate(spam_dir.glob("*.txt")):
        if i >= 20:
            break
        with open(email_file, 'r', encoding='utf-8') as f:
            email = parse_gmail_forward(f.read())
            training_data.append((email, True))  # True = spam
    
    print(f"📊 Loaded {len(training_data)} training samples")
    
    # Train the model with a small number of episodes
    print("🎯 Training RL orchestrator...")
    await orchestrator.train_on_dataset(training_data, episodes=10, validation_split=0.2)
    
    # Save the model to the models directory
    models_dir = Path("/app/data/models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "rl_orchestrator_model.pkl"
    orchestrator.save_model(str(model_path))
    
    print(f"✅ RL orchestrator model saved to: {model_path}")
    
    # Test loading the saved model
    print("🔄 Testing model loading...")
    test_orchestrator = RLEnhancedOrchestrator(model_path=str(model_path))
    await test_orchestrator.initialize()
    
    # Test prediction with loaded model
    test_email = training_data[0][0]  # Use first training email
    result = await test_orchestrator.predict(test_email)
    
    if result['success']:
        print(f"✅ Model loaded and tested successfully!")
        print(f"   Test prediction: {'SPAM' if result['is_spam'] else 'HAM'}")
        print(f"   Confidence: {result['confidence']:.3f}")
    else:
        print("❌ Model loading test failed")
    
    # Create summary
    stats = orchestrator.get_training_stats()
    
    print(f"\n📈 Model Statistics:")
    print(f"   Training episodes: {stats['rl_agent_stats']['episodes']}")
    print(f"   Q-table size: {stats['q_table_size']}")
    print(f"   Final exploration rate: {stats['current_epsilon']:.3f}")
    print(f"   Model file size: {model_path.stat().st_size / 1024:.1f} KB")
    
    print(f"\n🎯 Usage in CLI:")
    print(f"   The trained RL orchestrator model is now available at:")
    print(f"   {model_path}")
    print(f"   It can be loaded and used for enhanced email analysis.")

if __name__ == "__main__":
    asyncio.run(create_and_save_rl_model())