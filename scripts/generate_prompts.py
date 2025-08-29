# scripts/generate_prompts.py
"""Generate diverse prompts for BitNet POC testing."""

import json
from pathlib import Path
from typing import List, Dict

def generate_poc_prompts() -> List[Dict[str, str]]:
    """Generate a small but diverse set of prompts for POC testing."""
    
    prompts = [
        # Technical explanations (test knowledge distillation)
        {"prompt": "Explain how a neural network learns in simple terms"},
        {"prompt": "What is the difference between supervised and unsupervised learning?"},
        {"prompt": "Describe the transformer architecture in machine learning"},
        
        # Code generation (test structured output)
        {"prompt": "Write a Python function to calculate fibonacci numbers"},
        {"prompt": "Create a simple REST API endpoint in Python"},
        {"prompt": "Show me how to implement binary search in Python"},
        
        # Math/reasoning (test logical thinking)
        {"prompt": "If a train travels 120 miles in 2 hours, what is its average speed?"},
        {"prompt": "Solve this equation: 2x + 5 = 13"},
        {"prompt": "What is 15% of 240?"},
        
        # Creative writing (test generation quality)
        {"prompt": "Write a haiku about artificial intelligence"},
        {"prompt": "Create a short story opening about a robot learning to paint"},
        {"prompt": "Describe a futuristic city in three sentences"},
        
        # Factual Q&A (test knowledge retention)
        {"prompt": "What is the capital of France?"},
        {"prompt": "Who invented the telephone?"},
        {"prompt": "What year did World War II end?"},
        
        # Instructions/How-to (test instruction following)
        {"prompt": "How do you make a peanut butter sandwich?"},
        {"prompt": "List 5 tips for better sleep"},
        {"prompt": "Explain how to tie a shoelace step by step"},
        
        # Analysis/Comparison (test reasoning)
        {"prompt": "Compare and contrast cats and dogs as pets"},
        {"prompt": "What are the pros and cons of electric vehicles?"},
        {"prompt": "Analyze the benefits of remote work"},
        
        # JSON/Structured (test format following)
        {"prompt": "Create a JSON object representing a person with name, age, and city"},
        {"prompt": "Generate a markdown table with 3 programming languages and their uses"},
        {"prompt": "Write a SQL query to select all users older than 25"},
    ]
    
    return prompts

def generate_larger_prompt_set(n: int = 100) -> List[Dict[str, str]]:
    """Generate a larger set of prompts by creating variations."""
    
    base_templates = [
        "Explain {} in simple terms",
        "What is the difference between {} and {}?",
        "How does {} work?",
        "Write a Python function to {}",
        "Create a {} that {}",
        "List the benefits of {}",
        "What are the main features of {}?",
        "Describe {} in detail",
        "Compare {} with {}",
        "Generate a {} for {}",
        "Solve this problem: {}",
        "What would happen if {}?",
        "Design a {} system",
        "Optimize {} for better performance",
        "Debug this code: {}",
    ]
    
    topics = [
        ("machine learning", "deep learning"),
        ("Python", "JavaScript"),
        ("databases", "data warehouses"),
        ("REST APIs", "GraphQL"),
        ("containers", "virtual machines"),
        ("agile", "waterfall"),
        ("TCP", "UDP"),
        ("encryption", "hashing"),
        ("frontend", "backend"),
        ("microservices", "monoliths"),
    ]
    
    tasks = [
        "sort a list",
        "find duplicates",
        "calculate averages",
        "parse JSON",
        "validate emails",
        "generate passwords",
        "compress strings",
        "merge dictionaries",
        "filter arrays",
        "handle exceptions",
    ]
    
    prompts = []
    
    # Generate varied prompts
    for i in range(min(n, len(base_templates) * len(topics))):
        template_idx = i % len(base_templates)
        topic_idx = i % len(topics)
        task_idx = i % len(tasks)
        
        template = base_templates[template_idx]
        
        if "{}" in template and template.count("{}") == 2:
            # Two placeholder template
            prompt = template.format(topics[topic_idx][0], topics[topic_idx][1])
        elif "{}" in template and "function" in template:
            # Task-based template
            prompt = template.format(tasks[task_idx])
        elif "{}" in template:
            # Single placeholder template
            prompt = template.format(topics[topic_idx][0])
        else:
            prompt = template
        
        prompts.append({"prompt": prompt})
        
        if len(prompts) >= n:
            break
    
    return prompts

def save_prompts(prompts: List[Dict[str, str]], output_path: str = "data/prompts.json"):
    """Save prompts to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(prompts)} prompts to {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    # Show sample
    print(f"\n📝 Sample prompts:")
    for i, p in enumerate(prompts[:3]):
        print(f"  {i+1}. {p['prompt'][:60]}...")

def main():
    """Generate prompts for POC testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate prompts for BitNet POC")
    parser.add_argument("--count", type=int, default=25, 
                       help="Number of prompts to generate (default: 25 for POC)")
    parser.add_argument("--output", type=str, default="data/prompts.json",
                       help="Output JSON file path")
    parser.add_argument("--large", action="store_true",
                       help="Generate larger varied set")
    
    args = parser.parse_args()
    
    if args.large:
        prompts = generate_larger_prompt_set(args.count)
    else:
        prompts = generate_poc_prompts()
        if args.count > len(prompts):
            print(f"⚠️ Requested {args.count} prompts, but POC set has {len(prompts)}. Using all POC prompts.")
    
    save_prompts(prompts, args.output)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
