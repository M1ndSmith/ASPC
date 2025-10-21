"""
Agent Prompts Module
Centralized storage and loading of agent prompts from JSON configuration
"""
import json
from pathlib import Path


def load_prompts():
    """Load all agent prompts from prompts.json"""
    prompts_file = Path(__file__).parent / 'prompts.json'
    with open(prompts_file, 'r') as f:
        return json.load(f)


def get_prompt(agent_name):
    """
    Get prompt for a specific agent
    
    Args:
        agent_name: One of 'control_chart_agent', 'msa_agent', 'capability_agent'
    
    Returns:
        str: The prompt text for the specified agent
    """
    prompts = load_prompts()
    if agent_name not in prompts:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(prompts.keys())}")
    return prompts[agent_name]['prompt']


# Convenience exports
def get_control_chart_prompt():
    """Get Control Chart Agent prompt"""
    return get_prompt('control_chart_agent')


def get_msa_prompt():
    """Get MSA Agent prompt"""
    return get_prompt('msa_agent')


def get_capability_prompt():
    """Get Capability Agent prompt"""
    return get_prompt('capability_agent')


# Export for easy import
__all__ = [
    'load_prompts',
    'get_prompt',
    'get_control_chart_prompt',
    'get_msa_prompt',
    'get_capability_prompt'
]

