import sys
from pathlib import Path
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from process_capability_system.capability_tools import run_capability_analysis
from agent_config.agent_prompts import get_capability_prompt
from agent_config.config_loader import get_llm_model_string

# Load the agent prompt from JSON configuration
Capability_agent_prompt = get_capability_prompt()

# Initialize shared memory components
checkpointer = InMemorySaver()
store = InMemoryStore()

# Create the agent graph (exported for use in API)
capability_graph = create_agent(
    model=get_llm_model_string(),  # Load from config.yaml
    tools=[run_capability_analysis],  # Single comprehensive tool
    checkpointer=checkpointer,
    store=store,
    system_prompt=Capability_agent_prompt,
    name="Capability Agent"
)