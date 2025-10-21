import sys
from pathlib import Path
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from msa_system.msa_tools import run_msa_analysis
from agent_config.agent_prompts import get_msa_prompt
from agent_config.config_loader import get_llm_model_string

# Load the agent prompt from JSON configuration
MSA_agent_prompt = get_msa_prompt()

# Initialize shared memory components
checkpointer = InMemorySaver()
store = InMemoryStore()

# Create the agent graph (exported for use in API)
msa_graph = create_react_agent(
    model=get_llm_model_string(),  # Load from config.yaml
    tools=[run_msa_analysis],  # Single comprehensive tool
    checkpointer=checkpointer,
    store=store,
    prompt=MSA_agent_prompt,
    name="MSA Agent"
)