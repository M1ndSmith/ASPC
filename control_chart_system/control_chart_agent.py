import sys
from pathlib import Path
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from control_chart_system.control_chart_tools import run_control_chart_analysis
from agent_config.agent_prompts import get_control_chart_prompt
from agent_config.config_loader import get_llm_model_string

# Load the agent prompt from JSON configuration
Control_chart_agent = get_control_chart_prompt()

# Initialize shared memory components
checkpointer = InMemorySaver()
store = InMemoryStore()

# Create the agent graph (exported for use in API)
control_chart_graph = create_react_agent(
    model=get_llm_model_string(),  # Load from config.yaml
    tools=[run_control_chart_analysis],  # Single comprehensive tool
    checkpointer=checkpointer,
    store=store,
    prompt=Control_chart_agent,
    name="control charts ai generator",
)

