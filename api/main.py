"""
FastAPI Agent Chat API - Minimal & Clean

Three conversational endpoints:
- POST /chat/msa - Measurement System Analysis
- POST /chat/control-charts - Statistical Process Control  
- POST /chat/capability - Process Capability Analysis
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path
from langchain_core.messages import HumanMessage

# Import agent graphs
from control_chart_system.control_chart_agent import control_chart_graph
from msa_system.msa_agent import msa_graph
from process_capability_system.capability_agent import capability_graph

app = FastAPI(
    title="SPC & Quality Management API",
    description="AI Quality Consultants for MSA, SPC, and Capability Analysis",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent graphs mapping
AGENT_GRAPHS = {
    "msa": msa_graph,
    "control-charts": control_chart_graph,
    "capability": capability_graph
}


# ===== ENDPOINTS =====

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "SPC & Quality Management API",
        "version": "2.0.0",
        "endpoints": {
            "msa": "/chat/msa",
            "control_charts": "/chat/control-charts",
            "capability": "/chat/capability",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post("/chat/{agent_id}")
async def chat(
    agent_id: str,
    message: str = Form(...),
    thread_id: str = Form(...),
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Universal chat endpoint for all agents
    
    Args:
        agent_id: Agent identifier (msa, control-charts, capability)
        message: Your question or command
        thread_id: Thread ID for conversation continuity
        user_id: User ID for tracking
        file: Optional CSV file with data
    
    Returns:
        {
            "status": "success",
            "thread_id": "...",
            "user_id": "...",
            "response": "agent response text"
        }
    """
    # Validate agent exists
    if agent_id not in AGENT_GRAPHS:
        raise HTTPException(404, f"Agent not found. Available: {list(AGENT_GRAPHS.keys())}")
    
    try:
        # Save uploaded file if provided
        file_path = None
        if file:
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            file_path = str(file_path)
        
        # Prepend file path to message if provided
        if file_path:
            message = f"File: {file_path}\n\n{message}"
        
        # Create config for agent memory
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # Run agent
        agent_response = await AGENT_GRAPHS[agent_id].ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        # Extract response text
        response_text = "No response from agent"
        if "messages" in agent_response and len(agent_response["messages"]) > 0:
            last_message = agent_response["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                response_text = last_message.content
        
        # Return response
        return JSONResponse(content={
            "status": "success",
            "thread_id": thread_id,
            "user_id": user_id,
            "response": response_text
        })
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
