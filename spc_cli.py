#!/usr/bin/env python3
"""
SPC & Quality Management System - CLI Tool
Command-line interface for interacting with the SPC API
"""
import argparse
import requests
import json
import sys
from pathlib import Path
from typing import Optional
import time


class SPCClient:
    """Client for SPC API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> dict:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def chat(
        self,
        agent: str,
        message: str,
        file_path: Optional[str] = None,
        thread_id: str = "cli_session",
        user_id: str = "cli_user",
        timeout: int = 120
    ) -> dict:
        """Send a message to an agent"""
        url = f"{self.base_url}/chat/{agent}"
        
        data = {
            "message": message,
            "thread_id": thread_id,
            "user_id": user_id
        }
        
        files = None
        if file_path:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            files = {"file": (file_path.name, open(file_path, "rb"), "text/csv")}
        
        try:
            start_time = time.time()
            response = requests.post(url, data=data, files=files, timeout=timeout)
            elapsed_time = time.time() - start_time
            
            if files:
                files["file"][1].close()
            
            response.raise_for_status()
            result = response.json()
            result["elapsed_time"] = round(elapsed_time, 2)
            return result
            
        except requests.Timeout:
            return {"status": "error", "message": f"Request timed out after {timeout} seconds"}
        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def list_agents(self) -> dict:
        """List available agents"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}


def print_response(result: dict, verbose: bool = False):
    """Pretty print API response"""
    if result.get("status") == "error":
        print(f"\n[ERROR] {result.get('message', 'Unknown error')}\n")
        return
    
    print("\n" + "="*80)
    print("[SUCCESS]")
    print("="*80)
    
    if "response" in result:
        print("\nAgent Response:")
        print("-" * 80)
        print(result["response"])
        print("-" * 80)
    
    if verbose:
        print("\nMetadata:")
        for key, value in result.items():
            if key != "response":
                print(f"  {key}: {value}")
    
    if "elapsed_time" in result:
        print(f"\nResponse time: {result['elapsed_time']} seconds")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SPC & Quality Management System - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check API health
  %(prog)s health
  
  # List available agents
  %(prog)s list
  
  # Chat with Control Charts agent
  %(prog)s chat control-charts "Analyze my data" -f data.csv
  
  # MSA analysis
  %(prog)s chat msa "Run Gage R&R study" -f msa_data.csv -t session1 -u user1
  
  # Capability analysis with verbose output
  %(prog)s chat capability "Assess process capability" -f data.csv -v
  
Available Agents:
  - control-charts  : Control chart analysis and SPC
  - msa            : Measurement System Analysis (Gage R&R, Bias, Linearity, Stability)
  - capability     : Process Capability Analysis (Cp, Cpk, Pp, Ppk)
        """
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check API health status")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available agents")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with an agent")
    chat_parser.add_argument(
        "agent",
        choices=["control-charts", "msa", "capability"],
        help="Agent to chat with"
    )
    chat_parser.add_argument(
        "message",
        help="Message to send to the agent"
    )
    chat_parser.add_argument(
        "-f", "--file",
        help="CSV file to upload (optional)"
    )
    chat_parser.add_argument(
        "-t", "--thread-id",
        default="cli_session",
        help="Thread ID for conversation continuity (default: cli_session)"
    )
    chat_parser.add_argument(
        "-u", "--user-id",
        default="cli_user",
        help="User ID for tracking (default: cli_user)"
    )
    chat_parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)"
    )
    chat_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output with metadata"
    )
    chat_parser.add_argument(
        "-o", "--output",
        help="Save response to file (JSON format)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    client = SPCClient(base_url=args.url)
    
    if args.command == "health":
        result = client.health_check()
        if result.get("status") == "healthy":
            print("\n[OK] API is healthy and ready!")
            if "agents" in result:
                print("\nAvailable agents:")
                for agent, status in result["agents"].items():
                    print(f"  - {agent}: {status}")
            print()
        else:
            print(f"\n[FAIL] API health check failed: {result.get('message', 'Unknown error')}\n")
            sys.exit(1)
    
    elif args.command == "list":
        result = client.list_agents()
        if "endpoints" in result:
            print("\nAvailable Agents and Endpoints:")
            print("="*80)
            for agent_id, endpoint in result["endpoints"].items():
                print(f"\nAgent: {agent_id}")
                print(f"  Endpoint: {endpoint}")
            print("\n" + "="*80 + "\n")
        else:
            print(f"\n[FAIL] Failed to list agents: {result.get('message', 'Unknown error')}\n")
            sys.exit(1)
    
    elif args.command == "chat":
        print(f"\nChatting with {args.agent} agent...")
        if args.file:
            print(f"File: {args.file}")
        print(f"Message: {args.message}")
        print(f"Thread ID: {args.thread_id}")
        
        result = client.chat(
            agent=args.agent,
            message=args.message,
            file_path=args.file,
            thread_id=args.thread_id,
            user_id=args.user_id,
            timeout=args.timeout
        )
        
        print_response(result, verbose=args.verbose)
        
        # Save to file if requested
        if args.output and result.get("status") != "error":
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Response saved to: {output_path}\n")
        
        # Exit with error code if request failed
        if result.get("status") == "error":
            sys.exit(1)


if __name__ == "__main__":
    main()

