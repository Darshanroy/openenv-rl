import os
import time
import uuid
import docker
import requests
from typing import List, Optional, Dict, Any

class SupportObservation:
    def __init__(self, observation_data: Dict[str, Any]):
        # The example from BrowserGym has a 'goal' and 'last_action_error'
        # We'll map our environment messages to these fields.
        messages = observation_data.get("messages", [])
        self.messages = messages
        
        # Goal is usually the first customer message in our case
        self.goal = next((m["content"] for m in messages if m["role"] == "customer"), "Resolve customer issue")
        
        # URL isn't strictly applicable to our API env, but we'll provide a placeholder
        self.url = observation_data.get("url", "http://api-env:8000")
        
        # Check if the last system message was an error
        last_msg = messages[-1]["content"] if messages else ""
        self.last_action_error = "Error" in last_msg or "Unknown" in last_msg
        
        # Screenshot placeholder (required for vision-compatible structure)
        self.screenshot = None 

class SupportResult:
    def __init__(self, step_data: Dict[str, Any], session_id: str):
        self.observation = SupportObservation(step_data)
        self.done = step_data.get("done", False)
        self.reward = step_data.get("reward", 0.0)
        self.metadata = step_data.get("metadata", {})
        self.session_id = session_id

class SupportAction:
    def __init__(self, action_str: str):
        self.action_str = action_str

class SupportEnv:
    """
    Standardized Wrapper for OpenEnv CSA Environment.
    Manages Docker lifecycle to match the BrowserGym example flow.
    """
    def __init__(self, container, port: int, session_id: str):
        self.container = container
        self.port = port
        self.session_id = session_id
        self.base_url = f"http://localhost:{port}"

    @classmethod
    def from_docker_image(cls, image: str, env_vars: Optional[Dict[str, str]] = None):
        """
        Spawns a new Docker container for the environment and waits for health check.
        """
        client = docker.from_env()
        
        # Generate a unique session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        print(f"📦 Spawning environment container: {image}...")
        container = client.containers.run(
            image,
            detach=True,
            ports={'8000/tcp': ('127.0.0.1', None)}, # Map 8000 to a random free port
            environment=env_vars or {},
            remove=True # Auto-cleanup on stop
        )
        
        # Refetch container to get the assigned port
        container.reload()
        host_port = int(container.ports['8000/tcp'][0]['HostPort'])
        
        # Wait for health check
        max_retries = 30
        for i in range(max_retries):
            try:
                resp = requests.get(f"http://localhost:{host_port}/health", timeout=1)
                if resp.status_code == 200:
                    print(f"✅ Environment ready on 127.0.0.1:{host_port}")
                    return cls(container, host_port, session_id)
            except:
                pass
            time.sleep(1)
        
        # If we get here, it failed
        container.stop()
        raise Exception("Environment failed to start within timeout.")

    def reset(self, task_id: Optional[str] = None):
        """Reset the environment state."""
        # Note: task_id can be passed via env_vars or reset. 
        # We'll support both for flexibility.
        resp = requests.post(
            f"{self.base_url}/session/reset",
            json={"session_id": self.session_id, "task_id": task_id},
            timeout=10
        ).json()
        return SupportResult(resp, self.session_id)

    def step(self, action: SupportAction):
        """Execute one step in the environment."""
        resp = requests.post(
            f"{self.base_url}/session/step/{self.session_id}",
            json={"message": action.action_str},
            timeout=10
        ).json()
        return SupportResult(resp, self.session_id)

    def close(self):
        """Stop and remove the environment container."""
        if self.container:
            print("🛑 Stopping environment container...")
            self.container.stop()
