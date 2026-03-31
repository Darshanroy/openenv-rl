"""
OpenEnv-compliant FastAPI server for the Customer Support Environment.
Uses openenv.core.create_app() for full spec compliance.
"""
from openenv.core import create_app
from my_env.models import SupportAction, SupportObservation
from my_env.server.my_environment import SupportEnvironment


def create_environment():
    """Factory function that returns a fresh SupportEnvironment instance."""
    return SupportEnvironment(max_turns=8)


app = create_app(
    env=create_environment,
    action_cls=SupportAction,
    observation_cls=SupportObservation,
    env_name="CustomerSupport-v1",
    max_concurrent_envs=64,
)


def main():
    """Entry point for `openenv serve` and `[project.scripts]`."""
    import uvicorn
    uvicorn.run("my_env.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
