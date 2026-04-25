"""FastAPI app for CyberSelfPlay OpenEnv environment."""

from openenv.core.env_server.http_server import create_app

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction, CyberObservation

app = create_app(
    CyberSelfPlayEnvironment,
    CyberAction,
    CyberObservation,
    env_name="CyberSelfPlay",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7870):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
