import asyncio
from typing import Dict, Any

import httpx
import numpy as np

from collect_responses import simulate_user, launch_experiment


async def main(
    *, config: Dict[str, Any], hostname: str, n_responses=30_000, n_users=30,
) -> int:
    n_answers = (n_responses // n_users) + 1

    users = {
        k: {
            "secs_till_start": np.random.uniform(0, 10),
            "response_times": np.random.uniform(low=0.2, high=1.8, size=n_answers),
        }
        for k in range(n_users)
    }
    async with httpx.AsyncClient() as client:
        running_users = [
            simulate_user(
                config=config,
                client=client,
                hostname=hostname,
                responses=user_responses,
                puid=k + 1,
            )
            for k, user_responses in users.items()
        ]
        num_responses = await asyncio.gather(*running_users)
    return sum(num_responses)


if __name__ == "__main__":
    n = 180
    d = 2
    hostname = "http://localhost:8421"
    config = launch_experiment(hostname, n=n, d=d, seed=42, sampler="RandomSampling", reset=True)
    asyncio.run(main(config=config, hostname=hostname, n_responses=70_000))
