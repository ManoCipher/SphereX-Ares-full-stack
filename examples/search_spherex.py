"""Minimal live SPHEREx archive metadata search."""

from pprint import pprint

from spherex.spherex_client import SPHERExClient


def main() -> None:
    client = SPHERExClient(timeout=60)
    observations = client.search(
        ra=133.795,
        dec=-7.245,
        radius=0.05,
        max_results=50,
    )

    print(f"Found {len(observations)} observation(s)")
    for observation in observations:
        pprint(observation)


if __name__ == "__main__":
    main()
