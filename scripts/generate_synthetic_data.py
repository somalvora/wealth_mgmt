"""Write the synthetic demo statements to data/synthetic/ for inspection."""

from pathlib import Path

from household_agent.synthetic import write_demo_statements

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    for path in write_demo_statements(out):
        print(path.relative_to(out.parent.parent))
