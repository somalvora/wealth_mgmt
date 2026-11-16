"""Write a fictional, realistic-looking card statement you can upload in the chat demo.

    uv run python scripts/make_card_statement.py --out ~/Downloads/2026_10_sayali_card_realistic.csv

Upload it as sayali (the owner of acct_sayali_card). Nothing in it comes from a real statement.
"""

import argparse
from pathlib import Path

from household_agent.synthetic import realistic_card_statement

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--month", default="2026-10", help="YYYY-MM")
    ap.add_argument("--account", default="acct_sayali_card")
    ap.add_argument("--opening", default="-1200.00", help="opening balance; a card owes money, so it is negative")
    args = ap.parse_args()
    args.out.expanduser().write_text(realistic_card_statement(args.seed, args.month, args.account, args.opening))
    print(args.out.expanduser())
