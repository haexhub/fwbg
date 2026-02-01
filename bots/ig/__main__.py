"""
Entry point für den IG Trading Bot.

Usage:
    python -m bots.ig
"""
from .bot import discover_accounts, run_bot_for_account, logger
import sys


if __name__ == "__main__":
    accounts = discover_accounts()

    if not accounts:
        logger.error("No valid accounts found in 'accounts/' directory")
        logger.info("Each account needs: account_info.json and assets.json")
        sys.exit(1)

    logger.info(f"Found {len(accounts)} account(s): {accounts}")

    if len(accounts) == 1:
        run_bot_for_account(accounts[0])
    else:
        import threading
        threads = []
        for account_path in accounts:
            t = threading.Thread(target=run_bot_for_account, args=(account_path,), daemon=True)
            t.start()
            threads.append(t)
            logger.info(f"Started bot for {account_path}")

        for t in threads:
            t.join()
