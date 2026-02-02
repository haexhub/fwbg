"""
Entry point für den IG Trading Bot.

Usage:
    python -m bots.ig                  # Streaming mode (default)
    python -m bots.ig --no-streaming   # Polling mode
"""
from .bot import discover_accounts, EliteBot, logger
import sys
import argparse


def run_bot_for_account(account_path, use_streaming=True):
    """Run a bot instance for a specific account."""
    try:
        bot = EliteBot(account_path, use_streaming=use_streaming)
        bot.run()
    except Exception as e:
        logger.error(f"❌ Bot for {account_path} crashed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IG Trading Bot")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Use polling mode instead of streaming (for accounts without streaming permission)"
    )
    args = parser.parse_args()

    use_streaming = not args.no_streaming
    mode = "polling" if args.no_streaming else "streaming"
    logger.info(f"🚀 Starting bot in {mode} mode")

    accounts = discover_accounts()

    if not accounts:
        logger.error("No valid accounts found in 'accounts/' directory")
        logger.info("Each account needs: account_info.json and assets.json")
        sys.exit(1)

    logger.info(f"Found {len(accounts)} account(s): {accounts}")

    if len(accounts) == 1:
        run_bot_for_account(accounts[0], use_streaming=use_streaming)
    else:
        import threading
        threads = []
        for account_path in accounts:
            t = threading.Thread(
                target=run_bot_for_account,
                args=(account_path, use_streaming),
                daemon=True
            )
            t.start()
            threads.append(t)
            logger.info(f"Started bot for {account_path}")

        for t in threads:
            t.join()
