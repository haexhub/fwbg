"""
FWBG Trading Bot Entry Point.

Usage:
    python -m fwbg --broker ig [--account-dir PATH] [--no-streaming]

Environment Variables:
    ACCOUNTS_PATH: Base directory for account configs (default: accounts)
    LOG_DIR: Log directory (default: logs)
    STATS_DIR: Stats export directory (default: stats_export)
"""
import os
import sys
import json
import logging
import argparse

# Setup logging early
LOG_DIR = os.environ.get("LOG_DIR", "logs")
STATS_DIR = os.environ.get("STATS_DIR", "stats_export")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log")),
    ],
)
logger = logging.getLogger(__name__)


def discover_accounts(accounts_dir: str) -> list:
    """Findet alle Account-Verzeichnisse mit gültiger Konfiguration."""
    accounts = []
    if not os.path.exists(accounts_dir):
        logger.warning(f"Accounts directory '{accounts_dir}' does not exist")
        return accounts

    for name in os.listdir(accounts_dir):
        account_path = os.path.join(accounts_dir, name)
        if os.path.isdir(account_path):
            account_info = os.path.join(account_path, "account_info.json")
            assets_file = os.path.join(account_path, "assets.json")
            if os.path.exists(account_info) and os.path.exists(assets_file):
                accounts.append(account_path)
            else:
                logger.warning(f"Skipping '{name}': missing config files")

    return accounts


def create_adapter(broker: str, credentials: dict, currency: str):
    """Erstellt den passenden BrokerAdapter."""
    if broker == "ig":
        from fwbg.adapters import IGBrokerAdapter
        return IGBrokerAdapter(
            username=credentials["username"],
            password=credentials["password"],
            api_key=credentials["api_key"],
            env=credentials.get("env", "DEMO").upper(),
            currency=currency,
        )
    else:
        raise ValueError(f"Unknown broker: {broker}")


def run_bot_for_account(
    broker: str,
    account_path: str,
    use_streaming: bool = True,
    strategy_slug: str | None = None,
):
    """Startet den Bot für einen spezifischen Account.

    Args:
        broker: Broker identifier (e.g. "ig").
        account_path: Path to the account directory.
        use_streaming: Use streaming mode when available.
        strategy_slug: Optional. When set, propagates into every AssetConfig so
            paper-trade telemetry can isolate output under
            ``data/account-trades/<strategy_slug>/``. ``None`` → legacy mode.
    """
    from fwbg.bot import TradingBot

    logger.info(f"Starting bot for account: {os.path.basename(account_path)}")

    # Account config laden
    with open(os.path.join(account_path, "account_info.json")) as f:
        account_info = json.load(f)

    with open(os.path.join(account_path, "assets.json")) as f:
        assets_config = json.load(f)

    # Credentials extrahieren
    creds = account_info.get("credentials", {})
    if not all(k in creds for k in ["username", "password", "api_key"]):
        logger.error(f"Missing credentials in {account_path}")
        return

    # Account config für Bot
    account_config = {
        "account_id": os.path.basename(account_path),
        "currency": account_info.get("metadata", {}).get("currency", "EUR"),
        "min_lot_size": account_info.get("money_management", {}).get("min_lot_size", 0.1),
        "max_risk_percent": account_info.get("money_management", {}).get("max_risk_percent", 0.05),
    }

    # M6a: propagate strategy_slug into every asset config (1-strategy-per-bot).
    # None → legacy mode (no slug, no telemetry).
    if strategy_slug is not None:
        for sym_cfg in assets_config.values():
            sym_cfg["strategy_slug"] = strategy_slug

    # Adapter erstellen
    adapter = create_adapter(broker, creds, account_config["currency"])

    # Bot erstellen
    bot = TradingBot(
        adapter=adapter,
        assets_config=assets_config,
        account_config=account_config,
        stats_dir=STATS_DIR,
        use_streaming=use_streaming,
    )

    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        bot.stop()
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise
    finally:
        if adapter.is_connected:
            adapter.disconnect()


def main():
    parser = argparse.ArgumentParser(description="FWBG Trading Bot")
    parser.add_argument(
        "--broker",
        type=str,
        default="ig",
        choices=["ig"],
        help="Broker to use (default: ig)"
    )
    parser.add_argument(
        "--account-dir",
        type=str,
        help="Path to specific account directory"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (use polling instead)"
    )
    parser.add_argument(
        "--strategy-slug",
        type=str,
        default=None,
        help=(
            "Optional strategy slug for paper-trade telemetry isolation. "
            "When set, telemetry is written under data/account-trades/<slug>/. "
            "Omit for legacy mode."
        ),
    )
    args = parser.parse_args()

    use_streaming = not args.no_streaming
    mode = "streaming" if use_streaming else "polling"
    logger.info(f"Starting FWBG bot with {args.broker} broker in {mode} mode")

    if args.account_dir:
        if not os.path.exists(args.account_dir):
            logger.error(f"Account directory not found: {args.account_dir}")
            sys.exit(1)
        run_bot_for_account(
            args.broker, args.account_dir, use_streaming,
            strategy_slug=args.strategy_slug,
        )
    else:
        from fwbg.api.workspace import get_accounts_dir
        accounts_dir = os.environ.get("ACCOUNTS_PATH", str(get_accounts_dir()))
        accounts = discover_accounts(accounts_dir)

        if not accounts:
            logger.error(f"No valid accounts found in {accounts_dir}")
            logger.info("Each account needs: account_info.json and assets.json")
            sys.exit(1)

        logger.info(f"Found {len(accounts)} account(s)")

        if len(accounts) == 1:
            run_bot_for_account(
                args.broker, accounts[0], use_streaming,
                strategy_slug=args.strategy_slug,
            )
        else:
            import threading
            threads = []
            for account_path in accounts:
                t = threading.Thread(
                    target=run_bot_for_account,
                    args=(args.broker, account_path, use_streaming),
                    kwargs={"strategy_slug": args.strategy_slug},
                    daemon=True
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()


if __name__ == "__main__":
    main()
