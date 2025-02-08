import asyncio
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from handlers.command_handlers import start_command, shutdown_command, button_callback
from config import TELEGRAM_TOKEN
from database.db_config import init_database
import os


def run_bot():
    """Initialize and run the bot"""
    # Initialize database
    init_database()

    # Create application
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("shutdown", shutdown_command))
    app.add_handler(CallbackQueryHandler(button_callback))

    # Get port from environment variable
    port = int(os.environ.get("PORT", 8080))

    print(f"Starting bot on port {port}...")
    # Run the bot with webhook
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=os.environ.get("WEBHOOK_URL", "https://betwinner-bot.onrender.com"),
    )


if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nError running bot: {e}")
