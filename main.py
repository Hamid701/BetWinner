import asyncio
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from handlers.command_handlers import start_command, shutdown_command, button_callback
from config import TELEGRAM_TOKEN
from database.db_config import init_database
import os
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_bot():
    """Initialize and run the bot"""
    try:
        # Initialize database
        init_database()

        # Create application
        app = Application.builder().token(TELEGRAM_TOKEN).build()

        # Add handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("shutdown", shutdown_command))
        app.add_handler(CallbackQueryHandler(button_callback))

        logger.info("Starting bot in polling mode...")

        # Run the bot with polling (local mode)
        app.run_polling(drop_pending_updates=True)

    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise


if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
