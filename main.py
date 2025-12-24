#!/usr/bin/env python3
"""Main execution script for Capital Gains podcast automation.

This script orchestrates the full pipeline:
1. Fetch news articles from TBS NEWS DIG RSS
2. Select top articles using GPT-4o
3. Generate manzai script in Kansai dialect
4. Synthesize audio using OpenAI TTS
5. Mix audio with BGM into final podcast

Usage:
    python main.py              # Full production run
    python main.py --test       # Test run with limited articles
    python main.py --debug      # Debug mode with verbose logging
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.fetcher import fetch_news
from src.processor import NewsProcessor
from src.synthesizer import ScriptSynthesizer
from src.mixer import PodcastMixer


# Configure logging
def setup_logging(debug: bool = False):
    """Set up logging configuration.

    Args:
        debug: If True, set logging level to DEBUG.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Execute the full podcast generation pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate Capital Gains business manzai podcast'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: fetch fewer articles and use test output filename'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--num-articles',
        type=int,
        default=5,
        help='Number of articles to select for script (default: 5)'
    )
    parser.add_argument(
        '--fetch-limit',
        type=int,
        default=20,
        help='Maximum number of articles to fetch from RSS (default: 20)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Custom output filename (default: daily_YYYYMMDD.mp3 or test_YYYYMMDD.mp3)'
    )
    parser.add_argument(
        '--add-bgm',
        action='store_true',
        help='Add background music during dialogue (experimental)'
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.error("Please set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    try:
        # Print banner
        logger.info("=" * 60)
        logger.info("Capital Gains Podcast Generator")
        logger.info("ビジネス経済漫才ポッドキャスト自動生成")
        logger.info("=" * 60)
        logger.info(f"Mode: {'TEST' if args.test else 'PRODUCTION'}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

        # Step 1: Fetch news articles
        logger.info("STEP 1/5: Fetching news articles from TBS NEWS DIG...")
        logger.info("-" * 60)

        fetch_limit = 10 if args.test else args.fetch_limit
        articles = fetch_news(max_items=fetch_limit)

        logger.info(f"Successfully fetched {len(articles)} articles")
        logger.info("")

        # Step 2: Select articles and generate script
        logger.info("STEP 2/5: Selecting articles and generating script...")
        logger.info("-" * 60)

        processor = NewsProcessor()

        # Select articles
        logger.info(f"Selecting {args.num_articles} articles from {len(articles)} candidates...")
        selected_news = processor.select_news(articles, num_articles=args.num_articles)

        logger.info(f"Selected {len(selected_news)} articles:")
        for i, news in enumerate(selected_news, 1):
            logger.info(f"  {i}. {news['title'][:60]}...")
        logger.info("")

        # Generate script
        logger.info("Generating manzai script...")
        script = processor.generate_script(selected_news)

        logger.info(f"Generated script with {len(script)} dialogue lines")
        logger.info(f"Preview (first 5 lines):")
        for i, line in enumerate(script[:5], 1):
            logger.info(f"  {line['speaker']}: {line['text'][:50]}...")
        if len(script) > 5:
            logger.info(f"  ... and {len(script) - 5} more lines")
        logger.info("")

        # Step 3: Synthesize audio
        logger.info("STEP 3/5: Synthesizing audio with OpenAI TTS...")
        logger.info("-" * 60)

        synthesizer = ScriptSynthesizer()
        audio_data = synthesizer.synthesize_script(script)

        logger.info(f"Successfully synthesized {len(audio_data)} audio files")
        logger.info(f"Audio files location: {synthesizer.get_output_dir()}")
        logger.info("")

        # Step 4: Mix podcast
        logger.info("STEP 4/5: Mixing podcast with BGM...")
        logger.info("-" * 60)

        mixer = PodcastMixer()

        # Generate output filename
        if args.output:
            output_filename = args.output
        else:
            today = datetime.now().strftime("%Y%m%d")
            prefix = "test" if args.test else "daily"
            output_filename = f"{prefix}_{today}.mp3"

        podcast_path = mixer.mix_podcast(
            audio_data=audio_data,
            output_filename=output_filename,
            add_background_music=args.add_bgm,
            normalize_audio=True
        )

        logger.info(f"Podcast mixed successfully!")
        logger.info(f"Output file: {podcast_path}")
        logger.info(f"File size: {podcast_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info("")

        # Step 5: Summary
        logger.info("STEP 5/5: Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Articles fetched:     {len(articles)}")
        logger.info(f"Articles selected:    {len(selected_news)}")
        logger.info(f"Script lines:         {len(script)}")
        logger.info(f"Audio files:          {len(audio_data)}")
        logger.info(f"Final podcast:        {podcast_path}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("SUCCESS! Podcast generation completed.")
        logger.info(f"You can play the podcast with: ffplay {podcast_path}")
        logger.info("")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\nERROR: Pipeline failed")
        logger.error(f"Error: {str(e)}")
        if args.debug:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
