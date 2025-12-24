"""Audio mixer for combining dialogue and BGM into final podcast episode.

This module uses pydub to:
1. Combine individual dialogue audio files in sequence
2. Add opening BGM (出囃子)
3. Optionally add background music during dialogue
4. Export final podcast MP3 file
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from pydub import AudioSegment
from pydub.effects import normalize


# Configure logging
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_BGM_PATH = PROJECT_ROOT / "assets" / "bgm_op.mp3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "public" / "audio"

# Audio configuration
AFTER_BGM_PAUSE_MS = 800  # Pause after opening BGM
DIALOGUE_PAUSE_MS = 50  # Minimal pause between dialogue lines (nearly seamless)
BGM_FADE_OUT_MS = 1500  # Fade out duration for opening BGM
BGM_OPENING_VOLUME_DB = -9  # Reduce opening BGM volume by 6dB
FADE_OUT_DURATION_MS = 2000  # Fade out duration at the end
BGM_VOLUME_REDUCTION_DB = -20  # Reduce BGM volume to not interfere with dialogue


class PodcastMixer:
    """Mix dialogue audio files and BGM into final podcast episode."""

    def __init__(
        self,
        bgm_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """Initialize the PodcastMixer.

        Args:
            bgm_path: Path to opening BGM file. If None, uses assets/bgm_op.mp3.
            output_dir: Directory to save final audio. If None, uses public/audio/.

        Raises:
            FileNotFoundError: If BGM file doesn't exist.
        """
        # Set BGM path
        if bgm_path:
            self.bgm_path = Path(bgm_path)
        else:
            self.bgm_path = DEFAULT_BGM_PATH

        # Validate BGM file exists
        if not self.bgm_path.exists():
            raise FileNotFoundError(f"BGM file not found: {self.bgm_path}")

        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = DEFAULT_OUTPUT_DIR

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized PodcastMixer")
        logger.info(f"BGM file: {self.bgm_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def mix_podcast(
        self,
        audio_data: List[Dict[str, any]],
        output_filename: Optional[str] = None,
        add_background_music: bool = False,
        normalize_audio: bool = True
    ) -> Path:
        """Mix dialogue audio files and BGM into final podcast episode.

        Args:
            audio_data: List of audio info from synthesizer.
                Format: [{"speaker": "進藤", "text": "...", "audio_path": "..."}]
            output_filename: Custom output filename. If None, uses daily_YYYYMMDD.mp3.
            add_background_music: Whether to add subtle background music during dialogue.
            normalize_audio: Whether to normalize final audio output.

        Returns:
            Path to the generated podcast file.

        Raises:
            Exception: If audio mixing fails.

        Examples:
            >>> mixer = PodcastMixer()
            >>> audio_data = [{"audio_path": "line1.mp3"}, {"audio_path": "line2.mp3"}]
            >>> podcast_path = mixer.mix_podcast(audio_data)
            >>> print(f"Podcast saved to: {podcast_path}")
        """
        logger.info(f"Starting podcast mixing with {len(audio_data)} dialogue lines")

        try:
            # Load opening BGM
            logger.info("Loading opening BGM...")
            bgm_opening = self._load_bgm_opening()

            # Add pause after BGM
            pause_after_bgm = AudioSegment.silent(duration=AFTER_BGM_PAUSE_MS)
            logger.debug(f"Adding {AFTER_BGM_PAUSE_MS}ms pause after BGM")

            # Combine dialogue audio files
            logger.info("Combining dialogue audio files...")
            dialogue_audio = self._combine_dialogue(audio_data)

            # Create final podcast
            logger.info("Creating final podcast mix...")
            final_audio = bgm_opening + pause_after_bgm + dialogue_audio

            # Optionally add background music during dialogue
            if add_background_music:
                logger.info("Adding background music to dialogue...")
                final_audio = self._add_background_music(
                    final_audio,
                    bgm_start_offset=len(bgm_opening)
                )

            # Apply fade out at the end
            logger.info("Applying fade out...")
            final_audio = final_audio.fade_out(duration=FADE_OUT_DURATION_MS)

            # Normalize audio levels
            if normalize_audio:
                logger.info("Normalizing audio levels...")
                final_audio = normalize(final_audio)

            # Generate output filename if not provided
            if output_filename is None:
                today = datetime.now().strftime("%Y%m%d")
                output_filename = f"daily_{today}.mp3"

            # Export final podcast
            output_path = self.output_dir / output_filename
            logger.info(f"Exporting final podcast to: {output_path}")

            final_audio.export(
                output_path,
                format="mp3",
                bitrate="192k",
                tags={
                    "artist": "キャピタル・ゲインズ",
                    "album": "ビジネス経済漫才ポッドキャスト",
                    "genre": "Comedy/Business"
                }
            )

            # Log final audio stats
            duration_seconds = len(final_audio) / 1000.0
            logger.info(f"Podcast successfully created!")
            logger.info(f"Duration: {duration_seconds:.1f} seconds")
            logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

            return output_path

        except Exception as e:
            logger.error(f"Failed to mix podcast: {str(e)}")
            raise

    def _load_bgm_opening(self) -> AudioSegment:
        """Load and prepare opening BGM (full length with volume adjustment and fade out).

        Returns:
            AudioSegment of opening BGM with volume reduction and fade out applied.

        Raises:
            Exception: If BGM file cannot be loaded.
        """
        try:
            # Load BGM file at full length
            bgm = AudioSegment.from_mp3(self.bgm_path)

            # Reduce volume
            bgm = bgm + BGM_OPENING_VOLUME_DB

            # Apply fade out at the end
            bgm = bgm.fade_out(duration=BGM_FADE_OUT_MS)

            logger.debug(
                f"Loaded opening BGM: {len(bgm)}ms ({len(bgm)/1000:.1f}s), "
                f"volume: {BGM_OPENING_VOLUME_DB}dB, "
                f"fade out: {BGM_FADE_OUT_MS}ms"
            )
            return bgm

        except Exception as e:
            logger.error(f"Failed to load BGM: {str(e)}")
            raise

    def _combine_dialogue(
        self,
        audio_data: List[Dict[str, any]]
    ) -> AudioSegment:
        """Combine individual dialogue audio files into one continuous track.

        Args:
            audio_data: List of audio info with audio_path field.

        Returns:
            Combined AudioSegment with all dialogue.

        Raises:
            Exception: If audio files cannot be loaded or combined.
        """
        try:
            # Start with silence
            combined = AudioSegment.silent(duration=0)

            # Add each dialogue line with pause between
            for i, data in enumerate(audio_data):
                audio_path = data.get('audio_path')
                speaker = data.get('speaker', 'Unknown')

                if not audio_path:
                    logger.warning(f"Skipping line {i+1}: no audio_path")
                    continue

                # Load audio file
                logger.debug(f"Loading line {i+1}/{len(audio_data)}: {speaker}")
                audio = AudioSegment.from_mp3(audio_path)

                # Add to combined audio
                combined += audio

                # Add pause between lines (except after last line)
                if i < len(audio_data) - 1:
                    combined += AudioSegment.silent(duration=DIALOGUE_PAUSE_MS)

            logger.info(
                f"Combined {len(audio_data)} dialogue lines "
                f"(total: {len(combined)/1000:.1f}s)"
            )
            return combined

        except Exception as e:
            logger.error(f"Failed to combine dialogue: {str(e)}")
            raise

    def _add_background_music(
        self,
        audio: AudioSegment,
        bgm_start_offset: int
    ) -> AudioSegment:
        """Add subtle background music during dialogue sections.

        Args:
            audio: The main audio track.
            bgm_start_offset: When to start the background music (in ms).

        Returns:
            AudioSegment with background music overlaid.

        Raises:
            Exception: If background music cannot be added.
        """
        try:
            # Load full BGM
            bgm = AudioSegment.from_mp3(self.bgm_path)

            # Reduce BGM volume so it doesn't interfere with dialogue
            bgm_quiet = bgm + BGM_VOLUME_REDUCTION_DB

            # Calculate how much BGM we need (after the opening)
            dialogue_duration = len(audio) - bgm_start_offset

            # Loop BGM if needed to cover entire dialogue
            if len(bgm_quiet) < dialogue_duration:
                loops_needed = (dialogue_duration // len(bgm_quiet)) + 1
                bgm_quiet = bgm_quiet * loops_needed

            # Trim BGM to match dialogue duration
            bgm_quiet = bgm_quiet[:dialogue_duration]

            # Create the background track
            # Keep opening as-is, then add quiet BGM for dialogue
            background = audio[:bgm_start_offset] + bgm_quiet

            # Overlay background onto main audio
            result = audio.overlay(background, position=bgm_start_offset)

            logger.debug(f"Added background music (reduced by {BGM_VOLUME_REDUCTION_DB}dB)")
            return result

        except Exception as e:
            logger.error(f"Failed to add background music: {str(e)}")
            raise

    def get_output_dir(self) -> Path:
        """Get the output directory for podcast files.

        Returns:
            Path to the output directory.
        """
        return self.output_dir


def mix_manzai_podcast(
    audio_data: List[Dict[str, any]],
    bgm_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    output_filename: Optional[str] = None,
    add_background_music: bool = False
) -> Path:
    """Convenience function to mix manzai podcast.

    Args:
        audio_data: List of audio info from synthesizer.
        bgm_path: Path to opening BGM file. If None, uses assets/bgm_op.mp3.
        output_dir: Directory to save final audio. If None, uses public/audio/.
        output_filename: Custom output filename. If None, uses daily_YYYYMMDD.mp3.
        add_background_music: Whether to add subtle background music during dialogue.

    Returns:
        Path to the generated podcast file.

    Examples:
        >>> audio_data = [{"audio_path": "line1.mp3"}, {"audio_path": "line2.mp3"}]
        >>> podcast_path = mix_manzai_podcast(audio_data)
        >>> print(f"Podcast created: {podcast_path}")
    """
    mixer = PodcastMixer(bgm_path=bgm_path, output_dir=output_dir)

    return mixer.mix_podcast(
        audio_data=audio_data,
        output_filename=output_filename,
        add_background_music=add_background_music
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test the mixer with existing audio files
    try:
        logger.info("Testing mixer with sample audio files...")

        # Use existing test audio files from tmp/audio/
        test_audio_dir = PROJECT_ROOT / "tmp" / "audio"

        if not test_audio_dir.exists():
            logger.error(f"Test audio directory not found: {test_audio_dir}")
            logger.info("Please run synthesizer.py first to generate test audio files")
            exit(1)

        # Gather existing audio files
        audio_files = sorted(test_audio_dir.glob("line_*.mp3"))

        if not audio_files:
            logger.error("No audio files found in test directory")
            logger.info("Please run synthesizer.py first to generate test audio files")
            exit(1)

        # Create test audio data structure
        test_audio_data = []
        for audio_file in audio_files:
            # Try to extract speaker from filename (e.g., line_0000_onyx.mp3)
            if "onyx" in audio_file.name:
                speaker = "進藤"
            elif "alloy" in audio_file.name:
                speaker = "守屋"
            else:
                speaker = "Unknown"

            test_audio_data.append({
                "speaker": speaker,
                "text": f"Test line from {audio_file.name}",
                "audio_path": str(audio_file)
            })

        logger.info(f"Found {len(test_audio_data)} test audio files")

        # Create mixer instance
        mixer = PodcastMixer()

        # Mix podcast
        podcast_path = mixer.mix_podcast(
            audio_data=test_audio_data,
            output_filename="test_podcast.mp3",
            add_background_music=False,  # Set to True to test background music
            normalize_audio=True
        )

        print(f"\n=== Mixing Results ===")
        print(f"Podcast file: {podcast_path}")
        print(f"File size: {podcast_path.stat().st_size / 1024:.1f} KB")
        print(f"Input audio files: {len(test_audio_data)}")
        print()
        print("Test completed successfully!")
        print(f"You can play the podcast with: ffplay {podcast_path}")

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise
