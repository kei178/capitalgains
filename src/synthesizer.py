"""OpenAI TTS-based audio synthesizer for manzai script.

This module converts manzai script dialogue into audio files using OpenAI TTS API.
Each character has a specific voice:
- シンドウ (Shindo): Onyx (low, heavy tone)
- モリヤ (Moriya): Alloy (bright, clear tone)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI


# Configure logging
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "audio"

# Voice mapping for characters
VOICE_MAP = {
    "シンドウ": "onyx",      # Shindo: Deep, authoritative voice
    "モリヤ": "alloy",     # Moriya: Bright, energetic voice
}

# Default TTS configuration
DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"
DEFAULT_SPEED = 1.0


class ScriptSynthesizer:
    """Synthesize manzai script dialogue into audio files using OpenAI TTS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_TTS_MODEL,
        output_dir: Optional[Path] = None
    ):
        """Initialize the ScriptSynthesizer.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: TTS model to use. Defaults to tts-1.
            output_dir: Directory to save audio files. If None, uses tmp/audio/ in project.

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        self.model = model

        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # OpenAI client will automatically use OPENAI_API_KEY env var
            self.client = OpenAI()

        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Use project tmp directory if not specified
            self.output_dir = DEFAULT_OUTPUT_DIR

        # Create directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ScriptSynthesizer with model: {model}")
        logger.info(f"Audio output directory: {self.output_dir}")

    def synthesize_script(
        self,
        script: List[Dict[str, str]],
        speed: float = DEFAULT_SPEED
    ) -> List[Dict[str, any]]:
        """Synthesize all dialogue lines in the script to audio files.

        Args:
            script: List of dialogue lines in format [{"speaker": "シンドウ", "text": "..."}]
            speed: Speech speed multiplier (0.25 to 4.0). Defaults to 1.0.

        Returns:
            List of dictionaries with dialogue info and audio file paths:
            [{"speaker": "シンドウ", "text": "...", "audio_path": "path/to/file.mp3"}]

        Raises:
            Exception: If TTS API call fails or audio generation encounters errors.

        Examples:
            >>> synthesizer = ScriptSynthesizer()
            >>> script = [{"speaker": "シンドウ", "text": "おはようございます"}]
            >>> audio_data = synthesizer.synthesize_script(script)
            >>> print(audio_data[0]['audio_path'])
        """
        logger.info(f"Starting synthesis of {len(script)} dialogue lines")

        audio_data = []

        try:
            for i, line in enumerate(script):
                speaker = line.get('speaker', '')
                text = line.get('text', '')

                if not speaker or not text:
                    logger.warning(f"Skipping line {i+1}: missing speaker or text")
                    continue

                logger.info(f"Synthesizing line {i+1}/{len(script)}: {speaker}")
                logger.debug(f"Text: {text[:50]}...")

                # Get voice for speaker
                voice = self._get_voice_for_speaker(speaker)

                # Generate audio file
                audio_path = self._synthesize_line(
                    text=text,
                    voice=voice,
                    speed=speed,
                    index=i
                )

                # Add to results
                audio_data.append({
                    "speaker": speaker,
                    "text": text,
                    "audio_path": str(audio_path),
                    "voice": voice
                })

                logger.debug(f"Generated audio: {audio_path}")

            logger.info(f"Successfully synthesized {len(audio_data)} dialogue lines")
            return audio_data

        except Exception as e:
            logger.error(f"Failed to synthesize script: {str(e)}")
            raise

    def synthesize_line(
        self,
        text: str,
        speaker: str,
        speed: float = DEFAULT_SPEED,
        output_path: Optional[Path] = None
    ) -> Path:
        """Synthesize a single dialogue line to an audio file.

        Args:
            text: The dialogue text to synthesize.
            speaker: The speaker name (e.g., "シンドウ", "モリヤ").
            speed: Speech speed multiplier (0.25 to 4.0). Defaults to 1.0.
            output_path: Custom output file path. If None, auto-generates path.

        Returns:
            Path to the generated audio file.

        Raises:
            Exception: If TTS API call fails.

        Examples:
            >>> synthesizer = ScriptSynthesizer()
            >>> audio_path = synthesizer.synthesize_line(
            ...     text="おはようございます",
            ...     speaker="シンドウ"
            ... )
        """
        voice = self._get_voice_for_speaker(speaker)

        if output_path is None:
            output_path = self.output_dir / f"line_{speaker}_{hash(text) % 10000:04d}.mp3"

        return self._synthesize_line(
            text=text,
            voice=voice,
            speed=speed,
            output_path=output_path
        )

    def _get_voice_for_speaker(self, speaker: str) -> str:
        """Get the appropriate voice for a given speaker.

        Args:
            speaker: Speaker name (e.g., "シンドウ", "モリヤ").

        Returns:
            Voice name for OpenAI TTS (e.g., "onyx", "alloy").
        """
        voice = VOICE_MAP.get(speaker, DEFAULT_VOICE)

        if speaker not in VOICE_MAP:
            logger.warning(
                f"Unknown speaker '{speaker}', using default voice '{DEFAULT_VOICE}'"
            )

        return voice

    def _synthesize_line(
        self,
        text: str,
        voice: str,
        speed: float,
        index: Optional[int] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """Internal method to synthesize a single line using OpenAI TTS.

        Args:
            text: The text to synthesize.
            voice: Voice name for TTS.
            speed: Speech speed multiplier.
            index: Line index (for auto-generating filename).
            output_path: Custom output file path.

        Returns:
            Path to the generated audio file.

        Raises:
            Exception: If TTS API call fails or file writing fails.
        """
        try:
            # Validate speed parameter
            if not (0.25 <= speed <= 4.0):
                logger.warning(f"Speed {speed} out of range [0.25, 4.0], clamping")
                speed = max(0.25, min(4.0, speed))

            # Generate output path if not provided
            if output_path is None:
                if index is not None:
                    filename = f"line_{index:04d}_{voice}.mp3"
                else:
                    filename = f"line_{hash(text) % 10000:04d}_{voice}.mp3"
                output_path = self.output_dir / filename

            # Call OpenAI TTS API
            logger.debug(f"Calling TTS API with voice={voice}, speed={speed}")
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=text,
                speed=speed
            )

            # Write audio to file
            output_path = Path(output_path)
            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.debug(f"Saved audio to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to synthesize audio: {str(e)}")
            raise

    def get_output_dir(self) -> Path:
        """Get the output directory for audio files.

        Returns:
            Path to the output directory.
        """
        return self.output_dir


def synthesize_manzai_script(
    script: List[Dict[str, str]],
    output_dir: Optional[Path] = None,
    speed: float = DEFAULT_SPEED,
    api_key: Optional[str] = None
) -> List[Dict[str, any]]:
    """Convenience function to synthesize a manzai script.

    Args:
        script: List of dialogue lines in format [{"speaker": "シンドウ", "text": "..."}]
        output_dir: Directory to save audio files. If None, uses tmp/audio/ in project.
        speed: Speech speed multiplier (0.25 to 4.0). Defaults to 1.0.
        api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.

    Returns:
        List of dictionaries with dialogue info and audio file paths.

    Examples:
        >>> script = [
        ...     {"speaker": "シンドウ", "text": "おはようございます"},
        ...     {"speaker": "モリヤ", "text": "おはよう"}
        ... ]
        >>> audio_data = synthesize_manzai_script(script)
        >>> print(f"Generated {len(audio_data)} audio files")
    """
    synthesizer = ScriptSynthesizer(
        api_key=api_key,
        output_dir=output_dir
    )

    return synthesizer.synthesize_script(script, speed=speed)


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test the synthesizer with sample script
    test_script = [
        {
            "speaker": "シンドウ",
            "text": "おはようございます！今日のビジネスニュース、めっちゃ儲かりそうやで！"
        },
        {
            "speaker": "モリヤ",
            "text": "おはよう。また何か変な話やろ？ちゃんと説明してみい。"
        },
        {
            "speaker": "シンドウ",
            "text": "いやいや、これはガチやって。全財産突っ込むわ！"
        },
        {
            "speaker": "モリヤ",
            "text": "アホか！そんなんリスク管理できてへんやん。ちゃんと分散投資せなあかんで。"
        }
    ]

    try:
        logger.info("Testing synthesizer with sample manzai script...")

        # Create synthesizer instance
        synthesizer = ScriptSynthesizer()

        # Synthesize script
        audio_data = synthesizer.synthesize_script(test_script)

        print(f"\n=== Synthesis Results ===")
        print(f"Total audio files: {len(audio_data)}")
        print(f"Output directory: {synthesizer.get_output_dir()}")
        print()

        for i, data in enumerate(audio_data, 1):
            print(f"{i}. Speaker: {data['speaker']} (Voice: {data['voice']})")
            print(f"   Text: {data['text'][:60]}...")
            print(f"   Audio: {data['audio_path']}")
            print()

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise
