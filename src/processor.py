"""LLM-based news processing and script generation.

This module uses OpenAI GPT-4o to:
1. Select the most relevant business news articles
2. Generate manzai-style scripts in Kansai dialect
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI


# Configure logging
logger = logging.getLogger(__name__)

# OpenAI configuration
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
ASSETS_DIR = PROJECT_ROOT / "assets"
SELECTION_PROMPT_PATH = PROMPTS_DIR / "selection.md"
SCRIPT_PROMPT_PATH = PROMPTS_DIR / "script.md"
PATTERNS_PATH = ASSETS_DIR / "patterns.json"


class NewsProcessor:
    """Process news articles and generate manzai scripts using OpenAI GPT-4o."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """Initialize the NewsProcessor.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: OpenAI model to use. Defaults to gpt-4o.
            temperature: Temperature for generation. Defaults to 0.7.

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        self.model = model
        self.temperature = temperature

        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # OpenAI client will automatically use OPENAI_API_KEY env var
            self.client = OpenAI()

        logger.info(f"Initialized NewsProcessor with model: {model}")

    def select_news(
        self,
        articles: List[Dict[str, str]],
        num_articles: int = 5
    ) -> List[Dict[str, str]]:
        """Select the most relevant news articles using GPT-4o.

        Args:
            articles: List of news articles from fetcher.
            num_articles: Number of articles to select. Defaults to 5.

        Returns:
            List of selected articles with 'reason' field added.

        Raises:
            Exception: If API call fails or response parsing fails.

        Examples:
            >>> processor = NewsProcessor()
            >>> selected = processor.select_news(articles, num_articles=5)
            >>> print(f"Selected {len(selected)} articles")
        """
        logger.info(f"Selecting {num_articles} articles from {len(articles)} candidates")

        try:
            # Load selection prompt
            selection_prompt = self._load_prompt(SELECTION_PROMPT_PATH)

            # Format articles for the prompt
            articles_text = self._format_articles_for_prompt(articles)

            # Create system and user messages
            system_message = selection_prompt
            user_message = (
                f"以下のニュース記事から、必ず{num_articles}件を選んでください。\n\n"
                f"重要: 必ず{num_articles}件すべてを含むJSON配列として返してください。\n"
                f'出力形式: {{"articles": [{{"title": "...", "link": "...", "summary": "...", "reason": "..."}}, ...]}}\n\n'
                f"{articles_text}"
            )

            # Call OpenAI API
            logger.info("Calling OpenAI API for news selection...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Extract and parse response
            content = response.choices[0].message.content
            logger.debug(f"API response: {content}")

            # Log first 500 chars for debugging
            logger.info(f"Received response (first 500 chars): {content[:500]}")

            # Parse JSON response
            selected_articles = self._parse_selection_response(content)

            logger.info(f"Successfully selected {len(selected_articles)} articles")
            return selected_articles

        except Exception as e:
            logger.error(f"Failed to select news articles: {str(e)}")
            raise

    def generate_script(
        self,
        selected_news: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Generate manzai script from selected news articles.

        Args:
            selected_news: List of selected news articles.

        Returns:
            List of script lines in format: [{"speaker": "シンドウ", "text": "..."}]

        Raises:
            Exception: If API call fails or response parsing fails.

        Examples:
            >>> processor = NewsProcessor()
            >>> script = processor.generate_script(selected_news)
            >>> print(f"Generated script with {len(script)} lines")
        """
        logger.info(f"Generating manzai script from {len(selected_news)} news articles")

        try:
            # Load script generation prompt
            script_prompt = self._load_prompt(SCRIPT_PROMPT_PATH)

            # Format selected news for the prompt
            news_text = self._format_selected_news(selected_news)

            # Create system and user messages
            system_message = script_prompt
            user_message = (
                f"以下のニュースを使って漫才台本を作成してください。\n\n"
                f"重要: 必ず台本全体を1つのJSON配列として返してください。\n"
                f'出力形式: {{"script": [{{"speaker": "シンドウ", "text": "..."}}, {{"speaker": "守屋", "text": "..."}}]}}\n\n'
                f"{news_text}"
            )

            # Call OpenAI API
            logger.info("Calling OpenAI API for script generation...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Extract and parse response
            content = response.choices[0].message.content
            logger.debug(f"API response: {content[:200]}...")

            # Parse JSON response
            script = self._parse_script_response(content)

            logger.info(f"Successfully generated script with {len(script)} lines")
            return script

        except Exception as e:
            logger.error(f"Failed to generate script: {str(e)}")
            raise

    def generate_manzai_script(
        self,
        selected_news: List[Dict[str, str]],
        max_retries: int = 3
    ) -> List[Dict[str, str]]:
        """Generate detailed manzai script with pattern-based character interactions.

        This method:
        1. Loads manzai patterns from patterns.json
        2. Assigns random patterns to each news article (no duplication)
        3. Loads script prompt from prompts/script.md
        4. Injects news+patterns into the {{NEWS_WITH_PATTERNS}} placeholder
        5. Calls OpenAI API with strict JSON format requirements
        6. Validates and retries on parsing errors

        Args:
            selected_news: List of selected news articles (typically 5).
            max_retries: Maximum number of retries for API calls. Defaults to 3.

        Returns:
            List of script lines in format: [{"speaker": "モリヤ", "text": "..."}, ...]

        Raises:
            Exception: If all retry attempts fail or patterns cannot be loaded.

        Examples:
            >>> processor = NewsProcessor()
            >>> script = processor.generate_manzai_script(selected_news)
            >>> print(f"Generated script with {len(script)} lines")
        """
        logger.info(
            f"Generating pattern-based manzai script from {len(selected_news)} news articles"
        )

        try:
            # Step 1: Load manzai patterns
            patterns = self._load_patterns()

            # Step 2: Assign patterns to news articles
            news_with_patterns = self._assign_patterns_to_news(selected_news, patterns)

            # Step 3: Format news with patterns for prompt
            news_patterns_text = self._format_news_with_patterns(news_with_patterns)

            # Step 4: Load script prompt template from prompts/script.md
            script_prompt_template = self._load_prompt(SCRIPT_PROMPT_PATH)

            # Step 5: Inject news+patterns into the template
            prompt_with_data = script_prompt_template.replace(
                "{{NEWS_WITH_PATTERNS}}",
                news_patterns_text
            )

            # Step 6: Create system message (the entire prompt becomes the system message)
            system_message = prompt_with_data

            # User message just confirms the request
            user_message = (
                "上記のニュースとパターンを使って、関西弁の漫才台本を作成してください。\n"
                "必ずJSON形式で出力してください。"
            )

            # Step 7: Call OpenAI API with retry logic
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Calling OpenAI API for manzai script generation (attempt {attempt}/{max_retries})...")

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=self.temperature,
                        response_format={"type": "json_object"}
                    )

                    # Extract and parse response
                    content = response.choices[0].message.content
                    logger.debug(f"API response (first 300 chars): {content[:300]}...")

                    # Validate and parse JSON response
                    script = self._parse_script_response(content)

                    # Additional validation: check speaker names
                    self._validate_speaker_names(script)

                    logger.info(f"Successfully generated manzai script with {len(script)} lines")
                    return script

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed with parsing error: {e}"
                    )

                    if attempt == max_retries:
                        logger.error("All retry attempts exhausted")
                        raise

                    logger.info(f"Retrying... ({attempt + 1}/{max_retries})")

        except Exception as e:
            logger.error(f"Failed to generate manzai script: {str(e)}")
            raise

    def _load_prompt(self, prompt_path: Path) -> str:
        """Load prompt from markdown file.

        Args:
            prompt_path: Path to prompt file.

        Returns:
            Prompt content as string.

        Raises:
            FileNotFoundError: If prompt file doesn't exist.
        """
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.debug(f"Loaded prompt from: {prompt_path}")
        return content

    def _load_patterns(self) -> List[Dict[str, str]]:
        """Load manzai patterns from JSON file.

        Returns:
            List of pattern dictionaries with pattern_id, style, shindo_logic, moriya_tsukkomi.

        Raises:
            FileNotFoundError: If patterns file doesn't exist.
            ValueError: If patterns file is not valid JSON.
        """
        if not PATTERNS_PATH.exists():
            raise FileNotFoundError(f"Patterns file not found: {PATTERNS_PATH}")

        try:
            with open(PATTERNS_PATH, 'r', encoding='utf-8') as f:
                patterns = json.load(f)

            logger.info(f"Loaded {len(patterns)} manzai patterns from {PATTERNS_PATH}")
            return patterns

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse patterns JSON: {e}")
            raise ValueError(f"Invalid JSON in patterns file: {e}")

    def _assign_patterns_to_news(
        self,
        selected_news: List[Dict[str, str]],
        patterns: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Assign random manzai patterns to each news article without duplication.

        Args:
            selected_news: List of selected news articles.
            patterns: List of available manzai patterns.

        Returns:
            List of news articles with assigned patterns.

        Raises:
            ValueError: If not enough patterns available for the number of news articles.
        """
        num_news = len(selected_news)

        if len(patterns) < num_news:
            raise ValueError(
                f"Not enough patterns ({len(patterns)}) for {num_news} news articles"
            )

        # Randomly select patterns without duplication
        selected_patterns = random.sample(patterns, num_news)

        # Assign patterns to news articles
        news_with_patterns = []
        for news, pattern in zip(selected_news, selected_patterns):
            news_with_pattern = news.copy()
            news_with_pattern['pattern'] = pattern
            news_with_patterns.append(news_with_pattern)

            logger.debug(
                f"Assigned pattern '{pattern['style']}' (ID: {pattern['pattern_id']}) "
                f"to news: {news['title'][:50]}..."
            )

        logger.info(f"Successfully assigned {num_news} unique patterns to news articles")
        return news_with_patterns

    def _format_articles_for_prompt(self, articles: List[Dict[str, str]]) -> str:
        """Format articles list into readable text for prompt.

        Args:
            articles: List of article dictionaries.

        Returns:
            Formatted string with all articles.
        """
        formatted = []
        for i, article in enumerate(articles, 1):
            formatted.append(f"【記事{i}】")
            formatted.append(f"タイトル: {article['title']}")
            formatted.append(f"リンク: {article['link']}")
            formatted.append(f"要約: {article['summary']}")
            formatted.append("")  # Empty line between articles

        return "\n".join(formatted)

    def _format_selected_news(self, selected_news: List[Dict[str, str]]) -> str:
        """Format selected news for script generation prompt.

        Args:
            selected_news: List of selected news articles.

        Returns:
            Formatted string with selected news.
        """
        formatted = []
        for i, news in enumerate(selected_news, 1):
            formatted.append(f"【ニュース{i}】")
            formatted.append(f"タイトル: {news['title']}")
            formatted.append(f"要約: {news['summary']}")
            if 'reason' in news:
                formatted.append(f"選定理由: {news['reason']}")
            formatted.append("")

        return "\n".join(formatted)

    def _format_news_with_patterns(
        self,
        news_with_patterns: List[Dict[str, str]]
    ) -> str:
        """Format news with assigned patterns for detailed manzai script generation.

        Args:
            news_with_patterns: List of news articles with assigned patterns.

        Returns:
            Formatted string with news and pattern instructions for each topic.
        """
        formatted = []

        for i, news in enumerate(news_with_patterns, 1):
            pattern = news['pattern']

            formatted.append(f"【トピック{i}】")
            formatted.append(f"ニュース: {news['title']}")
            formatted.append(f"要約: {news['summary']}")
            formatted.append("")
            formatted.append(f"パターン: {pattern['style']}")
            formatted.append(f"シンドウの思考: {pattern['shindo_logic']}")
            formatted.append(f"モリヤのツッコミ方針: {pattern['moriya_tsukkomi']}")
            formatted.append("")
            formatted.append("---")
            formatted.append("")

        return "\n".join(formatted)

    def _build_character_instructions(self) -> str:
        """Build comprehensive character instructions for manzai script generation.

        Returns:
            Detailed character instructions including roles, dialect, and structure.
        """
        instructions = """
# 経済漫才台本生成システム - Capital Gains

あなたは関西弁の経済漫才台本を生成するAIアシスタントです。
ビジネスニュースを題材に、シンドウ（ボケ）とモリヤ（ツッコミ）の掛け合いで台本を作成してください。

## 登場人物

### モリヤ（ツッコミ役）
- **表記**: 必ず「モリヤ」のカタカナで統一してください（AIの読み間違い防止）
- **役割**: ツッコミ兼解説者
- **責任**:
  - 各ニュースの冒頭で、背景・数字・市場の反応など詳細を「しっかり目（4〜6セリフ分）」に解説する
  - シンドウのボケを鋭くツッコミ・訂正した直後に、リスナーが納得できる「正しい解説」を入れる
  - 各トピックの最後に「投資や生活への教訓」をまとめる
- **性格**: 冷静、論理的、でも関西人らしいキレのあるツッコミ
- **口調**: 「〜やねん」「〜やろ」「ちゃうねん！」など関西弁を多用

### シンドウ（ボケ役）
- **表記**: 必ず「シンドウ」のカタカナで統一してください（AIの読み間違い防止）
- **役割**: ボケ担当
- **責任**:
  - 割り当てられたパターンに従い、モリヤの詳細解説を台無しにする「斜め上の解釈」を大真面目に展開
  - 予測不能で、常識外れだが、パターンには忠実な発言
- **性格**: 天然、予測不能、独自の論理を持つ
- **口調**: 「〜や」「〜やで」「〜やろ」など関西弁

## 台本の構成ルール

### 基本フロー（各トピックごと）
1. **モリヤの導入解説**（4〜6セリフ）
   - ニュースの背景、重要な数字、市場の反応を詳しく説明
   - リスナーが状況を理解できるよう、丁寧に情報提供

2. **シンドウのボケ展開**（2〜4セリフ）
   - 割り当てられたパターンに従った「斜め上の解釈」
   - モリヤの解説を予想外の方向に持っていく

3. **モリヤのツッコミ＋正しい解説**（3〜5セリフ）
   - 「〜ちゃうねん！」「それは違うやろ！」などキレのあるツッコミ
   - その直後に、正確な経済知識・投資の視点を提供

4. **追加のやり取り**（必要に応じて2〜4往復）
   - テンポよく、でも情報は抜けなく

5. **モリヤのまとめ**（1〜2セリフ）
   - 投資家・リスナーへの教訓や示唆

### 関西弁の使い方
- **倒置法**: 「大変なことなっとるで、これ」
- **絶句**: 「....」（ツッコミ前の溜め）
- **独特の助詞**: 「〜やろ」「〜やねん」「〜しとるんや」「〜ちゃうねん」
- **強調**: 「めっちゃ」「ほんまに」「えらいこっちゃ」
- **ツッコミ**: 「何言うとんねん！」「アホか！」「そうはならんやろ！」

### 単調さの排除
- 5つのニュースすべてで「異なるパターン」が割り当てられています
- 各トピックで展開の仕方、テンポ、ボケの種類を変えてください
- 同じツッコミフレーズを連続で使わないこと

### 出力形式（厳守）
必ず以下のJSON形式で出力してください：

```json
{
  "script": [
    {"speaker": "モリヤ", "text": "さて、今日の最初のニュースやけどな..."},
    {"speaker": "シンドウ", "text": "お、何や何や？"},
    {"speaker": "モリヤ", "text": "..."}
  ]
}
```

- `speaker` は必ず「モリヤ」または「シンドウ」（カタカナ）
- `text` は1セリフの内容
- 配列の順序がそのまま台本の順序になります
"""
        return instructions.strip()

    def _validate_speaker_names(self, script: List[Dict[str, str]]) -> None:
        """Validate that speaker names are correct (モリヤ or シンドウ).

        Args:
            script: List of script lines with speaker and text.

        Raises:
            ValueError: If invalid speaker names are found.
        """
        valid_speakers = {"モリヤ", "シンドウ"}
        invalid_lines = []

        for i, line in enumerate(script):
            speaker = line.get("speaker", "")
            if speaker not in valid_speakers:
                invalid_lines.append(f"Line {i + 1}: '{speaker}'")

        if invalid_lines:
            error_msg = (
                f"Invalid speaker names found. Expected 'モリヤ' or 'シンドウ', "
                f"but found:\n" + "\n".join(invalid_lines)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Speaker name validation passed for {len(script)} lines")

    def _parse_selection_response(self, response: str) -> List[Dict[str, str]]:
        """Parse GPT response for news selection.

        Args:
            response: JSON string response from GPT.

        Returns:
            List of selected articles.

        Raises:
            ValueError: If response is not valid JSON or has unexpected format.
        """
        try:
            data = json.loads(response)

            # Log the response for debugging
            logger.debug(f"Parsed JSON data type: {type(data)}")
            logger.debug(f"Parsed JSON keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

            # Handle different possible response formats
            if isinstance(data, list):
                articles = data
            elif isinstance(data, dict) and 'articles' in data:
                articles = data['articles']
            elif isinstance(data, dict) and 'selected' in data:
                articles = data['selected']
            elif isinstance(data, dict) and 'selected_articles' in data:
                articles = data['selected_articles']
            elif isinstance(data, dict) and 'news' in data:
                articles = data['news']
            else:
                # If it's a dict with other keys, try to find a list value
                for key, value in data.items():
                    if isinstance(value, list):
                        logger.info(f"Found list in key '{key}', using as articles")
                        articles = value
                        break
                else:
                    # Last resort: check if this is a single article dict, log and error
                    if all(k in data for k in ['title', 'link', 'summary', 'reason']):
                        logger.error("Response appears to be a single article instead of a list")
                        logger.error(f"Full response: {json.dumps(data, indent=2, ensure_ascii=False)}")
                        raise ValueError(
                            "GPT returned a single article instead of a list. "
                            "Expected JSON array of articles."
                        )
                    else:
                        logger.error(f"Full response: {json.dumps(data, indent=2, ensure_ascii=False)}")
                        raise ValueError(f"Unexpected response format: {list(data.keys())}")

            # Validate article structure
            for article in articles:
                if not isinstance(article, dict):
                    raise ValueError(f"Invalid article format: {article}")
                if 'title' not in article:
                    raise ValueError(f"Article missing 'title' field: {article}")

            return articles

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response: {e}")

    def _parse_script_response(self, response: str) -> List[Dict[str, str]]:
        """Parse GPT response for script generation.

        Args:
            response: JSON string response from GPT.

        Returns:
            List of script lines with speaker and text.

        Raises:
            ValueError: If response is not valid JSON or has unexpected format.
        """
        try:
            data = json.loads(response)

            # Handle different possible response formats
            if isinstance(data, list):
                script = data
            elif isinstance(data, dict) and 'script' in data:
                script = data['script']
            elif isinstance(data, dict) and 'dialogue' in data:
                script = data['dialogue']
            else:
                # If it's a dict with other keys, try to find a list value
                for value in data.values():
                    if isinstance(value, list):
                        script = value
                        break
                else:
                    raise ValueError(f"Unexpected response format: {list(data.keys())}")

            # Validate script structure
            for line in script:
                if not isinstance(line, dict):
                    raise ValueError(f"Invalid script line format: {line}")
                if 'speaker' not in line or 'text' not in line:
                    raise ValueError(f"Script line missing required fields: {line}")

            return script

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response[:500]}...")
            raise ValueError(f"Invalid JSON response: {e}")


def select_and_generate(
    articles: List[Dict[str, str]],
    num_articles: int = 5,
    api_key: Optional[str] = None
) -> Dict[str, any]:
    """Convenience function to select news and generate script in one call.

    Args:
        articles: List of news articles from fetcher.
        num_articles: Number of articles to select. Defaults to 5.
        api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.

    Returns:
        Dictionary with 'selected_news' and 'script' keys.

    Examples:
        >>> from src.fetcher import fetch_news
        >>> articles = fetch_news()
        >>> result = select_and_generate(articles)
        >>> print(f"Selected: {len(result['selected_news'])} articles")
        >>> print(f"Script: {len(result['script'])} lines")
    """
    processor = NewsProcessor(api_key=api_key)

    selected_news = processor.select_news(articles, num_articles=num_articles)
    script = processor.generate_script(selected_news)

    return {
        'selected_news': selected_news,
        'script': script
    }


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Import fetcher to get test data
    from fetcher import fetch_news

    # Test the processor
    try:
        logger.info("Fetching news articles...")
        articles = fetch_news(max_items=20)

        logger.info("Processing articles...")
        result = select_and_generate(articles, num_articles=5)

        print("\n=== Selected News ===")
        for i, news in enumerate(result['selected_news'], 1):
            print(f"\n{i}. {news['title']}")
            if 'reason' in news:
                print(f"   理由: {news['reason']}")

        print("\n=== Generated Script ===")
        print(f"Total lines: {len(result['script'])}\n")
        for i, line in enumerate(result['script'][:10], 1):  # Show first 10 lines
            print(f"{i}. {line['speaker']}: {line['text']}")

        if len(result['script']) > 10:
            print(f"\n... and {len(result['script']) - 10} more lines")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
