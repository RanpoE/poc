#!/usr/bin/env python3
"""
Intent and Sentiment Analyzer using Claude Sonnet 4.5
Supports both synchronous and asynchronous analysis
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


class IntentAnalyzer:
    """
    Analyzes transcribed text for intent and sentiment using Claude
    """

    INTENTS = [
        "neutral_statement",  # Just sharing information
        "question_faq",  # Asking a question
        "request_command",  # Asking for something to be done
        "call_to_action",  # Prompting action or decision
        "feedback_opinion",  # Expressing opinion or giving feedback
    ]

    SENTIMENTS = ["positive", "neutral", "negative"]

    def __init__(self, api_key=None, max_workers=3):
        """Initialize the analyzer with Anthropic API key"""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"  # Claude Sonnet 4.5

        # Thread pool for async analysis
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def analyze(self, text):
        """
        Analyze text for intent and sentiment

        Args:
            text (str): The transcribed text to analyze

        Returns:
            dict: {"intent": str, "sentiment": str, "confidence": float}
        """
        if not text or not text.strip():
            return {
                "intent": "neutral_statement",
                "sentiment": "neutral",
                "confidence": 0.0,
            }

        prompt = f"""Analyze the following transcribed speech for intent and sentiment.

Text: "{text}"

Classify the intent into ONE of these categories:
1. neutral_statement - Just sharing information or making a statement
2. question_faq - Asking a question or seeking information
3. request_command - Asking for something to be done or giving a command
4. call_to_action - Prompting action, decision, or engagement
5. feedback_opinion - Expressing opinion, feedback, or evaluation

Also classify the sentiment as: positive, neutral, or negative

Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{{"intent": "<intent_category>", "sentiment": "<sentiment>", "confidence": <0.0-1.0>}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.1,  # Low temperature for consistent classification
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the response text
            result_text = response.content[0].text.strip()

            # Parse JSON response
            result = json.loads(result_text)

            # Validate the response
            if "intent" not in result or "sentiment" not in result:
                raise ValueError("Invalid response format")

            # Ensure valid values
            if result["intent"] not in self.INTENTS:
                result["intent"] = "neutral_statement"

            if result["sentiment"] not in self.SENTIMENTS:
                result["sentiment"] = "neutral"

            # Add confidence if not present
            if "confidence" not in result:
                result["confidence"] = 0.8

            return result

        except json.JSONDecodeError as e:
            print(f"Error parsing Claude response: {e}")
            print(
                f"Response was: {result_text if 'result_text' in locals() else 'N/A'}"
            )
            return {
                "intent": "neutral_statement",
                "sentiment": "neutral",
                "confidence": 0.0,
                "error": "parse_error",
            }
        except Exception as e:
            print(f"Error analyzing with Claude: {e}")
            return {
                "intent": "neutral_statement",
                "sentiment": "neutral",
                "confidence": 0.0,
                "error": str(e),
            }

    def analyze_async(self, text, callback=None):
        """
        Analyze text asynchronously without blocking

        Args:
            text (str): The transcribed text to analyze
            callback (function): Optional callback function(result) called when analysis completes

        Returns:
            Future: A Future object that will contain the result
        """

        def analyze_with_callback():
            result = self.analyze(text)
            if callback:
                callback(result)
            return result

        return self.executor.submit(analyze_with_callback)

    def analyze_batch(self, texts):
        """
        Analyze multiple texts in batch

        Args:
            texts (list): List of texts to analyze

        Returns:
            list: List of analysis results
        """
        return [self.analyze(text) for text in texts]

    def shutdown(self):
        """Shutdown the thread pool executor"""
        self.executor.shutdown(wait=True)


if __name__ == "__main__":
    # Test the analyzer
    analyzer = IntentAnalyzer()

    test_texts = [
        "This is me testing whisper live",
        "How does this feature work?",
        "Please send me the documentation",
        "You should definitely try this out",
        "I think Claude 4.5 has been great",
    ]

    print("Testing Intent Analyzer:")
    print("=" * 60)
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Intent: {result['intent']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
