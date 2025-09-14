"""
Claude AI Trading Strategy Module

Интеграция Claude API для принятия торговых решений на основе
анализа портфеля, цен и рыночных данных.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic
from config import Config
from utils import Logger, PerformanceTracker
from data_handler import DataHandler, PortfolioData, TokenPrice
from strategy_engine import BaseStrategy, TradingSignal, TradingAction, RiskLevel, MarketData


@dataclass
class ClaudeAPIResponse:
    """Claude API response container"""
    success: bool
    decision: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    response_time: Optional[float] = None
    tokens_used: Optional[int] = None


class ClaudeAIStrategy(BaseStrategy):
    """
    Claude AI-powered trading strategy

    Использует Claude API для анализа рыночной ситуации и принятия
    торговых решений на основе текущего портфеля и цен.
    """

    def __init__(self, config: Config, data_handler: DataHandler):
        super().__init__(config, data_handler)

        # Claude API конфигурация
        self.claude_api_key = config.get_env_var("CLAUDE_API_KEY")
        if not self.claude_api_key:
            raise ValueError("CLAUDE_API_KEY не найден в конфигурации!")

        # Инициализируем Claude client
        try:
            self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
            self.logger.info("✅ Claude API client инициализирован")
        except Exception as e:
            raise ValueError(f"Ошибка инициализации Claude client: {str(e)}")

        # Модель Claude
        self.claude_model = "claude-sonnet-4-20250514"

        # Trading parameters
        self.min_trade_amount = config.trading.min_trade_amount_usd
        self.max_trade_amount = config.trading.max_trade_amount_usd
        self.max_position_size = 0.3  # Максимум 30% портфеля в одну сделку
        self.confidence_threshold = config.ai.min_confidence_threshold

        # Rate limiting
        self.last_claude_call = 0
        self.min_call_interval = config.ai.rate_limit_seconds

        # Performance tracking
        self.api_call_count = 0
        self.total_cost_estimate = 0.0

        self.logger.info("ClaudeAIStrategy инициализирована",
                         model=self.claude_model,
                         min_trade_amount=self.min_trade_amount,
                         confidence_threshold=self.confidence_threshold)

    def get_strategy_name(self) -> str:
        """Get strategy name identifier"""
        return "ClaudeAI_Strategy"

    def _call_claude_api(self, prompt: str, max_tokens: int = 1500) -> ClaudeAPIResponse:
        """
        Безопасный вызов Claude API с error handling

        Args:
            prompt: Промпт для Claude
            max_tokens: Максимальное количество токенов в ответе

        Returns:
            ClaudeAPIResponse: Ответ Claude или ошибка
        """
        start_time = time.time()

        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_claude_call < self.min_call_interval:
                wait_time = self.min_call_interval - (current_time - self.last_claude_call)
                self.logger.info(f"Rate limiting: ждем {wait_time:.1f}s перед Claude API...")
                time.sleep(wait_time)

            self.logger.info("🧠 Отправляем запрос к Claude API...",
                           model=self.claude_model,
                           prompt_length=len(prompt))

            # Вызов Claude API через официальную библиотеку
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=max_tokens,
                temperature=self.config.ai.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            self.last_claude_call = time.time()
            response_time = time.time() - start_time

            if response.content and len(response.content) > 0:
                response_text = response.content[0].text
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

                # Оценка стоимости
                estimated_cost = (response.usage.input_tokens * 0.000008) + (response.usage.output_tokens * 0.000024)
                self.total_cost_estimate += estimated_cost
                self.api_call_count += 1

                self.logger.info("✅ Claude API ответ получен",
                               response_time=f"{response_time:.2f}s",
                               tokens_used=tokens_used,
                               estimated_cost=f"${estimated_cost:.4f}",
                               total_calls=self.api_call_count)

                return ClaudeAPIResponse(
                    success=True,
                    decision={"raw_response": response_text},
                    reasoning=response_text,
                    response_time=response_time,
                    tokens_used=tokens_used
                )
            else:
                error_msg = "Claude API вернул пустой ответ"
                self.logger.error("❌ Claude API error", error=error_msg)
                return ClaudeAPIResponse(success=False, error=error_msg)

        except anthropic.APIConnectionError as e:
            error_msg = f"Ошибка подключения к Claude API: {str(e)}"
            self.logger.error("❌ Claude API connection error", error=error_msg)
            return ClaudeAPIResponse(success=False, error=error_msg)

        except anthropic.RateLimitError as e:
            error_msg = f"Превышен лимит запросов Claude API: {str(e)}"
            self.logger.error("❌ Claude API rate limit", error=error_msg)
            return ClaudeAPIResponse(success=False, error=error_msg)

        except anthropic.AuthenticationError as e:
            error_msg = f"Ошибка аутентификации Claude API: {str(e)}"
            self.logger.error("❌ Claude API auth error", error=error_msg)
            return ClaudeAPIResponse(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"Неожиданная ошибка Claude API: {str(e)}"
            self.logger.error("❌ Claude API unexpected error", error=error_msg)
            return ClaudeAPIResponse(success=False, error=error_msg)

    def _create_market_analysis_prompt(self, market_data: MarketData) -> str:
        """
        Создает промпт для анализа рынка Claude

        Args:
            market_data: Данные рынка для анализа

        Returns:
            str: Промпт для Claude
        """
        # Подготовим данные портфеля
        portfolio_summary = "Портфель пуст"
        if market_data.portfolio and hasattr(market_data.portfolio, 'tokens') and market_data.portfolio.tokens:
            portfolio_lines = []
            total_value = 0.0

            for token in market_data.portfolio.tokens:
                symbol = token.get('symbol', 'UNKNOWN')
                amount = float(token.get('amount', 0.0))
                price = float(token.get('price', 0.0))
                value = float(token.get('value', 0.0))

                total_value += value
                portfolio_lines.append(
                    f"  {symbol}: {amount:.6f} токенов @ ${price:.6f} = ${value:.2f}"
                )

            portfolio_summary = f"""
Текущий портфель (общая стоимость: ${total_value:.2f}):
{chr(10).join(portfolio_lines)}"""

        # Подготовим данные цен
        prices_summary = "Данные цен недоступны"
        if market_data.token_prices:
            price_lines = []
            for address, price_data in market_data.token_prices.items():
                symbol = getattr(price_data, 'symbol', address[:10] + "...")
                price = getattr(price_data, 'price', 0.0)
                price_lines.append(f"  {symbol}: ${price:.6f}")
            prices_summary = f"""
Актуальные цены токенов:
{chr(10).join(price_lines)}"""

        # Создаем промпт
        prompt = f"""Ты профессиональный криптовалютный трейдер и советник по инвестициям. 
Твоя задача - проанализировать текущую рыночную ситуацию и дать рекомендацию по торговле.

ТЕКУЩАЯ СИТУАЦИЯ:
{portfolio_summary}

{prices_summary}

Временная метка: {datetime.fromtimestamp(market_data.timestamp).isoformat()}

ИНСТРУКЦИИ ПО АНАЛИЗУ:
1. Проанализируй текущее состояние портфеля
2. Оцени рыночную ситуацию на основе доступных данных
3. Определи есть ли возможность для выгодной сделки
4. Учти риски и диверсификацию

ПРАВИЛА ТОРГОВЛИ:
- Минимальная сумма сделки: ${self.min_trade_amount}
- Максимальный размер позиции: {self.max_position_size * 100}% от портфеля
- Всегда предоставляй детальное обоснование решения
- Если нет уверенности >= {self.confidence_threshold * 100}% - рекомендуй HOLD

ДОСТУПНЫЕ ТОКЕНЫ (адреса):
- USDC: {self.config.tokens.usdc_address}
- WETH: {self.config.tokens.weth_address}

ФОРМАТ ОТВЕТА:
Дай ответ в следующем JSON формате:
{{
    "action": "BUY|SELL|HOLD",
    "from_token": "адрес_токена_источника",
    "to_token": "адрес_токена_назначения", 
    "amount": "сумма_в_исходном_токене",
    "confidence": 0.85,
    "risk_level": "LOW|MEDIUM|HIGH",
    "reasoning": "Детальное обоснование решения с анализом рынка",
    "expected_return": 0.05
}}

Если рекомендуешь HOLD, то from_token, to_token, amount должны быть null.

ПРОВЕДИ АНАЛИЗ И ДАЙ РЕКОМЕНДАЦИЮ:"""

        return prompt

    def _parse_claude_response(self, claude_response: str) -> Optional[Dict[str, Any]]:
        """
        Парсит ответ Claude в структурированный формат

        Args:
            claude_response: Текстовый ответ от Claude

        Returns:
            Optional[Dict]: Структурированное решение или None при ошибке
        """
        try:
            # Ищем JSON в ответе
            import re

            # Убираем markdown форматирование если есть
            clean_response = claude_response.replace('```json', '').replace('```', '').strip()

            # Ищем JSON блок
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)

            if json_match:
                json_str = json_match.group()
                decision = json.loads(json_str)

                # Валидируем обязательные поля
                required_fields = ['action', 'confidence', 'risk_level', 'reasoning']
                missing_fields = [f for f in required_fields if f not in decision]

                if missing_fields:
                    self.logger.warning("❌ Отсутствуют обязательные поля в Claude ответе",
                                        missing_fields=missing_fields)
                    return None

                # Валидируем значения
                if decision.get('action') not in ['BUY', 'SELL', 'HOLD']:
                    self.logger.warning("❌ Неверное значение action", action=decision.get('action'))
                    return None

                confidence = decision.get('confidence')
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    self.logger.warning("❌ Неверное значение confidence", confidence=confidence)
                    return None

                if decision.get('risk_level') not in ['LOW', 'MEDIUM', 'HIGH']:
                    self.logger.warning("❌ Неверное значение risk_level", risk_level=decision.get('risk_level'))
                    return None

                self.logger.info("✅ Claude решение успешно распарсено",
                                 action=decision.get('action'),
                                 confidence=decision.get('confidence'),
                                 risk_level=decision.get('risk_level'))
                return decision
            else:
                self.logger.warning("❌ JSON не найден в Claude ответе")
                return None

        except json.JSONDecodeError as e:
            self.logger.error("❌ Ошибка парсинга JSON из Claude ответа", error=str(e))
            return None
        except Exception as e:
            self.logger.error("❌ Неожиданная ошибка при парсинге Claude ответа", error=str(e))
            return None

    def analyze_market(self, market_data: MarketData) -> TradingSignal:
        """
        Главный метод анализа рынка с использованием Claude AI

        Args:
            market_data: Данные рынка для анализа

        Returns:
            TradingSignal: Торговый сигнал на основе анализа Claude
        """
        try:
            self.logger.info("🧠 Начинаем анализ рынка с Claude AI...")

            # Создаем промпт для Claude
            prompt = self._create_market_analysis_prompt(market_data)

            # Вызываем Claude API
            claude_response = self._call_claude_api(prompt, max_tokens=self.config.ai.max_tokens)

            if not claude_response.success:
                # Fallback при ошибке Claude API
                self.logger.warning("Claude API недоступен, создаем HOLD сигнал",
                                    error=claude_response.error)
                return self._create_fallback_signal(market_data, claude_response.error)

            # Парсим ответ Claude
            decision = self._parse_claude_response(claude_response.reasoning)

            if not decision:
                self.logger.warning("Не удалось распарсить Claude ответ, создаем HOLD сигнал")
                return self._create_fallback_signal(market_data, "Parse error")

            # Создаем торговый сигнал
            trading_signal = self._create_trading_signal_from_decision(decision, market_data)

            if trading_signal:
                self.logger.info("✅ Торговый сигнал создан от Claude AI",
                                signal_action=trading_signal.action.value,
                                confidence=trading_signal.confidence,
                                risk_level=trading_signal.risk_level.value)
                return trading_signal
            else:
                return self._create_fallback_signal(market_data, "Signal creation failed")

        except Exception as e:
            self.logger.error("❌ Критическая ошибка в Claude анализе", error=str(e))
            return self._create_fallback_signal(market_data, f"Critical error: {str(e)}")

    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """
        Генерирует торговый сигнал (alias для analyze_market)

        Args:
            market_data: Рыночные данные для анализа

        Returns:
            TradingSignal: Торговый сигнал от Claude AI
        """
        return self.analyze_market(market_data)

    def _create_trading_signal_from_decision(self, decision: Dict[str, Any], market_data: MarketData) -> Optional[TradingSignal]:
        """
        Создает TradingSignal из решения Claude с валидацией

        Args:
            decision: Решение от Claude
            market_data: Рыночные данные

        Returns:
            Optional[TradingSignal]: Торговый сигнал или None при ошибке
        """
        try:
            action_str = decision.get('action', 'HOLD')
            action = TradingAction(action_str)

            confidence = float(decision.get('confidence', 0.0))
            risk_level_str = decision.get('risk_level', 'HIGH')
            risk_level = RiskLevel(risk_level_str)

            reasoning = decision.get('reasoning', 'Claude AI analysis')
            expected_return = decision.get('expected_return')

            # Валидация уверенности
            if confidence < self.confidence_threshold:
                self.logger.warning(f"Уверенность Claude ({confidence}) ниже порога ({self.confidence_threshold}), HOLD")
                action = TradingAction.HOLD

            # Определение токенов и суммы
            if action == TradingAction.HOLD:
                from_token = None
                to_token = None
                amount = "0"
            else:
                from_token = decision.get('from_token', self.config.tokens.usdc_address)
                to_token = decision.get('to_token', self.config.tokens.weth_address)
                amount = str(decision.get('amount', self.min_trade_amount))

                # Валидация суммы
                try:
                    amount_float = float(amount)
                    if amount_float < self.min_trade_amount:
                        amount = str(self.min_trade_amount)
                        self.logger.info(f"Скорректирована сумма до минимума: {amount}")
                    elif amount_float > self.max_trade_amount:
                        amount = str(self.max_trade_amount)
                        self.logger.info(f"Скорректирована сумма до максимума: {amount}")
                except ValueError:
                    amount = str(self.min_trade_amount)
                    self.logger.warning(f"Неверная сумма от Claude, используем минимум: {amount}")

            return TradingSignal(
                action=action,
                from_token=from_token or "",
                to_token=to_token or "",
                amount=amount,
                confidence=confidence,
                risk_level=risk_level,
                reasoning=reasoning,
                timestamp=time.time(),
                strategy_name=self.get_strategy_name(),
                expected_return=expected_return
            )

        except Exception as e:
            self.logger.error("❌ Ошибка создания торгового сигнала", error=str(e))
            return None

    def _create_fallback_signal(self, market_data: MarketData, error_reason: str) -> TradingSignal:
        """
        Создает fallback HOLD сигнал при ошибке Claude

        Args:
            market_data: Рыночные данные
            error_reason: Причина ошибки

        Returns:
            TradingSignal: HOLD сигнал
        """
        return TradingSignal(
            action=TradingAction.HOLD,
            from_token="",
            to_token="",
            amount="0",
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            reasoning=f"Claude AI недоступен - fallback HOLD. Причина: {error_reason}",
            timestamp=time.time(),
            strategy_name=self.get_strategy_name()
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получить метрики производительности Claude AI стратегии"""
        return {
            "strategy_name": self.get_strategy_name(),
            "api_calls_total": self.api_call_count,
            "estimated_cost_usd": round(self.total_cost_estimate, 4),
            "claude_model": self.claude_model,
            "confidence_threshold": self.confidence_threshold,
            "min_trade_amount": self.min_trade_amount,
            "max_trade_amount": self.max_trade_amount
        }