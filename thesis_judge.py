from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import threading

from loguru import logger
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


@dataclass
class GeminiJudgeConfig:
    model_id: str = "gemini-flash-lite-latest"
    pro_model_id: str = "gemini-flash-lite-latest"  # Используем ту же модель
    pro_thinking_budget: int = 8000  # Thinking budget для Pro (required for thinking mode)
    max_output_tokens: int = 64
    temperature: float = 0.0
    timeout: int = 10  # Таймаут для API вызова (секунды)
    max_retries: int = 1  # УСКОРЕНИЕ: 1 попытка вместо 3 (экономим ~2 сек на фейле)
    retry_delay: float = 0.5  # УСКОРЕНИЕ: быстрый retry (было 1.0)


class GeminiJudge:
    """
    Проверяет, покрывает ли речь пользователя заданный тезис (да/нет + уверенность 0..1),
    используя Google GenAI (Gemini Flash Lite + Pro для улучшения).
    
    Архитектура:
    1. Быстрый вызов Gemini Flash Lite → мгновенный ответ
    2. Параллельный вызов Gemini Pro → улучшенная оценка (приоритет если отличается)
    
    Метрики:
    - Количество успешных/неудачных вызовов
    - Время ответа Gemini
    - Количество retries
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_pro: bool = True,
        openai_api_key: Optional[str] = None,
    ) -> None:
        cfg = GeminiJudgeConfig()
        self.model_id = model_id or cfg.model_id
        self.pro_model_id = cfg.pro_model_id
        self.pro_thinking_budget = cfg.pro_thinking_budget
        self.max_output_tokens = int(max_output_tokens if max_output_tokens is not None else cfg.max_output_tokens)
        self.temperature = float(temperature if temperature is not None else cfg.temperature)
        self.timeout = cfg.timeout
        self.max_retries = cfg.max_retries
        self.retry_delay = cfg.retry_delay
        self.enable_pro = enable_pro

        import os
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY не найден для GeminiJudge")
        self.client = genai.Client(api_key=key)
        
        # Fallback на OpenAI
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
                logger.info("OpenAI fallback доступен")
            except ImportError:
                logger.warning("openai пакет не установлен, fallback недоступен")
        
        # Метрики
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retries": 0,
            "fallback_calls": 0,
            "pro_upgrades": 0,  # Сколько раз Pro изменил решение Flash
            "total_time": 0.0,
        }
        
        # Callback для обновлений от Pro модели
        self._upgrade_callback = None
        self._last_result_container = None
        
        logger.info(f"GeminiJudge инициализирован: model={self.model_id}, pro={self.pro_model_id}, enable_pro={enable_pro}")

    def judge(self, thesis: str, dialogue_context) -> Tuple[bool, float]:
        """
        Оценивает покрытие тезиса студентом на основе контекста диалога.
        
        Args:
            thesis: Текст тезиса-подсказки для проверки
            dialogue_context: Контекст диалога в одном из форматов:
                - str: простой текст (legacy API)
                - List[Tuple[str, str]]: список пар (роль, текст) где:
                    * роль: "студент" или "экзаменатор"
                    * текст: реплика участника
                    Пример: [("экзаменатор", "Кто первым полетел в космос?"), 
                             ("студент", "Гагарин")]
                
                ВАЖНО: Передаётся ВСЬ контекст диалога для обеих моделей
                       (Flash и Pro получают одинаковые данные).
        
        Returns:
            Tuple[bool, float]: (покрыт_тезис, уверенность_0_1)
                - covered: True если студент передал суть тезиса или тема сменилась
                - confidence: уровень уверенности от 0.0 до 1.0
        
        Примечание:
            Использует двухуровневую оценку:
            1. Gemini Flash Lite (быстрый ответ)
            2. Gemini Pro (параллельно, улучшает оценку если отличается)
        """
        self.metrics["total_calls"] += 1
        start_time = time.time()
        
        if not thesis:
            return (False, 0.0)
        
        # Формируем контекст диалога
        if isinstance(dialogue_context, str):
            # Старый API - просто текст
            dialogue_text = dialogue_context.strip()
        elif isinstance(dialogue_context, list):
            # Новый API - список (роль, текст)
            # Передаём ВСЕ реплики для максимального контекста
            lines = []
            for role, text in dialogue_context:
                if role == "студент":
                    lines.append(f"Студент: {text}")
                else:
                    lines.append(f"Экзаменатор: {text}")
            dialogue_text = "\n".join(lines)
        else:
            return (False, 0.0)
        
        if not dialogue_text:
            return (False, 0.0)

        # Быстрый вызов Flash Lite с retry
        flash_result = self._call_with_retry(
            model=self.model_id,
            thesis=thesis,
            dialogue_text=dialogue_text,
            is_pro=False
        )
        
        elapsed = time.time() - start_time
        self.metrics["total_time"] += elapsed
        
        if flash_result is None:
            # Gemini полностью упал
            self.metrics["failed_calls"] += 1  # Gemini упал
            
            # Пробуем OpenAI fallback
            logger.info("🔄 Используем OpenAI fallback")
            flash_result = self._call_openai_fallback(thesis, dialogue_text)
            if flash_result is None:
                # И OpenAI упал - полный провал, нет результата
                return (False, 0.0)
            # OpenAI fallback сработал - есть результат
            self.metrics["fallback_calls"] += 1
        
        self.metrics["successful_calls"] += 1
        covered, confidence = flash_result
        
        # Параллельный вызов Pro для улучшения (если включен)
        if self.enable_pro and dialogue_context:
            # Сохраняем snapshot для валидации ответа Pro
            snapshot_context = dialogue_text
            snapshot_thesis = thesis
            snapshot_timestamp = time.time()
            
            # Количество реплик на момент запроса (для проверки что контекст не изменился)
            if isinstance(dialogue_context, list):
                snapshot_context_size = len(dialogue_context)
            else:
                snapshot_context_size = len(dialogue_context.split('\n'))
            
            # Результат который может быть обновлён Pro
            result_container = {"covered": covered, "confidence": confidence, "upgraded": False}
            
            def pro_worker():
                pro_result = self._call_with_retry(
                    model=self.pro_model_id,
                    thesis=snapshot_thesis,
                    dialogue_text=snapshot_context,
                    is_pro=True
                )
                if pro_result:
                    pro_covered, pro_conf = pro_result
                    pro_response_time = time.time()
                    
                    # ВАЖНО: Проверяем валидность ответа Pro
                    # Вызываем callback с snapshot данными для валидации
                    if hasattr(self, '_upgrade_callback') and self._upgrade_callback:
                        try:
                            # Callback должен вернуть True если можно применить решение Pro
                            can_apply = self._upgrade_callback(
                                thesis=snapshot_thesis,
                                covered=pro_covered,
                                confidence=pro_conf,
                                snapshot_timestamp=snapshot_timestamp,
                                snapshot_context_size=snapshot_context_size,
                                response_timestamp=pro_response_time
                            )
                            
                            if can_apply and pro_covered != result_container["covered"]:
                                self.metrics["pro_upgrades"] += 1
                                logger.info(f"🔄 Pro upgrade: Flash={result_container['covered']} → Pro={pro_covered} (conf {pro_conf:.2f})")
                                result_container["covered"] = pro_covered
                                result_container["confidence"] = pro_conf
                                result_container["upgraded"] = True
                            elif not can_apply:
                                logger.debug(f"⏭️  Pro ответ проигнорирован (контекст устарел): thesis={snapshot_thesis[:50]}...")
                        except Exception as e:
                            logger.error(f"Callback ошибка: {e}")
            
            thread = threading.Thread(target=pro_worker, daemon=True)
            thread.start()
            
            # Сохраняем ссылку на контейнер для возможного доступа позже
            self._last_result_container = result_container
        
        return (covered, confidence)
    
    def _call_with_retry(
        self, 
        model: str, 
        thesis: str, 
        dialogue_text: str,
        is_pro: bool = False
    ) -> Optional[Tuple[bool, float]]:
        """Вызывает Gemini с retry и таймаутом."""
        sys_instr = (
            "Ты помощник на экзамене. ЗАДАЧА: проверь правильно ли студент ОТВЕТИЛ НА ВОПРОС экзаменатора. "
            "Тезис-подсказка - это ЭТАЛОННЫЙ ОТВЕТ для проверки. "
            "Студент НЕ должен повторять тезис дословно! Он должен правильно ответить на вопрос! "
            "ВАЖНО - ASR (распознавание речи) ошибается! Будь ОЧЕНЬ снисходительным: "
            "1) Фонетическое сходство = верно "
            "2) Опечатки/пропуски предлогов = верно "
            "3) Синонимы/перефразировки = верно "
            "4) Частичный ответ (назвал ключевое слово из тезиса) = верно "
            "5) Неправильный падеж/число = верно "
            "Если студент ПРАВИЛЬНО ответил на вопрос (даже не полностью) - covered=true. "
            "Если разговор ушёл в другую тему - covered=true. "
            "Ответ строго в JSON: {\"covered\": true|false, \"confidence\": 0..1}"
        )
        user_prompt = (
            "Эталонный ответ:\n" + thesis.strip() + "\n\n" +
            "Диалог:\n" + dialogue_text + "\n\n" +
            "Правильно ли студент ответил на вопрос? (сравни с эталоном, учитывай ASR-ошибки)"
        )

        # Для Pro модели используем thinking mode, для остальных - без thinking
        thinking_budget = self.pro_thinking_budget if is_pro else 0
        
        cfg = types.GenerateContentConfig(
            system_instruction=sys_instr,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=0.9,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget > 0 else None,
        )
        
        for attempt in range(self.max_retries):
            try:
                # Вызываем Gemini (без таймаута - signal не работает в потоках)
                resp = self.client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
                    config=cfg,
                )
                raw = (resp.text or "").strip()
                
                # Парсим ответ
                import json
                import re
                cleaned = raw
                if '```json' in cleaned:
                    match = re.search(r'```json\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                    if match:
                        cleaned = match.group(1)
                elif '```' in cleaned:
                    match = re.search(r'```\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                    if match:
                        cleaned = match.group(1)
                
                data = json.loads(cleaned)
                covered = bool(data.get("covered", False))
                conf = float(data.get("confidence", 0.0))
                conf = max(0.0, min(1.0, conf))
                
                model_label = "Pro" if is_pro else "Flash"
                logger.debug(f"{model_label} ответ: covered={covered}, conf={conf:.2f}")
                return (covered, conf)
                
            except Exception as e:  # noqa: BLE001
                if attempt < self.max_retries - 1:
                    self.metrics["retries"] += 1
                    logger.warning(f"Gemini {model} attempt {attempt+1} failed: {e}, retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Gemini {model} failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None
    
    def _call_openai_fallback(self, thesis: str, dialogue_text: str) -> Optional[Tuple[bool, float]]:
        """Fallback на OpenAI если Gemini упал."""
        if not self.openai_client:
            return None
        
        try:
            logger.info("🔄 Используем OpenAI fallback")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 
                     "Ты помощник на экзамене. Оцени покрыл ли студент тезис. "
                     "Будь мягким к опечаткам ASR, синонимам. "
                     "Ответ в JSON: {\"covered\": true|false, \"confidence\": 0..1}"},
                    {"role": "user", "content": 
                     f"Тезис:\n{thesis}\n\nДиалог:\n{dialogue_text}\n\nПокрыл ли студент тезис?"}
                ],
                temperature=0.0,
                max_tokens=64,
                timeout=10,
            )
            
            import json
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            covered = bool(data.get("covered", False))
            conf = float(data.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            return (covered, conf)
        except Exception as e:  # noqa: BLE001
            logger.error(f"OpenAI fallback failed: {e}")
            return None
    
    def set_upgrade_callback(self, callback):
        """
        Устанавливает callback для обновлений от Pro модели.
        
        Args:
            callback: функция с сигнатурой:
                (thesis: str, covered: bool, confidence: float, 
                 snapshot_timestamp: float, snapshot_context_size: int,
                 response_timestamp: float) -> bool
                
                Должна вернуть True если можно применить решение Pro,
                False если контекст устарел (тезис закрылся или появились новые реплики).
                
                Параметры:
                - thesis: текст тезиса который оценивался
                - covered: результат оценки Pro (True/False)
                - confidence: уверенность Pro (0..1)
                - snapshot_timestamp: когда был отправлен запрос Pro
                - snapshot_context_size: количество реплик на момент запроса
                - response_timestamp: когда пришёл ответ Pro
        """
        self._upgrade_callback = callback
    
    def get_metrics(self) -> dict:
        """Возвращает статистику работы судьи."""
        return self.metrics.copy()


__all__ = ["GeminiJudge", "GeminiJudgeConfig"]
