from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os

from loguru import logger
from dotenv import load_dotenv
from google import genai
from google.genai import types

# OpenAI fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

load_dotenv()


@dataclass
class ThesisGenConfig:
    model_id: str = "gemini-flash-lite-latest"
    max_output_tokens: int = 256
    temperature: float = 0.3
    n_theses: int = 8
    language: str = "ru"


class GeminiThesisGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        cfg = ThesisGenConfig()
        self.model_id = model_id or cfg.model_id
        self.max_output_tokens = int(max_output_tokens if max_output_tokens is not None else cfg.max_output_tokens)
        self.temperature = float(temperature if temperature is not None else cfg.temperature)

        # Gemini client
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY не найден для ThesisGenerator")
        self.client = genai.Client(api_key=key)
        logger.info(f"ThesisGenerator инициализирован: model={self.model_id}")
        
        # OpenAI fallback (опционально)
        self.openai_client = None
        if OPENAI_AVAILABLE:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    self.openai_client = OpenAI(api_key=openai_key)
                    logger.info("OpenAI fallback включен для генерации тезисов")
                except Exception as e:
                    logger.warning(f"OpenAI fallback недоступен: {e}")
            else:
                logger.debug("OPENAI_API_KEY не найден, OpenAI fallback отключен")

    def generate(self, question_text: str, n: int = 8, language: str = "ru", context: Optional[str] = None) -> List[str]:
        if not question_text:
            return []
        n = max(1, int(n))
        sys_instr = (
            "Ты генератор тезисов для устных экзаменов."
            " СТРОГИЕ ПРАВИЛА:"
            " 1. АГРЕССИВНО ПЫТАЙСЯ ПОНЯТЬ ВОПРОС даже с опечатками и ошибками распознавания:"
            "    - Система распознавания речи (ASR) может делать ошибки!"
            "    - 'убложить' = 'умножить', 'плюс' вместо 'плос', 'минос' вместо 'минус'"
            "    - Распознавай числа: '102', 'сто два', 'стодва' - все равно"
            "    - Исправляй опечатки: 'осовали' -> 'основали', 'рдился' -> 'родился'"
            "    - Пропущенные буквы, перестановки, созвучные слова - пытайся понять!"
            " 2. ВСЕГДА ОТВЕЧАЙ если хоть что-то понял, даже на математику!"
            "    - '102 убложить на 24' -> посчитай и дай ответ словами"
            "    - 'сколько будет 5 плюс 3' -> посчитай"
            " 3. ⚠️ ПЫТАЙСЯ ПОНЯТЬ ВОПРОС, НО НЕ ВЫДУМЫВАЙ ФАКТЫ!"
            "    - АГРЕССИВНО пытайся расшифровать искаженный вопрос от ASR"
            "    - Если смог понять вопрос - ОТВЕЧАЙ, даже если не уверен на 100%"
            "    - НО: НЕ ВЫДУМЫВАЙ факты! Если не знаешь точного ответа - скажи общую информацию"
            "    - ПУСТАЯ СТРОКА только если: полная бессмыслица (шум, набор букв, НЕ похоже на вопрос)"
            "    - Помни: лучше дать приблизительный ответ, чем выдумать неправильные факты"
            " 4. ⚠️⚠️⚠️ СТРУКТУРА ТЕЗИСА - ОТВЕТ ВСЕГДА ПЕРВЫМ СЛОВОМ:"
            "    - КТО? → ИМЯ первым: 'Юрий Гагарин - первый космонавт'"
            "    - КОГДА? → ДАТА первой: 'Двенадцатого апреля тысяча девятьсот шестьдесят первого года - полёт Гагарина'"
            "    - ЧТО? → КЛЮЧЕВОЕ СЛОВО первым: 'Кератин - основной компонент волос'"
            "    - ГДЕ? → МЕСТО первым: 'Москва - столица России'"
            "    - СКОЛЬКО? → ЧИСЛО первым: 'Двести двадцать пять - пятнадцать в квадрате'"
            "    - ❌ НЕ ПИШИ: 'Основной компонент волос - кератин' (неправильно!)"
            "    - ✅ ПИШИ: 'Кератин - основной компонент волос' (правильно!)"
            " 5. Каждый тезис - короткое предложение (до 15 слов)"
            " 6. ТОЛЬКО конкретные факты, без воды"
            " 7. ⚠️ КОНТЕКСТ ТОЛЬКО ДЛЯ МЕСТОИМЕНИЙ! Используй контекст ТОЛЬКО для понимания местоимений (ОН, ЕГО, ЭТО, ТАМ) в ТЕКУЩЕМ вопросе"
            " 8. ⚠️ ОТВЕЧАЙ ТОЛЬКО НА ПОСЛЕДНИЙ ВОПРОС! Контекст НЕ надо пересказывать, НЕ надо отвечать на старые вопросы!"
            " 9. ФОРМАТ ОТВЕТА: тезис1 ||| тезис2 ||| тезис3"
            " 10. ⚠️⚠️⚠️ СТРОГО ЗАПРЕЩЕНО:"
            "     - 'я не знаю', 'не понял', 'не похоже на вопрос', 'бессмыслица'"
            "     - Любые комментарии, объяснения, извинения, мета-информация"
            "     - Если не понял - ПУСТАЯ СТРОКА (NO TEXT), НЕ объяснение!"
            " 11. ТОЛЬКО ФАКТЫ В ТЕЗИСАХ - ничего больше!"
            " 12. ТОЛЬКО если СОВСЕМ ничего не понял (шум, набор букв) → вернуть ПУСТУЮ СТРОКУ"
            "     - ❌ НЕ ПИШИ: 'Я не понял вопрос, так как...'"
            "     - ✅ ПРОСТО ВЕРНИ: '' (пустая строка, никакого текста)"
            " 13. ВСЕ ЦИФРЫ ПИШИ СЛОВАМИ (не '1961', а 'тысяча девятьсот шестьдесят первый')"
            " 14. ДАТЫ ПИШИ СЛОВАМИ: 'двенадцатого апреля', 'первого января'"
            " 15. ЧИСЛА ПИШИ СЛОВАМИ: 'две тысячи четыреста сорок восемь'"
            " 16. АДАПТИРУЙ количество тезисов:"
            "    - Простой вопрос (год, дата, кто, что, где, математика) -> 1 тезис"
            "    - Развернутый вопрос (история, расскажи, опиши, ключевые события) -> СТРОГО 5 тезисов"
        )
        
        # Формируем промпт с явным разделением контекста и текущего вопроса
        context_section = ""
        if context and context.strip():
            context_section = (
                f"📚 КОНТЕКСТ (предыдущие вопросы - НЕ отвечай на них!):\n"
                f"{context.strip()}\n\n"
                f"⚠️ КОНТЕКСТ НУЖЕН ТОЛЬКО для понимания местоимений в текущем вопросе!\n\n"
            )
        
        user_prompt = (
            f"{context_section}"
            f"❓ ТЕКУЩИЙ ВОПРОС (отвечай ТОЛЬКО на него!):\n{question_text.strip()}\n\n"
            f"ЗАДАЧА: АГРЕССИВНО пытайся понять вопрос и ВСЕГДА отвечай!\n\n"
            f"⚠️⚠️⚠️ КРИТИЧНО: ОТВЕТ ВСЕГДА ПЕРВЫМ СЛОВОМ!\n\n"
            f"ПРИМЕРЫ СТРУКТУРЫ (ОТВЕТ ПЕРВЫМ):\n\n"
            f"Вопрос: 'Из чего состоят волосы?'\n"
            f"❌ НЕПРАВИЛЬНО: 'Основной компонент человеческого волоса - это белок кератин'\n"
            f"✅ ПРАВИЛЬНО: 'Кератин - основной компонент волос'\n\n"
            f"Вопрос: 'Где находится Эйфелева башня?'\n"
            f"❌ НЕПРАВИЛЬНО: 'Эйфелева башня находится в Париже'\n"
            f"✅ ПРАВИЛЬНО: 'Париж - место расположения Эйфелевой башни'\n\n"
            f"Вопрос: 'Кто изобрел радио?'\n"
            f"❌ НЕПРАВИЛЬНО: 'Радио изобрел Александр Попов'\n"
            f"✅ ПРАВИЛЬНО: 'Александр Попов - изобретатель радио'\n\n"
            f"Вопрос: 'Когда основали Москву?'\n"
            f"❌ НЕПРАВИЛЬНО: 'Москву основали в тысяча сто сорок седьмом году'\n"
            f"✅ ПРАВИЛЬНО: 'Тысяча сто сорок седьмой год - основание Москвы'\n\n"
            f"ПРИМЕРЫ С ОШИБКАМИ РАСПОЗНАВАНИЯ:\n\n"
            f"Вопрос: '102 убложить на 24 сколько будет' (ошибка: убложить→умножить)\n"
            f"✅ 'Две тысячи четыреста сорок восемь - результат умножения'\n\n"
            f"Вопрос: 'Кто осовал Украину' (ошибка: осовал→основал)\n"
            f"✅ Понимай как 'основал' и отвечай\n\n"
            f"ПРИМЕРЫ ДЛЯ НЕ ПОНЯЛ:\n\n"
            f"Вопрос: 'кстати лаппать бымышь' (полная бессмыслица)\n"
            f"❌ НЕПРАВИЛЬНО: 'Я не понял вопрос, так как...'\n"
            f"❌ НЕПРАВИЛЬНО: 'Не похоже на осмысленный запрос'\n"
            f"✅ ПРАВИЛЬНО: '' (пустая строка, никакого текста!)\n\n"
            f"ПРИМЕРЫ С КОНТЕКСТОМ (местоимения):\n\n"
            f"КОНТЕКСТ: 'Кто первым полетел в космос?'\n"
            f"ТЕКУЩИЙ ВОПРОС: 'Когда он полетел?'\n"
            f"✅ 'Двенадцатого апреля тысяча девятьсот шестьдесят первого года - полёт Гагарина'\n\n"
            f"ФОРМАТ ОТВЕТА: тезис1 ||| тезис2 ||| тезис3\n"
            f"ВСЕ ЦИФРЫ ПРОПИСЬЮ! ОТВЕТ ВСЕГДА ПЕРВЫМ! БЕЗ КОММЕНТАРИЕВ!\n"
            f"НЕ ПОНЯЛ → ПУСТАЯ СТРОКА (не объяснение!)"
        )
        # ✅ ОПТИМИЗАЦИЯ 3D: Отключаем thinking для снижения латентности
        # thinking_budget=0 экономит 100-200мс на генерации тезисов.
        # См. OPTIMIZATION_TABLE.md - код 3D
        cfg = types.GenerateContentConfig(
            system_instruction=sys_instr,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=0.9,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # НЕ МЕНЯТЬ!
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
                config=cfg,
            )
            raw = (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001
            error_str = str(e)
            is_server_overload = (
                "503" in error_str or 
                "504" in error_str or
                "overloaded" in error_str.lower() or
                "unavailable" in error_str.lower()
            )
            logger.error(f"ThesisGenerator (Gemini) ошибка: {e}")
            
            # Fallback на OpenAI при перегрузке Gemini
            if is_server_overload and self.openai_client:
                logger.warning("Gemini перегружен (503), переключаемся на OpenAI fallback")
                raw = self._generate_openai(sys_instr, user_prompt, n)
                if not raw:
                    return []
            else:
                return []

        # Парсинг ответа: ожидаем формат "тезис1 ||| тезис2 ||| тезис3"
        import json
        
        # Попытка 1: парсим по разделителю |||
        if "|||" in raw:
            theses = [t.strip() for t in raw.split("|||") if t.strip()]
            return theses[:n]
        
        # Попытка 2: парсим как JSON (fallback для старого формата)
        try:
            data = json.loads(raw)
            items = data.get("theses", [])
            out: List[str] = []
            for it in items:
                if not isinstance(it, str):
                    continue
                t = it.strip()
                if not t:
                    continue
                out.append(t)
            return out[:n]
        except Exception:
            pass
        
        # Попытка 3: построчный парсинг (fallback)
        lines = [ln.strip("-•* \t") for ln in raw.splitlines()]
        out: List[str] = [ln for ln in lines if ln]
        return out[:n]
    
    def _generate_openai(self, system_instruction: str, user_prompt: str, n: int) -> str:
        """Генерация через OpenAI (fallback при перегрузке Gemini)"""
        if not self.openai_client:
            return ""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=0.9,
            )
            
            raw = (response.choices[0].message.content or "").strip()
            logger.info(f"✅ OpenAI fallback вернул {len(raw)} символов")
            return raw
            
        except Exception as e:
            logger.error(f"OpenAI fallback ошибка: {e}")
            return ""
    
    def generate_deeper(
        self, 
        previous_theses: List[str], 
        question: str, 
        context: Optional[str] = None,
        n: int = 5,
        language: str = "ru"
    ) -> List[str]:
        """
        Генерирует дополнительные тезисы (углубление) на основе предыдущих.
        
        Args:
            previous_theses: Список предыдущих тезисов
            question: Исходный вопрос
            context: Контекст диалога
            n: Количество дополнительных тезисов (по умолчанию 5)
            language: Язык (по умолчанию ru)
        
        Returns:
            Список дополнительных тезисов
        """
        if not previous_theses or not question:
            return []
        
        n = max(1, int(n))
        
        # Системная инструкция для углубления
        sys_instr = (
            "Ты генератор дополнительных тезисов для устных экзаменов."
            " ЗАДАЧА: дать углубленные факты, детали, контекст по теме."
            " ПРАВИЛА:"
            " 1. НЕ ПОВТОРЯЙ предыдущие тезисы - только НОВАЯ информация"
            " 2. ⚠️ СТРУКТУРА: КЛЮЧЕВОЕ СЛОВО/ОТВЕТ ВСЕГДА ПЕРВЫМ!"
            "    - 'Кератин состоит из...' → 'Аминокислоты - строительные блоки кератина'"
            "    - 'Волосы содержат...' → 'Меланин - пигмент определяющий цвет волос'"
            " 3. Тезис: короткое предложение (до 15 слов), только факты"
            " 4. ВСЕ ЦИФРЫ ПРОПИСЬЮ: '1961'→'тысяча девятьсот шестьдесят первый'"
            " 5. Формат: тезис1 ||| тезис2 ||| тезис3"
            " 6. ⚠️ СТРОГО ЗАПРЕЩЕНО:"
            "    - Повторы предыдущих тезисов"
            "    - Комментарии, объяснения, 'я не знаю', 'не могу добавить'"
            "    - Если нечего добавить → пустая строка (NO TEXT)"
            " 7. Углубляйся: детали, связанные события, интересные факты"
        )
        
        # Формируем промпт с предыдущими тезисами
        theses_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(previous_theses)])
        
        context_section = ""
        if context and context.strip():
            context_section = f"📚 КОНТЕКСТ:\n{context.strip()}\n\n"
        
        user_prompt = (
            f"{context_section}"
            f"❓ ИСХОДНЫЙ ВОПРОС: {question.strip()}\n\n"
            f"📝 ПРЕДЫДУЩИЕ ТЕЗИСЫ:\n{theses_list}\n\n"
            f"ЗАДАЧА: Дай {n} ДОПОЛНИТЕЛЬНЫХ тезисов (углубление, детали, связанные факты).\n\n"
            f"⚠️ КРИТИЧНО: ОТВЕТ/КЛЮЧЕВОЕ СЛОВО ПЕРВЫМ!\n"
            f"НЕ ПОВТОРЯЙ предыдущие тезисы!\n"
            f"Нечего добавить → пустая строка (не объяснение!)\n\n"
            f"Формат: тезис1 ||| тезис2 ||| тезис3\n"
            f"ВСЕ ЦИФРЫ ПРОПИСЬЮ! БЕЗ КОММЕНТАРИЕВ!"
        )
        
        # ✅ ОПТИМИЗАЦИЯ 3D: Отключаем thinking (см. комментарий выше в generate())
        cfg = types.GenerateContentConfig(
            system_instruction=sys_instr,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=0.9,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # НЕ МЕНЯТЬ!
        )
        
        try:
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
                config=cfg,
            )
            raw = (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001
            error_str = str(e)
            is_server_overload = (
                "503" in error_str or 
                "504" in error_str or
                "overloaded" in error_str.lower() or
                "unavailable" in error_str.lower()
            )
            logger.error(f"ThesisGenerator.generate_deeper (Gemini) ошибка: {e}")
            
            # Fallback на OpenAI при перегрузке Gemini
            if is_server_overload and self.openai_client:
                logger.warning("Gemini перегружен (503) при углублении, переключаемся на OpenAI fallback")
                raw = self._generate_openai(sys_instr, user_prompt, n)
                if not raw:
                    return []
            else:
                return []
        
        # Парсинг ответа (аналогично generate())
        import json
        
        # Попытка 1: парсим по разделителю |||
        if "|||" in raw:
            theses = [t.strip() for t in raw.split("|||") if t.strip()]
            return theses[:n]
        
        # Попытка 2: парсим как JSON
        try:
            data = json.loads(raw)
            items = data.get("theses", [])
            out: List[str] = []
            for it in items:
                if not isinstance(it, str):
                    continue
                t = it.strip()
                if not t:
                    continue
                out.append(t)
            return out[:n]
        except Exception:
            pass
        
        # Попытка 3: построчный парсинг
        lines = [ln.strip("-•* \t") for ln in raw.splitlines()]
        out: List[str] = [ln for ln in lines if ln]
        return out[:n]


__all__ = ["GeminiThesisGenerator", "ThesisGenConfig"]
