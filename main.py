from __future__ import annotations

import argparse
import os
from pathlib import Path
try:
    from dotenv import load_dotenv, find_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None  # type: ignore
    find_dotenv = None  # type: ignore
try:
    from loguru import logger
except Exception:  # noqa: BLE001
    class _DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            print(*args)

    logger = _DummyLogger()  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diarization",
        description="Минимальный лайв-модуль: узнавание вашего голоса (pyannote + ECAPA)",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    p_enroll = sub.add_parser("enroll", help="Записать голосовой профиль")
    p_enroll.add_argument(
        "--profile",
        type=Path,
        default=Path("voice_profile.npz"),
        help="Путь к файлу профиля (npz)",
    )
    p_enroll.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Максимальная длительность записи для профиля (сек)",
    )
    p_enroll.add_argument(
        "--min-voiced-seconds",
        type=float,
        default=4.0,
        help="Минимальная длительность озвученной речи при записи профиля (сек)",
    )
    p_enroll.add_argument(
        "--vad-aggr",
        type=int,
        default=2,
        help="Агрессивность WebRTC VAD [0..3] (выше=строже к шумам)",
    )
    p_enroll.add_argument(
        "--min-consec",
        type=int,
        default=5,
        help="Мин. подряд речевых кадров (20мс каждый) для старта сегмента",
    )
    p_enroll.add_argument(
        "--flatness-th",
        type=float,
        default=0.60,
        help="Порог спектральной плоскостности для отбраковки шумов (меньше=строже)",
    )
    # VAD backend selection
    p_enroll.add_argument(
        "--vad-backend",
        type=str,
        choices=["webrtc", "silero"],
        default="webrtc",
        help="Выбор бэкенда VAD: webrtc (по умолчанию) или silero",
    )
    p_enroll.add_argument(
        "--silero-vad-th",
        type=float,
        default=0.5,
        help="Порог вероятности для Silero VAD (0..1)",
    )
    p_enroll.add_argument(
        "--silero-vad-window",
        type=int,
        default=100,
        help="Длина окна Silero VAD (мс)",
    )
    # Скрипт для чтения при записи профиля
    p_enroll.add_argument(
        "--read-script",
        type=str,
        default=None,
        help="Текст, который будет предложено прочитать при записи профиля",
    )
    p_enroll.add_argument(
        "--read-script-file",
        type=Path,
        default=None,
        help="Путь к файлу с текстом для чтения при записи профиля",
    )
    # ASR (для единообразия флаги есть, но при enroll ASR не используется)
    p_enroll.add_argument(
        "--asr",
        action="store_true",
        help="Включить ASR (используется только в live)",
    )
    p_enroll.add_argument(
        "--asr-model",
        type=str,
        default=os.getenv("ASR_MODEL", "large-v3-turbo"),
        help="Размер модели faster-whisper: tiny|base|small|medium|large-v3",
    )
    p_enroll.add_argument(
        "--asr-lang",
        type=str,
        default=None,
        help="Язык ASR, напр. 'ru' (по умолчанию авто)",
    )
    p_enroll.add_argument(
        "--asr-device",
        type=str,
        default="cuda",
        help="Устройство для ASR: cuda|cpu (по умолчанию авто)",
    )
    p_enroll.add_argument(
        "--asr-compute",
        type=str,
        default=None,
        help="Тип вычислений ASR: float16|int8 и т.п. (по умолчанию авто)",
    )

    p_live = sub.add_parser("live", help="Лайв-распознавание голоса")
    p_live.add_argument(
        "--profile",
        type=Path,
        default=Path("voice_profile.npz"),
        help="Путь к файлу профиля (npz)",
    )
    p_live.add_argument(
        "--select-profile",
        action="store_true",
        help="Интерактивно выбрать существующий профиль или создать новый перед запуском",
    )
    p_live.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("profiles"),
        help="Каталог с профилями (npz) для интерактивного выбора/создания",
    )
    p_live.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Имя профиля (без расширения) при создании нового",
    )
    p_live.add_argument(
        "--auto-enroll-if-missing",
        action="store_true",
        help="Если указанный профиль отсутствует, автоматически создать новый (запросит запись)",
    )
    p_live.add_argument(
        "--enroll-min-voiced-seconds",
        type=float,
        default=4.0,
        help="Минимальная озвученная речь при авто-энролле (сек)",
    )
    p_live.add_argument(
        "--enroll-seconds",
        type=float,
        default=20.0,
        help="Максимальная длительность записи при авто-энролле (сек)",
    )
    p_live.add_argument(
        "--enroll-read-script-file",
        type=Path,
        default=None,
        help="Файл с текстом для чтения при авто-энролле",
    )
    p_live.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Порог косинусной дистанции (<= мой голос)",
    )
    p_live.add_argument(
        "--vad-aggr",
        type=int,
        default=2,
        help="Агрессивность WebRTC VAD [0..3] (выше=строже к шумам)",
    )
    p_live.add_argument(
        "--min-consec",
        type=int,
        default=5,
        help="Мин. подряд речевых кадров (20мс каждый) для старта сегмента",
    )
    p_live.add_argument(
        "--flatness-th",
        type=float,
        default=0.60,
        help="Порог спектральной плоскостности для отбраковки шумов (меньше=строже)",
    )
    # VAD backend selection
    p_live.add_argument(
        "--vad-backend",
        type=str,
        choices=["webrtc", "silero"],
        default="webrtc",
        help="Выбор бэкенда VAD: webrtc (по умолчанию) или silero",
    )
    p_live.add_argument(
        "--silero-vad-th",
        type=float,
        default=0.5,
        help="Порог вероятности для Silero VAD (0..1)",
    )
    p_live.add_argument(
        "--silero-vad-window",
        type=int,
        default=100,
        help="Длина окна Silero VAD (мс)",
    )
    p_live.add_argument(
        "--min-segment-ms",
        type=int,
        default=500,
        help="Мин. длительность речевого сегмента (мс)",
    )
    p_live.add_argument(
        "--max-silence-ms",
        type=int,
        default=400,
        help="Пауза (мс) для завершения сегмента",
    )
    p_live.add_argument(
        "--pre-roll-ms",
        type=int,
        default=160,
        help="Предзахват (мс) до старта сегмента, чтобы не обрезать первые звуки",
    )
    p_live.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Авто-остановка лайв-режима через N секунд (0 — бесконечно). Полезно для смоук-тестов",
    )
    # ASR
    p_live.add_argument(
        "--asr",
        action="store_true",
        help="Включить транскрибацию незнакомых голосов (faster-whisper)",
    )
    p_live.add_argument(
        "--asr-model",
        type=str,
        default=os.getenv("ASR_MODEL", "small"),
        help="Размер модели faster-whisper: tiny|base|small|medium|large-v3",
    )
    p_live.add_argument(
        "--asr-lang",
        type=str,
        default=None,
        help="Язык ASR, напр. 'ru' (по умолчанию авто)",
    )
    p_live.add_argument(
        "--asr-device",
        type=str,
        default="cuda",
        help="Устройство для ASR: cuda|cpu (по умолчанию авто)",
    )
    p_live.add_argument(
        "--asr-compute",
        type=str,
        default=None,
        help="Тип вычислений ASR: float16|int8 и т.п. (по умолчанию авто)",
    )
    p_live.add_argument(
        "--llm",
        action="store_true",
        help="Включить LLM-ответ (Gemma)",
    )
    p_live.add_argument(
        "--theses",
        type=Path,
        default=Path("theses.txt"),
        help="Путь к файлу с тезисами (по одному в строке)",
    )
    p_live.add_argument(
        "--no-theses",
        action="store_true",
        help="Отключить тезисный помощник",
    )
    p_live.add_argument(
        "--thesis-match",
        type=float,
        default=0.6,
        help="Порог совпадения тезиса (0..1) при распознавании вашей речи",
    )
    p_live.add_argument(
        "--thesis-semantic",
        type=float,
        default=0.55,
        help="Порог семантического совпадения тезиса (0..1)",
    )
    p_live.add_argument(
        "--thesis-semantic-model",
        type=str,
        default=None,
        help="ИД модели эмбеддингов для семантического сравнения (по умолчанию multilingual-e5-small)",
    )
    p_live.add_argument(
        "--thesis-semantic-disable",
        action="store_true",
        help="Отключить семантический анализ (только токены)",
    )
    # Gemini judge
    p_live.add_argument(
        "--thesis-gemini-conf",
        type=float,
        default=0.60,
        help="Мин. уверенность Gemini для закрытия тезиса (0..1)",
    )
    p_live.add_argument(
        "--thesis-gemini-disable",
        action="store_true",
        help="Отключить судью Gemini (останутся токены/семантика)",
    )
    # Автогенерация тезисов
    p_live.add_argument(
        "--thesis-autogen-disable",
        action="store_true",
        help="Отключить автогенерацию тезисов из чужой речи",
    )
    p_live.add_argument(
        "--thesis-autogen-batch",
        type=int,
        default=4,
        help="Размер пакета автоматически генерируемых тезисов (обычно 3–4)",
    )

    # Тестовый режим: прогон текстов без аудио
    p_test = sub.add_parser("test", help="Тест: извлечь тезисы из текста без аудио")
    p_test.add_argument(
        "--text",
        type=str,
        default=None,
        help="Текст чужой реплики/диалога для извлечения тезисов",
    )
    p_test.add_argument(
        "--file",
        type=Path,
        default=Path("tests/examples.txt"),
        help="Файл с примерами, по одному кейсу в строке (пустые/начинающиеся с # игнорируются)",
    )
    p_test.add_argument(
        "--json",
        action="store_true",
        help="Печатать результаты в JSON-формате",
    )

    # Управление профилями: список и удаление
    p_profiles = sub.add_parser("profiles", help="Список и удаление голосовых профилей")
    prof_sub = p_profiles.add_subparsers(dest="profiles_cmd", required=True)
    p_list = prof_sub.add_parser("list", help="Показать доступные профили")
    p_list.add_argument("--dir", type=Path, default=Path("profiles"), help="Каталог профилей")
    p_del = prof_sub.add_parser("delete", help="Удалить профиль(и)")
    p_del.add_argument("--dir", type=Path, default=Path("profiles"), help="Каталог профилей")
    p_del.add_argument("--name", type=str, default=None, help="Имя профиля без .npz для удаления")
    p_del.add_argument("--all", action="store_true", help="Удалить все профили")

    return parser


def main() -> None:
    # Загрузим .env (ищем в текущей папке и выше)
    try:
        if load_dotenv and find_dotenv:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                load_dotenv(env_path, override=False)
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args()

    # Режим по умолчанию: без подкоманды запускаем интерактивный выбор/создание профиля
    # и стартуем ассистента (ASR + LLM + тезисы). Время авто-остановки читаем из env.
    if getattr(args, "command", None) is None:
        from live_recognizer import live_cli
        from live_recognizer import enroll_cli

        # Интерактивный выбор/создание профиля в каталоге profiles/
        prof_dir: Path = Path("profiles")
        prof_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted([p for p in prof_dir.glob("*.npz") if p.is_file()])
        if existing:
            print("Найдены профили:")
            for i, p in enumerate(existing, 1):
                print(f"  {i}) {p.name}")
            print("  n) создать новый профиль")
            choice = input("Выберите номер профиля или 'n' для нового: ").strip()
            if choice.lower() == "n":
                name = input("Имя нового профиля (латиницей): ").strip() or "my_profile"
                selected_profile = prof_dir / f"{name}.npz"
            else:
                try:
                    idx = int(choice)
                    if 1 <= idx <= len(existing):
                        selected_profile = existing[idx - 1]
                    else:
                        print("Некорректный номер, используем профиль по умолчанию")
                        selected_profile = prof_dir / "my_profile.npz"
                except Exception:
                    print("Некорректный ввод, используем профиль по умолчанию")
                    selected_profile = prof_dir / "my_profile.npz"
        else:
            print("Профили не найдены. Будет создан новый.")
            name = input("Имя нового профиля (латиницей): ").strip() or "my_profile"
            selected_profile = prof_dir / f"{name}.npz"

        # Если файла нет — проводим запись профиля (8 сек, строгий VAD к шумам)
        if not selected_profile.exists():
            print(f"Профиль {selected_profile} не найден. Запишем новый.")
            enroll_cli(
                profile_path=selected_profile,
                seconds=20.0,
                min_voiced_seconds=4.0,
                vad_aggr=2,
                min_consec=5,
                flatness_th=0.60,
                vad_backend="webrtc",
                silero_vad_threshold=0.5,
                silero_vad_window_ms=100,
                asr=False,
                asr_model="large-v3-turbo",
                asr_lang=None,
                asr_device=None,
                asr_compute=None,
                read_script=None,
                read_script_file=None,
            )

        # Авто-стоп для смоук-теста через переменные окружения
        run_seconds_env = os.getenv("ASSISTANT_RUN_SECONDS") or os.getenv("RUN_SECONDS")
        try:
            run_seconds = float(run_seconds_env) if run_seconds_env else 0.0
        except Exception:
            run_seconds = 0.0

        # Включаем LLM, только если есть ключ
        llm_enable = bool(os.getenv("GEMINI_API_KEY"))

        # Запускаем live-режим с разумными дефолтами
        live_cli(
            profile_path=selected_profile,
            threshold=0.75,
            vad_aggr=2,
            min_consec=5,
            flatness_th=0.60,
            vad_backend="webrtc",
            silero_vad_threshold=0.5,
            silero_vad_window_ms=100,
            min_segment_ms=500,
            max_silence_ms=400,
            pre_roll_ms=160,
            asr=True,
            asr_model=os.getenv("ASR_MODEL", "small"),
            asr_lang="ru",
            asr_device=None,
            asr_compute=None,
            llm=llm_enable,
            theses_path=Path("theses.txt"),
            thesis_match=0.6,
            thesis_semantic=0.55,
            thesis_semantic_model=None,
            thesis_semantic_disable=False,
            thesis_gemini_conf=0.60,
            thesis_gemini_disable=not llm_enable,
            thesis_autogen_disable=False,
            thesis_autogen_batch=4,
            run_seconds=run_seconds,
        )
        return

    if args.command == "enroll":
        # Ленивая загрузка тяжёлых зависимостей только при реальном запуске
        from live_recognizer import enroll_cli
        enroll_cli(
            profile_path=args.profile,
            seconds=args.seconds,
            min_voiced_seconds=args.min_voiced_seconds,
            vad_aggr=args.vad_aggr,
            min_consec=args.min_consec,
            flatness_th=args.flatness_th,
            vad_backend=args.vad_backend,
            silero_vad_threshold=args.silero_vad_th,
            silero_vad_window_ms=args.silero_vad_window,
            asr=args.asr,
            asr_model=args.asr_model,
            asr_lang=args.asr_lang,
            asr_device=args.asr_device,
            asr_compute=args.asr_compute,
            read_script=args.read_script,
            read_script_file=args.read_script_file,
        )
    elif args.command == "live":
        from live_recognizer import live_cli, VoiceProfile

        # Поддержка таймера авто-остановки через ENV, если флаг не задан
        if not args.run_seconds or float(args.run_seconds) <= 0.0:
            run_seconds_env = os.getenv("ASSISTANT_RUN_SECONDS") or os.getenv("RUN_SECONDS")
            if run_seconds_env:
                try:
                    args.run_seconds = float(run_seconds_env)
                except Exception:
                    pass

        # Интерактивный выбор/создание профиля
        selected_profile: Path = args.profile
        if args.select_profile:
            prof_dir: Path = args.profiles_dir
            prof_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted([p for p in prof_dir.glob("*.npz") if p.is_file()])
            if existing:
                print("Найдены профили:")
                for i, p in enumerate(existing, 1):
                    print(f"  {i}) {p.name}")
                print("  n) создать новый профиль")
                choice = input("Выберите номер профиля или 'n' для нового: ").strip()
                if choice.lower() == "n":
                    name = args.profile_name or input("Имя нового профиля (латиницей): ").strip() or "my_profile"
                    selected_profile = prof_dir / f"{name}.npz"
                else:
                    try:
                        idx = int(choice)
                        if 1 <= idx <= len(existing):
                            selected_profile = existing[idx - 1]
                        else:
                            print("Некорректный номер, используем по умолчанию")
                    except Exception:
                        print("Некорректный ввод, используем путь по умолчанию")
            else:
                print("Профили не найдены. Будет создан новый.")
                name = args.profile_name or input("Имя нового профиля (латиницей): ").strip() or "my_profile"
                selected_profile = prof_dir / f"{name}.npz"

        # Авто-энролл, если нет файла
        if (args.auto_enroll_if_missing or args.select_profile) and not selected_profile.exists():
            print(f"Профиль {selected_profile} не найден. Запишем новый.")
            from live_recognizer import enroll_cli
            # Параметры записи можно спросить у пользователя, но оставим дефолты
            enroll_cli(
                profile_path=selected_profile,
                seconds=args.enroll_seconds,
                min_voiced_seconds=args.enroll_min_voiced_seconds,
                vad_aggr=2,
                min_consec=5,
                flatness_th=0.60,
                vad_backend="webrtc",
                silero_vad_threshold=0.5,
                silero_vad_window_ms=100,
                asr=False,
                asr_model="large-v3-turbo",
                asr_lang=None,
                asr_device=None,
                asr_compute=None,
                read_script_file=args.enroll_read_script_file,
            )

        live_cli(
            profile_path=selected_profile,
            threshold=args.threshold,
            vad_aggr=args.vad_aggr,
            min_consec=args.min_consec,
            flatness_th=args.flatness_th,
            vad_backend=args.vad_backend,
            silero_vad_threshold=args.silero_vad_th,
            silero_vad_window_ms=args.silero_vad_window,
            min_segment_ms=args.min_segment_ms,
            max_silence_ms=args.max_silence_ms,
            pre_roll_ms=args.pre_roll_ms,
            asr=args.asr,
            asr_model=args.asr_model,
            asr_lang=args.asr_lang,
            asr_device=args.asr_device,
            asr_compute=args.asr_compute,
            llm=args.llm,
            theses_path=None if args.no_theses else args.theses,
            thesis_match=args.thesis_match,
            thesis_semantic=args.thesis_semantic,
            thesis_semantic_model=args.thesis_semantic_model,
            thesis_semantic_disable=args.thesis_semantic_disable,
            thesis_gemini_conf=args.thesis_gemini_conf,
            thesis_gemini_disable=args.thesis_gemini_disable,
            thesis_autogen_disable=args.thesis_autogen_disable,
            thesis_autogen_batch=args.thesis_autogen_batch,
            run_seconds=args.run_seconds,
        )
    elif args.command == "profiles":
        # Список/удаление профилей
        if args.profiles_cmd == "list":
            d: Path = args.dir
            d.mkdir(parents=True, exist_ok=True)
            items = sorted([p for p in d.glob("*.npz") if p.is_file()])
            if not items:
                print("Профили не найдены")
            else:
                print("Доступные профили:")
                for p in items:
                    print(" -", p.name)
        elif args.profiles_cmd == "delete":
            d: Path = args.dir
            d.mkdir(parents=True, exist_ok=True)
            if args.all:
                for p in d.glob("*.npz"):
                    try:
                        p.unlink()
                        print("Удалён:", p.name)
                    except Exception as e:  # noqa: BLE001
                        print("Не удалось удалить", p.name, e)
            elif args.name:
                target = d / f"{args.name}.npz"
                if target.exists():
                    try:
                        target.unlink()
                        print("Удалён:", target.name)
                    except Exception as e:  # noqa: BLE001
                        print("Не удалось удалить", target.name, e)
                else:
                    print("Не найден профиль:", target.name)
            else:
                print("Укажите --name <имя> или --all для удаления")
        else:
            logger.error("Неизвестная команда управления профилями")
    elif args.command == "test":
        from live_recognizer import extract_theses_from_text
        import json as _json

        cases: list[str] = []
        if args.text:
            cases = [args.text]
        else:
            f: Path = args.file
            if f.exists():
                raw = f.read_text(encoding="utf-8").splitlines()
                for line in raw:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    cases.append(s)
            else:
                print(f"Файл с примерами не найден: {f}")
                sys.exit(1)

        results: list[dict] = []
        for idx, sample in enumerate(cases, 1):
            theses = extract_theses_from_text(sample)
            item = {"id": idx, "text": sample, "theses": theses}
            results.append(item)
            if not args.json:
                print("== Кейс", idx)
                print(sample)
                print("Тезисы:")
                for i, t in enumerate(theses, 1):
                    print(f"  {i}) {t}")
                print()

        if args.json:
            print(_json.dumps({"results": results}, ensure_ascii=False, indent=2))
    else:
        logger.error("Неизвестная команда")


if __name__ == "__main__":
    main()
