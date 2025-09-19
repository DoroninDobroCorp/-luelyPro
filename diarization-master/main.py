from __future__ import annotations

import argparse
from pathlib import Path
from loguru import logger

from live_recognizer import enroll_cli, live_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diarization",
        description="Минимальный лайв-модуль: узнавание вашего голоса (pyannote + ECAPA)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
        default=5.0,
        help="Длительность записи для профиля (сек)",
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
    # ASR (для единообразия флаги есть, но при enroll ASR не используется)
    p_enroll.add_argument(
        "--asr",
        action="store_true",
        help="Включить ASR (используется только в live)",
    )
    p_enroll.add_argument(
        "--asr-model",
        type=str,
        default="small",
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
        default=None,
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
    # ASR
    p_live.add_argument(
        "--asr",
        action="store_true",
        help="Включить транскрибацию незнакомых голосов (faster-whisper)",
    )
    p_live.add_argument(
        "--asr-model",
        type=str,
        default="small",
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
        default=None,
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "enroll":
        enroll_cli(
            profile_path=args.profile,
            seconds=args.seconds,
            vad_aggr=args.vad_aggr,
            min_consec=args.min_consec,
            flatness_th=args.flatness_th,
            asr=args.asr,
            asr_model=args.asr_model,
            asr_lang=args.asr_lang,
            asr_device=args.asr_device,
            asr_compute=args.asr_compute,
        )
    elif args.command == "live":
        live_cli(
            profile_path=args.profile,
            threshold=args.threshold,
            vad_aggr=args.vad_aggr,
            min_consec=args.min_consec,
            flatness_th=args.flatness_th,
            min_segment_ms=args.min_segment_ms,
            max_silence_ms=args.max_silence_ms,
            asr=args.asr,
            asr_model=args.asr_model,
            asr_lang=args.asr_lang,
            asr_device=args.asr_device,
            asr_compute=args.asr_compute,
            llm=args.llm,
        )
    else:
        logger.error("Неизвестная команда")


if __name__ == "__main__":
    main()
