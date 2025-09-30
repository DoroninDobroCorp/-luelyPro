"""Модели данных CluelyPro."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from exceptions import ProfileError


@dataclass
class VoiceProfile:
    """Голосовой профиль пользователя (эмбеддинг)."""
    
    embedding: np.ndarray  # shape (d,) - вектор эмбеддинга
    
    @staticmethod
    def load(path: Path) -> Optional["VoiceProfile"]:
        """Загрузить профиль из файла.
        
        Args:
            path: Путь к .npz файлу профиля
            
        Returns:
            VoiceProfile или None если файл не существует
            
        Raises:
            ProfileError: Если файл повреждён
        """
        if not path.exists():
            return None
        
        try:
            data = np.load(path)
            if "embedding" not in data:
                raise ProfileError(f"Профиль {path} не содержит эмбеддинг")
            return VoiceProfile(embedding=data["embedding"])  # type: ignore[index]
        except Exception as e:
            raise ProfileError(f"Ошибка загрузки профиля {path}: {e}") from e
    
    def save(self, path: Path) -> None:
        """Сохранить профиль в файл.
        
        Args:
            path: Путь для сохранения .npz файла
            
        Raises:
            ProfileError: Если не удалось сохранить
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, embedding=self.embedding)
        except Exception as e:
            raise ProfileError(f"Ошибка сохранения профиля {path}: {e}") from e


@dataclass
class QueuedSegment:
    """Сегмент аудио в очереди на обработку."""
    
    kind: str  # "self" или "foreign"
    audio: np.ndarray  # аудио данные
    timestamp: float  # время создания
    distance: float = 0.0  # косинусная дистанция (для foreign)


__all__ = ["VoiceProfile", "QueuedSegment"]
