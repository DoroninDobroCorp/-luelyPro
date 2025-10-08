#!/usr/bin/env python3
"""
Утилита для проверки аудио устройств
Показывает все доступные устройства ввода/вывода
"""
import sounddevice as sd

def main():
    print("\n" + "="*70)
    print("🔊 ПРОВЕРКА АУДИО УСТРОЙСТВ")
    print("="*70 + "\n")
    
    # Получаем список всех устройств
    devices = sd.query_devices()
    
    # Текущее устройство по умолчанию
    default_input = sd.default.device[0]
    default_output = sd.default.device[1]
    
    print(f"📥 УСТРОЙСТВО ВВОДА ПО УМОЛЧАНИЮ: {default_input}")
    print(f"📤 УСТРОЙСТВО ВЫВОДА ПО УМОЛЧАНИЮ: {default_output}\n")
    
    print("="*70)
    print("СПИСОК ВСЕХ УСТРОЙСТВ:")
    print("="*70 + "\n")
    
    for idx, device in enumerate(devices):
        marker = ""
        if idx == default_input:
            marker = " ← ВХОД ПО УМОЛЧАНИЮ"
        elif idx == default_output:
            marker = " ← ВЫХОД ПО УМОЛЧАНИЮ"
        
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append(f"ВХОД ({device['max_input_channels']} каналов)")
        if device['max_output_channels'] > 0:
            device_type.append(f"ВЫХОД ({device['max_output_channels']} каналов)")
        
        type_str = " | ".join(device_type)
        
        print(f"[{idx}] {device['name']}")
        print(f"    Тип: {type_str}{marker}")
        print(f"    Sample Rate: {device['default_samplerate']} Hz")
        print()
    
    print("="*70)
    print("\n💡 СОВЕТ:")
    print("   Если наушники не работают, проверьте:")
    print("   1. Наушники подключены и выбраны в системных настройках")
    print("   2. macOS: Системные настройки → Звук → Выход → Выберите наушники")
    print("   3. Windows: Звук → Устройства воспроизведения → Установить по умолчанию")
    print()
    print("   После этого система будет использовать правильное устройство автоматически!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
