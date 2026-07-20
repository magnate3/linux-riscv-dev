#!/usr/bin/env python3
"""
YOLOv8 Channel Pruning Script

Применяет structured pruning к YOLOv8 модели с минимальным падением качества.

Требования:
    pip install ultralytics torch-pruning

Использование:
    python prune_yolo_model.py --model yolov8n.pt --ratio 0.2 --epochs 10
    
    Это создаст:
    - yolov8n_pruned.pt (PyTorch модель)
    - yolov8n_pruned.onnx (ONNX для конвертации)
"""

import argparse
from pathlib import Path


def check_dependencies():
    """Проверяет наличие необходимых библиотек"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append('torch')
    
    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append('ultralytics')
    
    try:
        import torch_pruning
    except ImportError:
        missing.append('torch-pruning')
    
    if missing:
        print("❌ Отсутствуют необходимые библиотеки:")
        print(f"   pip install {' '.join(missing)}")
        return False
    return True


def prune_yolov8(
    model_path: str,
    pruning_ratio: float = 0.2,
    finetune_epochs: int = 10,
    data_yaml: str = 'coco128.yaml',
    imgsz: int = 640,
    output_dir: str = './pruned_models'
):
    """
    Применяет channel pruning к YOLOv8 модели.
    
    Args:
        model_path: Путь к .pt модели
        pruning_ratio: Доля каналов для удаления (0.2 = 20%)
        finetune_epochs: Эпохи для дообучения
        data_yaml: Датасет для fine-tuning
        imgsz: Размер изображения
        output_dir: Директория для сохранения
    """
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🔪 YOLOv8 Channel Pruning")
    print(f"{'='*60}")
    print(f"Модель:          {model_path}")
    print(f"Pruning ratio:   {pruning_ratio*100:.0f}%")
    print(f"Fine-tune epochs: {finetune_epochs}")
    print(f"Dataset:         {data_yaml}")
    
    # 1. Загрузка модели
    print("\n📥 Загрузка модели...")
    model = YOLO(model_path)
    
    # Получаем базовую информацию
    pytorch_model = model.model
    
    # Подсчёт параметров до pruning
    params_before = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Параметров до pruning: {params_before:,} ({params_before/1e6:.2f}M)")
    
    # 2. Настройка pruner
    print("\n🔧 Настройка pruner...")
    
    # Пример входа для трассировки графа
    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(next(pytorch_model.parameters()).device)
    
    # Слои, которые НЕ нужно обрезать (detection heads и критичные)
    ignored_layers = []
    for name, module in pytorch_model.named_modules():
        # Пропускаем detection heads
        if 'detect' in name.lower() or 'cv2' in name or 'cv3' in name:
            ignored_layers.append(module)
    
    # Importance scorer на основе L1-нормы весов
    importance = tp.importance.MagnitudeImportance(p=1)  # L1 norm
    
    # Создаём pruner
    pruner = tp.pruner.MagnitudePruner(
        pytorch_model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=8,  # Округление до кратного 8 для SIMD
    )
    
    # 3. Применяем pruning
    print("\n✂️  Применяем pruning...")
    pruner.step()
    
    # Подсчёт параметров после pruning
    params_after = sum(p.numel() for p in pytorch_model.parameters())
    reduction = (1 - params_after / params_before) * 100
    print(f"Параметров после pruning: {params_after:,} ({params_after/1e6:.2f}M)")
    print(f"Сокращение: {reduction:.1f}%")
    
    # 4. Fine-tuning
    if finetune_epochs > 0:
        print(f"\n🎯 Fine-tuning на {finetune_epochs} эпохах...")
        model.train(
            data=data_yaml,
            epochs=finetune_epochs,
            imgsz=imgsz,
            batch=16,
            patience=5,
            pretrained=False,  # Уже есть веса
            optimizer='AdamW',
            lr0=0.001,
            warmup_epochs=1,
            cos_lr=True,
        )
    
    # 5. Сохранение
    pruned_name = Path(model_path).stem + '_pruned'
    
    # PyTorch
    pt_path = output_path / f"{pruned_name}.pt"
    model.save(str(pt_path))
    print(f"\n💾 Сохранено: {pt_path}")
    
    # ONNX для конвертации в ncnn
    onnx_path = output_path / f"{pruned_name}.onnx"
    model.export(format='onnx', imgsz=imgsz, simplify=True)
    print(f"💾 Экспортировано: {onnx_path}")
    
    # 6. Валидация
    print("\n📊 Валидация pruned модели...")
    metrics = model.val(data=data_yaml, imgsz=imgsz)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ Pruning завершён!")
    print(f"{'='*60}")
    print("\nСледующие шаги для Android:")
    print("1. Установите ncnn tools:")
    print("   git clone https://github.com/Tencent/ncnn && cd ncnn && mkdir build && cd build")
    print("   cmake .. && make -j$(nproc)")
    print("\n2. Конвертируйте ONNX → ncnn:")
    print(f"   ./onnx2ncnn {onnx_path} {pruned_name}.param {pruned_name}.bin")
    print(f"\n3. Скопируйте в assets:")
    print(f"   cp {pruned_name}.param {pruned_name}.bin app/src/main/assets/")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Channel Pruning')
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                        help='Path to YOLOv8 model (.pt)')
    parser.add_argument('--ratio', '-r', type=float, default=0.2,
                        help='Pruning ratio (0.2 = remove 20%% channels)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Fine-tuning epochs (0 to skip)')
    parser.add_argument('--data', '-d', default='coco128.yaml',
                        help='Dataset for fine-tuning')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--output', '-o', default='./pruned_models',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if not check_dependencies():
        return 1
    
    prune_yolov8(
        model_path=args.model,
        pruning_ratio=args.ratio,
        finetune_epochs=args.epochs,
        data_yaml=args.data,
        imgsz=args.imgsz,
        output_dir=args.output
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
