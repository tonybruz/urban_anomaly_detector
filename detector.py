#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аналитическое приложение для обнаружения борделя по данным перемещений.
Использует 5 взвешенных метрик для ранжирования клеток.

Автор: AI Assistant
Версия: 1.0.0
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')  # Без GUI для серверов
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# КОНФИГУРАЦИЯ АНАЛИЗА
# ============================================================================

ANALYSIS_DIR = Path("brothel_analysis")
CLIENT_MAPS_DIR = ANALYSIS_DIR / "client_maps"

# Веса метрик обнаружения
METRIC_WEIGHTS = {
    'night_ratio': 0.25,       # Доля ночных посещений (18:00-06:00)
    'short_visit_ratio': 0.25, # Доля коротких визитов (1-4 часа)
    'core_visitors': 0.20,     # Постоянное ядро (2-5 человек)
    'visit_regularity': 0.20,  # Регулярность (интервал 20-45 дней)
    'evening_peak': 0.10       # Вечерний пик (19:00-02:00)
}

# Параметры анализа
GRID_SIZE = 128
NIGHT_HOURS = set(list(range(18, 24)) + list(range(0, 6)))
PEAK_HOURS = set(range(19, 3))  # 19:00-02:00
SHORT_VISIT_MAX_HOURS = 4
CORE_VISITOR_MIN_VISITS = 5
CORE_VISITOR_COUNT_RANGE = (2, 5)
REGULARITY_INTERVAL = (20, 45)  # дней


def load_data():
    """Загрузка данных симуляции и ground truth."""
    try:
        df = pd.read_csv("output/enhanced_city_movement.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        with open("output/ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
            
        print(f"📥 Загружено записей: {len(df):,}")
        return df, ground_truth
    except FileNotFoundError as e:
        print(f"❌ Ошибка: не найдены файлы симуляции: {e}")
        print("   Сначала запустите: python simulator.py")
        raise


def calculate_cell_metrics(cell_data: pd.DataFrame) -> tuple:
    """
    Расчёт метрик для одной клетки.
    v3: добавлен дискриминатор "резиденты + массовые посетители".
    """
    if len(cell_data) == 0:
        return {k: 0 for k in METRIC_WEIGHTS}, 0
    
    metrics = {}
    total_steps = len(cell_data)
    
    # === Разделение резидентов и посетителей ===
    visitor_counts = cell_data.groupby('user_id').size()
    threshold_resident = max(total_steps * 0.05, 50)  # 5% или минимум 50 записей
    resident_ids = visitor_counts[visitor_counts >= threshold_resident].index
    visitor_ids = visitor_counts[visitor_counts < threshold_resident].index
    
    visitor_data = cell_data[cell_data['user_id'].isin(visitor_ids)]
    n_visitors_records = len(visitor_data)
    n_unique_visitors = len(visitor_ids)
    n_residents = len(resident_ids)
    
    # 1. Ночные визиты ПОСЕТИТЕЛЕЙ
    if n_visitors_records > 0:
        night_v = visitor_data[visitor_data['hour'].apply(lambda h: h in NIGHT_HOURS)].shape[0]
        metrics['night_ratio'] = night_v / n_visitors_records
    else:
        metrics['night_ratio'] = 0
    
    # 2. Короткие визиты ПОСЕТИТЕЛЕЙ
    short_count = 0
    total_sessions = 0
    if n_visitors_records > 0:
        vs = visitor_data.sort_values(['user_id', 'timestamp'])
        for uid, uv in vs.groupby('user_id'):
            if len(uv) < 2:
                total_sessions += 1
                short_count += 1  # одиночный шаг = короткий визит
                continue
            td = uv['timestamp'].diff().dt.total_seconds() / 3600
            sid = (td > 1).cumsum()
            for _, sess in uv.groupby(sid):
                total_sessions += 1
                dur = (sess['timestamp'].max() - sess['timestamp'].min()).total_seconds() / 3600
                if dur == 0:
                    dur = 0.5
                if 0.5 <= dur <= SHORT_VISIT_MAX_HOURS:
                    short_count += 1
        metrics['short_visit_ratio'] = short_count / total_sessions if total_sessions > 0 else 0
    else:
        metrics['short_visit_ratio'] = 0
    
    # 3. Постоянное ядро (резиденты)
    if CORE_VISITOR_COUNT_RANGE[0] <= n_residents <= CORE_VISITOR_COUNT_RANGE[1]:
        metrics['core_visitors'] = 1.0
    elif n_residents < CORE_VISITOR_COUNT_RANGE[0]:
        metrics['core_visitors'] = n_residents / CORE_VISITOR_COUNT_RANGE[0] if n_residents > 0 else 0
    else:
        metrics['core_visitors'] = max(0, 1 - (n_residents - CORE_VISITOR_COUNT_RANGE[1]) / 10)
    
    # 4. Регулярность посетителей
    reg_scores = []
    if n_visitors_records > 0:
        for uid, uv in visitor_data.groupby('user_id'):
            vd = uv['day'].drop_duplicates().sort_values()
            if len(vd) >= 2:
                intervals = vd.diff().dropna()
                in_range = ((intervals >= 15) & (intervals <= 45)).sum()
                reg_scores.append(in_range / len(intervals))
            elif len(vd) == 1:
                reg_scores.append(0.3)
    metrics['visit_regularity'] = np.mean(reg_scores) if reg_scores else 0
    
    # 5. Вечерний пик ПОСЕТИТЕЛЕЙ (расширенный диапазон 17-04)
    extended_peak = set(list(range(17, 24)) + list(range(0, 5)))
    if n_visitors_records > 0:
        peak_v = visitor_data[visitor_data['hour'].apply(lambda h: h in extended_peak)].shape[0]
        metrics['evening_peak'] = peak_v / n_visitors_records
    else:
        metrics['evening_peak'] = 0
    
    # === КЛЮЧЕВОЙ ДИСКРИМИНАТОР ===
    # Бордель = единственная клетка где:
    #   - Есть резиденты (≥2)
    #   - Много уникальных посетителей (≥20)
    #   - Посетители приходят преимущественно вечером/ночью
    # Шумовые клетки имеют 2-5 посетителей максимум
    
    discriminator = 0.0
    if n_residents >= 2 and n_unique_visitors >= 20:
        # Бонус пропорционален количеству уникальных посетителей
        # 20 посетителей → 0.15, 50 → 0.25, 100+ → 0.35
        visitor_bonus = min(0.35, 0.15 + (n_unique_visitors - 20) * 0.0025)
        
        # Усиление если посетители приходят вечером
        evening_factor = max(metrics['evening_peak'], metrics['night_ratio'])
        discriminator = visitor_bonus * (0.5 + 0.5 * evening_factor)
    
    # Взвешенная сумма + дискриминатор
    base_score = sum(metrics[k] * METRIC_WEIGHTS[k] for k in METRIC_WEIGHTS)
    score = base_score + discriminator
    
    return metrics, score


def process_cell_group(args):
    """Обработка одной клетки. Принимает кортеж (coords, dataframe)."""
    (x, y), cell_data = args
    metrics, score = calculate_cell_metrics(cell_data)
    return {
        'x': x, 'y': y, 'score': score,
        **{f'metric_{k}': v for k, v in metrics.items()},
        'total_visits': len(cell_data),
        'unique_visitors': cell_data['user_id'].nunique()
    }


def analyze_all_cells(df: pd.DataFrame, max_workers: int = 8):
    """
    Параллельный анализ всех клеток сетки.
    Координаты извлекаются ДО отправки в пул процессов.
    """
    print(f"🔍 Анализ {GRID_SIZE}×{GRID_SIZE} клеток...")
    print(f"⚙️  Используется {max_workers} процессов")

    # Извлекаем (координаты, данные) заранее — это ключевое исправление
    cell_groups = [
        (coords, group) 
        for coords, group in df.groupby(['x', 'y'])
    ]
    total_groups = len(cell_groups)
    print(f"   Клеток для обработки: {total_groups:,}")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_cell_group, item) for item in cell_groups]

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Обработка клеток"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"⚠️ Ошибка при обработке клетки: {e}")
                continue

    if not results:
        raise RuntimeError("❌ Ни одна клетка не была обработана успешно!")

    results_df = pd.DataFrame(results)
    print(f"✅ Обработано клеток: {len(results_df):,}")
    return results_df


def find_brothel(cell_scores: pd.DataFrame) -> tuple:
    """Поиск клетки с максимальным скором."""
    best = cell_scores.loc[cell_scores['score'].idxmax()]
    return int(best['x']), int(best['y']), best['score']


def identify_workers_and_clients(df: pd.DataFrame, brothel_cell: tuple) -> dict:
    """
    Идентификация работниц и клиентов по статистике посещений клетки.
    v2: не зависит от location_type, использует только частоту посещений.
    """
    x, y = brothel_cell
    cell_data = df[(df['x'] == x) & (df['y'] == y)]
    
    if len(cell_data) == 0:
        return {'detected_workers': [], 'detected_clients': []}
    
    total_steps = len(cell_data)
    visitor_stats = cell_data.groupby('user_id').size()
    
    # Порог резидента: >5% всех записей в клетке ИЛИ >100 записей
    resident_threshold = max(total_steps * 0.05, 100)
    
    # Работницы: резиденты (живут/работают в клетке)
    workers = visitor_stats[visitor_stats >= resident_threshold].index.tolist()
    
    # Клиенты: посетили клетку, но не являются резидентами
    # Минимум 2 записи (хотя бы один визит длительностью ≥1 часа)
    clients = visitor_stats[
        (visitor_stats < resident_threshold) & 
        (visitor_stats >= 2)
    ].index.tolist()
    
    print(f"   Найдено работниц: {len(workers)} → {workers}")
    print(f"   Найдено клиентов: {len(clients)}")
    
    return {
        'detected_workers': workers,
        'detected_clients': clients
    }


def manhattan_distance(p1: tuple, p2: tuple) -> int:
    """Расчёт манхэттенского расстояния между двумя точками."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def generate_detection_map(cell_scores: pd.DataFrame, ground_truth: dict, 
                          detected_cell: tuple):
    """Генерация карты сравнения: эталон vs обнаружение."""
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_title('Обнаружение борделя: сравнение с эталоном', fontsize=14, pad=20)
    ax.grid(alpha=0.1, linestyle='--')
    
    # Тепловая карта скоринга (топ 5% клеток)
    top_scores = cell_scores['score'].quantile(0.95)
    hot_cells = cell_scores[cell_scores['score'] >= top_scores]
    
    if len(hot_cells) > 0:
        scatter = ax.scatter(hot_cells['x'], hot_cells['y'], 
                           c=hot_cells['score'], cmap='Reds', 
                           s=30, alpha=0.6, edgecolors='none',
                           label='Подозрительные клетки')
        plt.colorbar(scatter, ax=ax, label='Score', shrink=0.7)
    
    # Эталонный бордель (B)
    gt_x, gt_y = ground_truth['brothel_location']
    ax.text(gt_x, gt_y, 'B', color='black', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='square,pad=0.3', 
            facecolor='yellow', edgecolor='black', alpha=0.8))
    
    # Обнаруженный бордель (!)
    det_x, det_y = detected_cell
    distance = manhattan_distance((gt_x, gt_y), (det_x, det_y))
    marker = '✓' if distance == 0 else '!'
    color = 'green' if distance <= 5 else 'red'
    ax.text(det_x, det_y, marker, color='white', ha='center', va='center',
            fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='square,pad=0.3', facecolor=color, 
                     edgecolor='black', alpha=0.9))
    
    # Редкие клетки (?)
    for rx, ry in ground_truth['rare_cells']:
        ax.text(rx, ry, '?', color='blue', ha='center', va='center',
                fontsize=10, fontweight='bold', alpha=0.7)
    
    # Линия между эталоном и находкой
    if (gt_x, gt_y) != (det_x, det_y):
        ax.plot([gt_x, det_x], [gt_y, det_y], 'r--', linewidth=1, alpha=0.5)
        ax.text((gt_x + det_x) // 2, (gt_y + det_y) // 2 + 2, 
               f'd={distance}', fontsize=9, ha='center', 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Легенда
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
                  markersize=15, label='Эталон (B)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                  markersize=15, label=f'Обнаружено ({marker})'),
        plt.Line2D([0], [0], marker='x', color='blue', markersize=10, 
                  label='Редкие клетки (?)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlabel('X координата')
    ax.set_ylabel('Y координата')
    
    plt.tight_layout()
    map_path = ANALYSIS_DIR / "brothel_detection_map.png"
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🗺️ Карта обнаружения сохранена: {map_path}")


def generate_client_maps(df: pd.DataFrame, clients: list, brothel_location: tuple, 
                        home_locations: dict, limit: int = 10):
    """Генерация карт перемещений для первых N клиентов."""
    print(f"👤 Генерация карт для {min(limit, len(clients))} клиентов...")
    
    for i, client_id in enumerate(clients[:limit]):
        client_data = df[df['user_id'] == client_id].sort_values('timestamp')
        if len(client_data) == 0:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.set_xlim(-1, GRID_SIZE)
        ax.set_ylim(-1, GRID_SIZE)
        ax.set_aspect('equal')
        ax.set_title(f'Клиент #{client_id}: маршрут к борделю', fontsize=12, pad=15)
        ax.grid(alpha=0.1)
        
        # Дом клиента (зелёный треугольник H)
        home = home_locations.get(client_id)
        if home:
            ax.plot(home[0], home[1], marker='^', color='green', markersize=15, 
                   markeredgecolor='darkgreen', label='Дом (H)', zorder=5)
            ax.text(home[0], home[1] + 2, 'H', color='green', ha='center', 
                   fontweight='bold', fontsize=10)
        
        # Бордель (чёрный квадрат B)
        bx, by = brothel_location
        ax.add_patch(plt.Rectangle((bx-1, by-1), 3, 3, facecolor='black', 
                                  alpha=0.7, edgecolor='red', linewidth=2))
        ax.text(bx, by, 'B', color='white', ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Маршрут клиента
        if len(client_data) > 1:
            # Фильтруем только значимые перемещения
            route_data = client_data[['x', 'y', 'timestamp']].drop_duplicates(subset=['x', 'y'])
            if len(route_data) > 1:
                ax.plot(route_data['x'], route_data['y'], 'b-', linewidth=0.5, 
                       alpha=0.4, label='Маршрут')
                ax.scatter(route_data['x'], route_data['y'], c='blue', s=10, alpha=0.5)
        
        # Выделение визитов в бордель
        brothel_visits = client_data[(client_data['x'] == bx) & (client_data['y'] == by)]
        if len(brothel_visits) > 0:
            ax.scatter([bx]*len(brothel_visits), [by]*len(brothel_visits), 
                      c='red', s=50, marker='*', edgecolors='yellow', 
                      label='Визиты в бордель', zorder=6)
        
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.tight_layout()
        client_path = CLIENT_MAPS_DIR / f"client_{client_id:03d}.png"
        plt.savefig(client_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Сгенерировано карт клиентов: {min(limit, len(clients))}")


def calculate_accuracy(ground_truth: dict, detected_cell: tuple, 
                      detected_workers: list, detected_clients: list) -> dict:
    """Расчёт метрик точности обнаружения."""
    gt_workers = set(ground_truth['brothel_workers'])
    gt_clients = set(ground_truth['brothel_clients'])
    
    # Точность обнаружения клетки
    cell_distance = manhattan_distance(ground_truth['brothel_location'], detected_cell)
    cell_correct = cell_distance <= 5
    
    # Точность идентификации работниц
    detected_workers_set = set(detected_workers)
    worker_tp = len(gt_workers & detected_workers_set)
    worker_precision = worker_tp / len(detected_workers_set) if detected_workers_set else 0
    worker_recall = worker_tp / len(gt_workers) if gt_workers else 0
    worker_f1 = (2 * worker_precision * worker_recall / (worker_precision + worker_recall) 
                if (worker_precision + worker_recall) > 0 else 0)
    
    # Точность идентификации клиентов
    detected_clients_set = set(detected_clients)
    client_tp = len(gt_clients & detected_clients_set)
    client_precision = client_tp / len(detected_clients_set) if detected_clients_set else 0
    client_recall = client_tp / len(gt_clients) if gt_clients else 0
    
    return {
        'cell_distance': cell_distance,
        'cell_within_threshold': cell_correct,
        'worker_precision': round(worker_precision, 3),
        'worker_recall': round(worker_recall, 3),
        'worker_f1': round(worker_f1, 3),
        'worker_match_rate': round(worker_tp / len(gt_workers) * 100, 1) if gt_workers else 0,
        'client_precision': round(client_precision, 3),
        'client_recall': round(client_recall, 3),
        'success': cell_correct and (worker_tp / len(gt_workers) >= 0.6 if gt_workers else True)
    }


def generate_report(ground_truth: dict, detected_cell: tuple, 
                   accuracy: dict, top_cells: pd.DataFrame):
    """Генерация итогового отчёта в формате Markdown."""
    report_path = ANALYSIS_DIR / "report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🔍 Отчёт об обнаружении скрытого объекта\n\n")
        
        f.write("## 📊 Параметры анализа\n")
        f.write(f"- **Сетка города**: {GRID_SIZE}×{GRID_SIZE}\n")
        f.write(f"- **Метрики**: {len(METRIC_WEIGHTS)} взвешенных показателей\n")
        f.write(f"- **Веса метрик**:\n")
        for k, v in METRIC_WEIGHTS.items():
            f.write(f"  - `{k}`: {v*100:.0f}%\n")
        f.write("\n")
        
        f.write("## 🎯 Результаты обнаружения\n")
        f.write(f"- **Эталонная клетка**: {ground_truth['brothel_location']}\n")
        f.write(f"- **Обнаруженная клетка**: {detected_cell}\n")
        f.write(f"- **Манхэттенское расстояние**: {accuracy['cell_distance']}\n")
        f.write(f"- **В пределах порога (≤5)**: {'✅ Да' if accuracy['cell_within_threshold'] else '❌ Нет'}\n\n")
        
        f.write("## 👥 Идентификация участников\n")
        f.write("### Работницы\n")
        f.write(f"- Эталон: {ground_truth['brothel_workers']}\n")
        f.write(f"- Совпадение: {accuracy['worker_match_rate']}% "
               f"({'✅' if accuracy['worker_match_rate'] >= 60 else '❌'} порог 60%)\n")
        f.write(f"- Precision: {accuracy['worker_precision']}, "
               f"Recall: {accuracy['worker_recall']}, "
               f"F1-score: {accuracy['worker_f1']}\n\n")
        
        f.write("### Клиенты\n")
        f.write(f"- Эталон: {len(ground_truth['brothel_clients'])} человек\n")
        f.write(f"- Precision: {accuracy['client_precision']}, "
               f"Recall: {accuracy['client_recall']}\n\n")
        
        f.write("## 📈 Топ-10 подозрительных клеток\n")
        f.write("| Ранг | Координаты | Score | Ночные % | Короткие % | Ядро | Регулярность | Пик |\n")
        f.write("|------|------------|-------|----------|------------|------|--------------|-----|\n")
        top10 = top_cells.nlargest(10, 'score')
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            f.write(f"| {rank} | ({int(row['x'])}, {int(row['y'])}) | "
                   f"{row['score']:.3f} | {row['metric_night_ratio']:.1%} | "
                   f"{row['metric_short_visit_ratio']:.1%} | "
                   f"{row['metric_core_visitors']:.2f} | "
                   f"{row['metric_visit_regularity']:.2f} | "
                   f"{row['metric_evening_peak']:.1%} |\n")
        f.write("\n")
        
        f.write("## 🗺️ Визуализации\n")
        f.write("![Карта обнаружения](brothel_detection_map.png)\n\n")
        f.write("### Примеры карт клиентов\n")
        for i in range(1, 4):
            f.write(f"![Клиент {i}](client_maps/client_{i:03d}.png) ")
        f.write("\n\n")
        
        f.write("## ✅ Итоговая оценка\n")
        status = "🎉 УСПЕХ" if accuracy['success'] else "⚠️ ТРЕБУЕТ ДОРАБОТКИ"
        f.write(f"**Статус**: {status}\n\n")
        f.write("### Критерии успеха:\n")
        f.write(f"- [{'x' if accuracy['cell_within_threshold'] else ' '}] "
               f"Обнаружение клетки с отклонением ≤5: "
               f"{'✅' if accuracy['cell_within_threshold'] else '❌'}\n")
        f.write(f"- [{'x' if accuracy['worker_match_rate'] >= 60 else ' '}] "
               f"Идентификация работниц ≥60%: "
               f"{'✅' if accuracy['worker_match_rate'] >= 60 else '❌'} "
               f"({accuracy['worker_match_rate']}%)\n")
        
    print(f"📄 Отчёт сохранён: {report_path}")


def main():
    """Точка входа детектора."""
    print("🔍 Запуск детектора аномальной активности")
    print("=" * 60)
    
    # Создание директорий
    ANALYSIS_DIR.mkdir(exist_ok=True)
    CLIENT_MAPS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Загрузка данных
    print("📥 Загрузка данных симуляции...")
    df, ground_truth = load_data()
    
    # Подготовка словаря домов для карт клиентов
    home_locations = {}
    home_data = df[df['location_type'] == 'home']
    if len(home_data) > 0:
        home_locations = home_data.groupby('user_id').apply(
            lambda x: (x.iloc[0]['x'], x.iloc[0]['y'])
        ).to_dict()
    
    # Анализ клеток
    cell_scores = analyze_all_cells(df, max_workers=8)
    scores_path = "output/cell_statistics_scored.csv"
    cell_scores.to_csv(scores_path, index=False)
    print(f"💾 Сохранено: {scores_path}")
    
    # Поиск борделя
    det_x, det_y, best_score = find_brothel(cell_scores)
    detected_cell = (det_x, det_y)
    print(f"🎯 Обнаруженная клетка: {detected_cell} (score: {best_score:.3f})")
    
    # Идентификация участников
    print("👥 Идентификация работниц и клиентов...")
    participants = identify_workers_and_clients(df, detected_cell)
    
    # Генерация визуализаций
    print("🎨 Генерация карт...")
    generate_detection_map(cell_scores, ground_truth, detected_cell)
    generate_client_maps(df, participants['detected_clients'], 
                        ground_truth['brothel_location'], home_locations)
    
    # Расчёт точности
    print("📊 Расчёт метрик точности...")
    accuracy = calculate_accuracy(
        ground_truth, detected_cell,
        participants['detected_workers'],
        participants['detected_clients']
    )
    
    # Генерация отчёта
    generate_report(ground_truth, detected_cell, accuracy, cell_scores)
    
    # Вывод итогов
    print("\n" + "=" * 60)
    print("📋 ИТОГОВЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print("=" * 60)
    print(f"   🏠 Эталонная клетка: {ground_truth['brothel_location']}")
    print(f"   🎯 Обнаруженная клетка: {detected_cell}")
    print(f"   📏 Расстояние: {accuracy['cell_distance']} клеток (порог: ≤5) "
          f"{'✅' if accuracy['cell_within_threshold'] else '❌'}")
    print(f"   👩‍💼 Работницы: {accuracy['worker_match_rate']}% совпадений "
          f"(порог: ≥60%) {'✅' if accuracy['worker_match_rate'] >= 60 else '❌'}")
    print(f"   🎉 Статус: {'УСПЕХ' if accuracy['success'] else 'ТРЕБУЕТ ДОРАБОТКИ'}")
    print("=" * 60)
    
    return accuracy


if __name__ == "__main__":
    main()