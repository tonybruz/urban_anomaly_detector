#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симулятор городской мобильности с скрытым объектом (бордель).
Генерирует датасет перемещений 2000 жителей города 128×128 за 30 дней.

Автор: AI Assistant
Версия: 1.0.0
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Без GUI для серверов
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# КОНФИГУРАЦИЯ СИМУЛЯЦИИ
# ============================================================================

RANDOM_SEED = 42
GRID_SIZE = 128
POPULATION = 2000
DAYS = 30
STEP_MINUTES = 30
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES  # 48 шагов в день
TOTAL_STEPS = DAYS * STEPS_PER_DAY

# Пути вывода
OUTPUT_DIR = Path("output")

# Параметры борделя
BROTHEL_WORKERS_COUNT = 3
CLIENT_PERCENTAGE = (0.04, 0.06)  # 4-6% = 80-120 человек
BROTHEL_VISIT_INTERVAL = (14, 30)  # дней между визитами
BROTHEL_VISIT_DURATION = (2, 4)    # часов на визит
BROTHEL_ACTIVE_HOURS = set(list(range(18, 24)) + list(range(0, 4)))  # 18:00-01:00
PEAK_HOURS = set(range(19, 3))      # 19:00-02:00
WEEKEND_MULTIPLIER = 2.0
RANDOM_VISIT_CHANCE = 0.10

# Параметры шума
RARE_CELLS_COUNT = (5, 8)
RARE_CELL_PROB = 0.001  # 0.1%

# Временные зоны
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 7))
WORK_HOURS = set(range(9, 17))
EVENING_HOURS = set(range(17, 22))


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class CityGrid:
    """Представление городской сетки."""
    
    def __init__(self, size: int):
        self.size = size
        self.center = size // 2
        self.radius_center = size // 4
        
    def is_center(self, x: int, y: int) -> bool:
        """Проверка, находится ли клетка в центральной зоне."""
        return (abs(x - self.center) < self.radius_center and 
                abs(y - self.center) < self.radius_center)
    
    def is_edge(self, x: int, y: int, margin: int = 10) -> bool:
        """Проверка, находится ли клетка на краю города."""
        return (x < margin or x >= self.size - margin or 
                y < margin or y >= self.size - margin)
    
    def get_remote_zone(self) -> tuple:
        """Получение координат для уединённого места (не центр, не край)."""
        while True:
            x = random.randint(15, self.size - 16)
            y = random.randint(15, self.size - 16)
            if not self.is_center(x, y) and not self.is_edge(x, y, margin=5):
                return x, y
    
    def get_home_location(self) -> tuple:
        """Генерация домашней локации (не в центре)."""
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if not self.is_center(x, y):
                return x, y
    
    def get_work_location(self) -> tuple:
        """Генерация рабочей локации (в центре, нормальное распределение)."""
        while True:
            x = int(np.random.normal(self.center, self.radius_center // 2))
            y = int(np.random.normal(self.center, self.radius_center // 2))
            if 0 <= x < self.size and 0 <= y < self.size and self.is_center(x, y):
                return x, y


class Resident:
    """Модель жителя города."""
    
    def __init__(self, user_id: int, city: CityGrid):
        self.user_id = user_id
        self.city = city
        self.home = city.get_home_location()
        self.work = city.get_work_location()
        self.is_brothel_worker = False
        self.is_brothel_client = False
        self.brothel_visit_interval = None
        self.next_brothel_visit = None
        self.current_location = self.home
        
    def set_as_brothel_worker(self, brothel_location: tuple):
        """Назначить жителя работницей борделя."""
        self.is_brothel_worker = True
        self.home = brothel_location  # Работницы живут в борделе
        self.work = brothel_location
        self.current_location = brothel_location
        
    def set_as_client(self):
        """Назначить жителя клиентом борделя."""
        self.is_brothel_client = True
        self.brothel_visit_interval = random.randint(*BROTHEL_VISIT_INTERVAL)
        # Первый визит ОБЯЗАТЕЛЬНО в первые 10 дней вечером
        first_visit_day = random.randint(1, 10)
        # Вечерний шаг: 18:00 = шаг 36 внутри дня
        evening_step = random.randint(36, 44)  # 18:00-22:00
        self.next_brothel_visit = first_visit_day * STEPS_PER_DAY + evening_step
        
    def get_location_at_step(self, step: int, brothel_location: tuple, 
                            rare_cells: list, day_of_week: int, hour: int) -> tuple:
        """
        Определение локации жителя на данном шаге времени.
        Возвращает (x, y, location_type, is_brothel_visit, is_rare_visit)
        """
        is_weekend = day_of_week >= 5  # Суббота, Воскресенье
        
        # Приоритет 1: работницы борделя всегда в борделе
        if self.is_brothel_worker:
            return (*brothel_location, 'brothel', True, False)
        
                # Приоритет 2: клиент борделя с запланированным визитом
        if self.is_brothel_client and step >= self.next_brothel_visit:
            if hour in BROTHEL_ACTIVE_HOURS:
                # Базовая вероятность визита (усилена)
                visit_prob = 0.8  # было min(visit_prob, 1.0) = 1.0 но только при условии
                if is_weekend:
                    visit_prob = 0.95
                
                # Дополнительные случайные вечерние визиты
                extra_visit = random.random() < RANDOM_VISIT_CHANCE
                
                if random.random() < visit_prob or extra_visit:
                    # Длительность визита: 2-4 часа (усилено для detectability)
                    duration_hours = random.randint(2, 4)
                    duration_steps = duration_hours * (60 // STEP_MINUTES)
                    
                    # Следующий визит через 20-35 дней (сужен диапазон)
                    next_interval = random.randint(20, 35)
                    self.next_brothel_visit = step + next_interval * STEPS_PER_DAY
                    
                    return (*brothel_location, 'brothel', True, False)
        
        # Приоритет 3: редкие клетки-шум
        for rare_cell in rare_cells:
            if random.random() < RARE_CELL_PROB and self.current_location != rare_cell:
                if random.random() < 0.3:  # 30% шанс задержаться
                    return (*rare_cell, 'rare_cell', False, True)
        
        # Стандартное расписание по времени суток
        if hour in NIGHT_HOURS:
            # Ночь: дома
            return (*self.home, 'home', False, False)
        
        elif hour in WORK_HOURS and day_of_week < 5:  # Будни, рабочее время
            return (*self.work, 'work', False, False)
        
        elif hour in range(7, 9) and day_of_week < 5:  # Утро будни
            # Путь на работу (линейная интерполяция)
            progress = (hour - 7) / 2
            x = int(self.home[0] + (self.work[0] - self.home[0]) * progress)
            y = int(self.home[1] + (self.work[1] - self.home[1]) * progress)
            return (x, y, 'transit', False, False)
        
        elif hour in range(17, 19) and day_of_week < 5:  # Вечер после работы
            # Путь домой (линейная интерполяция)
            progress = (hour - 17) / 2
            x = int(self.work[0] + (self.home[0] - self.work[0]) * progress)
            y = int(self.work[1] + (self.home[1] - self.work[1]) * progress)
            return (x, y, 'transit', False, False)
        
        elif hour in EVENING_HOURS:
            # Вечер: чаще дома, иногда прогулки
            if random.random() < 0.7 or is_weekend:
                return (*self.home, 'home', False, False)
            else:
                # Случайная прогулка в радиусе 10 клеток от дома
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                x = max(0, min(self.city.size - 1, self.home[0] + dx))
                y = max(0, min(self.city.size - 1, self.home[1] + dy))
                return (x, y, 'leisure', False, False)
        
        else:
            # Остальное время: дома
            return (*self.home, 'home', False, False)


class BrothelSimulator:
    """Основной класс симулятора городской мобильности."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        random.seed(seed)
        np.random.seed(seed)
        self.city = CityGrid(GRID_SIZE)
        self.residents: list[Resident] = []
        self.brothel_location = None
        self.brothel_workers: list[int] = []
        self.brothel_clients: list[int] = []
        self.rare_cells: list[tuple] = []
        self.records: list[dict] = []
        
    def initialize(self):
        """Инициализация симуляции."""
        logger.info("Инициализация симуляции...")
        
        # 1. Размещение борделя в уединённом месте
        self.brothel_location = self.city.get_remote_zone()
        logger.info(f"🏠 Бордель размещён: {self.brothel_location}")
        
        # 2. Создание всех жителей
        logger.info(f"👥 Создание {POPULATION} жителей...")
        for i in range(POPULATION):
            resident = Resident(i, self.city)
            self.residents.append(resident)
        
        # 3. Назначение работниц борделя (3 человека, живут там)
        worker_candidates = [r for r in self.residents if not r.is_brothel_worker]
        workers = random.sample(worker_candidates, BROTHEL_WORKERS_COUNT)
        for w in workers:
            w.set_as_brothel_worker(self.brothel_location)
            self.brothel_workers.append(w.user_id)
        logger.info(f"👩‍💼 Назначено {len(self.brothel_workers)} работниц")
        
        # 4. Назначение клиентов борделя (2-4% населения)
        client_count = random.randint(
            int(POPULATION * CLIENT_PERCENTAGE[0]),
            int(POPULATION * CLIENT_PERCENTAGE[1])
        )
        client_candidates = [r for r in self.residents 
                           if not r.is_brothel_worker and not r.is_brothel_client]
        clients = random.sample(client_candidates, client_count)
        for c in clients:
            c.set_as_client()
            self.brothel_clients.append(c.user_id)
        logger.info(f"👤 Назначено {len(self.brothel_clients)} клиентов")
        
        # 5. Генерация редких клеток-шума
        rare_count = random.randint(*RARE_CELLS_COUNT)
        while len(self.rare_cells) < rare_count:
            cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if cell != self.brothel_location and cell not in self.rare_cells:
                self.rare_cells.append(cell)
        logger.info(f"🔊 Создано {len(self.rare_cells)} редких клеток")
        
    def run_simulation(self):
        """Запуск симуляции на заданный период."""
        logger.info(f"🚀 Запуск симуляции: {DAYS} дней, {TOTAL_STEPS} шагов...")
        
        for step in tqdm(range(TOTAL_STEPS), desc="Симуляция"):
            # Вычисление времени
            total_minutes = step * STEP_MINUTES
            day = total_minutes // (24 * 60)
            hour = (total_minutes % (24 * 60)) // 60
            minute = (total_minutes % (24 * 60)) % 60
            day_of_week = day % 7
            
            timestamp = datetime(2026, 1, 1) + timedelta(minutes=total_minutes)
            
            # Генерация записей для всех жителей
            for resident in self.residents:
                x, y, loc_type, is_brothel, is_rare = resident.get_location_at_step(
                    step, self.brothel_location, self.rare_cells, day_of_week, hour
                )
                
                self.records.append({
                    'timestamp': timestamp.isoformat(),
                    'user_id': resident.user_id,
                    'x': x,
                    'y': y,
                    'hour': hour,
                    'minute': minute,
                    'day': day,
                    'day_of_week': day_of_week,
                    'location_type': loc_type,
                    'is_brothel_worker': resident.is_brothel_worker,
                    'is_brothel_client': resident.is_brothel_client,
                    'is_brothel_visit': is_brothel,
                    'is_rare_cell_visit': is_rare
                })
                
        logger.info(f"📊 Сгенерировано {len(self.records):,} записей")
        
    def save_data(self):
        """Сохранение результатов симуляции."""
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # 1. CSV с перемещениями
        csv_path = OUTPUT_DIR / "enhanced_city_movement.csv"
        df = pd.DataFrame(self.records)
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 Сохранено: {csv_path} ({len(df):,} строк)")
        
        # 2. Файл с параметрами симуляции
        info_path = OUTPUT_DIR / "simulation_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ПАРАМЕТРЫ СИМУЛЯЦИИ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"GRID_SIZE: {GRID_SIZE}×{GRID_SIZE}\n")
            f.write(f"POPULATION: {POPULATION}\n")
            f.write(f"DAYS: {DAYS}\n")
            f.write(f"STEP_MINUTES: {STEP_MINUTES}\n")
            f.write(f"RANDOM_SEED: {RANDOM_SEED}\n\n")
            f.write("СКРЫТЫЕ ОБЪЕКТЫ (GROUND TRUTH):\n")
            f.write(f"brothel_location: {self.brothel_location}\n")
            f.write(f"brothel_workers: {self.brothel_workers}\n")
            f.write(f"brothel_clients_count: {len(self.brothel_clients)}\n")
            f.write(f"brothel_clients_sample: {self.brothel_clients[:10]}...\n")
            f.write(f"rare_cells: {self.rare_cells}\n")
        logger.info(f"💾 Сохранено: {info_path}")
        
        # 3. Статистика по клеткам
        cell_stats = df.groupby(['x', 'y']).agg({
            'user_id': ['count', 'nunique'],
            'hour': lambda x: ((x >= 18) | (x < 6)).sum(),
            'is_brothel_visit': 'sum',
            'is_rare_cell_visit': 'sum'
        }).round(2)
        cell_stats.columns = ['total_visits', 'unique_visitors', 'night_visits', 
                             'brothel_visits', 'rare_visits']
        stats_path = OUTPUT_DIR / "cell_statistics.csv"
        cell_stats.reset_index().to_csv(stats_path, index=False)
        logger.info(f"💾 Сохранено: {stats_path}")
        
        # 4. Карта города
        self._generate_city_map()
        
    def _generate_city_map(self):
        """Генерация карты города с маркерами объектов."""
        fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
        ax.set_xlim(-1, GRID_SIZE)
        ax.set_ylim(-1, GRID_SIZE)
        ax.set_aspect('equal')
        ax.set_title('Карта города 128×128 с скрытым объектом', fontsize=14, pad=20)
        ax.grid(alpha=0.1, linestyle='--')
        
        # Центральная зона (работа)
        center_patch = plt.Rectangle(
            (self.city.center - self.city.radius_center, 
             self.city.center - self.city.radius_center),
            self.city.radius_center * 2, self.city.radius_center * 2,
            facecolor='lightblue', alpha=0.15, label='Центр (работа)', edgecolor='none'
        )
        ax.add_patch(center_patch)
        
        # Бордель (чёрный квадрат с B)
        bx, by = self.brothel_location
        brothel_square = plt.Rectangle((bx-1, by-1), 3, 3, 
                                       facecolor='black', alpha=0.8, 
                                       edgecolor='red', linewidth=2)
        ax.add_patch(brothel_square)
        ax.text(bx, by, 'B', color='white', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Редкие клетки (синие кресты)
        for i, (rx, ry) in enumerate(self.rare_cells):
            ax.plot(rx, ry, marker='x', color='blue', markersize=12, 
                   markeredgewidth=2, 
                   label='Редкая клетка (шум)' if i == 0 else '')
        
        # Дома и работы (выборка для визуализации)
        homes = np.array([r.home for r in self.residents if not r.is_brothel_worker])
        works = np.array([r.work for r in self.residents if not r.is_brothel_worker])
        
        ax.scatter(homes[::20, 0], homes[::20, 1], c='green', s=2, alpha=0.3, 
                  label='Дома (выборка)')
        ax.scatter(works[::20, 0], works[::20, 1], c='orange', s=2, alpha=0.3, 
                  label='Работы (выборка)')
        
        # Легенда
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.set_xlabel('X координата')
        ax.set_ylabel('Y координата')
        
        plt.tight_layout()
        map_path = OUTPUT_DIR / "city_map_with_brothel.png"
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"🗺️ Сохранена карта: {map_path}")
        
    def get_ground_truth(self) -> dict:
        """Возврат эталонных данных для проверки детектора."""
        return {
            'brothel_location': self.brothel_location,
            'brothel_workers': self.brothel_workers,
            'brothel_clients': self.brothel_clients,
            'rare_cells': self.rare_cells
        }


def main():
    """Точка входа симулятора."""
    print("🏙️  Запуск симулятора городской мобильности")
    print("=" * 60)
    
    simulator = BrothelSimulator(seed=RANDOM_SEED)
    simulator.initialize()
    simulator.run_simulation()
    simulator.save_data()
    
    # Сохранение ground truth для детектора
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(simulator.get_ground_truth(), f, indent=2)
    logger.info(f"🎯 Ground truth сохранён: {gt_path}")
    
    print("\n" + "=" * 60)
    print("✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"📊 Всего записей: {len(simulator.records):,}")
    print(f"🏠 Бордель в точке: {simulator.brothel_location}")
    print(f"👩‍💼 Работниц борделя: {len(simulator.brothel_workers)}")
    print(f"👤 Клиентов борделя: {len(simulator.brothel_clients)}")
    print(f"🔊 Редких клеток-шума: {len(simulator.rare_cells)}")
    print("=" * 60)
    
    return simulator.get_ground_truth()


if __name__ == "__main__":
    main()