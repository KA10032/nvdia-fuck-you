import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import random
from enum import Enum
import time
from collections import deque

# ==================== 修复中文字体问题 ====================
import matplotlib
try:
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("中文字体设置成功")
except:
    print("中文字体设置失败，使用默认字体")

# 安全导入torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("PyTorch可用，使用神经网络功能")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch不可用，使用简化版本")

class ProductionCategory(Enum):
    AGRICULTURE = "Agriculture"
    INDUSTRY = "Industry" 
    SERVICES = "Services"
    HIGH_TECH = "High_Tech"

@dataclass
class Commune:
    id: int
    name: str
    level: int
    parent_id: int
    population: int
    labor_force: int
    capital_stock: float
    natural_resources: Dict[str, float]
    production_capacity: Dict[ProductionCategory, float]
    current_production: Dict[ProductionCategory, float] = field(default_factory=dict)
    consumption_needs: Dict[ProductionCategory, float] = field(default_factory=dict)
    historical_production: List[Dict[ProductionCategory, float]] = field(default_factory=list)
    activation_level: float = 0.5
    connections: List[int] = field(default_factory=list)
    learning_parameters: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.current_production:
            self.current_production = {cat: 0.0 for cat in ProductionCategory}
        if not self.consumption_needs:
            self.consumption_needs = {cat: random.uniform(500, 5000) for cat in ProductionCategory}
        if not self.learning_parameters:
            self.learning_parameters = {
                'adaptation_rate': random.uniform(0.01, 0.1),
                'innovation_bias': random.uniform(0.1, 0.3),
                'cooperation_tendency': random.uniform(0.5, 0.8)
            }

class EconomicNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EconomicNeuralNetwork, self).__init__()
        print(f"初始化神经网络: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class MarxistEconomicModel:
    def __init__(self):
        self.communes = []
        self.cycles_completed = 0
        self.historical_data = []
        self.organic_composition = 0.6
        self.rate_of_exploitation = 0.5
        self.accumulation_rate = 0.25
        self.reproduction_balance_threshold = 0.1
        self.wave_length = 50
        self.innovation_cluster = 0.3
        self.long_wave_phase = 0
        self.learning_rate = 0.01
        self.prediction_accuracy_history = deque(maxlen=20)
    
    def calculate_surplus_value(self, commune: Commune) -> float:
        constant_capital = commune.capital_stock * 0.1
        variable_capital = commune.labor_force * 1.0
        surplus_value = variable_capital * self.rate_of_exploitation
        return surplus_value
    
    def calculate_profit_rate(self, commune: Commune) -> float:
        surplus_value = self.calculate_surplus_value(commune)
        constant_capital = commune.capital_stock * 0.1
        variable_capital = commune.labor_force * 1.0
        return surplus_value / (constant_capital + variable_capital) if (constant_capital + variable_capital) > 0 else 0
    
    def check_reproduction_balance(self, production_data: Dict[ProductionCategory, float]) -> Tuple[bool, float]:
        department_I = production_data[ProductionCategory.INDUSTRY] + production_data[ProductionCategory.HIGH_TECH]
        department_II = production_data[ProductionCategory.AGRICULTURE] + production_data[ProductionCategory.SERVICES]
        balance_ratio = department_I / department_II if department_II > 0 else 1.0
        is_balanced = abs(1 - balance_ratio) < self.reproduction_balance_threshold
        return is_balanced, balance_ratio
    
    def mandatory_economic_law(self, total_production: float, consumption: float) -> float:
        wave_effect = np.sin(2 * np.pi * self.cycles_completed / self.wave_length)
        optimal_ratio = 0.7 + 0.1 * wave_effect
        current_ratio = consumption / total_production if total_production > 0 else 0
        adjustment = optimal_ratio - current_ratio
        return adjustment * (1 + self.innovation_cluster)
    
    def update_long_wave_phase(self):
        self.long_wave_phase = (2 * np.pi * self.cycles_completed) / self.wave_length

class AdvancedNeuralPlanNetwork:
    def __init__(self, economic_model: MarxistEconomicModel):
        self.economic_model = economic_model
        self.connection_strength = {}
        self.learning_rate = 0.1
        self.prediction_model = None
        self.optimizer = None
        self.loss_function = nn.MSELoss()
        self.initialize_prediction_model()
        
    def initialize_prediction_model(self):
        num_categories = len(ProductionCategory)
        input_size = num_categories * 3 + 5
        hidden_size = 32
        output_size = num_categories
        
        print(f"神经网络参数: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        
        self.prediction_model = EconomicNeuralNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.prediction_model.parameters(), lr=0.001)
        
    def prepare_training_data(self, commune: Commune) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(commune.historical_production) < 2:
            return None, None
            
        input_data = []
        
        recent_data = commune.historical_production[-1]
        for category in ProductionCategory:
            input_data.append(recent_data.get(category, 0))
        
        for category in ProductionCategory:
            input_data.append(commune.production_capacity.get(category, 0))
        
        for category in ProductionCategory:
            input_data.append(commune.consumption_needs.get(category, 0))
        
        input_data.extend([
            commune.population / 100000,
            commune.labor_force / 50000,
            commune.capital_stock / 10000000,
            self.economic_model.organic_composition,
            self.economic_model.rate_of_exploitation
        ])
        
        expected_size = len(ProductionCategory) * 3 + 5
        if len(input_data) != expected_size:
            print(f"警告: 输入数据维度不匹配! 期望: {expected_size}, 实际: {len(input_data)}")
            if len(input_data) < expected_size:
                input_data.extend([0] * (expected_size - len(input_data)))
            else:
                input_data = input_data[:expected_size]
        
        input_tensor = torch.FloatTensor(input_data)
        
        target_data = [commune.current_production[category] for category in ProductionCategory]
        target_tensor = torch.FloatTensor(target_data)
        
        return input_tensor, target_tensor
    
    def train_prediction_model(self, commune: Commune):
        input_tensor, target_tensor = self.prepare_training_data(commune)
        if input_tensor is None:
            return float('inf')
            
        self.prediction_model.train()
        self.optimizer.zero_grad()
        
        prediction = self.prediction_model(input_tensor.unsqueeze(0))
        loss = self.loss_function(prediction.squeeze(), target_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_optimal_production(self, commune: Commune) -> Dict[ProductionCategory, float]:
        if len(commune.historical_production) < 1:
            return commune.production_capacity.copy()
            
        self.prediction_model.eval()
        with torch.no_grad():
            input_tensor, _ = self.prepare_training_data(commune)
            if input_tensor is None:
                return commune.production_capacity.copy()
            
            if input_tensor.shape[0] != self.prediction_model.layer1.in_features:
                print(f"维度不匹配: 输入{input_tensor.shape[0]}, 期望{self.prediction_model.layer1.in_features}")
                if input_tensor.shape[0] < self.prediction_model.layer1.in_features:
                    padding = torch.zeros(self.prediction_model.layer1.in_features - input_tensor.shape[0])
                    input_tensor = torch.cat([input_tensor, padding])
                else:
                    input_tensor = input_tensor[:self.prediction_model.layer1.in_features]
            
            prediction = self.prediction_model(input_tensor.unsqueeze(0))
            predicted_values = prediction.squeeze().numpy()
            
            optimal_production = {}
            total_predicted = np.sum(predicted_values)
            
            for i, category in enumerate(ProductionCategory):
                if i < len(predicted_values):
                    if total_predicted > 0:
                        proportion = predicted_values[i] / total_predicted
                        optimal_production[category] = proportion * sum(commune.production_capacity.values())
                    else:
                        optimal_production[category] = commune.production_capacity[category]
                else:
                    optimal_production[category] = commune.production_capacity[category]
            
            return optimal_production
    
    def initialize_network(self):
        communes = self.economic_model.communes
        for i, commune1 in enumerate(communes):
            for j, commune2 in enumerate(communes):
                if i != j and self.should_connect(commune1, commune2):
                    strength = self.calculate_connection_strength(commune1, commune2)
                    self.connection_strength[(commune1.id, commune2.id)] = strength
                    commune1.connections.append(commune2.id)
    
    def should_connect(self, commune1: Commune, commune2: Commune) -> bool:
        economic_complementarity = self.calculate_economic_complementarity(commune1, commune2)
        return (abs(commune1.level - commune2.level) <= 1 or 
                commune1.parent_id == commune2.id or 
                commune2.parent_id == commune1.id or
                economic_complementarity > 0.7)
    
    def calculate_economic_complementarity(self, commune1: Commune, commune2: Commune) -> float:
        complementarity = 0
        for category in ProductionCategory:
            production_ratio = commune1.production_capacity[category] / commune2.consumption_needs[category] if commune2.consumption_needs[category] > 0 else 0
            demand_ratio = commune2.production_capacity[category] / commune1.consumption_needs[category] if commune1.consumption_needs[category] > 0 else 0
            
            complementarity += min(production_ratio, 1.0) + min(demand_ratio, 1.0)
        
        return complementarity / (2 * len(ProductionCategory))
    
    def calculate_connection_strength(self, commune1: Commune, commune2: Commune) -> float:
        base_strength = self.calculate_economic_complementarity(commune1, commune2)
        learning_bonus = (commune1.learning_parameters['cooperation_tendency'] + 
                         commune2.learning_parameters['cooperation_tendency']) / 2
        return base_strength * (1 + learning_bonus)
    
    def propagate_signal(self, start_commune_id: int, signal_strength: float, signal_type: str = "adjustment"):
        visited = set()
        queue = deque([(start_commune_id, signal_strength, 0)])
        
        while queue:
            current_id, current_strength, depth = queue.popleft()
            if current_id in visited or current_strength < 0.05 or depth > 3:
                continue
                
            visited.add(current_id)
            current_commune = next(c for c in self.economic_model.communes if c.id == current_id)
            
            if signal_type == "adjustment":
                adjustment = current_strength * 0.1 * current_commune.learning_parameters['adaptation_rate']
                current_commune.activation_level = np.clip(
                    current_commune.activation_level + adjustment, 0.1, 1.0
                )
            elif signal_type == "innovation":
                current_commune.learning_parameters['innovation_bias'] = np.clip(
                    current_commune.learning_parameters['innovation_bias'] + current_strength * 0.05, 0.1, 0.5
                )
            
            for neighbor_id in current_commune.connections:
                if neighbor_id not in visited:
                    connection_key = (current_id, neighbor_id)
                    if connection_key in self.connection_strength:
                        decay_factor = 0.7 ** depth
                        strength = (current_strength * 
                                  self.connection_strength[connection_key] * 
                                  self.learning_rate * decay_factor)
                        queue.append((neighbor_id, strength, depth + 1))

class EnhancedPlanningSystem:
    def __init__(self):
        self.economic_model = MarxistEconomicModel()
        self.neural_network = AdvancedNeuralPlanNetwork(self.economic_model)
        self.plan_cycles = []
        self.accuracy_history = []
        
    def create_sample_communes(self, num_communes: int = 30):
        communes = []
        for i in range(num_communes):
            level = random.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0]
            parent_id = -1 if level == 3 else random.randint(0, num_communes-1)
            
            base_capacity = random.uniform(5000, 50000)
            production_capacity = {
                ProductionCategory.AGRICULTURE: base_capacity * random.uniform(0.8, 1.2),
                ProductionCategory.INDUSTRY: base_capacity * random.uniform(0.8, 1.2),
                ProductionCategory.SERVICES: base_capacity * random.uniform(0.8, 1.2),
                ProductionCategory.HIGH_TECH: base_capacity * random.uniform(0.1, 0.5)
            }
            
            commune = Commune(
                id=i,
                name=f"Commune_{i}",
                level=level,
                parent_id=parent_id,
                population=random.randint(10000, 100000),
                labor_force=random.randint(5000, 50000),
                capital_stock=random.uniform(1000000, 10000000),
                natural_resources={
                    'land': random.uniform(1000, 10000),
                    'minerals': random.uniform(0, 1000),
                    'energy': random.uniform(100, 5000)
                },
                production_capacity=production_capacity,
                consumption_needs={
                    cat: production_capacity[cat] * random.uniform(0.3, 0.7) 
                    for cat in ProductionCategory
                }
            )
            communes.append(commune)
        
        self.economic_model.communes = communes
        self.neural_network.initialize_network()
    
    def simulate_production_cycle(self, cycle_num: int, learning_phase: bool = False):
        print(f"开始模拟第 {cycle_num} 个生产周期...")
        
        self.economic_model.update_long_wave_phase()
        
        neural_predictions = {}
        total_prediction_accuracy = 0
        prediction_count = 0
        
        for commune in self.economic_model.communes:
            if commune.level == 0:
                try:
                    prediction = self.neural_network.predict_optimal_production(commune)
                    neural_predictions[commune.id] = prediction
                    
                    if len(commune.historical_production) > 0:
                        actual = commune.current_production
                        prediction_error = 0
                        valid_categories = 0
                        for cat in ProductionCategory:
                            if actual[cat] > 0:
                                error = abs(prediction[cat] - actual[cat]) / actual[cat]
                                prediction_error += error
                                valid_categories += 1
                        
                        if valid_categories > 0:
                            accuracy = 1 - (prediction_error / valid_categories)
                            total_prediction_accuracy += accuracy
                            prediction_count += 1
                except Exception as e:
                    print(f"Commune {commune.id} prediction failed: {e}")
                    neural_predictions[commune.id] = commune.production_capacity.copy()
        
        if prediction_count > 0:
            avg_accuracy = total_prediction_accuracy / prediction_count
            self.accuracy_history.append(avg_accuracy)
            self.economic_model.prediction_accuracy_history.append(avg_accuracy)
            print(f"Neural network prediction accuracy: {avg_accuracy:.3f}")
        
        base_production_data = self.collect_base_data(neural_predictions if not learning_phase else None)
        aggregated_data = self.aggregate_data(base_production_data)
        national_plan = self.create_national_plan(aggregated_data)
        execution_results = self.execute_plan(national_plan, neural_predictions if not learning_phase else None)
        
        if not learning_phase:
            self.neural_learning_phase(execution_results)
        
        self.neural_adjustment(execution_results)
        self.record_cycle_results(cycle_num, execution_results)
        
        return execution_results
    
    def collect_base_data(self, neural_predictions: Dict = None) -> Dict[int, Dict]:
        base_data = {}
        for commune in self.economic_model.communes:
            if commune.level == 0:
                actual_production = {}
                
                for category in ProductionCategory:
                    capacity = commune.production_capacity[category]
                    
                    if neural_predictions and commune.id in neural_predictions:
                        predicted = neural_predictions[commune.id][category]
                        base_target = predicted * 0.7 + capacity * 0.3
                    else:
                        base_target = capacity
                    
                    fluctuation = random.uniform(0.8, 1.2)
                    activation_effect = 0.5 + commune.activation_level * 0.5
                    
                    actual_production[category] = base_target * fluctuation * activation_effect
                
                base_data[commune.id] = actual_production
                commune.current_production = actual_production
                
                commune.historical_production.append(actual_production.copy())
                if len(commune.historical_production) > 10:
                    commune.historical_production.pop(0)
        
        return base_data
    
    def neural_learning_phase(self, execution_results: Dict):
        print("Neural network learning...")
        total_loss = 0
        learning_count = 0
        
        for commune in self.economic_model.communes:
            if commune.level == 0 and len(commune.historical_production) >= 2:
                try:
                    loss = self.neural_network.train_prediction_model(commune)
                    if loss != float('inf'):
                        total_loss += loss
                        learning_count += 1
                except Exception as e:
                    print(f"Commune {commune.id} training failed: {e}")
        
        if learning_count > 0:
            avg_loss = total_loss / learning_count
            print(f"Average training loss: {avg_loss:.6f}")
    
    def aggregate_data(self, base_data: Dict) -> Dict[int, Dict]:
        aggregated = {}
        
        for level in range(1, 4):
            level_communes = [c for c in self.economic_model.communes if c.level == level]
            for commune in level_communes:
                subordinates = [c for c in self.economic_model.communes 
                              if c.parent_id == commune.id]
                
                if subordinates:
                    aggregated_production = {cat: 0 for cat in ProductionCategory}
                    for sub in subordinates:
                        if sub.id in base_data:
                            for cat, value in base_data[sub.id].items():
                                aggregated_production[cat] += value
                    
                    aggregated[commune.id] = aggregated_production
        
        return aggregated
    
    def create_national_plan(self, aggregated_data: Dict) -> Dict:
        total_national_production = {cat: 0 for cat in ProductionCategory}
        
        for commune_data in aggregated_data.values():
            for cat, value in commune_data.items():
                total_national_production[cat] += value
        
        plan = {}
        total_population = sum(c.population for c in self.economic_model.communes)
        
        wave_effect = np.sin(self.economic_model.long_wave_phase)
        per_capita_needs = {
            ProductionCategory.AGRICULTURE: 0.4 + 0.1 * wave_effect,
            ProductionCategory.INDUSTRY: 0.3 + 0.1 * wave_effect,
            ProductionCategory.SERVICES: 0.2 + 0.1 * abs(wave_effect),
            ProductionCategory.HIGH_TECH: 0.1 + 0.05 * (1 + wave_effect)
        }
        
        total_weight = sum(per_capita_needs.values())
        per_capita_needs = {k: v/total_weight for k, v in per_capita_needs.items()}
        
        for category in ProductionCategory:
            social_need = total_population * per_capita_needs[category] * 10
            current_production = total_national_production[category]
            
            adjustment_factor = 1.0
            if current_production > 0:
                adjustment_factor = social_need / current_production
            
            innovation_effect = 1.0 + self.economic_model.innovation_cluster * random.uniform(0.8, 1.2)
            
            plan[category] = {
                'target': social_need,
                'adjustment': adjustment_factor * innovation_effect,
                'accumulation_share': self.economic_model.accumulation_rate,
                'wave_phase': self.economic_model.long_wave_phase
            }
        
        return plan
    
    def execute_plan(self, national_plan: Dict, neural_predictions: Dict = None) -> Dict:
        results = {}
        
        for commune in self.economic_model.communes:
            if commune.level == 0:
                commune_results = {}
                for category in ProductionCategory:
                    current = commune.current_production[category]
                    adjustment = national_plan[category]['adjustment']
                    
                    if neural_predictions and commune.id in neural_predictions:
                        neural_suggestion = neural_predictions[commune.id][category]
                        blended_target = (current * adjustment * 0.6 + neural_suggestion * 0.4)
                    else:
                        blended_target = current * adjustment
                    
                    capacity = commune.production_capacity[category]
                    actual_production = min(blended_target, capacity * 1.15)
                    
                    commune_results[category] = {
                        'planned': blended_target,
                        'actual': actual_production,
                        'efficiency': actual_production / blended_target if blended_target > 0 else 0,
                        'capacity_utilization': actual_production / capacity if capacity > 0 else 0
                    }
                
                results[commune.id] = commune_results
        
        return results
    
    def neural_adjustment(self, execution_results: Dict):
        efficiencies = []
        for commune_id, results in execution_results.items():
            avg_efficiency = np.mean([r['efficiency'] for r in results.values()])
            efficiencies.append((commune_id, avg_efficiency))
        
        efficiencies.sort(key=lambda x: x[1])
        
        if efficiencies:
            worst_commune = efficiencies[0][0]
            best_commune = efficiencies[-1][0]
            
            self.neural_network.propagate_signal(worst_commune, 0.5, "adjustment")
            self.neural_network.propagate_signal(best_commune, 0.3, "innovation")
    
    def record_cycle_results(self, cycle_num: int, results: Dict):
        total_efficiency = 0
        total_production = {cat: 0 for cat in ProductionCategory}
        total_capacity_utilization = 0
        count = 0
        
        for commune_results in results.values():
            for category, result in commune_results.items():
                total_efficiency += result['efficiency']
                total_production[category] += result['actual']
                total_capacity_utilization += result['capacity_utilization']
                count += 1
        
        avg_efficiency = total_efficiency / count if count > 0 else 0
        avg_capacity_utilization = total_capacity_utilization / count if count > 0 else 0
        
        is_balanced, balance_ratio = self.economic_model.check_reproduction_balance(total_production)
        
        cycle_data = {
            'cycle': cycle_num,
            'efficiency': avg_efficiency,
            'production': total_production.copy(),
            'capacity_utilization': avg_capacity_utilization,
            'surplus_value': sum(self.economic_model.calculate_surplus_value(c) for c in self.economic_model.communes),
            'reproduction_balanced': is_balanced,
            'balance_ratio': balance_ratio,
            'profit_rate': np.mean([self.economic_model.calculate_profit_rate(c) for c in self.economic_model.communes]),
            'prediction_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0
        }
        
        self.economic_model.historical_data.append(cycle_data)
        self.economic_model.cycles_completed += 1

    def run_extended_simulation(self, total_cycles: int = 80):
        print("Initializing Enhanced Planned Economy Simulation System...")
        self.create_sample_communes(30)
        
        print(f"Starting {total_cycles} cycle simulation...")
        
        # Phase 1: Basic Learning (1-20 cycles)
        print("\n=== Phase 1: Basic Learning (1-20 cycles) ===")
        for cycle in range(1, 21):
            start_time = time.time()
            try:
                self.simulate_production_cycle(cycle, learning_phase=True)
                elapsed = time.time() - start_time
                current_data = self.economic_model.historical_data[-1]
                print(f"Cycle {cycle:2d} completed | Efficiency: {current_data['efficiency']:.3f} | Time: {elapsed:.2f}s")
            except Exception as e:
                print(f"Cycle {cycle} failed: {e}")
                break
        
        # Phase 2: Neural Network Optimization (21-60 cycles)
        print("\n=== Phase 2: Neural Network Optimization (21-60 cycles) ===")
        for cycle in range(21, 61):
            start_time = time.time()
            try:
                self.simulate_production_cycle(cycle, learning_phase=False)
                elapsed = time.time() - start_time
                current_data = self.economic_model.historical_data[-1]
                print(f"Cycle {cycle:2d} completed | Efficiency: {current_data['efficiency']:.3f} | "
                      f"Prediction Accuracy: {current_data.get('prediction_accuracy', 0):.3f} | Time: {elapsed:.2f}s")
            except Exception as e:
                print(f"Cycle {cycle} failed: {e}")
                break
        
        # Phase 3: Stable Operation (61-80 cycles)
        print("\n=== Phase 3: Stable Operation (61-80 cycles) ===")
        for cycle in range(61, total_cycles + 1):
            start_time = time.time()
            try:
                self.simulate_production_cycle(cycle, learning_phase=False)
                elapsed = time.time() - start_time
                current_data = self.economic_model.historical_data[-1]
                print(f"Cycle {cycle:2d} completed | Efficiency: {current_data['efficiency']:.3f} | "
                      f"Profit Rate: {current_data['profit_rate']:.3f} | Time: {elapsed:.2f}s")
            except Exception as e:
                print(f"Cycle {cycle} failed: {e}")
                break

    def visualize_comprehensive_results(self):
        """修复可视化函数 - 使用英文标签避免字体问题"""
        if len(self.economic_model.historical_data) < 2:
            print("Not enough historical data for visualization")
            return
        
        data = self.economic_model.historical_data
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Marxist Planned Economy Simulation - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. Efficiency Trend
        cycles = [d['cycle'] for d in data]
        efficiencies = [d['efficiency'] for d in data]
        axes[0, 0].plot(cycles, efficiencies, 'b-', linewidth=2)
        axes[0, 0].set_title('Production Efficiency Trend')
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Efficiency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Production Structure
        categories = list(ProductionCategory)
        production_data = {cat: [d['production'][cat] for d in data] for cat in categories}
        
        bottom = np.zeros(len(cycles))
        colors = ['#2E8B57', '#4682B4', '#FF6347', '#FFD700']
        
        for i, cat in enumerate(categories):
            axes[0, 1].bar(cycles, production_data[cat], bottom=bottom, 
                          label=cat.value, color=colors[i], alpha=0.8)
            bottom += production_data[cat]
        
        axes[0, 1].set_title('Production Structure Evolution')
        axes[0, 1].set_xlabel('Cycle')
        axes[0, 1].set_ylabel('Total Production')
        axes[0, 1].legend()
        
        # 3. Prediction Accuracy
        if any('prediction_accuracy' in d for d in data):
            accuracies = [d.get('prediction_accuracy', 0) for d in data]
            axes[1, 0].plot(cycles, accuracies, 'g-', linewidth=2)
            axes[1, 0].set_title('Neural Network Prediction Accuracy')
            axes[1, 0].set_xlabel('Cycle')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Profit Rate Changes
        profit_rates = [d['profit_rate'] for d in data]
        axes[1, 1].plot(cycles, profit_rates, 'purple', linewidth=2)
        axes[1, 1].set_title('Average Profit Rate Changes')
        axes[1, 1].set_xlabel('Cycle')
        axes[1, 1].set_ylabel('Profit Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 运行模拟
if __name__ == "__main__":
    print("Starting Fixed Marxist Planned Economy Simulation System...")
    
    planning_system = EnhancedPlanningSystem()
    
    # 运行80周期模拟
    start_time = time.time()
    planning_system.run_extended_simulation(80)
    total_time = time.time() - start_time
    
    print(f"\nSimulation completed! Total time: {total_time:.2f} seconds")
    
    # 生成可视化报告
    print("\nGenerating visualization report...")
    planning_system.visualize_comprehensive_results()
    
    print("\nSimulation system finished!")
