import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from typing import Literal

class Plotter: 
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
        self.losses_data_by_label = {}  # 存储所有损失数据
        self.results_data_by_label = {}  # 存储所有结果数据
        self.results_data_by_x = {}  # 存储所有结果数据，按x值分组
    
    def clear_data(self):
        self.losses_data_by_label.clear()
        self.results_data_by_label.clear()
        self.results_data_by_x.clear()
    
    def add_losses_by_label(self, losses: list[float], label: str):
        self.losses_data_by_label[label] = {
            'label': label,
            'losses': losses,
        }

    def add_results_by_label(self, results: dict[str, list], label: str):
        self.results_data_by_label[label] = {
            'label': label,
            'results': results,
        }

    def add_results_by_x(self, results: dict[str, list], x: str):
        self.results_data_by_x[x] = {
            'x': x,
            'results': results,
        }
    
    def _compute_best_metrics(self, results: dict[str, list], primary_key: str = 'NDCG@20'):
        primary_values = results[primary_key]
        best_epoch_idx = max(range(len(primary_values)), key=lambda i: primary_values[i])
        best_metrics = {name: results[name][best_epoch_idx] for name in results.keys()}
        return best_metrics
    
    def show_results_by_x(self, type: Literal['separate', 'cumulative'] = 'separate'):
        if not self.results_data_by_x:
            raise ValueError("No results data by x to plot.")
        
        if type == 'separate':
            self._show_results_by_x_separate()
        elif type == 'cumulative':
            self._show_results_by_x_cumulative()
        else:
            raise ValueError(f"Invalid type: {type}. Expected 'separate' or 'cumulative'.")

    def _show_results_by_x_separate(self):
        if not self.results_data_by_x:
            raise ValueError("No results data by x to plot.")
        
        # 收集所有度量指标和x值
        lines = {}
        x_values_set = set()
        
        for x, data in self.results_data_by_x.items():
            x_values_set.add(x)
            metric_names = data['results'].keys()
            best_metrics = self._compute_best_metrics(data['results'])
            for metric_name in metric_names:
                if metric_name not in lines:
                    lines[metric_name] = {}
                lines[metric_name][x] = best_metrics[metric_name]
        
        # 标准化x值顺序
        x_values = sorted(list(x_values_set))
        metric_names = list(lines.keys())
        
        # 创建图表
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 设置柱状图的宽度和位置
        bar_width = 0.8 / len(metric_names)  # 每个度量指标的柱宽
        x_positions = np.arange(len(x_values))
        
        # 为每个度量指标绘制柱状图
        for i, metric_name in enumerate(metric_names):
            values = [lines[metric_name].get(x, 0) for x in x_values]
            offset = (i - len(metric_names) / 2 + 0.5) * bar_width
            ax.bar(x_positions + offset, values, bar_width, label=metric_name)
        
        # 设置标签和标题
        ax.set_xlabel('X')
        ax.set_ylabel('Best Metric Value')
        ax.set_title('Best Metrics by X')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_values)
        ax.legend()
        fig.tight_layout()

    def _show_results_by_x_cumulative(self):
        if not self.results_data_by_x:
            raise ValueError("No results data by x to plot.")
        lines = {}
        x_axis = list(self.results_data_by_x.keys())
        for x, data in self.results_data_by_x.items():
            metric_names = data['results'].keys()
            best_metrics = self._compute_best_metrics(data['results'])
            for metric_name in metric_names:
                if metric_name not in lines:
                    lines[metric_name] = [best_metrics[metric_name]]
                else:
                    lines[metric_name].append(best_metrics[metric_name])

        fig, ax = plt.subplots(figsize=self.figsize)
        for metric_name, values in lines.items():
            ax.plot(x_axis, values, marker='o', label=metric_name)
        ax.set_xlabel('X')
        ax.set_ylabel('Best Metric Value')
        ax.set_title('Best Metrics by X')
        ax.legend()
        fig.tight_layout()

    def show_losses(self):
        if not self.losses_data_by_label:
            raise ValueError("No losses data to plot.")
        fig, ax = plt.subplots(figsize=self.figsize)
        for _, data in self.losses_data_by_label.items():
            ax.plot(data['losses'], label=data['label'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs')
        ax.legend()
        fig.tight_layout()
    
    def show_performance_comparison(self):
        if not self.results_data_by_label:
            raise ValueError("No results data to plot.")
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison')
        
        lines = []
        line_info_list = []
        
        for label, data in self.results_data_by_label.items():
            metric_names = data['results'].keys()
            best_metrics = self._compute_best_metrics(data['results'])
            best_values = [best_metrics[name] for name in metric_names]
            
            # 画折线
            line, = ax.plot(metric_names, best_values, marker='o', label=label)
            lines.append(line)
            
            # 保存信息用于图例
            line_info_list.append({
                'label': label,
                'best_metrics': best_metrics,
                'color': line.get_color()
            })

        ax.set_ylim(bottom=0.0, top=max(v for info in line_info_list for v in info['best_metrics'].values()) * 1.1)
        # 右上角：模型名称图例
        ax.legend(loc='upper right', fontsize=9)
        legend1 = ax.legend(loc='upper left', fontsize=9)
        ax.add_artist(legend1)
        self._add_best_metrics(ax, line_info_list, location='lower left')
        
        fig.tight_layout()

    def show_avg_metric_evolution(self):
        if not self.results_data_by_label:
            raise ValueError("No results data to plot.")
        fig, ax = plt.subplots(figsize=self.figsize)
        line_info_list = []
        for key in self.results_data_by_label.keys():
            results = self.results_data_by_label[key]['results']
            avg_line = sum([np.array(v) for v in results.values()]) / len(results)
            line = ax.plot(avg_line, label=self.results_data_by_label[key]['label'])
            line_info_list.append({
                'label': self.results_data_by_label[key]['label'],
                'best_metrics': self._compute_best_metrics(results),
                'color': line[0].get_color()
            })

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Metric Value')
        ax.set_title('Average Metric Value Evolution Over Epochs')
        legend1 = ax.legend(loc='upper left', fontsize=9)
        ax.add_artist(legend1)
        self._add_best_metrics(ax, line_info_list, location='lower right')
        fig.tight_layout()


    def _add_best_metrics(self, ax, line_info_list: list[dict], location: str = 'lower left'):     
        
        if line_info_list:
            custom_handles = []
            custom_labels = []
            
            for info in line_info_list:
                handle = Line2D([0], [0], marker='o', color=info['color'], 
                               markersize=4, linestyle='')
                metrics_str = " | ".join([f"{name}: {info['best_metrics'][name]:.4f}" 
                                          for name in info['best_metrics'].keys()])
                custom_handles.append(handle)
                custom_labels.append(metrics_str)
            
            ax.legend(custom_handles, custom_labels, loc=location,
                      fontsize=6, framealpha=0.8, 
                      prop={'family': 'monospace', 'size': 8},
                      handletextpad=0.3, labelspacing=0.2)
            
def test_plotter():
    """测试 Plotter 类"""

    # 创建绘图器
    plotter = Plotter(figsize=(12, 6))
    # 假设你有多个模型的训练结果
    results_model_a = {
        'Recall@10': [0.1, 0.12, 0.15, 0.14],
        'Recall@20': [0.15, 0.18, 0.22, 0.21],
        'NDCG@10': [0.08, 0.10, 0.12, 0.11],
        'NDCG@20': [0.10, 0.13, 0.16, 0.15],  # epoch 2 最大
        "dd": [0.1, 0.2, 0.3, 0.4],
    }

    results_model_b = {
        'Recall@10': [0.11, 0.14, 0.13, 0.15],
        'Recall@20': [0.16, 0.20, 0.19, 0.22],
        'NDCG@10': [0.09, 0.11, 0.10, 0.12],
        'NDCG@20': [0.12, 0.15, 0.14, 0.16],  # epoch 1 最大
        "dd": [0.2, 0.3, 0.4, 0.5],
    }

    results_model_c = {
        'Recall@10': [0.09, 0.13, 0.14, 0.16],
        'Recall@20': [0.14, 0.17, 0.21, 0.23],
        'NDCG@10': [0.07, 0.09, 0.11, 0.13],
        'NDCG@20': [0.11, 0.14, 0.17, 0.19],  # epoch 3 最大
        "dd": [0.15, 0.25, 0.35, 0.45],
    }

    model1_losses = [2.0, 1.8, 1.5, 1.4]
    model2_losses = [2.1, 1.9, 1.6, 1.5]
    model3_losses = [2.2, 2.0, 1.7, 1.6]
    # 添加损失数据
    plotter.add_losses(model1_losses, label='LightGCN')
    plotter.add_losses(model2_losses, label='EASE')
    plotter.add_losses(model3_losses, label='Model C')
    # 显示损失曲线
    print("显示损失曲线...")
    plotter.show_losses()

    plotter.add_results(results_model_a, label='LightGCN')
    plotter.add_results(results_model_b, label='EASE')
    plotter.add_results(results_model_c, label='Model C')
    
    # 显示最佳 epoch 对比图
    print("\n显示最佳 epoch 对比图...")
    plotter.show_performance_comparison()
    
    # 显示指标随 epoch 演变的曲线图
    print("显示指标演变曲线图...")
    plotter.show_avg_metric_evolution()
    plt.show()

if __name__ == "__main__":
    test_plotter()