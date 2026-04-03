import torch

class Metric(object):
    def __init__(self, k_list=[10, 20], device='cpu'):
        """
        初始化 Metric 类
        :param k_list: 需要评估的 Top-K 列表，如 [10, 20]
        :param device: 运行设备
        """
        self.k_list = sorted(k_list)
        self.max_k = max(self.k_list)
        self.device = device
        
        # 预计算 NDCG 需要的折损因子 (1/log2(i+1))
        # 只需要计算到 max_k 即可
        # shape: [max_k]
        self.discounts = 1.0 / torch.log2(torch.arange(2, self.max_k + 2, device=self.device).float())
        
        # 预计算 IDCG 查找表 (用于隐式反馈)
        # 完美排序下，前 n 个位置都是 1，其 IDCG 就是前 n 个 discount 的累加
        # idcg_table[n] 表示只有 n 个正样本时的理想 DCG
        self.idcg_table = torch.cumsum(self.discounts, dim=0)
        # 补一个 0 在最前面，方便处理没有任何正样本的用户 (index=0)
        self.idcg_table = torch.cat([torch.tensor([0.0], device=self.device), self.idcg_table])

    def _mask_train_items(self, pred_matrix: torch.Tensor, train_matrix: torch.Tensor):
        """
        内部方法：将训练集中出现过的物品分数置为 -inf
        """
        return pred_matrix.masked_fill(train_matrix > 0, float('-inf'))

    def _calculate_recall(self, hits_k: torch.Tensor, ground_truth_count: torch.Tensor):
        """
        计算 Recall
        :param hits_k: Top-K 的命中矩阵 (0/1), shape: [Users, K]
        :param ground_truth_count: 每个用户真实喜欢的总数, shape: [Users]
        """
        # 分子：Top-K 中命中的数量
        hits_num = hits_k.sum(dim=1)
        # 分母：真实总数 (加 epsilon 防止除零)
        recall = hits_num / (ground_truth_count + 1e-10)
        return recall.mean().item()

    def _calculate_ndcg(self, hits_k, ground_truth_count, k):
        """
        计算 NDCG
        :param hits_k: Top-K 的命中矩阵 (0/1), shape: [Users, K]
        :param ground_truth_count: 每个用户真实喜欢的总数
        :param k: 当前的 K 值
        """
        # 1. 计算 DCG (分子)
        # 截取当前 k 个 discount
        discounts_k = self.discounts[:k]
        # 命中位置 * 折损权重
        dcg = (hits_k * discounts_k).sum(dim=1)

        # 2. 计算 IDCG (分母)
        # 隐式反馈中，最好的排列是把所有的 '1' 都排在最前面
        # 我们不能超过 k (因为最多只推荐 k 个)
        valid_count = torch.clamp(ground_truth_count, max=k).long()
        # 直接查表得到理想得分
        idcg = self.idcg_table[valid_count]

        # 3. 计算 NDCG
        ndcg = dcg / (idcg + 1e-10)
        return ndcg.mean().item()

    def eval(self, train_matrix: torch.Tensor, pred_matrix: torch.Tensor, test_matrix: torch.Tensor) -> dict[str, float]:
        """
        主评估函数
        :param train_matrix: 训练集交互 (Users, Items), 0/1
        :param pred_matrix: 模型预测分数 (Users, Items), float logits/probs
        :param test_matrix: 测试集交互 (Users, Items), 0/1
        """
        # 0. 数据转移到设备
        train_matrix = train_matrix.to(self.device)
        pred_matrix = pred_matrix.to(self.device)
        test_matrix = test_matrix.to(self.device)

        # 1. 剔除训练集数据 (Masking)
        masked_pred = self._mask_train_items(pred_matrix, train_matrix)

        # first return value is values, we don't need it
        # shape of topk_indices: [Users, max_k]，值是物品索引
        _, topk_indices = torch.topk(masked_pred, k=self.max_k, dim=1)

        # 根据index获取对应的tensor
        # shape of hits: [Users, max_k]，值是0/1
        hits = torch.gather(input=test_matrix, dim=1, index=topk_indices)

        # 4. 获取分母 (每个用户在测试集里的真实正样本数量)
        ground_truth_count = test_matrix.sum(dim=1)

        result = {}

        # 5. 遍历不同的 K 计算指标
        for k in self.k_list:
            # 截取前 k 列的命中情况
            current_hits = hits[:, :k]
            
            # 调用方法计算
            result[f'Recall@{k}'] = self._calculate_recall(current_hits, ground_truth_count)
            result[f'NDCG@{k}'] = self._calculate_ndcg(current_hits, ground_truth_count, k)

        return result



def test_metric():
    """测试 Metric 类"""
    # 模拟数据规模: 5个用户，10个物品
    N_USER, N_ITEM = 5, 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 准备三个矩阵
    # Train: 用户历史上看过的 (用来屏蔽)
    train_mat = torch.zeros(N_USER, N_ITEM)
    train_mat[0, [0, 1]] = 1 # 用户0看过物品0,1

    # Predict: 模型输出的 Logits
    pred_mat = torch.randn(N_USER, N_ITEM) 

    # Test: 用户未来看过的 (用来验证)
    test_mat = torch.zeros(N_USER, N_ITEM)
    test_mat[0, [2, 3]] = 1 # 用户0实际上喜欢物品2,3

    # 2. 初始化 Metric
    evaluator = Metric(k_list=[2, 5], device=device)

    # 3. 运行评估
    metrics = evaluator.eval(train_mat, pred_mat, test_mat)

    print("评估结果:", metrics)
if __name__ == "__main__":
    
    test_metric()
    