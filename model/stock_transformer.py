import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockTransformer(nn.Module):
    def __init__(self, input_features=6, seq_len=30, d_model=64, n_heads=4, num_layers=2, output_features=1):
        super(StockTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 输入特征映射到d_model维度
        self.input_projection = nn.Linear(input_features, d_model)
        
        # 位置编码
        self.positional_encoding = self._generate_positional_encoding(seq_len, d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_features)
        )
        
        # 数据标准化器
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _generate_positional_encoding(self, seq_len, d_model):
        """生成位置编码"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def preprocess_data(self, df, features=['open', 'close', 'high', 'low', 'volume', 'amount'], target='close', stock_id=None):
        """预处理股票数据，将其转换为序列数据

        Args:
            df: 股票数据DataFrame
            features: 特征列列表
            target: 目标列名称
            stock_id: 股票标识符，如果为None则处理单只股票

        Returns:
            如果stock_id为None: 返回(X, y)元组
            否则: 返回包含{stock_id: (X, y)}的字典
        """
        # 如果提供了stock_id，确保df有该列
        if stock_id is not None and stock_id not in df.columns:
            raise ValueError(f"DataFrame must contain '{stock_id}' column when processing multiple stocks")

        # 单只股票处理
        if stock_id is None:
            return self._preprocess_single_stock(df, features, target)

        # 多只股票处理
        result = {}
        for stock in df[stock_id].unique():
            stock_df = df[df[stock_id] == stock].copy()
            if len(stock_df) >= self.seq_len:
                result[stock] = self._preprocess_single_stock(stock_df, features, target)
            else:
                print(f"Warning: Stock {stock} has insufficient data ({len(stock_df)} rows < {self.seq_len} seq_len)")
        return result

    def _preprocess_single_stock(self, df, features, target):
        """处理单只股票数据"""
        # 选择特征列
        data = df[features].values
        
        # 标准化
        data_scaled = self.scaler.fit_transform(data)
        
        # 创建序列数据
        X, y = [], []
        for i in range(self.seq_len, len(data_scaled)):
            X.append(data_scaled[i-self.seq_len:i, :])
            y.append(data_scaled[i, features.index(target)])
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 输入特征映射
        x = self.input_projection(x)
        print(f"Input shape after projection: {x.shape}")
        
        # 动态生成位置编码以匹配输入序列长度
        positional_encoding = self._generate_positional_encoding(seq_len, self.d_model)
        print(f"Positional encoding shape before squeeze: {positional_encoding.shape}")
        
        # 去除多余的维度
        positional_encoding = positional_encoding.squeeze(1)
        print(f"Positional encoding shape after squeeze: {positional_encoding.shape}")
        
        # 添加位置编码
        x = x + positional_encoding
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取最后一个时间步的输出进行预测
        x = self.output_projection(x[:, -1, :])
        
        return x

    def predict(self, model, data_loader, device):
        """使用模型进行预测"""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                outputs = model(x)
                predictions.extend(outputs.cpu().numpy())
        
        # 反标准化预测结果
        # 注意：这里需要根据实际情况调整反标准化逻辑
        predictions = np.array(predictions).reshape(-1, 1)
        dummy = np.zeros((predictions.shape[0], self.scaler.n_features_in_))
        dummy[:, 0] = predictions[:, 0]
        predictions = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predictions

# 示例训练代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 示例1: 单只股票训练
    print("===== 单只股票训练示例 =====")
    # 模拟单只股票数据
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    single_stock_data = pd.DataFrame({
        'open': np.cumsum(np.random.randn(len(dates))),
        'close': np.cumsum(np.random.randn(len(dates))),
        'high': np.cumsum(np.random.randn(len(dates))),
        'low': np.cumsum(np.random.randn(len(dates))),
        'volume': np.random.randint(100000, 1000000, size=len(dates)),
        'amount': np.random.randint(1000000, 10000000, size=len(dates))
    }, index=dates)
    
    # 创建模型
    model = StockTransformer(input_features=6, seq_len=30, d_model=64, n_heads=4, num_layers=2)
    
    # 预处理单只股票数据
    X_single, y_single = model.preprocess_data(single_stock_data)
    print(f"单只股票数据预处理完成: X形状={X_single.shape}, y形状={y_single.shape}")
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    single_dataset = TensorDataset(X_single, y_single)
    single_data_loader = DataLoader(single_dataset, batch_size=32, shuffle=True)
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20  # 简化示例，减少训练轮数
    print("开始单只股票模型训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in single_data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(single_data_loader):.4f}')
    
    print('单只股票模型训练完成')
    
    # 示例2: 多只股票训练准备
    print("\n===== 多只股票训练示例 =====")
    # 模拟多只股票数据
    stocks = ['AAPL', 'GOOG', 'MSFT', 'TSLA']
    all_dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    multi_stock_data = []
    
    for stock in stocks:
        stock_df = pd.DataFrame({
            'date': all_dates,
            'stock_id': stock,
            'open': np.cumsum(np.random.randn(len(all_dates))),
            'close': np.cumsum(np.random.randn(len(all_dates))),
            'high': np.cumsum(np.random.randn(len(all_dates))),
            'low': np.cumsum(np.random.randn(len(all_dates))),
            'volume': np.random.randint(100000, 1000000, size=len(all_dates)),
            'amount': np.random.randint(1000000, 10000000, size=len(all_dates))
        })
        multi_stock_data.append(stock_df)
    
    multi_stock_df = pd.concat(multi_stock_data, ignore_index=True)
    print(f"生成多只股票数据: {len(stocks)}只股票，共{len(multi_stock_df)}条记录")
    
    # 预处理多只股票数据
    stock_data_dict = model.preprocess_data(multi_stock_df, stock_id='stock_id')
    print(f"多只股票数据预处理完成，成功处理{len(stock_data_dict)}只股票")
    
    # 这里可以继续实现多只股票的训练逻辑，例如为每只股票训练单独的模型
    # 或使用统一模型处理所有股票数据
    print("多只股票训练逻辑可以在此处实现")
    
    # 保存模型
    torch.save(model.state_dict(), 'stock_transformer.pth')
    print('模型已保存为 stock_transformer.pth')