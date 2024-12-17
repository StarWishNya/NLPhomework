import torch

class config():
    def __init__(self,
                 n_vocab=1002,  # 这里可以动态设置
                 embed_size=128,
                 hidden_size=128,
                 num_layers=3,
                 dropout=0.8,
                 num_classes=2,  # 这里可以动态设置
                 pad_size=32,
                 batch_size=128,
                 is_shuffle=True,
                 learn_rate=0.001,
                 num_epochs=100):
        self.n_vocab = n_vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.pad_size = pad_size
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = '../data/weibo_senti_100k.csv'
        self.stopwords_path = '../data/stopwords.txt'
