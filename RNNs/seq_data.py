import random
import torch


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        self.batch_size, self.num_steps = batch_size, num_steps
        self.load_corpus_time_machine(max_tokens)
        if use_random_iter:
            self.data_iter_fn = self.seq_data_iter_random
        else:
            self.data_iter_fn = self.seq_data_iter_sequential

    def seq_data_iter_random(self):
        """使用随机采样生成一个小批量子序列。"""
        corpus = self.corpus[random.randint(0, self.num_steps - 1):]
        num_seques = (len(corpus) - 1) // self.num_steps
        initial_indices = list(range(0, num_seques * self.num_steps, self.num_steps))
        random.shuffle(initial_indices)

        def data(pos):
            return corpus[pos, pos + self.num_steps]

        num_batch = num_seques // self.batch_size
        for i in range(0, num_batch * self.batch_size, self.batch_size):
            initial_indices_per_batch = initial_indices[i: i + self.batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            yield torch.tensor(X), torch.tensor(Y)

    def seq_data_iter_sequential(self):
        offset = random.randint(0, self.num_steps - 1)
        num_token = (len(self.corpus) - 1 - offset) // self.batch_size * self.batch_size
        Xs = torch.tensor(self.corpus[offset, offset + num_token]).reshape(self.batch_size, -1)
        Ys = torch.tensor(self.corpus[offset + 1, offset + 1 + num_token]).reshape(self.batch_size, -1)
        num_batch = num_token // self.batch_size
        for i in range(0, num_batch * self.num_steps, self.num_steps):
            X = Xs[:, i: i + self.num_steps]
            Y = Ys[:, i: i + self.num_steps]
            yield X, Y

    def load_corpus_time_machine(self, max_tokens):
        pass
