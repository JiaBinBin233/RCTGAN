import numpy as np
import pandas as pd
# DataSampler为CTGAN对条件向量和相应的数据进行采样
import torch


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data_set for CTGAN."""

    def __init__(self, data, output_info, log_frequency):
        # data_set = pd.DataFrame(data_set)
        # data_set.to_csv("../dataset/1234.csv")
        self._data = data

        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == "softmax")

        n_discrete_columns = sum(
            [1 for column_info in output_info if is_discrete_column(column_info)])    # 获得离散列的个数

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype="int32")

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim # ed=2

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])   # 转换为独热编码之后分别取出两列来
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed  # st=2
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]   # 断言函数  当表达式错误时触发异常

        # Prepare an interval matrix for efficiently sample conditional vector  为有效采样条件向量准备区间矩阵
        max_category = max(
            [column_info[0].dim for column_info in output_info  # 取离散列的最大维度
             if is_discrete_column(column_info)], default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(
            n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros(    # np.zeros((1, 2))=[[0,0]]
            (n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum(                           # _n_categories=2
            [column_info[0].dim for column_info in output_info
             if is_discrete_column(column_info)])
        # print(self._n_categories)

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = (
                    category_prob)
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    # 为训练生成条件向量
    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector. 条件向量 是包含所有离散列的One-hot编码，除了我们希望生成的样本满足的条件的离散列中的（一个）类别之外，所有值都是零。
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.  指示所选离散列的 one-hot 向量
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        # print('2')
        if self._n_discrete_columns == 0:
            return None

        # 在离散列中随机取batch个列
        # self._n_discrete_columns=1
        # discrete_column_id 表示离散列的id号，本实验就一个离散列 并且在第一行 所有这个变量的值是0
        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch)
        # discrete_column_id=[0,0,0,0,0````0,0,0````]  500个

        # self._n_categories=2
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1  # mask[c,l]=1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (self._discrete_column_cond_st[discrete_column_id]
                       + category_id_in_col)
        cond[np.arange(batch), category_id] = 1
        # print('cond:',cond)
        # print('mask:',mask)
        # print('discrete_column_id:',discrete_column_id)
        # print('category_id_in_col:',category_id_in_col)

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    # 从原始训练集中采样满足被采样的条件向量的数据
    def sample_data(self, n, col, opt):
        """Sample data_set from original training data_set satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data_set.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))
        # print(idx)
        return self._data[idx]

    def dim_cond_vec(self):
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id = self._discrete_column_matrix_st[condition_info["discrete_column_id"]
                                             ] + condition_info["value_id"]
        vec[:, id] = 1
        return vec
