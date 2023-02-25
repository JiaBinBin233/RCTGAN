import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim, nn
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from ctgan1.data_sampler import DataSampler
from ctgan1.data_transformer import DataTransformer
from matplotlib import pyplot as plt
"""
The key improvements over previous TGAN are applying the mode-specific normalization to overcome the non-Gaussian and multimodal distribution. 
Then a conditional generator and training-by-sampling to deal with the imbalanced discrete columns.

网络架构：本文中使用WGANGP+PacGAN
将其与三种替代品进行比较，仅WGANGP、仅原始GAN损失和原始GAN+PacGAN。我们观察到，WGANP比原始GAN更适合于合成数据任务，而PacGAN有助于原始GAN的损失，但对WGANP不那么重要。

判别器：
参考 PacGAN，将 pac 个样本作为一个包（packet），以期防止模式坍缩（mode collapse）
损失函数的定义参考 WGAN-GP（在 WGAN 原本的损失函数的基础上加入梯度惩罚（gradient penalty）以使训练收敛）

生成器：
损失函数的定义基于 WGAN 再加上部分的交叉熵 （生成器要最小化Wasserstein距离）
"""

# 增加的分类器
class Qdiscriminator(Module):
    def __init__(self, input_dim):
        super(Qdiscriminator, self).__init__()
        self.model = Sequential(
            Linear(input_dim, 256),
            LeakyReLU(0.2),
            Dropout(0.5),

            Linear(256, 128),
            LeakyReLU(0.2),
            Dropout(0.5),
        )

        self.linear1 = Sequential(
            Linear(128, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        out = self.model(input)
        out1 = self.linear1(out)
        return out1

class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim , 1))
        self.seq = Sequential(*seq)

   
    def calc_gradient_penalty(self, real_data, fake_data, device="cpu", pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))

class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

class CTGANSynthesizer(object):
# class CTGANSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data_set using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data_set samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        # self.c_dim = c_dim

        # 对cuda进行判断
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    # Deals with the instability of the gumbel_softmax for older versions of torch  处理旧版本torch的不稳定性
    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    # Apply proper activation function to the output of the generator  给生成器的输出应用一定的激活性
    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    assert 0

        return torch.cat(data_t, dim=1)

    # Check whether ``discrete_columns`` exists in ``train_data``  检查离散列是否在数据中
    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError('Invalid columns found: {}'.format(invalid_columns))

    # Compute the cross entropy loss on the fixed discrete column.  计算离散列上的交叉熵损失
    def _cond_loss(self, data, c, m):

        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    # 对训练集数据进行训练
    def fit(self, train_data, discrete_columns=tuple(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data_set.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        # 检查离散列是否在数据表中
        self._validate_discrete_columns(train_data, discrete_columns)
        epochs = self._epochs

        """
            使用BayesianGMM对连续列进行建模，并将其规格化为标量[0，1]和向量
            离散列使用OneHotEncoder进行编码
        """
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)   # 对训练集数据建模
        train_data = self._transformer.transform(train_data)

        # DataSampler为CTGAN对条件向量和相应的数据进行采样
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions  # 输出训练数据转换之后总共的维度  data_dim=70

        # 实例化生成器  生成器可以被解释为给定的特定列特定值 行的条件分布
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        # 实例化辨别器
        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        qdiscriminator = Qdiscriminator(
            data_dim + self._data_sampler.dim_cond_vec()
        ).to(self._device)

        # 创建生成器优化器
        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        # 创建辨别器优化器
        optimizerD = optim.Adam(
            [{'params': discriminator.parameters()}, {'params': qdiscriminator.parameters()}],
            lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        # add cross entropy loss function
        loss_function = nn.CrossEntropyLoss()
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)  # 矩阵500行128列
        std = mean + 1
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        loss_g_list = []
        loss_d_list = []

        for i in range(epochs):
            total_loss_g = 0
            total_loss_d = 0
            total_closs_d = 0
            total_closs_g = 0

            for id_ in range(steps_per_epoch):
                # for n in range(self._discriminator_steps):
                    # _discriminator_steps = 1
                ##################################训练辨别器################################
                #从正态分布中随机
                fakez = torch.normal(mean=mean, std=std)   # 产生[500,128]的符合高斯(正态)分布得到随机矩阵

                condvec = self._data_sampler.sample_condvec(self._batch_size)   #如果没有离散列则返回None  条件生成器为不平衡列做准备的

                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)  # 将数据类型转换为张量的形式
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

                perm = np.arange(self._batch_size)
                np.random.shuffle(perm)                # 打乱顺序函数
                real = self._data_sampler.sample_data(self._batch_size, col[perm], opt[perm])   # real = [500,]
                c2 = c1[perm]

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)   # 对生成的数据进行激活操作

                real = torch.from_numpy(real.astype('float32')).to(self._device)  # 对真实数据进行数据格式转换

                fake_cat = torch.cat([fakeact, c1], dim=1)  # 先加入c1样本生成假数据，用真实的laber_vector引导生成
                real_cat = torch.cat([real, c2], dim=1)

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                #############################################################
                c3 = torch.full((self._batch_size, 1), 0)
                # c3 = torch.zeros(self._batch_size).unsqueeze(1)
                real_label = torch.cat([c2, c3], dim=1)
                real_label = torch.argmax(real_label, dim=1)
                pre_real = qdiscriminator(real_cat)
                real_loss = loss_function(pre_real, real_label)

                c4 = torch.full((self._batch_size, 1), 2)
                fake_label = torch.cat([c1, c4], dim=1)
                fake_label = torch.argmax(fake_label, dim=1)
                pre_fake = qdiscriminator(fake_cat)
                fake_loss = loss_function(pre_fake, fake_label)

                q_loss = real_loss + fake_loss
                ###################################################
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                total_loss_d += loss_d.item()
                total_closs_d += q_loss.item()

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward(retain_graph=True)
                q_loss.backward()
                optimizerD.step()

                ##########################训练生成器####################################
                ##########################训练生成器####################################
                ##########################训练生成器####################################
                ##########################训练生成器####################################
                fakez = torch.normal(mean=mean, std=std)
                # Generate the conditional vector for training
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                #
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fakecat = torch.cat([fakeact, c1], dim=1)
                y_fake = discriminator(fakecat)
                ################ add classification ###############
                c5 = torch.full((self._batch_size, 1), 0)
                real = torch.cat([c1, c5], dim=1)
                real = torch.argmax(real, dim=1)
                pre_g = qdiscriminator(fakecat)
                loss_fake = loss_function(pre_g, real)
                ###################################################

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy


                optimizerG.zero_grad()
                loss_g.backward(retain_graph=True)
                loss_fake.backward()
                optimizerG.step()
                ###############################
                total_loss_g += loss_g.item()
                total_closs_g += loss_fake.item()
                ###############################
            total_loss_g /= steps_per_epoch
            total_loss_d /= steps_per_epoch
            total_closs_d /= steps_per_epoch
            total_closs_g /= steps_per_epoch

            print("Epoch {}  Loss G:{:.3f}  Loss D:{:.3f} Loss C_D:{:.3f} Loss C_G:{:.3f}"
                  .format(i+1, total_loss_g, total_loss_d, total_closs_d, total_closs_g))

            loss_d_list.append(total_loss_d)
            loss_g_list.append(total_loss_g)

        plt.figure()
        ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        l1, = plt.plot(np.linspace(1, epochs, num=epochs), loss_d_list, 'green', linewidth=1)
        l2, = plt.plot(np.linspace(1, epochs, num=epochs), loss_g_list, 'red', linewidth=1)
        # l3, = plt.plot(np.linspace(0, epochs, num=epochs), loss_mmd_list, 'red',linestyle ='--', linewidth=1)
        plt.legend(handles=[l1,l2], labels=['D_loss','G_loss'], loc='best')
        plt.show()

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data_set similar to the training data_set.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
