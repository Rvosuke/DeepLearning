Generative models have many applications in machine learning. To list a few, they have been used to generate **high-fidelity** images , **synthesize realistic** speech and music **fragments** , improve the performance of semi-supervised learning , detect **adversarial** examples and other **anomalous** data , **imitation** learning , and explore promising states in reinforcement
learning . Recent progress is mainly driven by two approaches: likelihood-based methods and generative adversarial networks (GAN ). The former uses log-likelihood (or a suitable **surrogate**) as the training objective, while the latter uses adversarial training to minimize $f$ -**divergences**  or **integral** probability metrics  between model and data distributions.

Although likelihood-based models and GANs have achieved great success, they have some **intrinsic** limitations. For example, likelihood-based models either have to use specialized architectures to build a normalized probability model (e.g., autoregressive models, flow models), or use surrogate losses (e.g., the evidence lower bound used in **variational** auto-encoders , **contrastive divergence** in energy-based models ) for training. GANs avoid some of the limitations of likelihood-based models, but their training can be unstable due to the adversarial training procedure. In addition, the GAN objective is not suitable for evaluating and comparing different GAN models. While other objectives exist for generative modeling, such as noise contrastive estimation  and minimum probability flow , these methods typically only work well for low-dimensional data.

In this paper, we explore a new principle for generative modeling based on estimating and sampling from the (Stein) score  of the **logarithmic** data density, which is the gradient of the log-density function at the input data point. This is a vector field pointing in the direction where the log data density grows the most. We use a neural network trained with score matching  to learn this vector field from data. We then produce samples using Langevin dynamics, which approximately works by gradually moving a random initial sample to high density regions along the (estimated) vector field of scores. However, there are two main challenges with this approach. First, if the data distribution is supported on a low dimensional manifold—as it is often assumed for many real world datasets—the score will be undefined in the **ambient** space, and score matching will fail to provide a **consistent** score estimator. Second, the **scarcity** of training data in low data density regions, e.g., far from the manifold, **hinders** the accuracy of score estimation and slows down the mixing of Langevin dynamics sampling. Since Langevin dynamics will often be initialized in low-density regions of the data distribution, inaccurate score estimation in these regions will negatively affect the sampling process. Moreover, mixing can be difficult because of the need of traversing low density regions to
transition between modes of the distribution.

To tackle these two challenges, we propose to **perturb** the data with random Gaussian noise of various magnitudes. Adding random noise ensures the resulting distribution does not **collapse** to a low dimensional manifold. Large noise levels will produce samples in low density regions of the original (unperturbed) data distribution, thus improving score estimation. Crucially, we train a single score network conditioned on the noise level and estimate the scores at all noise **magnitudes**. We then propose an **annealed** version of Langevin dynamics, where we initially use scores corresponding to the highest noise level, and gradually anneal down the noise level until it is small enough to be **indistinguishable** from the original data distribution. Our sampling strategy is inspired by simulated annealing  which **heuristically** improves optimization for **multimodal landscapes**.

Our approach has several **desirable** properties. First, our objective is **tractable** for almost all **parameterizations** of the score networks without the need of special constraints or architectures, and can be optimized without adversarial training, MCMC sampling, or other approximations during training. The objective can also be used to **quantitatively** compare different models on the same dataset. Experimentally, we demonstrate the efficacy of our approach on MNIST, CelebA , and CIFAR-10 . We show that the samples look comparable to those generated from modern likelihood-based models and GANs. On CIFAR-10, our model sets the new state-of-the-art **inception** score of 8.87 for unconditional generative models, and achieves a competitive FID score of 25.32. We show that the model learns meaningful representations of the data by image inpainting experiments.

1. **high-fidelity** /haɪ fɪˈdɛlɪti/ 高保真度
2. **synthesize realistic** /ˈsɪnθəsaɪz rɪəˈlɪstɪk/ 合成逼真的
3. **fragments** /ˈfræɡmənts/ 碎片
4. **adversarial** /ˌædvərˈsɛriəl/ 对抗性的
5. **anomalous** /əˈnɒmələs/ 异常的
6. **imitation** /ɪmɪˈteɪʃən/ 模仿
7. **surrogate** /ˈsʌrəɡət/ 替代品
8. **divergences** /dɪˈvɜrdʒənsɪz/ 散度
9. **integral** /ˈɪntɪɡrəl/ 积分
10. **intrinsic** /ɪnˈtrɪnsɪk/ 固有的
11. **variational** /ˌvɛəriˈeɪʃənl/ 变分的
12. **contrastive divergence** /kənˈtræstɪv daɪˈvɜrdʒəns/ 对比散度
13. **logarithmic** /ˌlɒɡəˈrɪðmɪk/ 对数的
14. **ambient** /ˈæmbiənt/ 环境的
15. **consistent** /kənˈsɪstənt/ 一致的
16. **scarcity** /ˈskɛrsɪti/ 稀缺性
17. **hinders** /ˈhɪndərz/ 阻碍
18. **perturb** /pəˈtɜrb/ 扰动
19. **collapse** /kəˈlæps/ 崩溃
20. **magnitudes** /ˈmæɡnɪtudz/ 大小
21. **annealed** /əˈnild/ 退火的
22. **indistinguishable** /ˌɪndɪˈstɪŋɡwɪʃəbəl/ 难以分辨的
23. **heuristically** /hjʊərɪˈstɪkli/ 启发式地
24. **multimodal landscapes** /ˈmʌltimoʊdl ˈlændˌskeɪps/ 多模态景观
25. **desirable** /dɪˈzaɪrəbl/ 令人满意的
26. **tractable** /ˈtræktəbl/ 易处理的
27. **parameterizations** /pəˌræmɪtəraɪˈzeɪʃənz/ 参数化
28. **quantitatively** /ˈkwɒntɪteɪtɪvli/ 定量地
29. **inception** /ɪnˈsɛpʃən/ 初始
30. **inpainting** /ˈɪnˌpeɪntɪŋ/ 图像修复