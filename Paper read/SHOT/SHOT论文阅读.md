# SHOT论文阅读

*15-3-2024 INTRODUCTION&RELATEDWORK*

## Unsupervised Domain Adaptation

先简要了解*Domain Adaptation(DA)*域适应问题，我们有一个源域的标注数据,以及一个目标域的未标注数据,两个域的数据分布可能有较大差异,如何利用源域的标注数据去学习一个适用于目标域的模型?这在很多实际应用中有重要意义,因为人工标注数据往往代价很高,而未标注的数据相对容易获得。

UDA的主要挑战在于如何缓解源域和目标域之间的*分布差异(domain shift)*。虽然两个域的数据分布不同,但通常我们假设它们共享某些判别性的特征表示。UDA的目标就是学习这种domain-invariant的特征表示,使得训练好的模型能够很好地泛化到目标域。

> Over the last decade, increasing efforts have been devoted to deep domain adaptation, especially under the vanilla unsupervised closed-set setting, where two domains share the same label space but the target data are not labeled.

UDA方法大致可分为以下几类:

1. 基于特征变换的方法:通过某种特征变换(如PCA,MMD),将源域和目标域数据映射到一个公共空间,拉近两个域的距离。

   > One prevailing paradigm is to mitigate the distribution divergence between domains by matching the distribution statistical moments at different orders.

2. 基于domain adversarial的方法:引入一个domain discriminator来判断样本来自源域还是目标域,通过adversarial training使学到的特征具有domain-invariant性。代表工作有DANN等。

   > Another popular paradigm leverages the idea of adversarial learning and introduces an additional domain classifier to minimize the Proxy A-distance across domains.

3. 基于pseudo label的方法:用源域训练好的分类器对目标域数据打伪标签,再用这些伪标签数据和源域数据一起训练模型,重复这个过程直到收敛。

   > (Zou et al., 2018) further designs an integrated framework to alternately solve target pseudo labels and perform model training.

4. 基于样本权重的方法:为源域样本赋予不同权重,减小与目标域差异大的样本权重,提高相似样本权重。

   >  there are several sample-based DA methods that estimate the importance weights of source samples or sample-to-sample similarity via optimal transport.

如今的DA技术有一个弊病：**需要通过源数据来学习调整**。一方面，传输效率低下；另一方面，涉及隐私问题。

## Hypothesis Transfer Learning(HTL,假设迁移学习)

HTL是迁移学习的另一个重要分支,与UDA的设置有所不同。在HTL中,我们假设源域和目标域的标注数据都是可用的,但源域数据非常丰富,而目标域的标注数据非常少。HTL的目标是利用源域的知识来辅助目标域的学习,从而提高目标域的性能。

HTL的基本思路是,先在源域的大量标注数据上训练一个强大的"teacher"模型,然后利用这个模型来指导目标域模型的学习。这种指导可以通过不同的方式实现,主要有以下几种主流范式:

1. Fine-tuning:即先在源域数据上训练好一个模型,然后把这个模型的参数作为目标域模型的初始化,在目标域数据上进行fine-tuning。这种方法简单直接,但可能受到源域和目标域差异的影响。

2. Knowledge Distillation:即用源域的teacher模型对目标域的大量无标注数据进行"软标注",生成pseudo label,然后用这些pseudo label作为目标域的训练数据,蒸馏到student模型中去。代表工作如DAKD等。

   > The main idea is to label unlabeled data with the maximum predicted probability and perform fine-tuning together with labeled data, which is quite efficient.

3. Regularization:即在目标域模型的训练过程中,加入源域模型的输出作为一种regularization项,引导目标域模型向源域模型的预测结果看齐。代表工作如DELTA等。

   > For DA methods, (Zhang et al., 2018b; Choi et al., 2019) directly incorporate pseudo labeling as a regularization

4. Meta-learning:将HTL问题建模为一个meta-learning问题,通过从源域学习一个meta-learner,来指导目标域的few-shot learning。代表工作如MAML等。

5. Curriculum Learning:利用源域模型对目标域样本的置信度,设计一个渐进的curriculum,先学习简单的样本,再逐步过渡到困难的样本。

HTL的优势在于,它不需要直接通过源数据进行学习。

> In terms of privacy protection, this setting seems somewhat similar to a recently proposed transfer learning setting in Hypothesis Transfer Learning (HTL), where the learner does not have direct access to the source domain data and can only operate on the hypotheses induced from the source data.

HTL在图像分类、语义分割、关键点检测等领域都有广泛应用。最近兴起的prompt tuning,如CoOp等,本质上也可以看作是将大模型的知识迁移到下游任务的一种HTL方法。

但是HTL问题也很显著，它无法做到真正的无监督。

> Like the famous fine-tuning strategy, HTL mostly acquires at least a small set of labeled target examples per class, limiting its applicability to the semi-supervised DA scenario.

## 生成源模型

SHOT采用的了交叉熵损失函数，并采用了*标签平滑技术(Label Smoothing, LS)*.

根据论文中的图2及相关描述,SHOT框架的具体研究方法如下:

SHOT假设源域和目标域共享相同的深度神经网络模型,包含一个特征编码模块(feature extractor)和一个分类器模块(classifier)。如图所示:

> SHOT freezes the classifier module (hypothesis) of the source model and learns the target-specific feature extraction module by exploiting both information maximization and self-supervised pseudo-labeling to implicitly align representations from the target domains to the source hypothesis.

SHOT冻结源域模型的分类器模块$h_{t}=h_{s}$,将其作为目标域分类器,然后通过以下两种方式优化目标域特定的特征提取模块$g_{t}$:

1. 信息最大化(Information Maximization):最小化特征表示和分类器输出之间的条件熵 $L_{ent}$,鼓励网络为目标域特征生成确定的预测;最大化预测的均匀性 $L_{div}$,使得不同类别的预测概率尽量平均。

2. 自监督伪标签学习:先通过源域分类器$h_{s}$对目标域数据生成伪标签,然后计算目标域类别原型(centroids)$c_{k}^{(0)}$,基于原型重新生成更可靠的伪标签 $\hat{y}_{t}$ 作为监督信号,优化$g_{t}$以拟合伪标签。

通过联合优化上述两个目标,SHOT可以在不访问源域数据的情况下,学习与源域分类器$h_{s}$一致的目标域特征表示,从而完成领域自适应。

此外,SHOT在源域模型的设计中使用了标签平滑(Label Smoothing)、权重归一化(Weight Normalization)和批量归一化(Batch Normalization)等技术,以进一步提升自适应性能。

## SHOT-IM(信息最大化)

先前的一些DA方法,如之前提到的MMD,对抗调整等,都是假设源域和目标域有着同样的特征编码器.

信息最大化包含两个目标:
1. 最小化条件熵 $L_{ent}$,其中$\delta_{k}(f_{t}(x))=\text{softmax}(h_{t}(g_{t}(x)))$ 表示目标域样本x的softmax输出:

$$
L_{ent}(f_{t};X_{t})=-E_{x_{t} \in X_{t}} \sum_{k=1}^{K} \delta_{k}(f_{t}(x_{t})) \log \delta_{k}(f_{t}(x_{t}))
$$

2. 最大化预测的均匀性 $L_{div}$,其中 $\hat{p}_{k} $表示整个目标域在类别k上的平均预测概率,$1_K$是全为1的K维向量:

$$
L_{div}(f_{t};X_{t})=\sum_{k=1}^{K} \hat{p}_{k} \log \hat{p}_{k}=D_{KL}(\hat{p},\frac{1}{K}\textbf{1}_{K})-\log K
$$

这里 $\hat{p}=E_{x_{t} \in X_{t}}[\delta(f_{t}^{(k)}(x_{t}))]$ 。

## 自监督伪标签学习

自监督伪标签学习分为两步:
1. 计算目标域类别原型(centroids)。原型$ c_{k}^{(0)} $定义为基于源域分类器$h_{s}$对目标域数据预测的类别$k$的样本特征 $\hat{g}_{t}(x_{t}) $的加权平均:

$$
c_{k}^{(0)}=\frac{\sum_{x_{t} \in X_{t}} \delta_{k}(\hat{f}_{t}(x_{t}))\hat{g}_{t}(x_{t})}{\sum_{x_{t} \in X_{t}} \delta_{k}(\hat{f}_{t}(x_{t}))}
$$

其中 $\hat{f}_{t}=\hat{g}_{t} \circ h_{t}$ 表示之前学习到的目标域分类器。

2. 基于原型生成伪标签。伪标签 $\hat{y}_{t}$ 定义为样本 $x_{t}$ 与各类别原型 $c_{k}^{(1)}$ 之间余弦距离 $D_{f}$ 最小的类别:

$$
\hat{y}_{t}=\arg \min _{k} D_{f}\left(\hat{g}_{t}\left(x_{t}\right), c_{k}^{(1)}\right)
$$

这里原型 $c_{k}^{(1)}$ 是基于伪标签 $\hat{y}_{t}$ 更新得到的:

$$
c_{k}^{(1)}=\frac{\sum_{x_{t} \in X_{t}} \mathbf{1}\left(\hat{y}_{t}=k\right) \hat{g}_{t}\left(x_{t}\right)}{\sum_{x_{t} \in X_{t}} \mathbf{1}\left(\hat{y}_{t}=k\right)}
$$
SHOT通过加权交叉熵损失利用伪标签 $\hat{y}_{t}$ 作为目标域特征提取器 $g_{t}$ 的监督信号,从而实现自适应对齐。 



综上,信息最大化通过最小化条件熵和最大化预测均匀性,使得目标域特征在源域分类器下可以得到低熵、类别平衡的预测;而自监督伪标签学习通过源域分类器在目标域生成原型和伪标签,为目标域特征学习提供更精准的监督信息。两种方法相辅相成,共同促进SHOT的自适应学习过程。

## 实验设计

根据论文,封闭集(closed set)、开放集(open set)和部分集(partial set)是三种不同的领域自适应问题设定:

1. 封闭集:源域和目标域共享完全相同的类别标签空间,即$Y_{s}=Y_{t}$。这是最常见的无监督领域自适应设定。
2. 开放集:目标域中包含一些源域中没有的未知类别,即$Y_{t} \supset Y_{s}$。开放集DA需要识别目标域中的未知类别样本。
3. 部分集:源域的类别标签空间是目标域的一个超集,即$Y_{s} \supset Y_{t}$。部分集DA假设目标域只包含源域类别的一个子集。



SHOT在多个领域分别进行了实验,涵盖了上述三种DA问题:

1. 数字识别:在MNIST、USPS和SVHN数据集上进行封闭集DA实验。SHOT以98.4%的平均准确率超越了CDAN+E、CyCADA等方法。

> SHOT obtains the best mean accuracies for each task and outperforms prior work in terms of the average accuracy.

2. 物体识别:在Office、Office-Home和VisDA-C数据集上进行封闭集DA实验:
- Office数据集:SHOT以88.6%的平均准确率与现有最佳方法相当。
- Office-Home数据集:SHOT将平均准确率从67.6%提升至71.8%,超越了SAFN、TransNorm等。  
- VisDA-C数据集:SHOT将每类准确率从76.4%提升至82.9%,超越多数基于对抗学习的方法。

> SHOT significantly outperforms previously published state-of-the-art approaches, advancing the average accuracy from 67.6% (Wang et al., 2019) to 71.8% in Table 4.

在Office-Home上进行部分集和开放集DA实验:
- 部分集DA:SHOT将平均准确率从71.8%提升至79.3%,超越ETN、IWAN等。  
- 开放集DA:SHOT将平均准确率从69.5%提升至72.8%,超越ATI-λ、OSBP等。

此外,SHOT还在Office-Caltech数据集上进行了多源和多目标DA实验,也取得了领先的结果。

综上,SHOT在多个基准数据集上sistemically系统地验证了其在封闭集、开放集和部分集等不同DA场景下的有效性,并与当前最好方法进行了广泛对比,在多数任务中实现了新的最优性能。这些实验充分说明了SHOT作为一种新颖的源域假设迁移框架,在无法访问源域数据情况下解决无监督DA问题的优越性。