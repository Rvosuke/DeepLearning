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

## SHOT-IM(信息最大化)

先前的一些DA方法,如之前提到的MMD,对抗调整等,都是假设源域和目标域有着同样的特征编码器.