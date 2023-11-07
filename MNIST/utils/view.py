from matplotlib import pyplot as plt


def plot_history(train_losses, val_acc, title):
    epochs = range(1, len(train_losses) + 1)

    # 使用subplot创建1行2列的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 第一个subplot用于绘制训练损失
    ax1.plot(epochs, train_losses, 'r', label='Training loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 第二个subplot用于绘制验证精度
    ax2.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 先保存图像，再显示
    plt.savefig(f"{title}.png")
    plt.show()
