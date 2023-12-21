import matplotlib.pyplot as plt


def visualize_metric(train_f1_viz, eval_f1_viz, epochs):
    epochs = [i for i in range(0, epochs)]
    plt.plot(epochs, train_f1_viz, 'bo', label='Training F1')
    plt.plot(epochs, eval_f1_viz, 'r', label='Validation F1')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    y_arrow = max(eval_f1_viz)
    x_arrow = eval_f1_viz.index(y_arrow) + 1
    plt.annotate(str(y_arrow)[:6],
                 (x_arrow, y_arrow),
                 xytext=(x_arrow + 5, y_arrow + .02),
                 arrowprops=dict(facecolor='orange', shrink=0.05))
    plt.show()


def visualize_losses(train_loss_viz, eval_loss_viz, epochs):
    epochs = [i for i in range(0, epochs)]
    plt.plot(epochs, train_loss_viz, 'bo', label='Training Loss')
    plt.plot(epochs, eval_loss_viz, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
