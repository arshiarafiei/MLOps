import matplotlib.pyplot as plt

def bar_plot(mse_1, mse_2, save_path='mse_comparison.png'):
    mse_values = [mse_1, mse_2 ]
    model_names = ['1 X (x1)', '2 X (x1, x2)']

    plt.bar(model_names, mse_values, color=['blue', 'green'])
    plt.ylabel('Mean Squared Error')
    plt.title('Model Comparison: MSE')
    plt.savefig(save_path)
    plt.close()