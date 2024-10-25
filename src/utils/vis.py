# utils/plot.py
import matplotlib.pyplot as plt

def plot_predictions(timesteps, predictions):
    plt.plot(timesteps.numpy(), predictions.squeeze().numpy(), label='Predicted')
    plt.title('Model Prediction vs Actual Timestep')
    plt.xlabel('Actual Timestep')
    plt.ylabel('Predicted Value')
    plt.legend()
    plt.show()
