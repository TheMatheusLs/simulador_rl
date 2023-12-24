from stable_baselines3 import A2C
from OpticalSimFasterEnv.Enviroment.Manager import Enviroment

if __name__ == "__main__":

    # Cria o ambiente de simulação
    env = Enviroment(network_load = 80, k_routes = 3, number_of_slots = 64)

    # Cria o modelo
    model = A2C('MlpPolicy', env, verbose=2)

    # Treina o modelo
    model.learn(total_timesteps=100000, progress_bar=True)

    # Salva o modelo
    model.save("A2C_RSA_SAR")

    # Gráica a recompensa
    env.plot_reward()