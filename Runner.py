
from OpticalSimFasterEnv.Enviroment.Manager import Enviroment



if __name__ == "__main__":

    # Cria o ambiente de simulação
    env = Enviroment(network_load = 100, k_routes = 3)

    for _ in range(2):
        # # Reseta o ambiente
        env.reset()

        reward_sum = 0.0

        done = False

        action = 0

        while not done:

            next_state, reward, done, tru, info = env.step(action)

            #print("Reward:", reward)
            reward_sum += reward

        print("Reward sum:", reward_sum)

