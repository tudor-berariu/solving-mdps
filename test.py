import numpy as np
import gym
from envs import MDP
from policies import Policy, UniformPolicy

from algorithms.closed_form import cf_state_dist, cf_policy_eval
from algorithms.closed_form import cf_policy_eval_linear_approx
from algorithms.closed_form import cf_min_tderror_policy_eval
from algorithms.dynamic_programming import dp_state_dist, dp_policy_eval
from algorithms.error_functions import mean_squared_value_error
from algorithms.online_algorithms import GradientMonteCarlo


def show_state_dist(env: MDP,
                    policy: Policy) -> None:
    returns = []
    visits = np.zeros(env.n_nstates)

    for ep in range(10000):
        obs, done = env.reset(), False
        ret = 0
        while not done:
            visits += obs
            obs, reward, done, _ = env.step(np.random.randint(2))
            ret += reward
        returns.append(ret)
    visits /= np.sum(visits)

    print("Visits (empiric): ", visits)
    state_dist = cf_state_dist(env, policy)
    print("State dist closed_form: ", state_dist)

    state_dist = dp_state_dist(env, policy)
    print("State dist dynamic programming: ", state_dist)

    # print(np.mean(returns))


def show_values(env: MDP, policy: Policy) -> None:
    cf_values = cf_policy_eval(env, policy)
    dp_values = dp_policy_eval(env, policy)

    error = mean_squared_value_error(env, policy, values=cf_values)

    print(cf_values)
    print(dp_values)

    features = np.array([[1,  0], [0, 1], [.25, -.75]])
    weight = cf_policy_eval_linear_approx(env, policy, features)
    w_values = features @ weight
    print("CF projection: ", w_values, error(w_values))

    weight = cf_min_tderror_policy_eval(env, policy, features)
    w_values = features @ weight
    print("CF minTDerr: ", w_values, error(w_values))

    tabular = np.eye(env.n_nstates)
    weight = cf_policy_eval_linear_approx(env, policy, tabular)
    w_values = tabular @ weight
    print(w_values, error(w_values))

    gmc = GradientMonteCarlo(env, policy, features, episodes_no=10)
    results = gmc.train()

    print(results["values"])
    print(results["msve"])
    print(results["berr"])
    print(results["tderr"])


def main() -> None:
    env = gym.make('ThreeState-v0')
    policy = UniformPolicy(env)
    show_state_dist(env, policy)
    show_values(env, policy)


if __name__ == "__main__":
    main()
