import numpy as np
import pandas as pd

def evaluate_step_wis(eval_policy, state_dict, action_probs, action_dict, 
                      rewards_dict, eta=0.99):
    """Evaluates eval_policy based on offline data.

    This function uses the methof described in [1] to evaluate a particular
    policy based on the data given in the state and action variables.

    [1] Reinforcement Learning Evaluation of Treatment Policies for Patients with
    Hepatitus C Virus, Oselio et. al.

    Args:
        eval_policy (function): this function takes a state and action as input
        and returns a reward.
        state_dict (dict): A dict where the keys are samples and the values are state vectors.
        action_probs (dict): A dict where the keys are samples and the values are 2xN arrays, 
        with the first column equaling the probability of no treatment according to the baseline policy
        and the second column the probability of treatment.
        action_dict (dict): A dict where the keys are samples and the values are the actions observed.  
        rewards_dict (dict): A dict where the keys are samples and the values are a vector of the observed rewards.
        eta (float, default=0.99): Controls the amount of hitorical discounting.

    Returns:
        utility (float): the expected utility based on the off-line policy evaluation.
    """
    rho_dict = {}
    for i, (k, rewards) in enumerate(rewards_dict.items()):
        #if i % 500 == 0:
            #print(i)
            
        states = state_dict[k]
        actions = action_dict[k]

        log_rho = 0
        temp = action_probs[k]
        for i, (r, p_baseline, a) in enumerate(zip(rewards, temp, actions.astype(int))):
            baseline = p_baseline[int(a)]

            s = states[i]

            eval_num = eval_policy(s, a)
            if eval_num != 0:
                #eval_num = 1e-8
                log_rho = log_rho + np.log(eval_num) - np.log(baseline)
                
            if a == 1:
                break


        rho_dict[k] = np.exp(log_rho)

    reward_length = {k: len(v) for k, v in rewards_dict.items()}
    ids, lengths = zip(*[(k, v) for k, v in reward_length.items()])
    ids = np.array(ids)
    lengths = np.array(lengths)

    n = len(reward_length)
    utility = 0
    max_length = lengths.max()
    min_length = lengths.min()
    
    for i in range(min_length, max_length+1):
        discounts = np.array([eta**t for t in range(i, 0, -1)])
        id_subset = ids[lengths == i]
        if len(id_subset) == 0:
            continue
        W = len(id_subset) / n
        rho_sum = np.sum([rho_dict[k] for k in id_subset])

        reward_sum = 0
        for id_ in id_subset:
            reward_sum += rho_dict[id_] * np.sum(rewards_dict[id_] * discounts)

        if rho_sum == 0:
            continue
        utility += W / rho_sum * reward_sum
    return utility