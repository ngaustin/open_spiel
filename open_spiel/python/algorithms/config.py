#PPO Training returns
actor_loss_list = []
value_loss_list = []
entropy_list = []
kl_list = []
ppo_training_data = []

def check_new_training_rets():
    '''
        Will only return true if in fine-tune mode and ppo training data is returning. 
    '''
    return len(kl_list) > 0