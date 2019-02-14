#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CODE BASED ON IMPLEMENTATION FROM http://karpathy.github.io/2016/05/31/rl/
# SOURCE CODE ADJUSTED FROM https://github.com/thinkingparticle/deep_rl_pong_keras

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import gym
import numpy as np
import keras


# In[3]:


ENV_ID = "Breakout-v4"

env = gym.make(ENV_ID)
frame_shape = (105,80)
frame_shape_channel = (105,80,1)
n_actions = env.action_space.n


# In[4]:


#network parameters
learning_rate = 0.001
#gradient_momentum = 0.95
#min_sq_gradient = 0.01


# In[5]:


def m_loss(episode_reward):
    def loss(y_true,y_pred):
        # feed in y_true as actual action taken 
        # if actual action was up, we feed 1 as y_true and otherwise 0
        # y_pred is the network output(probablity of taking up action)
        # note that we dont feed y_pred to network. keras computes it
        
        vmax = np.max(y_pred)
        
        # first we clip y_pred between some values because log(0) and log(1) are undefined
        tmp_pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.05,0.95))(vmax)
        
        # we calculate log of probablity. y_pred is the probablity of taking up action
        # note that y_true is 1 when we actually chose up, and 0 when we chose down
        # this is probably similar to cross enthropy formula in keras, but here we write it manually to multiply it by the reward value
        tmp_loss = keras.layers.Lambda(lambda x:-keras.backend.log(x))(tmp_pred)
        # multiply log of policy by reward
        policy_loss=keras.layers.Multiply()([tmp_loss,episode_reward])
        return policy_loss
    return loss


# In[6]:


def to_grayscale(img):return np.mean(img, axis=2).astype(np.uint8)
def downsample(img):return img[::2, ::2]
def preprocess(img):return to_grayscale(downsample(img))


# In[7]:


env.unwrapped.get_action_meanings()


# In[18]:



input_layer = keras.layers.Input(frame_shape_channel)

h_layer_1 = keras.layers.Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(input_layer)
h_layer_2 = keras.layers.Conv2D(32, (8, 8), activation="relu", strides=(4, 4))(h_layer_1)

flattened_layer = keras.layers.core.Flatten()(h_layer_2)


sigmoid_output = keras.layers.Dense(1,activation='sigmoid',use_bias=False)(flattened_layer)
ddpg = keras.models.Model(inputs=input_layer,outputs=sigmoid_output)

ddpg.summary()
    
episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')
ddpg_train = keras.models.Model(inputs=[input_layer,episode_reward],outputs=sigmoid_output)

my_optimizer = keras.optimizers.RMSprop(lr=0.001)
ddpg_train.compile(optimizer=my_optimizer,loss=m_loss(episode_reward),)


# In[9]:


def process_rewards(r_list):
    reward_decay=0.99
    tmp_r=0
    rew=np.zeros_like(r_list,dtype=np.float32)
    for i in range(len(r_list)-1,-1,-1):
        if r_list[i]==0:
            tmp_r=tmp_r*reward_decay
            rew[i]=tmp_r
        else: 
            tmp_r = r_list[i]
            rew[i]=tmp_r
  #  rew -= np.mean(rew) # subtract by average
  #  rew /= np.std(rew) # divide by std
    return rew


# In[10]:


def clip_reward(r):return np.sign(r)


# In[12]:


env.step


# In[31]:


def generate_episode(env):
    states_list = [] # shape = (x,80,80)
    action_list=[] # 1 if we chose up. 0 if down
    rewards_list=[]
    network_output_list=[]
    observation = env.reset()
    observation, reward, done, info = env.step(1)
    lives = info['ale.lives']
    done = False
    policy_output_list = []
    lost_live = True
    
    
    while done == False:
        if lost_live: observation, reward, done, info = env.step(1)
        
        #processed_network_input = preprocess_frames(new_frame=new_observation,last_frame=observation)
        processed_image = preprocess(observation)
        states_list.append(processed_image)
        reshaped_input = np.expand_dims(processed_image,axis=2) # x shape is (80,80) so we need similar reshape(x,(1,80,80))
        reshaped_input = np.expand_dims(reshaped_input,axis=0)
        
        prediction = ddpg.predict(reshaped_input)
        network_output_list.append(prediction[0])
        action = np.argmax(prediction<0.5) + 2 
        action_list.append(action)
        
        observation, reward, done, info = env.step(action)
        lost_live = True if info['ale.lives'] < lives else False
        lives = info['ale.lives']
        rewards_list.append(reward + lost_live*-1)
        
        if done:
            break
            
    env.close()
    return states_list,action_list,rewards_list, network_output_list


# In[32]:


a, b, c, d= generate_episode(env)


# In[30]:


for i in c:
    if i!= 0.0: print (i)


# In[17]:


def get_one_hot(targets):
    return np.eye(n_actions)[np.array(targets).reshape(-1)]


# In[24]:


# we define a helper function to create a batch of simulations
# and after the batch simulations, preprocess data and fit the network
def train_ddpg(ddpg, env, n_batches=10):
    batch_state_list=[]
    batch_action_list=[]
    batch_rewards_list=[]
    batch_network_output_list = []
    for i in range(n_batches):
        states_list,action_list,rewards_list, network_output_list = generate_episode(ddpg, env) 
        batch_state_list.extend(states_list)
        batch_action_list.extend(action_list)
        batch_rewards_list.extend(rewards_list)
        batch_network_output_list.extend(network_output_list)
    
    episode_reward=process_rewards(batch_rewards_list)
    print(episode_reward)
    x=np.array(batch_state_list)
    x = np.expand_dims(x, 3)
    bnol = np.asarray(batch_network_output_list)
    #max_indexes = np.zeros_like(bnol)
    er = np.asarray(episode_reward)
    

    arr = get_one_hot(batch_action_list)
    y = bnol + bnol*arr
    ddpg_train.fit(x,y)

    return batch_state_list, action_list, batch_rewards_list, network_output_list


# In[25]:


train_n_times = 21 # for actual training, about 5000 may be a good start. 
for i in range(train_n_times):
    states_list,up_or_down_action_list,rewards_list,network_output_list=train_ddpg(ddpg, env, 10)
    if i%10==0:
        print("i="+str(i))
        rr=np.array(rewards_list)
        # i keep how many times we won in batch. you can use log more details more frequently
        print('count win='+str(len(rr[rr>0]))) 
        ddpg.save("policy_network_model_simple.h5")
        ddpg.save("policy_network_model_simple"+str(i)+".h5")
        with open('rews_model_simple.txt','a') as f_rew:
            f_rew.write("i="+str(i)+'       reward= '+str(len(rr[rr > 0])))
            f_rew.write("\n")


# In[ ]:


a,b,c = train_ddpg(ddpg, env)


# In[ ]:


import time
def play_and_show_episode(policy_network, env):
    done=False
    observation = env.reset()
    observation, reward, done, info = env.step(1)
    lives = info['ale.lives']
    done = False
    lost_live = True
    env.render()
    while done==False:
        time.sleep(1/80)
        if lost_live: 
            observation, reward, done, info = env.step(1)
            env.render()
            time.sleep(1/80)
        
        processed_image = preprocess(observation)
        reshaped_input = np.expand_dims(processed_image,axis=0)
        reshaped_input = np.expand_dims(reshaped_input,axis=3) # x shape is (80,80) so we need similar reshape(x,(1,80,80))

        prediction = ddpg.predict(reshaped_input)
        action = np.argmax(prediction)
        
        env.render()
        
        observation, reward, done, info = env.step(action)
        lost_live = True if info['ale.lives'] < lives else False
        lives = info['ale.lives']
        if reward!=0:
            print(reward)
        if done:
            break


# In[ ]:


play_and_show_episode(ddpg, env)


# In[ ]:


env.unwrapped.get_action_meanings()


# In[ ]:




