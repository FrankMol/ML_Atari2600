#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np 
import datetime
import keras 


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def preprocess_frames(new_frame,last_frame):
    # inputs are 2 numpy 2d arrays
    n_frame = new_frame.astype(np.int32)
    n_frame[(n_frame==2000)|(n_frame==180)]=0 # remove backgound colors
    l_frame = last_frame.astype(np.int32)
    l_frame[(l_frame==200)|(l_frame==180)]=0 # remove backgound colors
    diff = n_frame - l_frame
    # crop top and bot 
    diff = diff[35:195]
    # down sample 
    diff=diff[::2,::2]
    # convert to grayscale
    diff = diff[:,:,0] * 299. / 1000 + diff[:,:,1] * 587. / 1000 + diff[:,:,2] * 114. / 1000
    # rescale numbers between 0 and 1
    max_val =diff.max() if diff.max()> abs(diff.min()) else abs(diff.min())
    if max_val != 0:
        diff=diff/max_val
    return diff


# In[4]:


# inputs = keras.layers.Input(shape=(80,80))
# flattened_layer = keras.layers.Flatten()(inputs)
# full_connect_1 = keras.layers.Dense(units=200,activation='relu',use_bias=False,)(flattened_layer)
# softmax_output = keras.layers.Dense(4,activation='softmax',use_bias=False)(full_connect_1)
# policy_network_model = keras.models.Model(inputs=inputs,outputs=softmax_output)
# policy_network_model.summary()


# In[6]:


inputs = keras.layers.Input(shape=(80,80))
channeled_input = keras.layers.Reshape((80,80,1))(inputs) # Conv2D requries (batch, height, width, channels)  so we need to create a dummy channel 
conv_1 = keras.layers.Conv2D(filters=10,kernel_size=20,padding='valid',activation='relu',strides=(4,4),use_bias=False)(channeled_input)
conv_2 = keras.layers.Conv2D(filters=20,kernel_size=10,padding='valid',activation='relu',strides=(2,2),use_bias=False)(conv_1)
conv_3 = keras.layers.Conv2D(filters=40,kernel_size=3,padding='valid',activation='relu',use_bias=False)(conv_2)
flattened_layer = keras.layers.Flatten()(conv_3)
softmax_output = keras.layers.Dense(4,activation='softmax',use_bias=False)(flattened_layer)
policy_network_model = keras.models.Model(inputs=inputs,outputs=softmax_output)
policy_network_model.summary()


# In[9]:


def m_loss(episode_reward):
    def loss(y_true,y_pred):
        # feed in y_true as actual action taken 
        # if actual action was up, we feed 1 as y_true and otherwise 0
        # y_pred is the network output(probablity of taking up action)
        # note that we dont feed y_pred to network. keras computes it
        
        # first we clip y_pred between some values because log(0) and log(1) are undefined
        tmp_pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.05,0.95))(y_pred)
        # we calculate log of probablity. y_pred is the probablity of taking up action
        # note that y_true is 1 when we actually chose up, and 0 when we chose down
        # this is probably similar to cross enthropy formula in keras, but here we write it manually to multiply it by the reward value
        tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(tmp_pred)
        # multiply log of policy by reward
        policy_loss=keras.layers.Multiply()([tmp_loss,episode_reward])
        return policy_loss
    return loss


# In[11]:


episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')
policy_network_train = keras.models.Model(inputs=[inputs,episode_reward],outputs=softmax_output)

my_optimizer = keras.optimizers.RMSprop(lr=0.0001)
policy_network_train.compile(optimizer=my_optimizer,loss=m_loss(episode_reward),)


# In[12]:


def generate_episode(policy_network):
    states_list = [] # shape = (x,80,80)
    action_list=[] # 1 if we chose up. 0 if down
    rewards_list=[]
    network_output_list=[]
    env=gym.make("Breakout-v0")
    observation = env.reset()
    new_observation = observation
    done = False
    policy_output_list = []
    p=np.zeros(4)
    
    while done == False:
    
        processed_network_input = preprocess_frames(new_frame=new_observation,last_frame=observation)
        states_list.append(processed_network_input)
        reshaped_input = np.expand_dims(processed_network_input,axis=0) # x shape is (80,80) so we need similar reshape(x,(1,80,80))

        p[0] = policy_network.predict(reshaped_input,batch_size=1)[0][0]
        p[1] = policy_network.predict(reshaped_input,batch_size=1)[0][1]
        p[2] = policy_network.predict(reshaped_input,batch_size=1)[0][2]
        p[3] = policy_network.predict(reshaped_input,batch_size=1)[0][3]
        p = p/np.sum(p)
        network_output_list.append(p[0])
        policy_output_list.append(p[0])
        actual_action = np.random.choice(a=[0,1,2,3],size=1,p=p) # 2 is up. 3 is down 
        action_list.append(int(actual_action))


        
        observation= new_observation
        new_observation, reward, done, info = env.step(actual_action)
        reward = np.sign(reward)
        rewards_list.append(reward)
        
        if done:
            break
            
    env.close()
    return states_list,action_list,rewards_list,network_output_list


# In[13]:


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
    rew -= np.mean(rew) # subtract by average
    rew /= np.std(rew) # divide by std
    return rew


# In[14]:


def generate_episode_batches_and_train_network(n_batches=10):
    env = gym.make('Pong-v0')
    batch_state_list=[]
    batch_up_or_down_action_list=[]
    batch_rewards_list=[]
    batch_network_output_list=[]
    for i in range(n_batches):
        states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)   
        batch_state_list.extend(states_list[15:])
        batch_network_output_list.extend(network_output_list[15:])
        batch_up_or_down_action_list.extend(up_or_down_action_list[15:])
        batch_rewards_list.extend(rewards_list[15:])
    
    episode_reward=np.expand_dims(process_rewards(batch_rewards_list),1)
    x=np.array(batch_state_list)
    y_tmp = np.array(batch_up_or_down_action_list)
    y_true = np.expand_dims(y_tmp,1)
    policy_network_train.fit(x=[x,episode_reward],y=y_true)

    return batch_state_list,batch_up_or_down_action_list,batch_rewards_list,batch_network_output_list


# In[ ]:


train_n_times = 100 # for actual training, about 5000 may be a good start. 
for i in range(train_n_times):
    states_list,up_or_down_action_list,rewards_list,network_output_list=generate_episode_batches_and_train_network(10)
    if i%10==0:
        print("i="+str(i))
        rr=np.array(rewards_list)
        # i keep how many times we won in batch. you can use log more details more frequently
        print('count win='+str(len(rr[rr>0]))) 
        policy_network_model.save("policy_network_model_simple.h5")
        policy_network_model.save("policy_network_model_simple"+str(i)+".h5")
        with open('rews_model_simple.txt','a') as f_rew:
            f_rew.write("i="+str(i)+'       reward= '+str(len(rr[rr > 0])))
            f_rew.write("\n")


# In[ ]:


import time
def play_and_show_episode(policy_network):
    env = gym.make('Breakout-v0')
    done=False
    observation = env.reset()
    new_observation = observation
    while done==False:
        time.sleep(1/80)
        
        processed_network_input = preprocess_frames(new_frame=new_observation,last_frame=observation)
        reshaped_input = np.expand_dims(processed_network_input,axis=0) # x shape is (80,80) so we need similar reshape(x,(1,80,80))

        
        p0 = policy_network.predict(reshaped_input,batch_size=1)[0][0]
        p1 = policy_network.predict(reshaped_input,batch_size=1)[0][1]
        p2 = policy_network.predict(reshaped_input,batch_size=1)[0][2]
        p3 = policy_network.predict(reshaped_input,batch_size=1)[0][3]
        sump = p0+p1+p2+p3;
        actual_action = np.random.choice(a=[0,1,2,3],size=1,p=[p0, p1, p2, 1-p0-p1-p2]) # 2 is up. 3 is down 
        print(actual_action)
        env.render()
        
        observation= new_observation
        new_observation, reward, done, info = env.step(actual_action)
        if reward!=0:
            print(reward)
        if done:
            break
        
    env.close()


# In[ ]:


play_and_show_episode(policy_network_model)

