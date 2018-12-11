#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import numpy as np
import random
import os
import os.path
import json
from tensorflow import flags

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ATARI_SHAPE = (84, 84, 4)  # tensor flow backend -> channels last
FLAGS = flags.FLAGS
MODEL_PATH = 'trained_models/'

# define hyper parameters -> these can all be passed as command line arguments!
flags.DEFINE_float('discount_factor', 0.99, "discount factor used in q-learning")
flags.DEFINE_float('learning_rate', 0.00025, "learning rate used by CNN")
flags.DEFINE_float('gradient_momentum', 0.95, "gradient momentum used by CNN")
flags.DEFINE_float('sq_gradient_momentum', 0.95, "squared gradient momentum used by CNN")
flags.DEFINE_float('min_sq_gradient', 0.01, "constant added to squared gradient")


class PolicyGradientAgent:

    def __init__(self, env, model_id):
        self.n_actions = env.action_space.n
        self.model_name = os.path.join(MODEL_PATH, model_id)
        if os.path.exists(self.model_name + '.h5'):
            # load model and parameters
            self.model = keras.models.load_model(self.model_name + '.h5')
            self.load_parameters(self.model_name + '.json')
            print("\nLoaded model '{}'".format(model_id))
        else:
            # make new model
            self.build_model()
            self.save_checkpoint(0)
            print("\nCreated new model with name '{}'".format(model_id))

    def write_parameters(self, config_file):
        items = FLAGS.flag_values_dict()
        with open(config_file, 'w') as f:
            json.dump(items, f, indent=4)

    def load_parameters(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(FLAGS, key):
                    setattr(FLAGS, key, value)

    def write_iteration(self, config_file, iteration):
        if not os.path.exists(config_file):
            self.write_parameters(config_file)

        with open(config_file, "r") as f:
            items = json.load(f)

        items["iteration"] = iteration

        with open(config_file, "w+") as f:
            f.write(json.dumps(items, indent=4))

    def build_model(self):
        # from keras import backend as K
        # K.set_image_dim_ordering('th')

        # With the functional API we need to define the inputs.
        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        actions_input = keras.layers.Input((self.n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        conv_1 = keras.layers.Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)

        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = keras.layers.Conv2D(32, (4, 4), activation="relu", strides=(2, 2))(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(self.n_actions)(hidden)
        # Finally, we multiply the output by the mask!
        filtered_output = keras.layers.merge.Multiply()([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        optimizer = keras.optimizers.RMSprop(lr=FLAGS.learning_rate,
                                             rho=FLAGS.gradient_momentum,
                                             epsilon=FLAGS.min_sq_gradient)
        self.model.compile(optimizer, loss='mse')

    def get_one_hot(self, targets):
        return np.eye(self.n_actions)[np.array(targets).reshape(-1)]

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
    
    def generate_episode_batches_and_train_network(env, n_batches=10):
        env = gym.make('Breakout-v4')
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



    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            mask = np.ones(self.n_actions).reshape(1, self.n_actions)
            q_values = self.model.predict([state, mask])
            action = np.argmax(q_values)
        return action

    def save_checkpoint(self, iteration, new_best=False):
        self.model.save(self.model_name + '.h5')
        self.write_iteration(self.model_name + '.json', iteration)
        if new_best:
            self.model.save(self.model_name + '_best.h5')
            self.write_iteration(self.model_name + '_best.json', iteration)

