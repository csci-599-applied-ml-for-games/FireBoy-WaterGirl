import numpy as np
import time
import os
from getkeys import key_check
from ExperienceReplay import ExperienceReplay
from math import exp
from pynput.mouse import Button, Controller

# parameters
# epsilon = .2  # exploration
#num_actions = 4  # [ shoot_low, shoot_high, left_arrow, right_arrow]
num_actions = 4  # [ move_left, move_right, jump_left, jummp_right]
max_memory = 1000  # Maximum number of experiences we are storing
batch_size = 4  # Number of experiences we use for training per batch

exp_replay = ExperienceReplay(max_memory=max_memory)


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_epoch1000/Z1_model.json", "w") as json_file:  #change Z_model to model for fifa
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_epoch1000/Z1_model.h5")    #same as above
    # print("Saved model to disk")


def train(game, model, epochs,num_of_games, verbose=1):
    # Train
    # Reseting the win counter
    mouse=Controller()
    
    # We want to keep track of the progress of the AI over time, so we save its win count history
    loss_hist = []
    average_points=[]
    current_step=0

    for ga in range(num_of_games):
        time.sleep(1)
        reward_count=0
        #os.system('TASKKILL /F /IM zed.exe')
        os.startfile('C:/Users/naman/Desktop/zed.exe')
        #win_cnt = 0
        game.reset()
        game_over = False
        print('Training paused! will begin in 10 seconds')
        time.sleep(10)
        loss=0.
        average_value=0
        

        for e in range(1,epochs+1):
            current_step+=1       
            epsilon = 0.01 + 0.99*exp(-1.*current_step*0.001)
            # get tensorflow running first to acquire cudnn handle
            input_t = game.observe()
            if e==1:
                time.sleep(1)
#            if e == 0:
#                paused = True
#                print('Training is paused. Press N once game is loaded and is ready to be played.')
#            else:
#                paused = False
            #while not game_over:
            #if not paused:
                # The learner is acting on the last observed game screen
                # input_t is a vector containing representing the game screen
            input_tm1 = input_t

            n=np.random.rand()
            if n <= epsilon:
                # Eat something random from the menu
                action = int(np.random.randint(0, num_actions, size=1))
                #print('random action')
            else:
                # Choose yourself
                # q contains the expected rewards for the actions
                q = model.predict(input_tm1)
                # We pick the action with the highest expected reward
                action = np.argmax(q[0])
                #print('optimal action')

            # apply action, get rewards and new state
            input_t, reward, game_over = game.act(action)
            print("ingame_reward",reward)
            if reward==0.2 or reward==1:
                reward_count+=1
                average_value+=reward_count/e
                
            # If we managed to catch the fruit we add 1 to our win counter
#            if reward == 1:
#                win_cnt += 1
 
            """
            The experiences < s, a, r, s’ > we make during gameplay are our training data.
            Here we first save the last experience, and then load a batch of experiences to train our model
            """

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)
            # Load batch of experiences
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
        
            # train model on experiences
            batch_loss = model.train_on_batch(inputs, targets)
            print('batch loss',batch_loss)
            loss += batch_loss
            # menu control
            keys = key_check()
#            if 'N' in keys:
#                if paused:
#                    paused = False
#                    print('unpaused!') 
#                    time.sleep(1)
#                else:
#                    print('Pausing!')
#                    paused = True
#                    time.sleep(1)
            if 'O' in keys:
                print('Quitting!')
                return
            print("Step {:03d}/{:03d}".format(e,epochs))
            if game_over:
                print('game over! restarting')
                mouse.position=(1280,280)
                mouse.click(Button.left,1)
                break
                

        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f}".format(ga, num_of_games, loss))
            #print('Epoch: {}, Loss: {}, Accuracy: {}'.format(e+1,loss,win_cnt/e*100))
        
        loss_hist.append(loss)
        average_points.append(average_value)
        mouse.position=(1280,280)
        mouse.click(Button.left,1)
    save_model(model)
    return loss_hist,average_points
