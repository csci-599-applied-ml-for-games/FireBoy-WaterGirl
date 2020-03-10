import numpy as np
import pytesseract as pt
import cv2
from CNN import CNN
from PIL import Image
from PIL import ImageGrab
from pynput.mouse import Button, Controller
from grabscreen import grab_screen
from directkeys import *


mouse=Controller()
class Zed(object):

    cnn_graph = CNN()

    def __init__(self):
        self.rward=0
        self.reset()
        self.visit=0
        self.previous_screen=[]
        self.previous_compare_screen=[]
        self.previous_gold_screen=[]

    def _get_reward(self, action):
        flag1=False
        flag2=False
        flag3=False
        self.visit+=1

        screen = np.asarray(ImageGrab.grab(bbox=(611.5,276,1308.5,801)))
        
        gold_screen=screen[84:100,158:279]
        if self.visit>1:
            diff2=cv2.subtract(gold_screen,self.previous_gold_screen)
            b,g,r=cv2.split(diff2)
            #print(cv2.countNonZero(b),cv2.countNonZero(g),cv2.countNonZero(r))
            max1=max(cv2.countNonZero(b) , cv2.countNonZero(g) , cv2.countNonZero(r))
            if max1>30:
                flag1=True
        self.previous_gold_screen=gold_screen
                
        
        lives_screen= screen[55:75,70:150]
        if self.visit>1:
            diff1=cv2.subtract(lives_screen,self.previous_screen)
            b1,g1,r1=cv2.split(diff1)          
            if cv2.countNonZero(b1) > 0 or cv2.countNonZero(g1) > 0 or cv2.countNonZero(r1) > 0:
                flag2=True                
        self.previous_screen=lives_screen
        
        
        compare_screen=screen[170:,40:]
        if self.visit>1:
            diff2=cv2.subtract(compare_screen,self.previous_compare_screen)
            b2,g2,r2=cv2.split(diff2)
            #print(cv2.countNonZero(b1) , cv2.countNonZero(g1) , cv2.countNonZero(r1))
            max2=max(cv2.countNonZero(b2) , cv2.countNonZero(g2) , cv2.countNonZero(r2))
            if max2<1900:
                flag3=True  
# =============================================================================
#         if self.visit>1:
#             i = Image.fromarray(self.previous_compare_screen.astype('uint8'), 'RGB').show()
#             time.sleep(1)
#             i = Image.fromarray(compare_screen.astype('uint8'), 'RGB').show()
#             time.sleep(1)

# 
# =============================================================================
        self.previous_compare_screen=compare_screen
        if flag2:
            return -1
        if flag1:
            return 1        
        if flag3:
            return -0.1
 
        reward_screen=screen[50:80,600:673]
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')

        try:
            ocr_result = pt.image_to_string(i,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            if ocr_result=='':
                ingame_reward=0
            else:
                ingame_reward = int(''.join(c for c in ocr_result if c.isdigit()))
                r=ingame_reward
                if ingame_reward>self.reward:
                    ingame_reward=0.2
                else:
                    ingame_reward=0
                self.reward=r
        except:
            ingame_reward = -1 if self._is_over(action) else 0
            print('exception q-learning reward: ' + str(ingame_reward))

        return ingame_reward

    def _is_over(self, reward):
        if reward==-1:
            is_over=True
        else:
            is_over=False
        return is_over
    
    
    def observe(self):
        print('observe\n')
        # get current state s from screen using screen-grab
        screen = np.asarray(ImageGrab.grab(bbox=(611.5,276,1308.5,801)))
        quest_screen=screen[465:490,600:680]
        quest_screen=cv2.bitwise_not(quest_screen)
        i = Image.fromarray(quest_screen.astype('uint8'), 'RGB')
        restart_text = pt.image_to_string(i)
  
        if "Begin guest" in restart_text:
            self.reward=0
            mouse.position=(1256,758.5)
            time.sleep(0.1)
            mouse.click(Button.left,1)
            time.sleep(3)
            #mouse.position=(10,10) #move cursor out of the window
            #time.sleep(0.1)
            mouse.position=(950,710)
            time.sleep(0.1)
            mouse.click(Button.left,1)
            time.sleep(4)
            mouse.position=(10,10)
            screen = np.asarray(ImageGrab.grab(bbox=(611.5,276,1308.5,801)))
        
        # process through CNN to get the feature map from the raw image        
        state = self.cnn_graph.get_image_feature_map(screen)
        return state

    def act(self, action):
        #display_action = ['shoot_low', 'shoot_high', 'left_arrow', 'right_arrow']
        display_action = ['move_left', 'move_right','jump_left','jump_right']
        print('action: ' + str(display_action[action]))
        keys_to_press = [[leftarrow],[rightarrow],[leftarrow,uparrow],[rightarrow,uparrow]]
        # need to keep all keys pressed for some time before releasing them otherwise fifa considers them as accidental
        # key presses.
        for key in keys_to_press[action]:
            PressKey(key)
        time.sleep(0.27)
        for key in keys_to_press[action]:
            ReleaseKey(key)
        # wait until some time after taking action
        time.sleep(0.6)
        reward = self._get_reward(action)
        game_over = self._is_over(reward)
        return self.observe(), reward, game_over

    def reset(self):
        return
