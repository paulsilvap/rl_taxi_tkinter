import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from gym import spaces
import pdb

PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 5
WIDTH = 5
NUM_NODES = HEIGHT * WIDTH - 1
MAX_SOC = 24
MAX_STEPS = 240
CHARGING_REWARD = 0
PICKUP_REWARD = 1
DROP_REWARD = 25
DISCHARGE_RATE = 0.2
NOBATTERY_PENALTY = MAX_STEPS - (MAX_SOC / DISCHARGE_RATE)
CHARGING_RATE = 2.0

# np.random.seed(1)

class Env(tk.Tk):
    def __init__(self, normal_render = False):
        super(Env, self).__init__()
        self.action_space = ['u','d','l','r','stay']
        self.action_size = len(self.action_space)
        self.observation_space = spaces.Box(
            # low= np.array([0.]* (NUM_NODES+1 + 1 + 4)), 
            # high = np.concatenate(
            #     (np.concatenate((np.array([np.float32(MAX_SOC)]), np.array([np.float32(NUM_NODES)]*4))), 
            #     np.array([np.float32(5)]*(NUM_NODES+1)))), 
            low= np.array([0.]* (1 + 4)), 
            high =np.concatenate((np.array([np.float32(MAX_SOC)]), np.array([np.float32(NUM_NODES)]*4))), 
            dtype= np.float32)

        self.title('Taxi')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, WIDTH * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.normal_render = normal_render
        
        self.counter = 0
        self.rewards = []
        self.goal = []
        self.ev_soc = MAX_SOC
        self.passenger = False
        self.no_pickup_counter = 0
        self.no_drop_counter = 0

        self.car_location = self.canvas.coords(self.car)
        self.cs_location = [4, 4]
        self.set_reward(self.cs_location, CHARGING_REWARD)
        self.user_location = self.random_location()
        self.set_reward(self.user_location, PICKUP_REWARD)
        self.user_dest_location = self.random_location()
        self.set_reward(self.user_dest_location, DROP_REWARD)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                            height=HEIGHT * UNIT,
                            width=WIDTH * UNIT)

        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.rewards = []
        self.goal = []
        x, y = UNIT/2, UNIT/2

        self.car = canvas.create_image(x, y, image=self.shapes[0])

        self.taken_locations = [[x,y]]

        canvas.pack()

        return canvas

    def random_location(self):
        location = [np.random.randint(WIDTH-1), np.random.randint(HEIGHT-1)]
        while [(UNIT * x) + UNIT / 2 for x in location] in self.taken_locations:
            location = [np.random.randint(WIDTH-1), np.random.randint(HEIGHT-1)]
        return location

    def load_images(self):
        img_unit = int(UNIT*0.6)
        car_image = Image.open("./img/car.png").convert('RGBA').resize((img_unit, img_unit))
        station_image = Image.open("./img/green-energy.png").convert('RGBA').resize((img_unit, img_unit))
        # User image taken from https://uxwing.com/person-profile-image-icon/
        user_image = Image.open("./img/person.png").convert('RGBA').resize((img_unit, img_unit))
        # Dest image taken from https://uxwing.com/area-icon/
        user_dest_image = Image.open("./img/area.png").convert('RGBA').resize((img_unit,img_unit))
        car = PhotoImage(car_image)
        station = PhotoImage(station_image)
        user = PhotoImage(user_image)
        dest = PhotoImage(user_dest_image)
        #TODO add image for destination

        return car, station, user, dest

    #TODO
    def reset_rewards(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()

        self.no_pickup_counter = 0
        self.no_drop_counter = 0

        self.taken_locations = [self.taken_locations[0]]

        self.set_reward(self.cs_location, CHARGING_REWARD)
        self.user_location = self.random_location()
        self.set_reward(self.user_location, PICKUP_REWARD)
        self.user_dest_location = self.random_location()
        self.set_reward(self.user_dest_location, DROP_REWARD)

        self.passenger = False

    #TODO complete rewards
    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward == CHARGING_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[1])
            self.taken_locations.append([(UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2])

        elif reward == PICKUP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[2])
            self.taken_locations.append([(UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2])

        elif reward == DROP_REWARD:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])
            self.taken_locations.append([(UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2])

            self.goal.append(temp['figure'])

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        self.rewards.append(temp)

    #TODO
    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_done'] = False
        check_list['if_goal'] = False
        rewards = 0

        # pdb.set_trace()

        for reward in self.rewards:
            if reward['state'] == state:
                if reward['reward'] == PICKUP_REWARD and not self.passenger:
                    rewards += reward['reward']
                    self.passenger = True
                    self.canvas.itemconfigure(reward['figure'], state='hidden')
                if reward['reward'] == DROP_REWARD and self.passenger:
                    rewards += reward['reward']
                    check_list['if_goal'] = True
                    self.passenger = False
                if reward['reward'] == CHARGING_REWARD and not self.passenger:
                    self.ev_soc += CHARGING_RATE

        if not self.passenger:
            rewards -= PICKUP_REWARD

        if self.passenger and self.no_drop_counter >= 8:
            rewards -= PICKUP_REWARD * 2

        if self.ev_soc <= 0:
            check_list['if_done'] = True
            rewards -= (MAX_STEPS - self.counter) * 1

        if self.counter == MAX_STEPS:
            check_list['if_done'] = True

        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x,y]

    def state_to_node(self, coords):
        x, y = coords
        node = int((y * WIDTH) + x)
        return node
    
    #TODO
    def get_state(self):
        states = np.array(np.zeros(self.observation_space.shape[0], dtype=np.float32))
        states[0] = self.ev_soc
        states[1] = self.state_to_node(self.coords_to_state(self.canvas.coords(self.car)))
        if self.passenger:
            self.user_location = self.coords_to_state(self.canvas.coords(self.car))
        states[2] = self.state_to_node(self.user_location)
        states[3] = self.state_to_node(self.user_dest_location)
        states[4] = self.state_to_node(self.cs_location)
        return states

    def reset(self):
        self.update()
        if self.normal_render:
            time.sleep(0.5)
        x, y = self.canvas.coords(self.car)
        self.canvas.move(self.car, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_rewards()
        self.ev_soc = MAX_SOC
        self.counter = 0
        self.no_pickup_counter = 0
        self.no_drop_counter = 0
        return self.get_state()

    #TODO
    def step(self, action):
        self.counter += 1
        self.render()

        next_coords = self.move(self.car, action)
        check = self.check_if_reward(self.coords_to_state(next_coords))
        done = check['if_done']
        
        reward = check['rewards']

        if check['if_goal']:
            self.reset_rewards()

        if not self.passenger:
            self.no_pickup_counter += 1
        elif self.passenger:
            self.no_drop_counter += 1

        self.canvas.tag_raise(self.car)

        s_ = self.get_state()

        return s_, reward, done

    def move(self, target, action):
        s = self.canvas.coords(target)

        base_action = np.array([0, 0])

        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.ev_soc -= DISCHARGE_RATE

        self.canvas.move(target, base_action[0], base_action[1])

        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        if self.normal_render:
            time.sleep(0.07)
        self.update()