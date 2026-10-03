import pygame
import sys
import numpy as np
import pandas as pd
import torch
import os
import cv2
import matplotlib.pyplot as plt
import math
import time
from scipy.special import factorial
from DI_func import Directed_Information, build_DI_inputs
from PMIMEsigClass import PMIMEsig
from PMIMEsigClass import run_PMIMEsig as pmimesigclass
from simulation_systems import sysProdOnly, check_if_unstable
from scipy.io import savemat
    
def build_episode_pairs_from_reward(R: np.ndarray, thr:float = 50.0):
        R = np.asarray(R,dtype=float).ravel()
        reward_idx = np.where(R > thr)[0]
        start = 0
        episode_ends = []
        for idx in reward_idx:
            end = idx + 1
            episode_ends.append((start,end))
            start = end
        if start < len(R):
            episode_ends.append((start,len(R)))
        return episode_ends, reward_idx
    
def discounted_return_per_episode(R,episode_ends, gamma = 0.99):
    R = np.asarray(R,dtype=float).ravel()
    G = np.zeros_like(R, dtype=float)

    for (s,e) in episode_ends:
        running = 0.0
        for t in range(e-1, s - 1, -1):
            running = R[t] + gamma * running
            G[t] = running
    return G


class MyObject(pygame.sprite.Sprite):
    def __init__(self, screen=None):
        super().__init__()
        self.moving = True
        self.image = None
        self.rect = None
        if screen is None:
            self.screen = pygame.display.set_mode((500, 500))
        else:
            self.screen = screen
        self.screen_rect = screen.get_rect()
        self.pos = np.random.random(2) * np.array([self.screen_rect.width, self.screen_rect.height])
        self.angle = np.random.random(1) * 2 * np.pi
        self.speed = 5

    def next_pos(self):
    # Define the safe zone margin
        margin = 10 
        angle_change = np.random.uniform(-1,1)
        self.angle += angle_change

    
        potential_x = self.pos[0] + self.speed * np.cos(self.angle)
        potential_y = self.pos[1] + self.speed * np.sin(self.angle)

    
        half_width = self.rect.width / 2
        half_height = self.rect.height / 2

    # Define the boundaries of the safe "play area"
        min_x_bound = margin + half_width
        max_x_bound = self.screen_rect.width - margin - half_width
        min_y_bound = margin + half_height
        max_y_bound = self.screen_rect.height - margin - half_height

    # Check for upcoming collisions with the SAFE ZONE
    
    # Check horizontal boundaries
    
        if potential_x > max_x_bound:
            potential_x = max_x_bound # Clamp position to the safe boundary
            self.angle = np.pi - self.angle # Reflect horizontal velocity
            self.angle += np.random.uniform(-0.1, 0.1) # Add noise to prevent getting stuck
        
    
        elif potential_x < min_x_bound:
            potential_x = min_x_bound # Clamp position to the safe boundary
            self.angle = np.pi - self.angle # Reflect horizontal velocity
            self.angle += np.random.uniform(-0.1, 0.1)

    # Check vertical boundaries
    
        if potential_y > max_y_bound:
            potential_y = max_y_bound # Clamp position
            self.angle = -self.angle # Reflect vertical velocity
            self.angle += np.random.uniform(-0.1, 0.1)
        
    
        elif potential_y < min_y_bound:
            potential_y = min_y_bound 
            self.angle = -self.angle 
            self.angle += np.random.uniform(-0.1, 0.1)

        self.pos[0] = float(potential_x)
        self.pos[1] = float(potential_y)
        self.rect.centerx = int(self.pos[0])
        self.rect.centery = int(self.pos[1])

    def blit(self):
        self.screen.blit(self.image, self.rect)


class Food(MyObject):
    def __init__(self, screen):
        super().__init__(screen)
        self.image = pygame.image.load('food.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.centerx = int(self.pos[0])
        self.rect.centery = int(self.pos[1])
        self.reward = 100

    def update(self, display=True):
        if self.moving is True:
            self.next_pos()
        if display:
            self.screen.blit(self.image, self.rect)


class Killer(MyObject):
    def __init__(self, screen):
        super().__init__(screen)
        self.image = pygame.image.load('killer.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.centerx = int(self.pos[0])
        self.rect.centery = int(self.pos[1])
        self.reward = -100

    def update(self, display=True):
        if self.moving is True:
            self.next_pos()
        if display:
            self.screen.blit(self.image, self.rect)


class NoneObject(MyObject):
    def __init__(self, screen,image_path):
        super().__init__(screen)
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.centerx = int(self.pos[0])
        self.rect.centery = int(self.pos[1])
        self.reward = 0

    def update(self, display=True):
        if self.moving is True:
            self.next_pos()
        if display:
            self.screen.blit(self.image, self.rect)

class Agent(MyObject):
    def __init__(self, screen):
        super().__init__(screen)
        self.screen_rect = screen.get_rect()
        self.moving = False
        self.event_key = None
        self.image = pygame.image.load('agent.png')
        self.rect = self.image.get_rect()
        self.rect.centery = int(self.screen_rect.height * 0.7)
        self.rect.centerx = self.screen_rect.width // 2
        self.pos = np.array([self.rect.centerx, self.rect.centery], dtype=float)

    def blit(self):
        self.screen.blit(self.image, self.rect)

    def move(self, direction):
        if direction == pygame.K_RIGHT:
            if self.rect.centerx < self.screen_rect.width:
                self.rect.centerx += 1
        elif direction == pygame.K_LEFT:
            if self.rect.centerx > 0:
                self.rect.centerx -= 1
        elif direction == pygame.K_UP:
            if self.rect.centery > 0:
                self.rect.centery -= 1
        elif direction == pygame.K_DOWN:
            if self.rect.centery < self.screen_rect.height:
                self.rect.centery += 1

    def update(self, display=True):
        if self.event_key is not None:
            if self.moving:
                self.move(self.event_key)
        if display:
            self.blit()

    #  0 right 1 left 2 up 3 down 4 no action
    def step(self, action, display=False):
        if action == 0:
            if self.rect.centerx < self.screen_rect.width:
                self.rect.centerx += 3
        elif action == 1:
            if self.rect.centerx > 0:
                self.rect.centerx -= 3
        elif action == 2:
            if self.rect.centery > 0:
                self.rect.centery -= 3
        elif action == 3:
            if self.rect.centery < self.screen_rect.height:
                self.rect.centery += 3
        if display:
            self.blit()
        return self.pos
    
class VKCAM:
    def __init__(self, sigma = 10.0, similarity_thresh = 0.7, max_pos = 80):
        self.sigma = sigma
        self.similarity_thresh = similarity_thresh
        #Memory which contains feature_vector, one_hot, current_position_tuple
        self.memory = []
        self.next_index = 0
        self.max_total_obj = max_pos
    
    def gaussian_kernel(self, x, y):
        x_np = np.asarray(x)
        y_np = np.asarray(y)
        diff = x_np - y_np
        return np.exp(-np.dot(diff, diff) / (2 * self.sigma ** 2))
    
    def add_proto_object(self, feature_vector, pos):
        one_hot = np.zeros(self.max_total_obj)
        one_hot[self.next_index] = 1
        self.memory.append((feature_vector, one_hot, pos))
        assigned_index = self.next_index
        self.next_index += 1
        return assigned_index # Index of the object that just got assigned

    def check_patch_similarity(self, query_vector):
        if not self.memory:
            return None,0.0,-1
        highest_sim_found = 0.0 
        most_similar_po_id = None
        memory_idx_of_most_sim = -1

        for current_memory_idx, (feat_vector, one_hot, pos) in enumerate(self.memory):
            sim_with_this_po = self.gaussian_kernel(query_vector, feat_vector)
            #print(sim_with_this_po)
            if sim_with_this_po > highest_sim_found:
                highest_sim_found = sim_with_this_po
                most_similar_po_id = np.argmax(one_hot)
                memory_idx_of_most_sim = current_memory_idx
        
        if highest_sim_found >= self.similarity_thresh and most_similar_po_id is not None:
            
            return most_similar_po_id, highest_sim_found, memory_idx_of_most_sim
        return None, 0.0, -1
    
    def update_or_store(self, feature_vector, pos):
            existing_po_id, similarity, memory_idx = self.check_patch_similarity(feature_vector)
            if existing_po_id is not None:
                old_feat, old_one_hot, _ = self.memory[memory_idx]
                self.memory[memory_idx] = (old_feat,old_one_hot, pos)
                #print(f"UPDATED PO ID {existing_po_id}")
                return existing_po_id
            else:
                new_po_id = self.add_proto_object(feature_vector,pos)
                return new_po_id
                

    
    def get_memory(self):
        return self.memory


class GKCAM:
    def __init__(self):
        self.agent_patch = None 
        self.agent_position = None

    def store_agent_patch(self, feature_vector, data):
        self.agent_patch = feature_vector
        self.agent_position = data

    def update_agent_position(self, new_pos):
        if self.agent_position is not None:
            self.agent_position = new_pos

    def get_agent_patch(self):
        return self.agent_patch, self.agent_position


class MyGame:
    def __init__(self, wnd_sz=None, bg_color=None, n_turtles = 2, n_grass = 3):
        self.wnd_sz = wnd_sz
        self.bg_color = bg_color
        if wnd_sz is None:
            self.wnd_sz = (240, 240)
        if bg_color is None:
            self.bg_color = (0, 0, 0)

        self.screen = pygame.display.set_mode((self.wnd_sz[0], self.wnd_sz[1]))
        self.agent = Agent(self.screen)
        self.foods = pygame.sprite.Group()
        self.none_objects = pygame.sprite.Group()
        self.killers = pygame.sprite.Group()
        for _ in range(n_turtles):
            self.foods.add(Food(self.screen))
        grass_files = ['grass4.png', 'grass3.png', 'grass2.png']
        for i in range(n_grass):
            self.none_objects.add(NoneObject(self.screen, grass_files[i % len(grass_files)]))


        self.score = 0
        self.prev_frame = None
        self.prev_gray_frame = None

    def bounce_back(self):
    # Get all objects to check for collisions
        all_objs = list(self.foods.sprites()) + list(self.none_objects.sprites()) 
    
    # Check each pair of objects for collision
        for i in range(len(all_objs)):
            for j in range(i + 1, len(all_objs)):
                obj_i = all_objs[i]
                obj_j = all_objs[j]
            
           
                expanded_rect_i = obj_i.rect.inflate(15, 15)
                expanded_rect_j = obj_j.rect.inflate(15, 15)
            
                if expanded_rect_i.colliderect(expanded_rect_j):
                # Get object positions
                    pos_i = np.array([obj_i.pos[0], obj_i.pos[1]], dtype=float)
                    pos_j = np.array([obj_j.pos[0], obj_j.pos[1]], dtype=float)
                
                # Compute collision normal (vector from object i to j)
                    n = pos_j - pos_i
                    norm = np.linalg.norm(n)
                
                # Handle case where objects are at the same position
                    if norm < 1.0:
                    # Create random direction if objects are too close
                        n = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                        norm = np.linalg.norm(n)
                
                    n /= norm  # normalize
                
                # Calculate velocities from angles
                    v_i = np.array([np.cos(obj_i.angle), np.sin(obj_i.angle)]) * obj_i.speed
                    v_j = np.array([np.cos(obj_j.angle), np.sin(obj_j.angle)]) * obj_j.speed
                
                # Exchange momentum 
                    v_i = v_i.flatten()
                    v_j = v_j.flatten()

                    v_i_new = v_i - np.dot(v_i - v_j, n) * n
                    v_j_new = v_j - np.dot(v_j - v_i, -n) * n
                
                # Convert back to angle and speed
                    obj_i.angle = math.atan2(v_i_new[1], v_i_new[0])
                    obj_j.angle = math.atan2(v_j_new[1], v_j_new[0])
                
                # Add small random adjustment to prevent repeated collisions
                    obj_i.angle += np.random.uniform(-0.3, 0.3)
                    obj_j.angle += np.random.uniform(-0.3, 0.3)
                
                # Calculate minimum separation needed (use the sum of half-widths/heights)
                    min_sep = max(obj_i.rect.width, obj_i.rect.height) / 2 + max(obj_j.rect.width, obj_j.rect.height) / 2
                    min_sep += 15  # Add extra buffer
                
                # If objects are too close, separate them immediately
                    if norm < min_sep:
                    # Move objects apart along the normal vector
                        movement = (min_sep - norm + 2.0) * 1.1  # Extra factor for safety
                    
                    # Move objects in opposite directions
                        new_pos_i = pos_i - n * movement / 2
                        new_pos_j = pos_j + n * movement / 2
                    
                    # objects stay within screen boundaries
                        screen_w = self.screen.get_width()
                        screen_h = self.screen.get_height()
                    
                    # Apply boundary constraints
                        new_pos_i[0] = max(obj_i.rect.width/2, min(new_pos_i[0], screen_w - obj_i.rect.width/2))
                        new_pos_i[1] = max(obj_i.rect.height/2, min(new_pos_i[1], screen_h - obj_i.rect.height/2))
                        new_pos_j[0] = max(obj_j.rect.width/2, min(new_pos_j[0], screen_w - obj_j.rect.width/2))
                        new_pos_j[1] = max(obj_j.rect.height/2, min(new_pos_j[1], screen_h - obj_j.rect.height/2))
                    
                    # Update positions
                        obj_i.pos = new_pos_i
                        obj_i.rect.centerx = int(new_pos_i[0])
                        obj_i.rect.centery = int(new_pos_i[1])
                    
                        obj_j.pos = new_pos_j
                        obj_j.rect.centerx = int(new_pos_j[0])
                        obj_j.rect.centery = int(new_pos_j[1])



    def update(self, score):
        self.screen.fill(self.bg_color)
        self.agent.update()
        for food in self.foods.sprites():
            food.update()
        for killer in self.killers.sprites():
            killer.update()
        for o in self.none_objects.sprites():
            o.update()

        self.bounce_back()

        self.screen.blit(score.render("scores: " + str(self.score), True, pygame.Color(255, 0, 0), pygame.Color(230, 230, 230)),
                         (50, 50))

    def check_events(self, keyboard=True):
        if keyboard:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    self.agent.moving = True
                    self.agent.event_key = event.key
                elif event.type == pygame.KEYUP:
                    self.agent.moving = False
        reward = 0
        ls = pygame.sprite.spritecollide(self.agent, self.foods, True)
        n = len(ls)
        if n != 0:
            self.score += n
            reward = 100
            for i in range(n):
                self.foods.add(Food(self.screen))

        ls = pygame.sprite.spritecollide(self.agent, self.killers, False)
        if len(ls) != 0:
            reward = -100
            if not keyboard:
                return reward
            text = pygame.font.Font('freesansbold.ttf', 115)
            self.screen.blit(text.render('killed', True, pygame.Color(255, 0, 0), pygame.Color(230, 230, 230)), (500, 300))
            pygame.display.flip()
            close = False
            while not close:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
        return reward

    def run(self):
        pygame.init()
        pygame.display.set_caption("myGame")
        score = pygame.font.Font('freesansbold.ttf', 30)

        while True:
            # fill color
            self.check_events()

            self.update(score)

            # visualiaze the window
            pygame.display.flip()

    
    def gamma_kernel(self,k,m,size):
        # make a grid of size size
        x,y = np.meshgrid(

            np.linspace(-size // 2 , size // 2, size),
            np.linspace(-size // 2, size // 2, size)

        )
        r = np.sqrt(x**2 + y**2)
        holder = (m**(k+1)) / (2 * np.pi * factorial(k))
        gamma = holder * (r**(k - 1)) * np.exp(-m * r)
        return gamma 

    def multiscale_kernel_adaptation(self,grayscale_image, gamma_total):

      from scipy.ndimage import convolve
      self.grayscale_image = grayscale_image
      self.response = convolve(grayscale_image,gamma_total,mode = 'reflect')
      return self.response
    

    
    def build_bounding_box(self, center, boundLength, h, w):
     R, C = center
     half = boundLength // 2
     R1 = max(0, R - half)
     C1 = max(0, C - half)
     R2 = min(h, R + half)
     C2 = min(w, C + half)
     return R1, R2, C1, C2

    def extract_salient_patches(self, saliency_map, grayscale_image, patch_size, top_n, THs, zero_out_size):
     display = False
     smap = saliency_map.copy()
     h, w = smap.shape
     patches = []

     for i in range(top_n):
        # find global max above threshold
        if display: 
            cv2.imshow(f"Smap Iter {i}", (smap * 255).astype(np.uint8))
            cv2.waitKey(0) 
        peak_val = smap.max()
        if peak_val < THs:
            break
        # linear index → 2d coords
        idx = np.argmax(smap)
        R, C = divmod(idx, w)
        center = (R,C)
        # get clamped bounding box
        R1, R2, C1, C2 = self.build_bounding_box(center, patch_size, h, w)
        
        image_patch_gray = grayscale_image[R1:R2, C1:C2]
        if image_patch_gray.shape != (patch_size, patch_size):
            R1_zero_err, R2_zero_err, C1_zero_err, C2_zero_err = self.build_bounding_box(center,patch_size,h,w)
            smap[R1_zero_err:R2_zero_err, C1_zero_err:C2_zero_err] = 0 
            continue
        patches.append(((C,R), image_patch_gray, saliency_map[R,C]))
        R1_zero, R2_zero, C1_zero,C2_zero = self.build_bounding_box((R,C), zero_out_size, h, w)
        smap[R1_zero:R2_zero, C1_zero: C2_zero] = 0
     return patches


    def compute_motion_saliency(self, prev_frame, current_frame):

        prev_gray = np.dot(prev_frame[..., :3], [0.2989, 0.5870, 0.1140])
        curr_gray = np.dot(current_frame[..., :3], [0.2989, 0.5870, 0.1140])
        motion_diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion_saliency = cv2.threshold(motion_diff, 25, 255, cv2.THRESH_BINARY)
        return motion_saliency.astype(np.float32) / 255.0
    


    def step(self, action):
        self.screen.fill(self.bg_color)
        agent_pos = self.agent.step(action, display=True)
        food_pos = []
        killer_pos = []
        o_pos = []
        for food in self.foods.sprites():
            food.update()
            food_pos.append(food.rect.center)
        for killer in self.killers.sprites():
            killer.update()
            killer_pos.append(killer.rect.center)
        for o in self.none_objects.sprites():
            o.update()
            o_pos.append(o.rect.center)
        
        self.bounce_back()
        #Get the raw frame after action is done
        raw_frame = pygame.surfarray.array3d(self.screen)
        raw_frame = np.transpose(raw_frame, (1,0,2))
        reward = self.check_events(keyboard=False)
        return reward, agent_pos, food_pos, killer_pos, o_pos, raw_frame


if __name__ == "__main__":

    display = True
    if not display:
      os.environ["SDL_VIDEODRIVER"] = "dummy"
    gm = MyGame()
    pygame.init()
    pygame.display.set_caption("myGame")
    score = pygame.font.Font('freesansbold.ttf', 30)
    flag = True

    #Create the Kernels for convolution
    kernels = [
        gm.gamma_kernel(1, 4, 10),
        gm.gamma_kernel(10, 4, 10),
        gm.gamma_kernel(20, 4, 10),
        gm.gamma_kernel(30, 4, 10)
    ]
    kernels_normalized = []
    for k_component in kernels:
        sum_k_comp = np.sum(k_component)
        if np.abs(sum_k_comp) > 1e-9:
            kernels_normalized.append(k_component / sum_k_comp)
        else:
            kernels_normalized.append(k_component)
    total_gamma_kernel = np.zeros_like(kernels_normalized[0])
    for l, k_norm in enumerate(kernels_normalized):
        total_gamma_kernel += ((-1)**l) * k_norm



    vkcam = VKCAM(sigma = 10.0, similarity_thresh = 0.85, max_pos=80)
    gkcam = GKCAM()

    po_dist_relative_time_series = {}
    reward_time_series = []

    

    frame_count = 0 

    PMIME_params = dict(
        Lmax = 3,
        T = 1,
        nsur = 100,
        alpha = 0.10,
        showtxt = 2,
        nnei = 5,
        type = 'rp',
        mi_method = 'nonexact',
        cmi_method = 'nonexact',
        full_logs = False,
        cpu_pool = 3,
        execution = 'serial',
        include_interactions = False,
        use_backward_revision = False,
        method = 0
    )
    max_score = 0

    while flag: 
        step_choose = np.random.randint(0,5)
        reward, agent_pos, food_rect, killer_rect, o_rect,raw_frame = gm.step(4)
        noise = 0
        reward_time_series.append(reward + noise)
        if reward == 100: max_score +=1

        #Make the frame grayscale and apply the gamma kernel to extract the saliency map

        current_grayscale_uint8 = np.dot(raw_frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        current_grayscale = current_grayscale_uint8.astype(np.float32)



        #print(f"Stage 0: Grayscale Input | Min={np.min(current_grayscale):.1f}, Max={np.max(current_grayscale):.1f}, Shape={current_grayscale.shape}, Dtype={current_grayscale.dtype}")
        #if display: cv2.imshow("0. Grayscale Input", current_grayscale_uint8); cv2.waitKey(30)





        gamma_saliency_raw = gm.multiscale_kernel_adaptation(current_grayscale, total_gamma_kernel)
        display_map_raw_convo = gamma_saliency_raw.copy()
        cv2.normalize(display_map_raw_convo, display_map_raw_convo, 0, 255, cv2.NORM_MINMAX)
        #if display: cv2.imshow("1. Raw Convo (Scaled)", display_map_raw_convo.astype(np.uint8)); cv2.waitKey(30)
        #print(np.max(gamma_saliency_raw),np.min(gamma_saliency_raw))


        



        gamma_saliency_abs = np.abs(gamma_saliency_raw)
        display_map_abs = gamma_saliency_abs.copy()
        cv2.normalize(display_map_abs, display_map_abs, 0, 255, cv2.NORM_MINMAX) # Already positive
        #if display: cv2.imshow("2. Abs Value of Convo", display_map_abs.astype(np.uint8)); cv2.waitKey(30)



        blurred_saliency = cv2.GaussianBlur(gamma_saliency_abs,(25,25), 5.2)
        #print(f"Stage 3: Blurred Saliency | Min={np.min(blurred_saliency):.4f}, Max={np.max(blurred_saliency):.4f}")
        display_map_blur = blurred_saliency.copy()
        cv2.normalize(display_map_blur, display_map_blur, 0, 255, cv2.NORM_MINMAX)
        #if display: cv2.imshow("3. Blurred Saliency", display_map_blur.astype(np.uint8)); cv2.waitKey(30)


        



        min_val = np.min(blurred_saliency)
        max_val = np.max(blurred_saliency)
        if ( max_val - min_val) > 1e-9:
            blurred_saliency_norm = (blurred_saliency - min_val) / (max_val - min_val)
        else:
            blurred_saliency_norm = np.zeros_like(blurred_saliency)

        #print(f"Stage 3.5: Blurred Saliency Normalized to [0,1] | Min={np.min(blurred_saliency_norm):.2f}, Max={np.max(blurred_saliency_norm):.2f}")
        #if display: cv2.imshow("3.5 Blurred Saliency Norm [0,1]", (blurred_saliency_norm * 255).astype(np.uint8)); cv2.waitKey(30)

        
        alpha_exp = 2.5
        exponentiated_saliency = np.power(blurred_saliency_norm, alpha_exp)
        #print(f"Stage 4: Exponentiated Saliency | Min={np.min(exponentiated_saliency):.4e}, Max={np.max(exponentiated_saliency):.4e}") # Use scientific for potentially small/large
        display_map_exp = exponentiated_saliency.copy()
        #if display: cv2.imshow("4. Exponentiated Saliency (Scaled)", display_map_exp.astype(np.uint8)); cv2.waitKey(30)

        min_final_sal = np.min(exponentiated_saliency)
        max_final_sal = np.max(exponentiated_saliency)
        if (max_final_sal - min_final_sal) > 1e-9:
            final_normalized_gamma_map = (exponentiated_saliency - min_final_sal) / (max_final_sal - min_final_sal)
        else:
            final_normalized_gamma_map = np.zeros_like(exponentiated_saliency)

        normalized_gamma_saliency = final_normalized_gamma_map
    



        


                                          ################# MOTION DETECTION ###################
        motion_saliency = np.zeros_like(current_grayscale, dtype=np.float32)
        if gm.prev_gray_frame is not None:
            diff_frame = cv2.absdiff(gm.prev_gray_frame, current_grayscale)
            _, motion_saliency_binary = cv2.threshold(diff_frame, 25,255,cv2.THRESH_BINARY)
            motion_saliency = motion_saliency_binary.astype(np.float32) / 255.0
        combined_saliency_unnorm =    1 * normalized_gamma_saliency +  0 * motion_saliency
        min_cs, max_cs = np.min(combined_saliency_unnorm), np.max(combined_saliency_unnorm)
        if (max_cs - min_cs) > 1e-9:
            combined_saliency = (combined_saliency_unnorm - min_cs) / (max_cs - min_cs)
        else:
            combined_saliency = np.zeros_like(combined_saliency_unnorm)
        
        gm.prev_gray_frame = current_grayscale_uint8.copy()
        gm.prev_gray_frame = current_grayscale.copy()

        salient_patches = gm.extract_salient_patches(combined_saliency, current_grayscale, 16,7,0.05, 32)



        if gkcam.get_agent_patch() is None:
          
          agent_center_x, agent_center_y = agent_pos
          center_r, center_c = int(agent_center_y), int(agent_center_x)
          r1, r2, c1, c2 = gm.build_bounding_box((center_r,center_c),16,
                                                 current_grayscale.shape[0],current_grayscale.shape[1])
          PA_patch = current_grayscale[r1:r2, c1:c2]
          if PA_patch.shape == (16,16):
            flat_PA_patch = PA_patch.flatten()
            gkcam.store_agent_patch(flat_PA_patch,agent_pos)
          else:
              print("AGENT PATCH NOT 16X16")
        else:
            gkcam.update_agent_position(agent_pos)


        filtered = []
        for (x, y), patch_gray, score in salient_patches:
        # build a pygame.Rect for this 16×16 patch
            patch_rect = pygame.Rect(x - 8, y - 8, 16, 16)
        # if it overlaps the agent, drop it
            if patch_rect.colliderect(gm.agent.rect):
                continue
            filtered.append(((x, y), patch_gray, score))
        salient_patches = filtered


           # Convert grayscale to BGR 
        saliency_color = cv2.cvtColor((combined_saliency*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Draw a green rectangle for each patch
        for ((x, y), _, _) in salient_patches:
         cv2.rectangle(saliency_color, (x - 8, y - 8 ), (x + 8, y + 8), (0, 255, 0), 1)
        cv2.imshow("Saliency Map with Patches", saliency_color)
        cv2.waitKey(1)

        curr_frame_seen_ids = set()
        po_min_dist_this_frame = {}
        


        for ((x, y), patch_gray, score) in salient_patches:
            flattened_patch = patch_gray.flatten()  # feature vect
            min_p, max_p = flattened_patch.min(),  flattened_patch.max()
            flattened_patch_normalized = (flattened_patch - min_p) / (max_p - min_p)
            po_id = vkcam.update_or_store(flattened_patch_normalized,(x,y))
            if po_id is not None:
                curr_frame_seen_ids.add(po_id)
                dist = math.sqrt((x - agent_pos[0])**2 + (y - agent_pos[1])**2)
                if po_id not in po_min_dist_this_frame:
                    po_min_dist_this_frame[po_id] = dist
                else:
                    po_min_dist_this_frame[po_id] = min(po_min_dist_this_frame[po_id], dist)

        all_ids_ever = range(vkcam.next_index)
        for po_id in all_ids_ever:
            if po_id not in po_dist_relative_time_series:
                po_dist_relative_time_series[po_id] = [np.nan] * frame_count
            
            if po_id in curr_frame_seen_ids and po_id in po_min_dist_this_frame:
                po_dist_relative_time_series[po_id].append(po_min_dist_this_frame[po_id])
            else:
                po_dist_relative_time_series[po_id].append(np.nan)




        if max_score == 10:
           flag = False
        pygame.display.flip()

        frame_count += 1
    
    mem = vkcam.get_memory()
    num_objs = len(mem)
    if num_objs > 0:
        fig1, axes1 = plt.subplots(1, num_objs, figsize=(num_objs * 2.5, 3))
    # If only one proto-object exists, make sure axes1 is iterable.
        if num_objs == 1:
           axes1 = [axes1]
        for idx, (fv, one_hot, pos) in enumerate(mem):
            try:
                patch_img = fv.reshape((16, 16))
            except Exception as e:
               print(f"Error reshaping proto-object {idx}: {e}")
               patch_img = np.zeros((16, 16))
            axes1[idx].imshow(patch_img, cmap="gray")
            axes1[idx].set_title(f"Object ID {idx}")
            axes1[idx].axis("off")
    plt.tight_layout()
    
                ########## CAUSALITY ANALYSIS ##########



   ########## DATA PREP ##########

    stride = 1

    all_po_ids = sorted(po_dist_relative_time_series.keys())
    total_frames = len(reward_time_series)
    
    obj_indices = []
    for po_id in all_po_ids:
        actual_seen_count = pd.Series(po_dist_relative_time_series[po_id]).count()
        if actual_seen_count > (total_frames * 0.15): 
            obj_indices.append(po_id)
        else:
            print(f"Skipping Flicker PO {po_id}: Only seen for {actual_seen_count} frames.")

    # Impute missing observations
    imputed_trajectories = {}
    for po_id in obj_indices:
        raw_traj_series = pd.Series(po_dist_relative_time_series[po_id])
        clean_traj = raw_traj_series.interpolate(limit=5).fillna(1.0).to_numpy()
        imputed_trajectories[po_id] = clean_traj[::stride]


    R_original = np.array(reward_time_series, dtype=float).ravel()

    # DI target: smoothed binary reward 
    binary_r = np.where(R_original > 50, 1.0, 0.0)[::stride]
    R_di = np.convolve(binary_r, np.ones(4)/4, mode='same')
    if R_di.max() > 0:
        R_di = R_di / R_di.max()

    # PMIME target: discounted return 
    R_pmime = np.array(reward_time_series, dtype=float).ravel()[::stride]

    # Global Min-Max Normalized Trajectories
    all_trajs_stacked = np.array(list(imputed_trajectories.values()))
    g_min, g_max = all_trajs_stacked.min(), all_trajs_stacked.max()
    
    norm_trajs = []
    for po_id in obj_indices:
        t = imputed_trajectories[po_id]
        norm_t = (t - g_min) / (g_max - g_min) if (g_max - g_min) > 1e-9 else np.full_like(t, 0.5)
        norm_trajs.append(norm_t)

    n_episodes = len(np.where(R_original > 50)[0])
    print(f"[ANALYSIS] Frames: {len(R_original)} | Strided: {len(R_di)} | "
          f"Episodes (rewards): {n_episodes} | Objects: {len(obj_indices)}")

    plt.figure(figsize=(10, 4))
    for idx, series in enumerate(norm_trajs):
        plt.plot(series, label=f"PO {obj_indices[idx]}")
    plt.plot(R_di, 'k--', linewidth=2, alpha=0.7, label="Reward (smoothed binary)")
    plt.title("Unified Inputs for Causality Analysis")
    plt.legend(fontsize=8)
    plt.show()



    ########## DIRECTED INFORMATION ANALYSIS ##########
    
    lag = 2
    alpha_renyi = 1.01
    di_scores_raw = np.zeros(len(obj_indices))

    print("\n--- Running DI (Matrix Renyi Entropy) ---")
    di_start = time.perf_counter()
    for m in range(len(obj_indices)):
        X, Y, Z = build_DI_inputs(norm_trajs[m], R_di, lag)
        with torch.no_grad():
            di_val = Directed_Information(X, Y, Z, alpha_renyi)
        di_scores_raw[m] = float(di_val.cpu().item())
    di_elapsed = time.perf_counter() - di_start
    print(f"DI completed in {di_elapsed:.3f} seconds")


    ########## PMIME ANALYSIS ##########

    print("\n--- Running PMIME (Greedy Mixed Embedding) ---")
    # Build matrix: [Target | Obj0 | Obj1 | ...]
    data_matrix = np.hstack([R_pmime.reshape(-1, 1)] + [t.reshape(-1, 1) for t in norm_trajs])
    savemat('pmime_input_data6.mat', {'data_matrix': data_matrix})

    print(data_matrix)
    
    allM = PMIMEsig.normalize_data(data_matrix)
    task_params = PMIME_params.copy()
    task_params.update({"data": data_matrix, "allM": allM, "seeds": {0: 12345}})

    pmime_start = time.perf_counter()
    pmime_output = pmimesigclass(0, **task_params)
    pmime_elapsed = time.perf_counter() - pmime_start
    print(f"PMIME completed in {pmime_elapsed:.3f} seconds")


    # Extract R-measure scores
    K = data_matrix.shape[1]
    pmime_scores_raw = np.zeros(len(obj_indices))
    rm_col = None
    for item in pmime_output:
        if hasattr(item, "shape") and item.shape in [(K, 1), (K,)]:
            rm_col = item.flatten()
            break
    
    if rm_col is not None:
        for i in range(len(obj_indices)):
            pmime_scores_raw[i] = float(rm_col[1 + i])

    print(rm_col)

    ########## 4. COMPARISON & RESULTS ##########

    di_final = di_scores_raw.copy()
    pmime_final = pmime_scores_raw.copy()

    
    labels = [f"PO {po_id}" for po_id in obj_indices]

    print("\n" + "="*60)
    print(f"{'Object':<20} | {'DI':<15} | {'PMIME':<15}")
    print("-" * 60)
    for i, po_id in enumerate(obj_indices):
        print(f"{labels[i]:<20} | {di_final[i]:<15.4f} | {pmime_final[i]:<15.4f}")
    print("="*60)
    print(f"\nDI top pick: {labels[np.argmax(di_final)]}")
    print(f"PMIME top pick: {labels[np.argmax(pmime_final)]}")
    print(f"\nTiming:  DI = {di_elapsed:.3f}s  |  PMIME = {pmime_elapsed:.3f}s")

    ############## HEATMAP VISUALIZATION ############
    import seaborn as sns
    
    combined_mat = np.stack([di_final, pmime_final], axis=1)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(combined_mat, annot=True, fmt='.3f', cmap='viridis', 
                yticklabels=labels, xticklabels=['DI Score', 'PMIME Score'])
    plt.title("Causality Heatmap: DI vs PMIME")
    plt.ylabel("Proto-Objects")
    plt.show()

    cv2.destroyAllWindows()
    sys.exit()
