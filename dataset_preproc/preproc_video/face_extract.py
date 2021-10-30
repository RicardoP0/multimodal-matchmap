#%%
#https://github.com/timesler/facenet-pytorch
from facenet_pytorch import MTCNN, extract_face 
import torch
import numpy as np
import mmcv, cv2
import os
import matplotlib.pyplot as plt
from PIL import Image


     

# %%

#%%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
print(os.getcwd())
mtcnn = MTCNN(keep_all=True, device=device,image_size=100)
video_dir = "VIDEO_FILES/"
dest_path = 'VIDEO_PROCESSED/'
dir_list = os.listdir(video_dir)
dir_list.sort()


if not os.path.exists(dest_path):
    os.makedirs(dest_path)

        
#%%

# %%
#iemocap
k = 1 #session to process
video_dir = "IEMOCAP_full_release.tar/IEMOCAP_full_release/Session{}/dialog/avi/DivX".format(k)
dir_list = os.listdir(video_dir)
dir_list.sort()
dir_list = [x for x in dir_list if x[0] =='S']
i=0

#%%
dir_list
path = 'datasets/IEMOCAP/CLIPPED_VIDEOS/' + 'Session{}/'.format(k) 
if not os.path.exists(path):
    os.makedirs(path)
dir_list
#%%
#divide each video and manually crop around face
video_dir = "IEMOCAP_full_release.tar/IEMOCAP_full_release/Session{}/dialog/avi/DivX".format(k)
dir_list = os.listdir(video_dir)
dir_list.sort()
dir_list = [x for x in dir_list if x[0] =='S']
path = 'IEMOCAP/CLIPPED_VIDEOS/' + 'Session{}/'.format(k) 
if not os.path.exists(path):
    os.makedirs(path)
for file_name in dir_list:
    print(file_name)
    video = mmcv.VideoReader(video_dir + '/'+file_name)
    if 'F_' in file_name:
        new_file_left = path + file_name[:-4] + '_F.avi'
        new_file_right = path +file_name[:-4] + '_M.avi'
    else:
        new_file_left = path +file_name[:-4] + '_M.avi'
        new_file_right = path + file_name[:-4] + '_F.avi'

    h,w,c = video[0].shape
    
    dim = (300,280)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')  
    #left  
    
    video_tracked = cv2.VideoWriter(new_file_left, fourcc, 25.0, dim)
    i=0
    for frame in video:
        h,w,c = frame.shape
        #left
        #different boxes for each session
        #box (left, upper, right, lower)-tuple
        #ses1 [120:int(h- 690),120:int(w/2.4)]
        #ses2 [150:int(h - 660),120:int(w/2.4)]
        #ses5 [120:int(h - 690),120:int(w/2.4)]
        #[130:int(h/2.18),120:int(w/2.4)]
        video_tracked.write(frame[100:h-100,:300])
    
    video_tracked.release()
    del video_tracked
    print(h,w,c)
    dim = (370,280)
    # #right
    video_tracked = cv2.VideoWriter(new_file_right, fourcc, 25.0, dim)
    for frame in video:
        h,w,c = frame.shape
        #right
        #ses1 [150:int(h - 660),int(w/1.5):int(w-60)]
        #ses2 [150:int(h - 660),int(w/1.5):int(w-60)]
        #ses5 [150:int(h - 660),int(w/1.5):int(w-60)]
        
        video_tracked.write(frame[100:h-100,350:])
    video_tracked.release()
    
    del video, video_tracked    


#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
print(os.getcwd())
mtcnn = MTCNN(keep_all=True, device=device,image_size=2000,margin=5)
i = 1
video_dir = "../../../../datasets/IEMOCAP/CLIPPED_VIDEOS/Session{}/".format(i)
dir_list = os.listdir(video_dir)
dir_list.sort()
dir_list = [x for x in dir_list if x[0] =='S']

dir_list
#%%
file_list = dir_list
path = '../datasets/IEMOCAP/FACE_VIDEOS/Session{}/'.format(i)
if not os.path.exists(path):
    os.makedirs(path)
#%%
#%%
#track using mtcnn
for file_name in file_list:
    video = mmcv.VideoReader(video_dir + file_name)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    frames_tracked = []
    
    for x, frame in enumerate(frames):
        #print('\rTracking frame: {}'.format(i + 1), end='')
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if not boxes is None:
            # print(boxes[0])
            im_array = extract_face(frame, boxes[0],image_size=112,margin=50)
            #im_array = im_array.permute(1,2,0)
            img = im_array #Image.fromar ray(np.uint8(im_array.numpy()))
            # Add to frame list
            frames_tracked.append(img)
        else:
            frames_tracked.append(img)

        
    dim = frames_tracked[0].size
    print(len(frames),len(frames_tracked))
    new_file = path + '/' + file_name
    print(new_file)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
    video_tracked = cv2.VideoWriter(new_file, fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()
    del video, video_tracked, frames_tracked, frames

