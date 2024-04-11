#housekeeping
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import winsound
import playsound
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) #set line thickness and landmark point size

# compute mouth aspect ratio
def MAR(img, landmarks):
  mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178] #these are in order of p1 to p8
  height, width, _ = img.shape #get the height and width of the frame
  coords = []
  for index in mouth_indices:
    #denormalize each landmark point to get actual coords of the point
    x = int(landmarks.landmark[index].x * width) # is casting to int necessary???
    y = int(landmarks.landmark[index].y * height)
    
    #store each coord tuple in list
    coords.append((x, y))
    
  #calculate distances
  a = dist.euclidean(coords[1], coords[7]) #p2 and p8
  b = dist.euclidean(coords[2], coords[6]) #p3 and p7
  c = dist.euclidean(coords[3], coords[5]) #p4 and p6
  d = dist.euclidean(coords[0], coords[4]) #p1 and p5
    
	#calculate the mar
  mar = (a + b + c)/(2*d)
  return mar

def EAR(img, landmarks):
  height, width, _ = img.shape #get the height and width of the frame
  #RIGHT EYE
  r_indices = [33, 160, 158, 133, 153, 144]
  r_coords = []
  
	#LEFT EYE
  l_indices = [362, 385, 387, 263, 373, 380]
  l_coords = []
  for i in range(6):
    r_x = int(landmarks.landmark[r_indices[i]].x * width)
    r_y = int(landmarks.landmark[r_indices[i]].y * height)
    l_x = int(landmarks.landmark[l_indices[i]].x * width)
    l_y = int(landmarks.landmark[l_indices[i]].y * height)
    #store each coord tuple in left/right eye list
    r_coords.append((r_x, r_y))
    l_coords.append((l_x, l_y))
    
  #calculate distances for right eye
  r_a = dist.euclidean(r_coords[1], r_coords[5]) #p2 and p6
  r_b = dist.euclidean(r_coords[2], r_coords[4]) #p3 and p5
  r_c = dist.euclidean(r_coords[0], r_coords[3]) #p1 and p4
  #calculate distances for left eye
  l_a = dist.euclidean(l_coords[1], l_coords[5]) #p2 and p6
  l_b = dist.euclidean(l_coords[2], l_coords[4]) #p3 and p5
  l_c = dist.euclidean(l_coords[0], l_coords[3]) #p1 and p4
  #calculate ear for both right and left eye
  r_ear = (r_a + r_b)/(2*r_c)
  l_ear = (l_a + l_b)/(2*l_c)
  #get avg between the two
  ear = (r_ear + l_ear)/2.0
  return ear

#initialize constants and thresholds
EAR_THRESH = 0.20
MAR_THRESH = 0.60
EAR_CONSECUTIVE_FRAMES = 15
yawns = 0
yawn_thresh = 2
yawn_status = False 
COUNTER = 0
ALARM_ON = False
frames_since_last_yawn = 0
max_frames_without_yawn = 300

cap = cv2.VideoCapture(0)

#initialize face mesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    ret, image = cap.read()
    #image = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
    #mark the img as not writeable to pass by reference
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # draw the face mesh annotations on the img
    image.flags.writeable = True
    previous_frame_status = yawn_status
    frames_since_last_yawn += 1 #iterate frames to count how long without a yawn
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #change img back to bgr for better viewing
    #image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks( #draws the face mesh on the img
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        
        #GET MAR
        mar = MAR(image, face_landmarks)
        cv2.putText(image, "mouth aspect ratio: " + str(mar), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
        if mar > MAR_THRESH:
           frames_since_last_yawn = 0 #reset frames without yawn count
           #yawns += 1
           yawn_status = True
           cv2.putText(image, "yawn count: " + str(yawns+1), (0,350),cv2.FONT_HERSHEY_SIMPLEX, 
                       1,(0,255,127),2)
        else:
           yawn_status = False
        if previous_frame_status == True and yawn_status == False:
                    yawns += 1 #iterates yawn count when the mouth closes
        if frames_since_last_yawn >= max_frames_without_yawn:
           #frames_since_last_yawn = 0 #reset
           yawns = 0
        if yawns >= yawn_thresh:
           cv2.putText(image, "yawn threshold passed", (0,370), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,127), 2)
        
        
        #cv2.putText(image, "mouth aspect ratio: " + str(mar), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #if mar > 0.6:
             #yawn_status = True
             #output_text = "yawn count: " + str(yawns+1)
             #cv2.putText(image, output_text, (0,350),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,127),2)
        #else:
             #yawn_status = False
             
				#counts each yawn
        #if previous_frame_status == True and yawn_status == False:
                    #yawns += 1
                    
        #GET EAR
        ear = EAR(image, face_landmarks)
        cv2.putText(image, "eye aspect ratio: " + str(ear), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if ear < EAR_THRESH:
           COUNTER+=1
           if COUNTER >= EAR_CONSECUTIVE_FRAMES:
              if not ALARM_ON:
                 ALARM_ON = True 
                 winsound.PlaySound("alarm1.wav", winsound.SND_ASYNC | winsound.SND_ALIAS)


              cv2.putText(image, "drowsy detected", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
           COUNTER = 0
           ALARM_ON = False 
           
        
    #display 
    cv2.imshow("video capture", image)

    if cv2.waitKey(25) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()