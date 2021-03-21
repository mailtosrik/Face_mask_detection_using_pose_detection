# CS510_Final_Project

TO VIEW SAMPLE OUTPUT
----------------------
We have uploaded sample output snippets that was generated using our solution under `Output_snippets` folder

TO TRAIN THE MODEL
-------------------

1. Please keep the face_mask_detector_model_training.py and requirements.txt in the folder root directory.
2. Create a folder with the name ‘dataset’ in the root directory
3. Under the dataset, folder create two new folders called ‘with_mask’ and ‘without_mask’
4. Collect the images from dataset sources (mentioned in the final paper) and paste it in the corresponding folder.
5. To install the requirements, 'pip install -r requirements.txt' from the application root directory.
6. To train and save the model run the below query in the application root directory,
        'python face_mask_detector_model_training.py --dataset dataset'
7. Model and the graph plot will get saved in the application root directory once the run completes.

FOR DETECTION and SOLUTION
---------------------------
We have provided a way to execute our solution on video in Collab and view the detected results with GPU runtime enabled. 

Since we have developed this as a part of experimentation and research, its not packaged as a one-click solution.
1. Clone AlphaPose solution from the git. this presents with necessary solutions to generate poses
2. PIP installations as mentioned in notebook.
3. Place the video __1048231489-preview.mp4__
        in the location
       ``` /content/gdrive/MyDrive/Alphapose/AlphaPose/examples/Video/1048231489-preview.mp4
        ```
4. Create output folder `examples/res/random_videos/1048231489` where the output of alphapose will be saved.
5. Run cell mentioned under __`GENERATE ALPHA POSE RESULTS AS JSON ON VIDEO`__
6. Similar cells are provided to render the pose detection video to view or to run on images.
7. Place the saved model (from above code) into this path `/content/gdrive/MyDrive/Alphapose/AlphaPose/MaskDetectionModel/`
8. Under __`EXTRACT FACE FROM POSE`__, run __`FUNCTION TO CALL`__
9. Run the cell __`EXTRACT FACE WITH HEURISTIC 3 - THIS TRIAL WAS USED AS IT PRODUCES BEST RESULTS`__
10. Once the above code is run, execute the cell __`REGENERATE VIDEO FROM FRAMES`__.
11. This generates a video named __'full_scale_video_integration.avi'__ in the folder `"/content/gdrive/MyDrive/Alphapose/AlphaPose/examples/res/"`
