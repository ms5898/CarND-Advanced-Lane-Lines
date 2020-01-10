## CarND-Advanced-Lane-Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./img/camera_cal.png "Camera Calibration"
[image2]: ./img/distortion_correction.png "distortion correction"
[image3]: ./img/color_transforms_gradients.png "color transforms gradients"
[image4]: ./img/region_of_interest.png "region of interest"
[image5]: ./img/perspective_transform.png "perspective transform"
[image6]: ./img/histogram_of_image.png "histogram of image"
[image7]: ./img/curvature_of_lane.png "curvature of lane"
[image8]: ./img/result.jpg "result"

---

### Camera Calibration

#### 1. Example of a distortion corrected calibration image.

The code of camera calibration is write in `Show process .ipynb` 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Apply a distortion correction to raw images.

Use `cv2.undistort` and r`camera matrix & distortion coefficients` which get from camera calibration make original image to undistorted image
![alt text][image2]

#### 2. Use color transforms, gradients, etc., to create a thresholded binary image.

* Limit magnitude of the gradient

* Limit direction of the gradient

* Color Thresholding (Use S channel of HLS image)

The parameters used are as follows:
```python
sx_thresh=(15, 255)
dir_thresh=(0, np.pi/32)
s_thresh=(180, 255)
```

![alt text][image3]

#### 3. select region of interest.

Only onsider a part of iamge, the vertices of region of interest are as follow:
```python
vertices = np.array([[(550, 450),
                      (750, 450),
                      (1120, 700),
                      (180, 700)]], dtype=np.int32)
```

![alt text][image4]

#### 4. Apply a perspective transform to rectify binary image ("birds-eye view").

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 106 through 110 in the file `utils/helper.py`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
offset = 300
w, h = img.shape[1], img.shape[0]
dst = np.float32([[offset, 0], [w - offset, 0], [offset, h], [w - offset, h]])
src = np.float32([(582, 460), (705, 460), (260, 680), (1050, 680)])
```

![alt text][image5]

#### 5. Determine the curvature of the lane and vehicle position with respect to center.
![alt text][image6]




---

### Pipeline (video)

Here's a [link to my video result](./output_images/test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
