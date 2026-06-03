# Interactive Vision Experiments

A collection of interactive, real-time computer vision prototypes designed to bridge the gap between physical movement and digital environments. These scripts experiment with mapping real-world user data (face position, hand gestures) into physics simulations and simulated 3D rendering pipelines.

## 🛠 Tech Stack
* **Computer Vision:** OpenCV, MediaPipe (Face Mesh & Hands)
* **Physics & Kinematics:** Pymunk (Rigid-body 2D physics)
* **Rendering & Logic:** Pygame

## 🧪 The Prototypes

### 1. Spatial Parallax & Off-Axis Projection
**Files:** `FaceParallax.py`, `FaceParallaxSkybox.py`, `ParticleParallax.py`
These experiments simulate 3D depth on a 2D screen without VR hardware[cite: 1]. 
* Tracks the user's face in real-time to calculate eye position[cite: 1].
* Dynamically generates an off-axis projection frustum based on the viewer's physical location[cite: 1, 2].
* Renders a 3D grid and skybox that shifts naturally as the user moves their head, creating a "window into another world" effect[cite: 1, 2].

<img width="476" height="596" alt="Head-ar-GIF (2)" src="https://github.com/user-attachments/assets/e7210de5-5dec-49b6-9ecf-b60159051d65" />

### 2. Kinematic Hand Physics & Interaction
**Files:** `HandBrickBreaker.py`, `Hand3DBrickBreaker.py`, `FaceParticles.py`, `ParticleWell.py`
These scripts translate physical hand tracking into rigid-body physics objects.
* Maps MediaPipe hand landmarks into Pymunk kinematic colliders[cite: 3, 5, 8].
* Allows the user to physically "push" thousands of simulated particles using the natural shape of their hands[cite: 8].
* Includes a fully playable Brick Breaker clone where the user's hand acts as the paddle, utilizing collision detection and velocity reflection[cite: 5].

[Insert GIF here: 5-second clip showing you hitting the ball with your hand in Brick Breaker]

### 3. Real-Time Gesture Classification
**Files:** `HandPoses.py`
* Maps and analyzes joint angles to classify specific hand gestures on the fly (e.g., Open Hand, Fist, Peace Sign, Pointing)[cite: 6].
* Renders dynamically tracking bounding boxes and classification labels over the webcam feed[cite: 6].

[Insert GIF here: 5-second clip showing the gesture recognizer classifying your hand shapes]

## 🚀 How to Run Locally

**Requirements:**
Ensure you have Python 3.x installed, along with a connected webcam.

1. Clone the repository:
```bash
   git clone [https://github.com/your-username/interactive-vision-experiments.git](https://github.com/your-username/interactive-vision-experiments.git)
