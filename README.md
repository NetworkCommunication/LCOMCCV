# LCOMCCV

## Project Introduction

In Mixed Connected and Connectionless Vehicles (MCCV) scenarios, traditional vehicle control and traffic management systems struggle to adapt to the dynamic interactions between these different types of vehicles complicating critical driving decisions like lane-changing overtaking. Our system addresses these issues by developing a collaborative strategy for connected vehicles in the MCCV scenario. First, we design a priority detection and triggering mechanism to facilitate efficient collaboration among connected vehicles, optimizing decision-making and reducing conflicts. Second, we introduce the IDP-FCM algorithm to dynamically identify and adapt to different driving styles, thereby improving safety. Finally, addressing the challenge of hybrid action space, our proposed MCPDDQN algorithm enhances strategy stability in complex driving scenarios.

## Environmental Dependence

The code requires python3 (>=3.8) with the development headers. The code also need system packages as bellow:

numpy == 1.24.3

matplotlib == 3.7.1

pandas == 1.5.3

pytorch == 2.0.0

gym == 0.22.0

sumolib == 1.17.0

traci == 1.16.0

If users encounter environmental problems and reference package version problems that prevent the program from running, please refer to the above installation package and corresponding version.

## Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-repo/LCOMCCV.git
   cd LCOMCCV
   
2. **Create a virtual environment:**
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install dependencies:**
    Set up the necessary environment variables as described in 'config/environment.md'.

## Project structure introduction
LCOMCCV is the root directory of the project, and the secondary directory includes two folders and three python files. The folder "agents" is the model code we designed. The folder "data" is the simulation environment we built using sumo, which includes: 1) "xxx.net.xml" is the defined road network structure; 2) "xxx.rou.xml" is the defined road traffic flow; 3) "xxx.sumocfg" is the configuration file of sumo. For the three python files in the secondary directory: 1) "driverStyleCluster.py" is the IDP-FCM algorithm we designed; 2) "my_EnvCluster.py" is the RL environment we designed.


## Statement

In this project, due to the different parameter settings such as the location of the connected vehicle, the location of the target vehicle, and the state of surrounding vehicles, etc., the parameters of the reinforcement learning algorithm are set differently, and the reinforcement learning process is different, resulting in different experimental results. In addition, the parameters of the specific network model refer to the parameter settings in the experimental part of the paper. If you want to know more, please refer to our paper "Collaborative Overtaking Strategy for Enhancing Overall Effectiveness of Mixed Connected and Connectionless Vehicles".