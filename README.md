# Installation
- [Install Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
- Install the requirements: `pip3 install -r requirements.txt`

# Usage
## Start robot
In the `blossom` directory:
- Start the robot controller: `./tele.sh`

## Start interface
In the `blossom-app` directory (as of 2021-08-31, must be in the `fb` branch):
- Start the ngrok tunnels: `./ngrok.sh`
- Start the interface: `./start.sh`
- Navigate to the command center: `localhost:8000/broadcaster.html`

## User
- Have the user navigate to the interface: `blossombot.com`
  - The username and password are set in `blossom-app/passwords.json`

## Record
In the command center (`localhost:8000/broadcast.html`):
- Direct the user (either manually or with a prompt from the command center)
- Start recording with the `Record` button.
- Stop recording to save the video to `~/Downloads/`

## Infer (video -> human poses -> robot poses -> robot sequence)
In the **local** `videopose3d` directory:
- Start the transfer script: `python3 vp_transfer.py`
- Use `t` to transfer the most recent `~/Downloads/*.webm` file, or specify the file manually (e.g. drag-and-drop from Finder)
In the **remote** `videopose3d` directory:
- The transfer from the **local** directory will put the video in `videopose3d/inference/input_dir`
- Convert and infer in one script: `python3 inference_main.py`
- The inference will output to `videopose3d/inference/output_dir`
In the **local** `videopose3d` directory:
- Use `p` to pull any new files from the **remote** `videopose3d` directory
  - Pulling will automatically run the human-to-robot retargeting script: `python3 control/control.py --video newest`
    - Possible flags: `--show_video` (to show the video), `--robot` (to control the robot), `--height` (to add knee-bend-based height control)
    - Pulling will also move the most recent video from `~/Downloads/*.webm` -> `blossom-app/public/video.webm`

## Playback
In the command center, toggle `Show Webm` to show the user the video and playback control.
When the user clicks `Play Movement`, the robot and video will start playing (<em>there is a small delay between the robot starting and the video starting, to help with synchronization</em>)
