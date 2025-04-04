import subprocess
import re
import os
import sys
from datetime import datetime
import polars as pl
import time

# LLM Setup
dataPrompt = """
\"\"\"\"
I want you to help me create synthetic data. It will be in the form of a JSON file that includes potential user prompts as keys and links to webpages and why you linked those webpages. Also include some potential other webpages. Here's an example:
```json
{
    "Does Aren have any experience with Unity?": "[github.com/ArenKDesai/Boat] Yes, here's a cool-looking boat he made in Blender and rendered in Unity! He's also made made a **CyberCity**, although that's in Three.js.",
    "show me data science": "[arendesai.com/LMPF-README.md] I think you might find this Locational Market Price forecasting project interesting! You might also like his **Google Cloud Platform** work.",
    "about page": [arendesai.com/portfolio] Here you can find Aren's work and his github link! If you want to see his **coursework**, you can also ask me about that.",
    "What projects has Aren worked on in computer graphics?": [
        "[github.com/ArenKDesai/Boat] Here is his boat project in Blender and Unity",
        "[arendesai.com/bookofjoe] Book of Joe, his first portfolio project",
        "[arendesai.com/cybercity] And here's the CyberCity, a 3D cyberpunk city"
    ],
    "What robotics projects has Aren been involved with?": "[github.com/WisconsinRobotics/WRoverSoftware] Programming for the Wisconsin Rover project at UW-Madison.",
    "how is this website hosted?": "[github.com/ArenKDesai/ArenWebsite] This website is hosted on GCP, with the LLM also running on GCP.",
    "What simulation projects has Aren worked on?": "[arendesai.com/MSO-README.md] Check out his MarketSimOptimizer project for a complex MISO market simulation.",
    "show me something cool!
    ...
}
```
Please make at least 15 examples, but be unique and don't use the examples above. Use the third person ("he" instead of "I") Here's the website details for your reference:
```
arendesai.com (
├── /portfolio (Includes a description of what I do and what degrees / certifications I have, and links to my linkedin and github)
    ├── resume.pdf (link is arendesai.com/resume_nodetails.pdf)
    └── cv.pdf (link is arendesai.com/cv_nodetails.pdf)
├── /datascience (My data science projects)
    ├── MarketSimOptimizer README.md (a complex simulation of the MISO market on a generation unit. closed source) (link is arendesai.com/MSO-README.md)
    ├── This GCP Website (the website is hosted on GCP, and the LLM is hosted and rate limited on GCP as well) (link is github.com/ArenKDesai/ArenWebsite)
    └── LMP Forecasting README.md (two models, one LEAR and one DNN, that model LMPs based on whitepapers. closed source) (link is arendesai.com/LMPF-README.md)
├── /coursework (The coursework I completed in college)
├── /computergraphics (The computer graphics projects I've made)
    ├── Boat (a boat I made in blender and rendered in Unity) (link is github.com/ArenKDesai/Boat)
    ├── Book of Joe (my first portfolio project I made for Honors Graphics, its a web game) (link is arendesai.com/bookofjoe)
    └── CyberCity (my third portfolio project for Honors Graphics, its a 3D cyberpunk city) (link is arendesai.com/cybercity)
└── /robotics (my robotics projects)
    └── WRoverSoftware (the programming for the rover I worked on for UW - Madison) (link is github.com/WisconsinRobotics/WRoverSoftware)
```
Some details you might want to know about me:
- My name is Aren Desai
- I know Python, Java, JavaScript, Julia, C, C++, C#, R, SQL, and MatLab
- I was the Finance Lead of the Google Developer Student Club from 2023-2024
- I've worked at Compeer Financial and Madison Gas & Electric in data analytics
\"\"\"\"
"""
modelCommand = ["ollama", "run", "deepseek-r1:14b", dataPrompt]

# Data Creation
def create_data(modelCommand, force=False):
    """
    Creates JSON files based on the prompt. Assumes the cwd is the "data" directory. 
    """
    start = time.time()
    try:
        # LLM Generates Data
        modelStdout = subprocess.run(modelCommand, capture_output=True, text=True, encoding='utf-8').stdout

        # Data Cleaning
        jsonPattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
        jsonString = jsonPattern.search(modelStdout).group(1)

        # Save the Data
        i = 0
        fpath = f"data{i}.json"
        while fpath in os.listdir():
            i += 1 # NOTE: keeping all files discrete in case LLM's output isn't a JSON
            fpath = f"data{i}.json"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(jsonString)

        # logging
        print(f"{datetime.now().ctime()} {fpath} written, {time.time()-start:.2f}s")
    except Exception as e:
        errors = pl.DataFrame({
            "datetime": [datetime.now().ctime()],
            "prompt": [dataPrompt],
            "model_response": [modelStdout],
            "error": [str(e)]
        })
        if os.path.exists("log.csv"):
            errors.write_csv("log.csv", has_header=False, append=True)
        else:
            errors.write_csv("log.csv")
        print(f"{datetime.now().ctime()} error")
        if not force:
            sys.exit()

# TODO switch to proper argument handling library and add -f --force
if __name__ == "__main__":
    # Argument Handling
    if len(sys.argv) == 1: # Run Once
        create_data(modelCommand=modelCommand)
    elif sys.argv[1] == "-l" or sys.argv == "--loop":
        while True: # Run on a permanent loop
            create_data(modelCommand=modelCommand)
    else:
        print("Usage: data-builder (optional: -l / --loop)")