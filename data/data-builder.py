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
Task: Generate synthetic conversational data for a personal website assistant. Create at least 20 unique question-answer pairs in JSON format.

Output Format:
{
    "user question 1": "[link] with answer and explanation",
    "user question 2": ["[link1] explanation1", "[link2] explanation2"],
    ...
}

Guidelines:
1. Always include a relevant link in [square brackets] at the start of answers. This should be a real, working link, and not a placeholder
2. Use third-person perspective ("he" not "I")
3. Keep answers concise but informative (1-3 sentences)
4. For multiple relevant resources, use an array format
5. Bold related topics with **asterisks** to suggest follow-up questions
6. Create diverse questions covering education, projects, skills, and experience
7. Include simple questions ("tell me about Aren") and specific technical queries

Website Structure:
arendesai.com
├── /portfolio - Professional overview, education, LinkedIn and GitHub links
    ├── resume.pdf (arendesai.com/resume_nodetails.pdf)
    └── cv.pdf (arendesai.com/cv_nodetails.pdf)
├── /datascience - Data science projects
    ├── MarketSimOptimizer (arendesai.com/MSO-README.md) - MISO market simulation
    ├── LMP Forecasting (arendesai.com/LMPF-README.md) - LEAR & DNN price models
    └── GCP Website (github.com/ArenKDesai/ArenWebsite) - Website with GCP-hosted LLM
├── /computergraphics - Graphics projects
    ├── Boat (github.com/ArenKDesai/Boat) - Blender/Unity 3D boat model
    ├── Book of Joe (arendesai.com/bookofjoe) - Web-based portfolio game
    └── CyberCity (arendesai.com/cybercity) - 3D cyberpunk city environment
├── /robotics - Robotics work
    └── WRoverSoftware (github.com/WisconsinRobotics/WRoverSoftware) - UW-Madison rover
├── /coursework - Academic coursework at UW-Madison

About Aren Desai:
- Education: Computer Science & Data Science BS at UW-Madison (Spring 2025)
- Technical Skills: Python, SQL, R, Java, C/C++/C#, JavaScript, Julia, MATLAB
- Graphics & Robotics: OpenGL, WebGL, Unity, Blender, Gazebo, ROS/ROS2
- Data Science: Pandas, NumPy, PyTorch, TensorFlow, Scikit-Learn, Keras
- Experience: Madison Gas & Electric (Energy Supply Intern), Compeer Financial (Data Analytics)
- Projects: WebsiteLLM (Synthetic Data and Fine-Tuning), Wisconsin Rover software (ROS2)
- Leadership: Google Developer Student Club (Finance Lead), Wisconsin Robotics (Arm Developer)

Sample Questions (DO NOT REUSE THESE EXACT QUESTIONS):
- What are Aren's technical skills?
- Can you tell me about Aren's education?
- What data science projects has Aren worked on?
- Does Aren have experience with machine learning?
- What programming languages does Aren know?
- Tell me about Aren's work experience
- What is the WRoverSoftware project?
- What graphics projects has Aren created?
- What courses has Aren taken in data science?
- How can I see Aren's resume?

Remember to generate diverse, natural-sounding questions that website visitors might ask. Vary between general and specific questions about different aspects of Aren's background and projects.
\"\"\"\"
"""
modelCommand = ["ollama", "run", "deepseek-r1:8b", dataPrompt]

# Data Creation
def create_data(modelCommand, force=False):
    """
    Creates JSON files based on the prompt. Assumes the cwd is the "data" directory. 
    """
    start = time.time()
    try:
        # LLM Generates Data
        # TODO this would be a lot faster if the LLM spun-up once and was repeatedly asked questions
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
            old_errors = pl.read_csv("log.csv")
            new_errors = old_errors.vstack(errors)
            new_errors.write_csv("log.csv")
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
