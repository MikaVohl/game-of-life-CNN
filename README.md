# Solving Conway's Game of Life using a Convolutional Neural Network

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a iteration-based game with simple rules which yield suprisingly chaotic behaviour. A consequence of this chaotic behaviour is that there is no closed-form solution for finding the Nth next game state based on the current grid.

This restriction motivated me to explore using deep learning methods to attempt to predict the Nth next game state with a high level of accuracy. I quickly realized that building a convolutional neural network was the best fit for the task, as Conway's Game of Life's rules are inherently localized to a 3x3 grid, making convolutional layers with a 3x3 kernel an obvious choice.

I have deployed my implementation at [life.mikavohl.ca](life.mikavohl.ca) in the form of a blank canvas on which the user can input the initial state, then compare the results of direct simulation and the prediction by the CNN.


## Setup

### Prerequisites

- **Node.js & npm** (v18+ recommended)  
- **Python** (3.11+)  
- **Docker** (for building/running the Lambda image)  
- **AWS CLI v2** configured with an IAM user that can push to ECR & update Lambda

---

### Running Locally

1. **Clone & enter repo**  
   ```bash
   git clone https://github.com/MikaVohl/Game-of-Life-CNN.git
   cd Game-of-Life-CNN
   ```

2. **Install & link the Python package**

   ```bash
   pip install -e .
   ```

   This makes `life_sim` importable in both the Flask API and any notebooks.

3. **Install front-end dependencies**

   ```bash
   cd frontend
   npm install
   ```

4. **Create `frontend/.env`**

   ```bash
   VITE_API_URL=http://localhost:5001
   ```

5. **Run the Flask API**

   ```bash
   cd ../api
   pip install -r requirements.txt  # only once
   python server.py
   ```

6. **Run the front-end**

   ```bash
   cd ../frontend
   npm run dev
   ```

   Visit [http://localhost:5173](http://localhost:5173) to paint cells and compare simulation vs. CNN prediction.

---

### Building & Deploying on AWS Lambda

1. **Build the Docker image**
   From your project root:

   ```bash
   docker build -t game-of-life:latest -f infra/lambda/Dockerfile .
   ```

2. **Tag & push to ECR**

   ```bash
   # replace placeholders:
   ACCOUNT=123456789012
   REGION=us-east-1
   REPO=game-of-life-lambda

   docker tag game-of-life:latest \
     ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest

   aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

   docker push ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest
   ```

3. **Create or update your Lambda**

   ```bash
   # create (once):
   aws lambda create-function \
     --function-name life-sim \
     --package-type Image \
     --code ImageUri=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest \
     --role arn:aws:iam::${ACCOUNT}:role/YourLambdaExecRole

   # update (after each push):
   aws lambda update-function-code \
     --function-name life-sim \
     --image-uri ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest
   ```

4. **Hook up your front-end to Lambda**
   In `frontend/.env`, set:

   ```bash
   VITE_API_URL=https://<your-lambda-url>.lambda-url.${REGION}.on.aws
   ```
   
   Deploy your frontend however you please. I used Vercel's Vite deployment.