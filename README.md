### Examples for running AtomAI

#### Step 1. Create and configure conda environment
Using python3.10 or later:
```
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2 --no-cache-dir
```

#### Step 2. Confirm that you can run AtomAI (on Frontier)
```
python3 atomai_runner.py
```
You should see a file called 'model.pth' created. 


#### Step 3. Confirm that you can run the analysis script (on ACE). 

#### Step 4. Confirm that you can run data transfers (Frontier <--> ACE). 
// TODO TYLER ADD THIS



#### Step 5. Confirm that you can run the end-to-end workflow using Zambeze.
