# Day 1 Vector Addition
## Usage
1. Clone the repo 
2. `cd 100DayCudaChallenge/Day01`
3. Run the following commands:

Clean previous builds
`rm -rf build`

Build the extension
`python setup.py develop`

verify the file exists
`ls -lh *.so`

Then you can run test_run.py with
`python test_run.py`

## Explanation
This is a simple vector addition kernel, that adds two torch tensors and returns an output.

