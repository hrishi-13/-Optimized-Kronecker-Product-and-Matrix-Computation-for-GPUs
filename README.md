# -Optimized-Kronecker-Product-and-Matrix-Computation-for-GPUs
Enhanced the computation runtime for (C = A⊗B^T ) and (AB + CD^T ) by effectively leveraging Memory Coalescing and Shared Memory optimization techniques, while working with a 1024x1024 sized matrix.

Before running "evaluate.sh", make sure of the following things:
  1. Place your solution code "<ROLL_NO>.cu" in the "Submit" directory
  2. Input files must follow the naming format: "input##.txt"
  3. Output files must follow the naming format: "output##.txt"
  4. Place the input files in the "Evaluate Script/Input" directory
  5. Place the output files in the "Evaluate Script/Output" directory
  6. Output in A2-Marks.txt: rollNumber, NumberOfTestCasesPassed
