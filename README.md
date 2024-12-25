# Welcom to `c-nn`!
This is a repository made just for fun. I came up with the idea of realizing some basic linear algebra calculations using C and probably may be able to build a neural network out of the foundation.

Compile:

Windows:
```bash
gcc -o ./exec_win/main main.c linalg.c xlinalg.c -lm
```

Mac:
```bash
gcc -o ./exec_macos/main main.c linalg.c xlinalg.c -lm
```

Sample result of running `main.c`:

```zsh
===== Basic Matrix Operations ===
Left Matrix (L):
1.00 2.00 3.00 
4.00 5.00 6.00 

Right Matrix (R):
7.00 8.00 
9.00 10.00 
11.00 12.00 

Multiplied Matrix (M=LR):
58.00 64.00 
139.00 154.00 

Added Matrix (A=L+RT):
8.00 11.00 14.00 
12.00 15.00 18.00 

===== Basic Matrix Equation Solving ===
Solving Ax=b. Where:
 A:
3.00 2.00 1.00 
-1.00 -3.00 -1.00 
1.00 -2.00 -2.00 

b:
-7.00 
5.00 
4.00 

Solved that x:
-1.27 
-0.55 
-2.09 

===== Matrix Determinant Calculation ===
Matrix C (5 x 5):
4.00 0.00 -7.00 3.00 -5.00 
0.00 0.00 2.00 0.00 0.00 
7.00 3.00 -6.00 4.00 -8.00 
5.00 0.00 5.00 2.00 -3.00 
0.00 0.00 9.00 -1.00 2.00 

Calculated Determinant: 6.000000
Matrix acquired from: https://www.youtube.com/watch?v=crCsJy1lKXI%
```