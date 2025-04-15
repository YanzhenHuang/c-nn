# Welcom to `c-nn`!
This is a repository made just for fun. I came up with the idea of realizing some basic linear algebra calculations using C and probably may be able to build a neural network out of the foundation.

```mermaid
graph TD
    A["Main Application (main.c)"]:::entry
    subgraph "Linear Algebra Subsystem"
        B["Basic Linear Algebra (linalg)"]:::library
        C["Extended Linear Algebra (xlinalg)"]:::library
    end
    D["Neural Network Module (nn)"]:::module
    subgraph "Execution Environments"
        E["macOS Build Artifacts"]:::artifact
        F["Windows Build Artifacts"]:::artifact
    end

    A -->|"demo:xlinalg"| C
    A -->|"demo:nn"| D
    D -->|"uses"| B
    D -->|"uses"| C
    A -->|"build_for"| E
    A -->|"build_for"| F

    classDef entry fill:#f9c,stroke:#333,stroke-width:2px;
    classDef library fill:#cfc,stroke:#333,stroke-width:2px;
    classDef module fill:#ccf,stroke:#333,stroke-width:2px;
    classDef artifact fill:#ffc,stroke:#333,stroke-width:2px;

    click A "https://github.com/yanzhenhuang/c-nn/blob/main/main.c"
    click B "https://github.com/yanzhenhuang/c-nn/blob/main/linalg.c"
    click C "https://github.com/yanzhenhuang/c-nn/blob/main/xlinalg.c"
    click D "https://github.com/yanzhenhuang/c-nn/blob/main/nn.c"
    click E "https://github.com/yanzhenhuang/c-nn/tree/main/exec_macos/"
    click F "https://github.com/yanzhenhuang/c-nn/tree/main/exec_win/"
```

Compile:

Windows:
```bash
gcc -o ./exec_win/main main.c linalg.c xlinalg.c nn.c -lm
```

Mac:
```bash
gcc -o ./exec_macos/main main.c linalg.c xlinalg.c nn.c -lm
```

---

## Run `main.c` (Take macOS as an example)

Run demo of `xlinalg`:

```zsh
./exec_macos/main -demo xlinalg    
```

Run demo of `nn`:

```zsh
./exec_macos/main -demo nn    
```
