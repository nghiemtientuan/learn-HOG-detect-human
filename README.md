# 1. Introduction

# 2. Prerequisites

- make ```sudo apt install make```
- python3-dev python3-pip ```sudo apt install python3-dev python3-pip```
- Python >= 3.8

# 3. Document

# 4. Installation

- Install the requirements inside of a Python virtualenv (recommend)
```BASH
    pip install virtualenv
    virtualenv -p python3.8 venv
    source venv/bin/activate
```

- Make requirements
```BASH
    make requirements
```

# 5. Commands

- Train modal
```
    python hog_train.py
```

- Detect
```
    python hog_test.py <đường dẫn ảnh>
```

# 6. Debug
