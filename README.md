# CAWDR : Computer Aided Weld Defect Recognition
### Installation
1. Download the project zip file.
2. Download the segmentation model weights and put it in the Server Code folder. From this url: https://drive.google.com/file/d/1-oRGvv7b-CIeD4ib-Oy21rk0EeeDe2ex/view?usp=sharing
3. Create a virtual environment using anaconda.
4. Install the requirements.txt dependencies using the following command:
```python
pip install -r /path/to/requirements.txt
```

### Usage
1- If you test the application locally then run the server code in separate terminal  using the following command:
``` python
python server3.py
```
2- If you test the application with AWS please change the hostname and the port number. 
3- Open the project in the virtual environment you created and run the following command:
``` python
python main.py
```

