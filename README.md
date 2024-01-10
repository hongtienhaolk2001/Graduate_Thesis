# News Analysis Project
![Dashboard Layout](./documents/index.png)
## Documents
- [Report](documents/19133022_HongTienHao_Report.pdf)
- [Slide](documents/Presentation_Slide.pdf)

## Usage
1. Install package
```commandline
pip install -r requiremets.txt
```
2. Install Java 8 (or Higher)

3. Import data to database
- Create a mysql database with name "news"
- Run file `add2Databse.py` to add data to databse
```commandline
python3 .app/data/add2Database.py
```
4. Get trained model
- Download my model at [here](https://drive.google.com/file/d/1Bh0PjjOteQl9OLHaLKFJUo7pV2pKZmkg/view?usp=sharing)
- Or train for your own by running this [train.ipynb](./train/train.ipynb) (training faster if having GPU)

Finally, move `pt` file to folder app/weights
5. Run flask app
```commandline
python3 main.py
```
## Result
![Analysis Layout](./documents/analysis.gif)
