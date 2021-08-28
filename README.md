# Cat-Rabbit-Classifier
AI trying to guess what doodle is this, a cat or a rabbit the accuracy is 84% dataset from google quick draw

there is also API for it
here is sample request using python
```python
import requests

file = {"file": open("img.png" ,"rb")}

resp = requests.post("https://cat-rabbit-doodle-classifier.annasvirtual.repl.co/predict", files=file)
print(resp.text)
```
