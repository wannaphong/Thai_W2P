# Thai G2P

<a href="https://www.buymeacoffee.com/wannaphong"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

Thai Grapheme-to-Phoneme (G2P) converter

Code base from [AzamRabiee/Persian_G2P](https://github.com/AzamRabiee/Persian_G2P)

Dataset License: <img src="http://i.creativecommons.org/p/mark/1.0/88x31.png"
     style="border-style: none;" alt="Public Domain Mark" />

Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LXbXg2tJv5KkTWFv3hqmsCJk0g0PU6ea?usp=sharing)

## usage
**Install requirements**
```
pip3 install -r requirements.txt
```

with word segment (many words)
```python
python3 g2p.py --text "เป็นไลบรารีภาษาไพทอนสำหรับประมวลผลภาษาธรรมชาติ"
```
with one word (not word segment)
```python
python3 g2p.py --text "คนดี" --wordcut n
```
