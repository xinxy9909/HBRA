# HBRA: A BIDIRECTIONAL RECOMMENDER ALGORITHM BASED ON HETEROGENEOUS GRAPH
Our code implementation is modified from the  [Blurring-Sharpening Process Models for Collaborative Filtering](https://github.com/jeongwhanchoi/BSPM/) model.
# How do I use it?
- Gowalla
```bash
python main.py --dataset="gowalla" --topks="[20]" --simple_model="hbra" --testbatch=2048
```
- Yelp2018

```bash
# GPU
python main.py --dataset="yelp2018" --topks="[20]" --simple_model="hbra" --testbatch=2048
```

- Amazon-book

```bash
# GPU
python main.py --dataset="amazon-book" --topks="[20]" --simple_model="hbra"
```
