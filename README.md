# Low Rank Token Linear Extension


The TokenLinear's a custom linear transformation module that expand TokenFormer capacity via an additional set of low rank virtual tokens. 

There is a bunch of ways to formulate this, this is just one.

```python
# Replace any linear layer with this, expand the parameters postraining by init ing new ones as zeros.
linear = TokenLinear(in_features, out_features, num_tokens, rank=32):
```


```
@misc{algomancer2024,
  author = {@algomancer},
  title  = {Some Dumb Shit},
  year   = {2024}
}
```

```
@article{wang2024tokenformer,
  title={TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters},
  author={Wang, Haiyang and Yue, Fan and Naeem, Muhammad Ferjad and Xian, Yongqin and Lenssen, Jan Eric and Wang, Liwei and Tombari, Federico and Schiele, Bernt},
  journal={arXiv preprint arXiv:2410.23168},
  year={2024}
}
```
