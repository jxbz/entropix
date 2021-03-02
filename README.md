<h1 align="center">Computing the Neural Information Content</h1>

<p align="center">
  <img src="orthant.svg" width="150"/>
</p>

<p align="center">
  <a href="https://jeremybernste.in" target="_blank">Jeremy&nbsp;Bernstein</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://www.yisongyue.com" target="_blank">Yisong&nbsp;Yue</a> &emsp; &emsp;
</p>

## Getting started
- Run the unit tests:
```bash
python unit_test.py
```
- Run the main script:
```bash
python main.py
```
- Generate the plots using the Jupyter notebook `make_plots.ipynb`.

## Environment details
The code was run on:
- Pytorch 1.5.0
- Using docker container nvcr.io/nvidia/pytorch:20.03-py3
- On an NVIDIA Titan RTX GPU, with driver version 440.82, CUDA Version: 10.2

## Citation

If you find this code useful, feel free to cite [the paper](https://arxiv.org/abs/2103.01045):

```bibtex
@misc{entropix,
  title={Computing the Information Content of Trained Neural Networks},
  author={Jeremy Bernstein and Yisong Yue},
  year={2021},
  eprint={arXiv:2103.01045}
}
```

## License

We are making our algorithm available under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
