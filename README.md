# LDE

This repository is the official implementation of the paper [Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient](https://ieeexplore.ieee.org/document/9359652).

## Requirements

- Python 3.5.4
- Torch 1.3.1

## Run 
 
To train the parameter controller on CEC'13 benchmark functions and then run LDE on CEC'17, excute this command:

```sh
$ python PGnet_torch.py
```

## Results

The **trained agent** will be `saved` and the optimization **results** on CEC'17 benchmarks are also stored as a `.txt file`. 

There are two files in the `./Results` folder. These two textfiles `LDE_CEC17_10D.txt` and `LDE_CEC17_30D.txt` are the **raw results of LDE on CEC'17 in 10D and 30D** respectively, as reported in Table.VII and Table.VIII in the original paper. You can make a comparision with yours immediately.

To load the result files, run the following command:
```sh
x = numpy.loadtxt('LDE_CEC17_29Fs_10D_51runs_MAXNFE.txt')
```

Then *x* is a [NumFunctions, NumRuns] (i.e. [29, 51]) matrix, and the i-th row of *x* records all error values of *Fi* for 10D.

Note that error value smaller than *1e-8* should be taken as **zero**.

## Citation

If you find this repository useful for your work, please cite:

```sh
@ARTICLE{LDE,  
 author={Sun, Jianyong and Liu, Xin and Bäck, Thomas and Xu, Zongben},  
 journal={IEEE Transactions on Evolutionary Computation},   
 title={Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient},  
 year={2021}, volume={25},  number={4},  pages={666-680},  
 doi={10.1109/TEVC.2021.3060811}  
}
```

## License

[MIT © Richard McRichface.](../LICENSE)
