# M-Lines Fitting for Thin Film Waveguides: Prism Coupler Analysis Tools

This repository provides tools for fitting the parameters of a prism coupling technique model by [Sokolov et al.](https://iopscience.iop.org/article/10.1070/QE2015v045n09ABEH015852) to experimental data obtained from Metricon Prism Coupler machines. This model's parameters define the properties of thin film waveguides. 

They take as input CSV files as output by the Metricon machine and output the model's fitted parameters.

The fitting process is performed using either SciPy's `curve_fit` function or a genetic algorithm implemented using the PyGAD package.

## How to Use

To use the fitting script `reflection_fit.py`, follow these steps:


1. Prepare your configuration file by specifying the model's fixed parameters, and the bounds for the parameters to be fitted.

1. Prepare your measurement curve data in CSV format as output by the Metricon machine.

1. Run the main script using the following command-line arguments:

    ```
    python reflection_fit.py --config=config_file.txt --curve="New Data/T3_TE.csv" --fit_method=scipy --pol=s
    ```

### Author and Citation
The codes provided in the repository were prepared by [Nacer eddine Belaloui](https://github.com/Belaloui), researcher at [Constantine Quantum Technologies](https://cqtech.org).<br>
If you use these codes in your work, please cite as per the included [BibTeX file](citation.bib).