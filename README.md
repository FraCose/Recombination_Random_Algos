---------------------------------------------------
This Repository contains the Algorithms explained in<br />
Cosentino, Oberhauser, Abate<br />
"A randomized algorithm to reduce the support of discrete measures "<br />
---------------------------------------------------

The files are divided in the following way:<br />
- The ipython notebooks contain the experiments to be run.<br />
- recombination.py is the library with all the code of the Algorithms presented in<br />
 the cited work.<br />

Some general notes:<br />
- The names of the ipynb files refer directly to the experiments in the cited work.<br />
- The last cells of the notebooks produce the pictures of the pdf.<br />
- To reduce the running time the parameters can be easily changed, e.g. decreasing N, n or sample.<br />

---------------------------------------------------
Library - recombination.py
---------------------------------------------------
It contains the algorithms relative to the reduction of the measure presented in this work,<br />
see the pdf for more details. In recombination.py we have rewritten in Python the algorithm presented<br />
in Tchernychova, Lyons "Caratheodory cubature measures", PhD thesis, University of Oxford, 2016.<br />
Note that we do not consider their studies relative to different trees/data structure,<br />
read the cited work for more details.<br />

----------------------------------------------------
Special Note to run the experiments
----------------------------------------------------
The notebooks "Comparison_random_algos.ipynb", "Comparison_literature_algos.ipynb", "Running_time_ratio.ipynb"<br />
and "Running_times_vs_n.ipynb" contain multiple experiments: symmetric vs non-symmetric.<br />
You have to comment/uncomment the respective parts of the code as indicated to reproduce the <br />
wanted experiments. <br />

---------------------------------------------------
To Run the Experiments - Requested file
---------------------------------------------------
To run the comparisons, please download the following file and name it "Maalouf_Jubran_Feldman.py".<br />
"Fast and Accurate Least-Mean-Squares Solvers"<br />
(NIPS19' - Oral presentation + Outstanding Paper Honorable Mention) by Alaa Maalouf and Ibrahim<br />
Jubran and Dan Feldman”, which you can also find here<br />
https://github.com/ibramjub/Fast-and-Accurate-Least-Mean-Squares-Solvers<br />

---------------------------------------------------
To Run the Experiments - Datasets
---------------------------------------------------
To run the experiments, the following dataset need to be donwloaded and saved in the /Datasets folder:<br />
- 3D_spatial_network.txt -<br />
https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt<br />
- household_power_consumption.txt -<br />
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip<br />
(extract the .txt file)<br />
- kc_house_data.csv -<br />
https://www.kaggle.com/harlfoxem/housesalesprediction<br />

---------------------------------------------------
Funding
---------------------------------------------------
The authors want to thank The Alan Turing Institute and the University of Oxford<br /> 
for the financial support given. FC is supported by The Alan Turing Institute, TU/C/000021,<br />
under the EPSRC Grant No. EP/N510129/1. HO is supported by the EPSRC grant Datasig<br />
[EP/S026347/1], The Alan Turing Institute, and the Oxford-Man Institute.
