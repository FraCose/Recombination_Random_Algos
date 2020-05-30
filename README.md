THIS REPOSITORY CONTATINS THE ALGORITHMS EXPLAINED IN THE WORK
Cosentino, Oberhauser, Abate
"A randomized algorithm to reduce the support of discrete measures "

-The ipython notebooks contain the experiments to be run
-recombination.py is the library with all the necessary functions,

Some general notes:
-The name of the ipynb files refers directly to the experiment in the cited work.
-The last cells of the notebooks produce the pictures of the pdf.
-To reduce the running time the parameters can be easiliy changed, e.g. decreasing N, n or sample.

---------------------------------------------------
LIBRARIES - recombination.py
---------------------------------------------------
It contains the algorithms relative to the reduction of the measure presented in this work,
see the pdf for more details. In recombination.py we have rewritten in Python the algorithm presented
in Tchernychova, Lyons "Caratheodory cubature measures", PhD thesis, University of Oxford, 2016.
Note that we do not consider their studies relative to different trees/data structure,
due to numerical instability in some cases as declared in the work; another
reason is that their analysis was focused for specific cases. See the reference for more details.

---------------------------------------------------
TO RUN THE EXPERIMENTS - REQUESTED FILE
---------------------------------------------------
To run the comparisons, please download the following file and name it "Maalouf_Jubran_Feldman.py".
"Fast and Accurate Least-Mean-Squares Solvers"
(NIPS19' - Oral presentation + Outstanding Paper Honorable Mention) by Alaa Maalouf and Ibrahim
Jubran and Dan Feldman”, which you can also find here
https://github.com/ibramjub/Fast-and-Accurate-Least-Mean-Squares-Solvers

---------------------------------------------------
TO RUN THE EXPERIMENTS - DATASETS
---------------------------------------------------
Please, to run the experiments donwload the following dataset and put them in the Dataset folder:
	- 3D_spatial_network.txt -
      https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt
	- household_power_consumption.txt -
      https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
      (extract the .txt file)

---------------------------------------------------
ACKNOLEDGEMENTS AND DISCLOSURE OF FUNDING
---------------------------------------------------
The authors want to thank The Alan Turing Institute for the financial support given.
Grant: The Alan Turing Institute, TU/C/000021, under the EPSRC Grant No. EP/N510129/1.
