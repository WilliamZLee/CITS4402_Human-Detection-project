

      DaimlerChrysler Pedestrian Classification Benchmark Dataset
      ===========================================================

			S. Munder, D. M. Gavrila
	          dariu.gavrila@daimlerchrysler.com
			    May 10, 2006

		     (C) 2006 by DaimlerChrysler AG



This README describes the DaimlerChrysler Pedestrian Classification
Benchmark Dataset introduced in the publication

* S. Munder and D. M. Gavrila, "An Experimental Study on Pedestrian
  Classification", IEEE Trans. on Pattern Analysis and Machine
  Intelligence, 2006.

This dataset contains a collection of pedestrian and non-pedestrian images.
It is made publicly available for benchmarking purposes, aimed to advance
further research in pedestrian classification.



1. LICENSE AGREEMENT
====================

This dataset is made available to the scientific community for non-commercial
research purposes such as academic research, teaching, scientific publications,
or personal experimentation. Permission is granted to use, copy, and distribute
the data given that you agree:

 1. That the dataset comes "AS IS", without express or implied warranty.
    Although every effort has been made to ensure accuracy, DaimlerChrysler
    does not accept any responsibility for errors or omissions.

 2. That you include a reference to the above publication in any published work
    that makes use of the dataset.

 3. That if you have altered the content of the dataset or created derivative
    work, prominent notices are made so that any recipients know that they are
    not receiving the original data.

 4. That you may not use or distribute the dataset or any derivative work for
    commercial purposes as, for example, licensing or selling the data, or
    using the data with a purpose to procure a commercial gain.

 5. That this license agreement is retained with all copies of the dataset.

 6. That all rights not expressly granted to you are reserved by
    DaimlerChrysler.




2. DATA SETS
============

This dataset contains a collection of pedestrian and non-pedestrian images
in PGM format.

2.1 Base Data Set
-----------------

The base data set contains small example images of pedestrians and
non-pedestrians cut out from video images and scaled to common size.
This data set has been used in Section VII-A of the paper referenced above.

Pedestrian images were obtained from manually labeling and extracting the
rectangular positions of pedestrians in video images.  Video images were
recorded at various (day) times and locations with no particular constraints
on pedestrian pose or clothing, except that pedestrians are standing in
upright position and are fully visible.  In order to make maximum use of
these (valuable) labels, images were mirrored, and the bounding boxes were
shifted randomly by a few pixels in horizontal and vertical directions. The
latter is to account for small errors in ROI localization within an
application system. Six pedestrian examples are thus obtained from each
label.

As non-pedestrian images, we extracted patterns representative for typical
preprocessing steps within a pedestrian classification application, from
video images known not to contain any pedestrians. We chose to use a
shape-based pedestrian detector that matches a given set of pedestrian shape
templates to distance transformed edge images.

Given the bounding box locations of interest in video images, example images
were cut out after adding a border of 2 pixels to preserve contour
information, and scaled to common size 18x36.

The resulting data base is split into five fully disjoint sets, three for
training (named "1", "2", and "3") and two for testing (named "T1" and
"T2"). Each set consists of 4800 pedestrian examples obtained from 800
pedestrian labels by mirroring and shifting, plus 5000 non-pedestrian
examples. See the table below for a summary.  

Images recorded at the same time and location are kept within the same set,
so that e.g. a pedestrian captured in a sequence of images does not show up
in multiple data sets.  This ensures truly independent training and test
sets, but also implies that examples within a single data set are not
necessarily independent.

  Dataset   Purpose   Pedestrian   Pedestrian   Non-ped.   Storage
  Name                Labels       Examples     Examples   Size
  ---------------------------------------------------------------------
   "1"      Training  800          4800         5000       39 MB
   "2"      Training  800          4800         5000       39 MB
   "3"      Training  800          4800         5000       39 MB
   "T1"     Test      800          4800         5000       39 MB
   "T2"     Test      800          4800         5000       39 MB

For installation, simply "untar" the archive file DC-ped-dataset_base.tar.gz .
This will create 5 directories, one for each data set, with sub-directories
named ped_examples and non-ped_examples containing the pedestrian and
non-pedestrian example images, respectively.


2.2 Additional non-pedestrian images
------------------------------------

The three training sets each come with an additional collection of 1200 video
images NOT containing any pedestrians, intended for the extraction of
additional negative training examples. Section V of the paper referenced above
describes two methods on how to increase the training sample size from these
images, and Section VII-B lists experimental results.

These images are provided in 3 separat tar files named
DC-ped-dataset_add-{1,2,3}.tar.gz . When unpacked to the same directory as the
base data set, an additional sub-directory named add_non-ped_images is
installed in the corresponding training set directory.

Overall storage size of the additional image sets is 586 MB.



3. BENCHMARKING PROCEDURE
=========================

Authors who wish to evaluate pattern classifiers on this dataset are encouraged
to follow the benchmarking procedure as detailed in the publication given
above. Mean and variance of ROC performance are determined by varying training
and test sets, aimed to distinguish significant performance differences from
random effects.

More specifically, the benchmarking procedure starts with adjusting parameter
settings of the pattern classifier or the training algorithm via 3-fold 
cross-validation on the training sets. Having found optimal settings, three 
classifiers are generated, each by using 2 out of the three training data sets. 
Applying these classifiers to both test sets yields 6 ROC curves, from which mean 
and variance are computed, see section VI-B.


4. TALLYING OF RESULTS
======================

The original authors would like to hear about other publications that
make use of the benchmark data set in order to include corresponding 
references on the benchmark website.


