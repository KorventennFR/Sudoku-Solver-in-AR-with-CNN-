#Sudoku Solver in AR with CNN

****************************************************************************************************************
                                 Sudoku Solver in Augmented Reality with CNN      
****************************************************************************************************************

AUTHORS :

LUSSIEZ Corentin

*********************
DESCRIPTION :
*********************
This project is designed to provide a real-time Sudoku solver for augmented reality. It can complete partially finished sudoku grids, even those written by hand.
OpenCV is used for the grid preprocessing, then each square is labeled with either [empty, 1, 2 ..., 9] using a Convolutional Neural Network trained on a modified MNIST.
The database used contains pieces of a grid for each squares and adds a written number, a non-written number or nothing.
The database is not provided within this git repo, however the code to create it is as well as some trained models.
The model integretion is not done here (should be in main.cpp) since I couldn't retrive the last version of this code (this is a 4yo side project).
Though this should be an easy thing to do since everything else is present and functional.
 If you need more info feel free to contact me.
 

*********************
CONTACT :
*********************
If you have any questions you can contact me at: lussiez.corentin@gmx.com
