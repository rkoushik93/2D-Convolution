1. The programs are in the home directory/project on Comet.

2. Sequential program is also uploaded in the same directory.

3. Program that uses MPI_Send and Recv is under the name MPIsendAndRecv.c

4. Program that uses collective communication is under the name MPIcollcomm.c

5. Program that uses task and data parallelism is under the name MPItaskdataparallel.c

6. Separate bash files for each of the above mentioned programs are uploaded to the same directory.

7. C File			->		Bash FIle
   
   MPIsendAndRecv.c		->		sendrecvBash.sh
   MPIcollcomm.c		->		MPIcollcommBash.sh
   MPItaskdataparallel.c	->		MPItaskdataparallelbash.sh

8. The programs can be compiled using the command
	mpicc -o MPIsendAndRecv MPIsendAndRecv.c

9. The jobs can be submitted using the command
	sbatch sendrecvBash.sh

10. Check the queue status by the command
	squeue -u <username>

11. The output to be displayed will be stored in a file under the name
	MPIsendAndRecv.%j.%N.out

12. This file will mention the filename in which the output matrix
	will be saved.

13. The same procedure is to be followed to execute and get the ouput
	for all other programs.

NOTE:	For the sequential program, use the following command to compile
		gcc sequentialprogram.c -o sequentialprogram -lm

	The -lm at the end is used to link the math.h header file
