#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<time.h>
#include<mpi.h>

typedef struct {float r; float i;} complex;
static complex ctmp;
#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}
void c_fft1d(complex *r, int n, int isign);

#define N 512

int main (int argc, char **argv) {
	int myid,nproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
   
	const char* file1="im1";
   	const char* file2="im2";   
	complex A[N][N],B[N][N],C[N][N],tempComplex;
	int x,y;
	double startTime,endTime,timeTaken,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,commTime;
	MPI_Status *status;

	if(myid==0){

		//printf("\n\nCS546 Project - Koushik Raman A20388858\n\n\n");
   		printf("\n\nProgram using MPI_Send and MPI_Recv\n\n");

   		FILE *fp1 = fopen(file1,"r");
   		if ( !fp1 ) {
      			printf("ERROR!! This file does NOT exist\n\n");
      			exit(1);
   		}

   		for (x=0;x<N;x++){
      			for (y=0;y<N;y++) {
         			A[x][y].i=0;
				fscanf(fp1,"%g",&A[x][y].r);
      			}
		}
   		fclose(fp1);


		FILE *fp2=fopen(file2,"r");
		if(!fp2){
			printf("ERROR!! This file does NOT exist\n\n");
			exit(1);
		}

		for(x=0;x<N;x++){
			for(y=0;y<N;y++){
				B[x][y].i=0;
				fscanf(fp2,"%g",&B[x][y].r);
			}
		}
		fclose(fp2);
		//printf("\nFiles read\n\n");
	}

	t11=MPI_Wtime();
	if(myid==0){
		for(x=1;x<nproc;x++){
			MPI_Send(&A[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD);
			MPI_Send(&B[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD);
		}
		//printf("\n\nInitial data sent from 0 to others\n\n");
	}

	else{
		MPI_Recv(&A[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(&B[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		//printf("\nInitial data received from 0\n");
	}
	t12=MPI_Wtime();

	t1=MPI_Wtime();
 	//Performing N, N-point 1D FFT along rows 
 	//startTime=clock();		
	for (x=0;x<N/nproc;x++) {
      	c_fft1d(A[x], N, -1);
      	c_fft1d(B[x], N, -1);
   	}
	t2=MPI_Wtime(); 
	//printf("N, N-point 1d fft has been done  on A and B for the 1st time\n");

	t13=MPI_Wtime();

	if(myid==0){
		for(x=1;x<nproc;x++){
			MPI_Recv(&A[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			MPI_Recv(&B[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
		//printf("\nA and B has been received by 0 after first fft\n");
	}
	else{
		MPI_Send(&A[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD);
		MPI_Send(&B[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD);
		//printf("\nA and B has been sent to 0 after first fft\n");
	}
	t14=MPI_Wtime();

	t3=MPI_Wtime();
	if(myid==0){
		//Transposing the matrix
   		for (x=0;x<N;x++) {
      			for (y=x;y<N;y++) {
         			tempComplex= A[x][y];
         			A[x][y] = A[y][x];
         			A[y][x] = tempComplex;

         			tempComplex=B[x][y];
         			B[x][y]=B[y][x];
         			B[y][x]=tempComplex;
      			}
   		}
		//printf("\nA and B has been transposed by 0\n");
	}
	t4=MPI_Wtime();

	t15=MPI_Wtime();
	if(myid==0){
		for(x=1;x<nproc;x++){
			MPI_Send(&A[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD);
			MPI_Send(&B[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD);
		}
		//printf("\nA and B has been sent by 0 for the 2nd time\n");
	}

	else{
		MPI_Recv(&A[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(&B[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		//printf("\nA and B has been received from 0 for the 2nd time\n");
	}
	t16=MPI_Wtime();
   
	t5=MPI_Wtime();
	//Performing N, N-point 1D FFT along columns as we have transposed the matrices
   	for (x=0;x<N/nproc;x++) {
      		c_fft1d(A[x], N, -1);
      		c_fft1d(B[x], N, -1);
   	}
	//printf("\n2nd time fft on A and B has been done\n");

  
	//Code to do point-wise multiplication of A and B
	//for example C[x][y]=A[x][y]*B[x][y]
	//Complex numbers are multiplied as follows
	//Real part of result is difference between the product of real parts of the 2 inputs and product of both imaginary part
   	for (x=0;x<N/nproc;x++) {
      		for (y=0;y<N;y++) {
         		C[x][y].r = A[x][y].r*B[x][y].r - A[x][y].i*B[x][y].i;
         		C[x][y].i = A[x][y].r*B[x][y].i + A[x][y].i*B[x][y].r;
      		}
   	}

	//Performing N, N-point 1D inverse FFT along rows
     	for (x=0;x<N/nproc;x++) {
      		c_fft1d(C[x], N, 1);
   	}
	t6=MPI_Wtime();

	t17=MPI_Wtime();
	if(myid==0){
		for(x=1;x<nproc;x++){
			MPI_Recv(&C[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			//MPI_Recv(&B[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD,*status);
		}
	}
	else{
		MPI_Send(&C[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD);
		//MPI_Send(&B[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD);
	}
	t18=MPI_Wtime();

	t7=MPI_Wtime();
	if(myid==0){
	//Transposing the matrix
   	for (x=0;x<N;x++) {
      		for (y=x;y<N;y++) {
         		tempComplex=C[x][y];
         		C[x][y]=C[y][x];
         		C[y][x]=tempComplex;
      		}
   	}
	}
	t8=MPI_Wtime();

	t19=MPI_Wtime();
	if(myid==0){
		for(x=1;x<nproc;x++){
			MPI_Send(&C[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD);
		}
	}
	else{
		MPI_Recv(&C[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
	t20=MPI_Wtime();

	t9=MPI_Wtime();
	//Performing N. N-point 1D inverse FFT along columns
      	for (x=0;x<N/nproc;x++) {
      		c_fft1d(C[x], N, 1);
   	}
	//endTime=clock();
	//timeTaken=(double)(endTime-startTime)/CLOCKS_PER_SEC;
   	t10=MPI_Wtime();

	t21=MPI_Wtime();
	if(myid==0){
		for(x=1;x<nproc;x++){
			MPI_Recv(&C[x*N/nproc][0],N*N/nproc,MPI_COMPLEX,x,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
	}
	else{
		MPI_Send(&C[0][0],N*N/nproc,MPI_COMPLEX,0,0,MPI_COMM_WORLD);
	}
	t22=MPI_Wtime();


	if(myid==0){
		FILE *fp3 = fopen("outputMatrixSendRecv","w");

   		for (x=0;x<N;x++) {
      			for (y=0;y<N;y++)
         			fprintf(fp3,"   %e",C[x][y].r);
      			fprintf(fp3,"\n");
   		}

   		fclose(fp3);

		timeTaken=t2-t1+t4-t3+t6-t5+t8-t7+t10-t9;
		commTime=t12-t11+t14-t13+t16-t15+t18-t17+t20-t19+t22-t11;
		printf("\nOutput matrix has been generated and stored in outputMatrixSendRecv file\nTime taken for the computation = %f ms\n",timeTaken*1000);
		printf("\nCommunication time = %f ms\n\n",commTime*1000);
	}

	MPI_Finalize();
	return 0;  
}



/*
 ------------------------------------------------------------------------
 FFT1D            c_fft1d(r,i,-1)
 Inverse FFT1D    c_fft1d(r,i,+1)
 ------------------------------------------------------------------------
*/
/* ---------- FFT 1D
   This computes an in-place complex-to-complex FFT
   r is the real and imaginary arrays of n=2^m points.
   isign = -1 gives forward transform
   isign =  1 gives inverse transform
*/

void c_fft1d(complex *r, int      n, int      isign)
{
   int     m,i,i1,j,k,i2,l,l1,l2;
   float   c1,c2,z;
   complex t, u;

   if (isign == 0) return;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j)
         C_SWAP(r[i], r[j]);
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* m = (int) log2((double)n); */
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;

            /* t = u * r[i1] */
            t.r = u.r * r[i1].r - u.i * r[i1].i;
            t.i = u.r * r[i1].i + u.i * r[i1].r;

            /* r[i1] = r[i] - t */
            r[i1].r = r[i].r - t.r;
            r[i1].i = r[i].i - t.i;

            /* r[i] = r[i] + t */
            r[i].r += t.r;
            r[i].i += t.i;
         }
         z =  u.r * c1 - u.i * c2;

         u.i = u.r * c2 + u.i * c1;
         u.r = z;
      }
      c2 =sqrt((1.0 - c1) / 2.0);
      if (isign == -1) /* FWD FFT */
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for inverse transform */
   if (isign == 1) {       /* IFFT*/
      for (i=0;i<n;i++) {
         r[i].r /= n;
         r[i].i /= n;
      }
   }
}
