#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include<time.h>

typedef struct {float r; float i;} complex;
static complex ctmp;
#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}
void c_fft1d(complex *r, int n, int isign);

const int N = 512;

int main (int argc, char **argv) {

   
	const char* file1="im1";
   	const char* file2="im2";

	printf("\n\nCS546 Project - Koushik Raman A20388858\n\n\n");
   	printf("Sequential program\n\n");
   
	complex A[N][N],B[N][N],C[N][N],tempComplex;
	int x,y;
   	FILE *fp1 = fopen(file1,"r");
	double startTime,endTime,timeTaken;

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

 	//Performing N, N-point 1D FFT along rows 
 	startTime=clock();
	for (x=0;x<N;x++) {
      	c_fft1d(A[x], N, -1);
      	c_fft1d(B[x], N, -1);
   	}
 
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
   
	//Performing N, N-point 1D FFT along columns as we have transposed the matrices
   	for (x=0;x<N;x++) {
      		c_fft1d(A[x], N, -1);
      		c_fft1d(B[x], N, -1);
   	}

  
	//Code to do point-wise multiplication of A and B
	//for example C[x][y]=A[x][y]*B[x][y]
	//Complex numbers are multiplied as follows
	//Real part of result is difference between the product of real parts of the 2 inputs and product of both imaginary part
   	for (x=0;x<N;x++) {
      		for (y=0;y<N;y++) {
         		C[x][y].r = A[x][y].r*B[x][y].r - A[x][y].i*B[x][y].i;
         		C[x][y].i = A[x][y].r*B[x][y].i + A[x][y].i*B[x][y].r;
      		}
   	}

	//Performing N, N-point 1D inverse FFT along rows
     	for (x=0;x<N;x++) {
      		c_fft1d(C[x], N, 1);
   	}

	//Transposing the matrix
   	for (x=0;x<N;x++) {
      		for (y=x;y<N;y++) {
         		tempComplex=C[x][y];
         		C[x][y]=C[y][x];
         		C[y][x]=tempComplex;
      		}
   	}

	//Performing N. N-point 1D inverse FFT along columns
      	for (x=0;x<N;x++) {
      		c_fft1d(C[x], N, 1);
   	}
	endTime=clock();
	timeTaken=(double)(endTime-startTime)/CLOCKS_PER_SEC;
   
	FILE *fp3 = fopen("outputMatrix","w");

   	for (x=0;x<N;x++) {
      		for (y=0;y<N;y++)
         		fprintf(fp3,"   %e",C[x][y].r);
      		fprintf(fp3,"\n");
   	}

   	fclose(fp3);

	printf("\nOutput matrix has been generated and stored in outputMatrix file\nTime taken for the serial computation = %f sec\n",timeTaken);
	
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
