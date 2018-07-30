#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include<time.h>
#include<mpi.h>

typedef struct {float r; float i;} complex;
static complex ctmp;
#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}
void c_fft1d(complex *r, int n, int isign);

const int N = 512;

int main (int argc, char **argv) {
	int nproc,myid;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
   
	const char* file1="im1";
   	const char* file2="im2";

	if(myid==0){
	//printf("\n\nCS546 Project - Koushik Raman A20388858\n\n\n");
   	printf("Program for Task and Data Parallel\n\n");
   	}

	complex A[N][N],B[N][N],C[N][N],tempComplex;
	int x,y;
	double startTime,endTime,timeTaken,commTime,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23, t24,t25,t26,t27,t28,t29,t30;

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
	
	if(myid!=0){
		for(x=0;x<N;x++){
			for(y=0;y<N;y++){
				A[x][y].i=0;	A[x][y].r=0;
				B[x][y].i=0;	B[x][y].r=0;
				C[x][y].i=0;	C[x][y].r=0;
			}
		}
	}

	int groupSize = nproc/4;
   	int myGroupRank;
   	int P1_array[groupSize], P2_array[groupSize], P3_array[groupSize], P4_array[groupSize];
	int processorGroup;
	for(x=0; x<nproc; x++) {
      	processorGroup = x / groupSize;

	if(processorGroup==0)
		P1_array[x%groupSize]=x;
	else if(processorGroup==1)
		P2_array[x%groupSize]=x;
	else if(processorGroup==2)
		P3_array[x%groupSize]=x;
	else if(processorGroup==3)
		P4_array[x%groupSize]=x;
   	}

	MPI_Group worldGroup, P1, P2, P3, P4; 
   	MPI_Comm P1_comm, P2_comm, P3_comm, P4_comm;
   	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

	int myGroup = myid/groupSize;

	if ( myGroup == 0 )      {       
      		MPI_Group_incl(worldGroup, nproc/4, P1_array, &P1);
      		MPI_Comm_create( MPI_COMM_WORLD, P1, &P1_comm);
      		MPI_Group_rank(P1, &myGroupRank);
   	} 
   	else if ( myGroup == 1 ) { 
      		MPI_Group_incl(worldGroup, nproc/4, P2_array, &P2); 
      		MPI_Comm_create( MPI_COMM_WORLD, P2, &P2_comm);
      		MPI_Group_rank(P2, &myGroupRank);
   	} 
   	else if ( myGroup == 2 ) { 
      		MPI_Group_incl(worldGroup, nproc/4, P3_array, &P3); 
      		MPI_Comm_create( MPI_COMM_WORLD, P3, &P3_comm);
      		MPI_Group_rank(P3, &myGroupRank);
   	} 
   	else if ( myGroup == 3 ) { 
      		MPI_Group_incl(worldGroup, nproc/4, P4_array, &P4); 
      		MPI_Comm_create( MPI_COMM_WORLD, P4, &P4_comm);
      		MPI_Group_rank(P4, &myGroupRank);
   	}

	int chunk = N / groupSize;

	if(myid==0)	t15=MPI_Wtime();
	if ( myid == 0 ){

      		for ( x=0; x<groupSize; x++ ) {
        		if ( P1_array[x]==0 ) continue;
        		MPI_Send( &A[chunk*x][0], chunk*N, MPI_COMPLEX, P1_array[x], 0, MPI_COMM_WORLD );
      		}
      
      		for ( x=0; x<groupSize; x++ ) {
        		if ( P2_array[x]==0 ) continue;
        		MPI_Send( &B[chunk*x][0], chunk*N, MPI_COMPLEX, P2_array[x], 0, MPI_COMM_WORLD );
      		}
   	}
   	else {
      		if ( myGroup == 0 )
         		MPI_Recv( &A[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
      
      		if ( myGroup == 1 )
         		MPI_Recv( &B[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
   	}
	if(myid==0)	t16=MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t1=MPI_Wtime();
	if ( myGroup == 0 )
      		for ( x=chunk*myGroupRank; x<chunk*(myGroupRank+1); x++ )
         		c_fft1d(A[x], N, -1);

   
   	if ( myGroup == 1 )
      		for ( x=chunk*myGroupRank; x<chunk*(myGroupRank+1); x++ )
         		c_fft1d(B[x], N, -1);
	
	if(myid==0)	t2=MPI_Wtime();

   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t17=MPI_Wtime();
	if ( myGroup == 0 ) {
      		if ( myGroupRank == 0 ) {
         		for ( x=1; x<groupSize; x++ ) {
            			MPI_Recv( &A[chunk*x][0], chunk*N, MPI_COMPLEX, x, 0, P1_comm, MPI_STATUS_IGNORE );
         		}
         
      		}
      		else 
         		MPI_Send( &A[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, P1_comm );
   	}

   	if ( myGroup == 1 ) {
      		if ( myGroupRank == 0 ) {
         		for ( x=1; x<groupSize; x++ ) {
            			MPI_Recv( &B[chunk*x][0], chunk*N, MPI_COMPLEX, x, 0, P2_comm, MPI_STATUS_IGNORE );
         		}
         
      		}
      		else 
        		MPI_Send( &B[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, P2_comm );
   	}
	if(myid==0)	t18=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t3=MPI_Wtime();
	if ( myGroup == 0 && myGroupRank == 0 ) {
      		for (x=0;x<N;x++) {
         		for (y=x;y<N;y++) {
            			tempComplex = A[x][y];
            			A[x][y] = A[y][x];
            			A[y][x] = tempComplex;
         		}
      		}
   	}

    
   	if ( myGroup == 1 && myGroupRank == 0 ) {
      		for (x=0;x<N;x++) {
         		for (y=x;y<N;y++) {
            			tempComplex = B[x][y];
            			B[x][y] = B[y][x];
            			B[y][x] = tempComplex;
         		}
      		}
   	}
	if(myid==0)	t4=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t19=MPI_Wtime();
	if ( myGroup == 0 ) {
      		if ( myGroupRank == 0 ) {
         		for ( x=1; x<groupSize; x++ ) {
            			MPI_Send( &A[chunk*x][0], chunk*N, MPI_COMPLEX, x, 0, P1_comm );
         		}
         
      		}
      		else 
         		MPI_Recv( &A[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, P1_comm, MPI_STATUS_IGNORE );
   	}
  

   	if ( myGroup == 1 ) {
      		if ( myGroupRank == 0 ) {
         		for ( x=1; x<groupSize; x++ ) {
            			MPI_Send( &B[chunk*x][0], chunk*N, MPI_COMPLEX, x, 0, P2_comm );
         		}
         
      		}
      		else 
         		MPI_Recv( &B[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, P2_comm, MPI_STATUS_IGNORE );
   	}
	if(myid==0)	t20=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t5=MPI_Wtime();
	if ( myGroup == 0 )
      		for ( x=chunk*myGroupRank; x<chunk*(myGroupRank+1); x++ )
         		c_fft1d(A[x], N, -1);


  	if ( myGroup == 1 )
      		for ( x=chunk*myGroupRank; x<chunk*(myGroupRank+1); x++ )
         		c_fft1d(B[x], N, -1);

	if(myid==0)	t6=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t21=MPI_Wtime();
	if ( myGroup == 0 )
      		MPI_Send ( &A[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, P3_array[myGroupRank], 0, MPI_COMM_WORLD );
   	else if ( myGroup == 1 )
      		MPI_Send ( &B[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, P3_array[myGroupRank], 0, MPI_COMM_WORLD );

   	else if ( myGroup == 2 ) {
      		MPI_Recv( &A[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, P1_array[myGroupRank], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      		MPI_Recv( &B[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, P2_array[myGroupRank], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
   	}
	if(myid==0)	t22=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t7=MPI_Wtime();
	if ( myGroup == 2 ) {
      		for (x= chunk*myGroupRank ;x< chunk*(myGroupRank+1);x++) {
         		for (y=0;y<N;y++) {
            			C[x][y].r = A[x][y].r*B[x][y].r - A[x][y].i*B[x][y].i;
            			C[x][y].i = A[x][y].r*B[x][y].i + A[x][y].i*B[x][y].r;
         		}
      		}
   	}
	if(myid==0)	t8=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t23=MPI_Wtime();
	if ( myGroup == 2 ) {
      		MPI_Send ( &C[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, P4_array[myGroupRank], 0, MPI_COMM_WORLD );
   	}
   	else if ( myGroup == 3 ) {
      		MPI_Recv ( &C[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, P3_array[myGroupRank], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
   	}
  	if(myid==0)	t24=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t9=MPI_Wtime();
	if ( myGroup == 3 )
      		for ( x=chunk*myGroupRank; x<chunk*(myGroupRank+1); x++ )
         		c_fft1d(C[x], N, 1);
   
	if(myid==0)	t10=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);


	if(myid==0)	t25=MPI_Wtime();
	if ( myGroup == 3 ) {
      		if ( myGroupRank == 0 ) {
         		for ( x=1; x<groupSize; x++ ) {
            			MPI_Recv( &C[chunk*x][0], chunk*N, MPI_COMPLEX, x, 0, P4_comm, MPI_STATUS_IGNORE );
         		}
         
      		}
      		else 
         		MPI_Send( &C[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, P4_comm );
   	}
	if(myid==0)	t26=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t11=MPI_Wtime();
	if ( myGroup == 3 && myGroupRank == 0 ) {
      		for (x=0;x<N;x++) {
         		for (y=x;y<N;y++) {
            			tempComplex = C[x][y];
            			C[x][y] = C[y][x];
            			C[y][x] = tempComplex;
         		}
      		}
   	}
	if(myid==0)	t12=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);


	if(myid==0)	t27=MPI_Wtime();
	if ( myGroup == 3 ) {
      		if ( myGroupRank == 0 ) {
         		for ( x=1; x<groupSize; x++ ) {
            			MPI_Send( &C[chunk*x][0], chunk*N, MPI_COMPLEX, x, 0, P4_comm );
         		}
      		}
      		else 
         		MPI_Recv( &C[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, P4_comm, MPI_STATUS_IGNORE );
   	}
	if(myid==0)	t28=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)	t13=MPI_Wtime();
	if ( myGroup == 3 )
      		for ( x=chunk*myGroupRank; x<chunk*(myGroupRank+1); x++ )
         		c_fft1d(C[x], N, 1);

	if(myid==0)	t14=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);


	if(myid==0)	t29=MPI_Wtime();
	if ( myid == 0 ){
      		for ( x=0; x<groupSize; x++ ) {
         		if ( P4_array[x]==0 ) continue; 

         		MPI_Recv( &C[chunk*x][0], chunk*N, MPI_COMPLEX, P4_array[x], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      		}
   	}
   	else if ( myGroup == 3 )
      		MPI_Send( &C[chunk*myGroupRank][0], chunk*N, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD );

	if(myid==0)	t30=MPI_Wtime();
   	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0){
		FILE *fpfinal=fopen("outputTaskDataParallel","w");
		for(x=0;x<N;x++){
			for(y=0;y<N;y++){
				fprintf(fpfinal,"   %e",C[x][y].r);
			}
			fprintf(fpfinal,"\n");
		}
		fclose(fpfinal);
		

		timeTaken=t2-t1+t4-t3+t6-t5+t8-t7+t10-t9+t12-t11+t14-t13;
		commTime=t16-t15+t18-t17+t20-t19+t22-t21+t24-t23+t26-t25+t28-t27+t30-t29;
		printf("\n\nOutput matrix has been generated and stored in the file outputTaskDataParallel\n\n");
		printf("Time taken for the computation = %f ms\nTime taken for communication = %f ms\n\n",timeTaken,commTime);


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
