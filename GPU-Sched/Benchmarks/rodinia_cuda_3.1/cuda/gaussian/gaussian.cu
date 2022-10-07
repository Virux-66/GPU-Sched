/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>

#ifdef RD_WG_SIZE_0_0
        #define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define MAXBLOCKSIZE RD_WG_SIZE
#else
        #define MAXBLOCKSIZE 512
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_XY RD_WG_SIZE
#else
        #define BLOCK_SIZE_XY 4
#endif

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;
FILE *fp_out;

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
__global__ void Fan1(float *m, float *a, int Size, int t);
__global__ void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}


int main(int argc, char *argv[])
{
    fp_out = fopen("gt_gaussian_result.txt", "w");
  fprintf(fp_out, "WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    int verbose = 1;
    int i, j;
    char flag;
    if (argc < 2) {
        fprintf(fp_out, "Usage: gaussian -f filename / -s size [-q]\n\n");
        fprintf(fp_out, "-q (quiet) suppresses printing the matrix and result values.\n");
        fprintf(fp_out, "-f (filename) path of input file\n");
        fprintf(fp_out, "-s (size) size of matrix. Create matrix and rhs in this program \n");
        fprintf(fp_out, "The first line of the file contains the dimension of the matrix, n.");
        fprintf(fp_out, "The second line of the file is a newline.\n");
        fprintf(fp_out, "The next n lines contain n tab separated values for the matrix.");
        fprintf(fp_out, "The next line of the file is a newline.\n");
        fprintf(fp_out, "The next line of the file is a 1xn vector with tab separated values.\n");
        fprintf(fp_out, "The next line of the file is a newline. (optional)\n");
        fprintf(fp_out, "The final line of the file is the pre-computed solution. (optional)\n");
        fprintf(fp_out, "Example: matrix4.txt:\n");
        fprintf(fp_out, "4\n");
        fprintf(fp_out, "\n");
        fprintf(fp_out, "-0.6	-0.5	0.7	0.3\n");
        fprintf(fp_out, "-0.3	-0.9	0.3	0.7\n");
        fprintf(fp_out, "-0.4	-0.5	-0.3	-0.8\n");	
        fprintf(fp_out, "0.0	-0.1	0.2	0.9\n");
        fprintf(fp_out, "\n");
        fprintf(fp_out, "-0.85	-0.68	0.24	-0.53\n");	
        fprintf(fp_out, "\n");
        fprintf(fp_out, "0.7	0.0	-0.4	-0.5\n");
        fclose(fp_out);
        exit(0);
    }
    
    //PrintDeviceProperties();
    //char filename[100];
    //sfprintf(fp_out, filename,"matrices/matrix%d.txt",size);

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // platform
              i++;
              Size = atoi(argv[i]);
	      fprintf(fp_out, "Create matrix internally in parse, size = %d \n", Size);

	      a = (float *) malloc(Size * Size * sizeof(float));
	      create_matrix(a, Size);

	      b = (float *) malloc(Size * sizeof(float));
	      for (j =0; j< Size; j++)
	    	b[j]=1.0;

	      m = (float *) malloc(Size * Size * sizeof(float));
              break;
            case 'f': // platform
              i++;
	      fprintf(fp_out, "Read file from %s \n", argv[i]);
	      InitProblemOnce(argv[i]);
              break;
            case 'q': // quiet
	      verbose = 0;
              break;
	  }
      }
    }

    //InitProblemOnce(filename);
    InitPerRun();
    //begin timing
    struct timeval time_start;
    gettimeofday(&time_start, NULL);	
    
    // run kernels
    ForwardSub();
    
    //end timing
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
    
    if (verbose) {
        fprintf(fp_out, "Matrix m is: \n");
        PrintMat(m, Size, Size);

        fprintf(fp_out, "Matrix a is: \n");
        PrintMat(a, Size, Size);

        fprintf(fp_out, "Array b is: \n");
        PrintAry(b, Size);
    }
    BackSub();
    if (verbose) {
        fprintf(fp_out, "The final solution is: \n");
        PrintAry(finalVec,Size);
    }
    fprintf(fp_out, "\nTime total (including memory transfers)\t%f sec\n", time_total * 1e-6);
    fprintf(fp_out, "Time for CUDA kernels:\t%f sec\n",totalKernelTime * 1e-6);
    
    /*fprintf(fp_out, "%d,%d\n",size,time_total);
    fprintf(stderr,"%d,%d\n",size,time_total);*/
    
    free(m);
    free(a);
    free(b);
    fclose(fp_out);
}
/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties(){
	cudaDeviceProp deviceProp;  
	int nDevCount = 0;  
	
	cudaGetDeviceCount( &nDevCount );  
	fprintf(fp_out,  "Total Device found: %d", nDevCount );  
	for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx )  
	{  
	    memset( &deviceProp, 0, sizeof(deviceProp));  
	    if( cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx))  
	        {
				fprintf(fp_out,  "\nDevice Name \t\t - %s ", deviceProp.name );  
			    fprintf(fp_out,  "\n**************************************");  
			    fprintf(fp_out,  "\nTotal Global Memory\t\t\t - %lu KB", deviceProp.totalGlobalMem/1024 );  
			    fprintf(fp_out,  "\nShared memory available per block \t - %lu KB", deviceProp.sharedMemPerBlock/1024 );  
			    fprintf(fp_out,  "\nNumber of registers per thread block \t - %d", deviceProp.regsPerBlock );  
			    fprintf(fp_out,  "\nWarp size in threads \t\t\t - %d", deviceProp.warpSize );  
			    fprintf(fp_out,  "\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch );  
			    fprintf(fp_out,  "\nMaximum threads per block \t\t - %d", deviceProp.maxThreadsPerBlock );  
			    fprintf(fp_out,  "\nMaximum Thread Dimension (block) \t - %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );  
			    fprintf(fp_out,  "\nMaximum Thread Dimension (grid) \t - %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );  
			    fprintf(fp_out,  "\nTotal constant memory \t\t\t - %zu bytes", deviceProp.totalConstMem );  
			    fprintf(fp_out,  "\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major, deviceProp.minor );  
			    fprintf(fp_out,  "\nClock rate \t\t\t\t - %d KHz", deviceProp.clockRate );  
			    fprintf(fp_out,  "\nTexture Alignment \t\t\t - %zu bytes", deviceProp.textureAlignment );  
			    fprintf(fp_out,  "\nDevice Overlap \t\t\t\t - %s", deviceProp. deviceOverlap?"Allowed":"Not Allowed" );  
			    fprintf(fp_out,  "\nNumber of Multi processors \t\t - %d\n\n", deviceProp.multiProcessorCount );  
			}  
	    else  
	        fprintf(fp_out,  "\n%s", cudaGetErrorString(cudaGetLastError()));  
	}  
}
 
 
/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename)
{
	//char *filename = argv[1];
	
	//fprintf(fp_out, "Enter the data file name: ");
	//scanf("%s", filename);
	//fprintf(fp_out, "The file name is: %s\n", filename);
	
	fp = fopen(filename, "r");
	
	fscanf(fp, "%d", &Size);	
	 
	a = (float *) malloc(Size * Size * sizeof(float));
	 
	InitMat(a, Size, Size);
	//fprintf(fp_out, "The input matrix a is:\n");
	//PrintMat(a, Size, Size);
	b = (float *) malloc(Size * sizeof(float));
	
	InitAry(b, Size);
	//fprintf(fp_out, "The input array b is:\n");
	//PrintAry(b, Size);
		
	 m = (float *) malloc(Size * Size * sizeof(float));
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{   
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) fprintf(fp_out, ".");
	//fprintf(fp_out, "blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */ 

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;
	
	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	//fprintf(fp_out, "blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
	
	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if(yidx == 0){
		//fprintf(fp_out, "blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//fprintf(fp_out, "xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub()
{
	int t;
    float *m_cuda,*a_cuda,*b_cuda;

	int block_size,grid_size;
	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	//fprintf(fp_out, "1d grid size: %d\n",grid_size);

	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	
	int blockSize2d, gridSize2d;
	blockSize2d = BLOCK_SIZE_XY;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
	
	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);
	
	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	 
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	
	cudaMalloc((void **) &b_cuda, Size * sizeof(float));	

	// copy memory to GPU
	cudaMemcpy(m_cuda, m, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(a_cuda, a, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(b_cuda, b, Size * sizeof(float),cudaMemcpyHostToDevice );
	
    // begin timing kernels
    struct timeval time_start;
    gettimeofday(&time_start, NULL);
	for (t=0; t<(Size-1); t++) {
		Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		cudaThreadSynchronize();
		Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		cudaThreadSynchronize();
		checkCUDAError("Fan2");
	}
	// end timing kernels
	struct timeval time_end;
    gettimeofday(&time_end, NULL);
    totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
	
	// copy memory back to CPU
	cudaMemcpy(m, m_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(a, a_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaFree(m_cuda);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
	// create a new vector to hold the final answer
	finalVec = (float *) malloc(Size * sizeof(float));
	// solve "bottom up"
	int i,j;
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}

void InitMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+Size*i+j);
		}
	}  
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fprintf(fp_out, "%8.2f ", *(ary+Size*i+j));
		}
		fprintf(fp_out, "\n");
	}
	fprintf(fp_out, "\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}  

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		fprintf(fp_out, "%.2f ", ary[i]);
	}
	fprintf(fp_out, "\n\n");
}
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        fclose(fp_out);
        exit(EXIT_FAILURE);
    }                         
}

