#include <cuda_runtime.h>

#define UINT64 long long

__global__ void	
sumKernel ( UINT64 * a, UINT64 * b, UINT64 * m, UINT64 * c )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	c[idx] = ( a[idx] + b[idx] ) % m[idx];
}

__global__ void	
diffKernel ( UINT64 * a, UINT64 * b, UINT64 * m, UINT64 * c )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	c[idx] = ( m[idx] + a[idx] - b[idx] ) % m[idx];
}

__global__ void	
mulKernel ( UINT64 * a, UINT64 * b, UINT64 * m, UINT64 * c )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	c[idx] = ( a[idx] * b[idx] ) % m[idx];
}

__global__ void
divKernel ( UINT64 * a, UINT64 * b, UINT64 * m, UINT64 * c )
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for ( int i = 0; i < m[ index ]; i++ )
		if ( ( i*b[ index ] + m[ index ] ) % m[ index ] == ( a[ index ] % m[ index ] ) )
		{
			c[ index ] = i;
			break;
		}	
}

void doOperation( UINT64 * aDev, UINT64 * bDev, UINT64 * mDev, UINT64 * cDev, int operationType, const dim3 & threads, const dim3 & blocks )
{
	if ( operationType == 1 )
		sumKernel<<<blocks, threads>>> (aDev, bDev, mDev, cDev);
	else if ( operationType == 2 )
		diffKernel<<<blocks, threads>>> (aDev, bDev, mDev, cDev);
	else if ( operationType == 3 ) 
		mulKernel<<<blocks, threads>>> (aDev, bDev, mDev, cDev);
	else if ( operationType == 4 )
		mulKernel<<<blocks, threads>>> (aDev, bDev, mDev, cDev);
}
