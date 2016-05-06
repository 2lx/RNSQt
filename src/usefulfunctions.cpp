#include "usefulfunctions.h"

UINT64 getGCDEuclid( const UINT64 a, const UINT64 b )
{
	UINT64 gcd, oper, tmp;

	gcd = ( a < b ) ? a : b;
	oper = ( a > b ) ? a : b;

	while( gcd > 0 )
		if ( !( oper % gcd ) )
			break;
		else 
		{
			tmp = oper % gcd;
			oper = gcd;
			gcd = tmp;
		}

		return gcd;
}

UINT64 getDeterminantGauss( int** Arr, int size )
{
	int i, j;
	UINT64 det=0;
	int** matr;
	if ( size == 0 )
	{
		det = 1;
	}
	else if( size == 1 )
	{
		det = Arr[ 0 ][ 0 ];
	}
	else if( size == 2 )
	{
		det = Arr[ 0 ][ 0 ]*Arr[ 1 ][ 1 ] - Arr[ 0 ][ 1 ]*Arr[ 1 ][ 0 ];
	}
	else
	{
		matr = new int*[ size - 1 ];
		for( i = 0; i < size; ++i )
		{
			for( j = 0; j < size - 1; ++j )
			{
				if( j < i ) 
					matr[ j ] = Arr[ j ];
				else matr[ j ] = Arr[ j + 1 ];
			}
			if ( ( ( i + j ) % 2 ) == 1 )
				det += -1 * getDeterminantGauss( matr, size - 1) * Arr[ i ][ size - 1 ];
			else det += getDeterminantGauss( matr, size - 1 ) * Arr[ i ][ size - 1 ];
		}
		delete[] matr;
	}
	return det;
}

