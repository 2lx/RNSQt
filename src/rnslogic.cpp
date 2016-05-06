#include "rnslogic.h"
#include "usefulfunctions.h"
#include <cuda_runtime_api.h>

void doOperation( UINT64 * aDev, UINT64 * bDev, UINT64 * mDev, UINT64 * cDev, int operationType, const dim3 & threads, const dim3 & blocks );

RNSLogic::RNSLogic()
{
    // инициализация массива континуант
	contArray = new int*[ MAX_CONT_COUNT-1 ];
	for ( int i = 0; i < MAX_CONT_COUNT; i++ )
	{
		contArray[ i ] = new int[ MAX_CONT_COUNT - 1];
		for ( int j = 0; j < MAX_CONT_COUNT; j++ )
		{	if ( ( i - j ) == 1 ) 
		contArray[ i ][ j ] = -1;
		else if ( ( j - i ) == 1 ) 
			contArray[ i ][ j ] = 1;
		else contArray[ i ][ j ] = 0;
		}
	}
}

bool RNSLogic::checkModuliCoprime()
{
	bool isCoprime = true;
	
	for ( unsigned int i = 0; i < moduliCount - 1; i++ )
	{
		for ( unsigned int j = i + 1; j < moduliCount; j++ )
			if ( getGCDEuclid( moduli[ i ], moduli[ j ] ) > 1 )
			{
				isCoprime = false;
				emit consoleOut ( "Ошибка, модули должны быть взаимно простые! \n" );
				emit consoleOut ( "НОД чисел " + QString::number( moduli[ i ] ) + ", " 
					+ QString::number( moduli[ j ] ) + " = " + QString::number( getGCDEuclid( moduli[ i ], moduli[ j ] ) ) );
				break;
			}
		if ( !isCoprime ) break;
	}
	
	if ( isCoprime )
		emit consoleOut ( "Модули взаимно простые" );

	return isCoprime;
};

UINT64 RNSLogic::getMaxRNSValue() const
{
	UINT64 greatestRNSValue = 1;
	for ( unsigned int i = 0; i < moduliCount; i++ )
		greatestRNSValue *= moduli[ i ];
	greatestRNSValue--;	

	return greatestRNSValue;
}

UINT64 RNSLogic::getContinuantEuclid( const UINT64 a, const UINT64 b ) const
{
	UINT64 gcd, oper, tmp, cont;
	unsigned int contSize = 0;

	gcd = ( a < b ) ? a : b;
	oper = ( a > b ) ? a : b;

    while( gcd > 0 ) {
		if ( !( oper % gcd ) )
			break;
		else 
		{				
			contArray[ contSize ][ contSize ] = oper / gcd;
			contSize++;	

			tmp = oper % gcd;
			oper = gcd;
			gcd = tmp;
		}
    }

    if ( ( contSize % 2 ) == 1 )
         cont = ( -1 * getDeterminantGauss( contArray, contSize ) + b ) % b;
    else cont = getDeterminantGauss( contArray, contSize );

    return cont;
}

UINT64 getContinuantEuclid2( const UINT64 a, const UINT64 b )
{
	if ( !a ) return 0;

	UINT64 r1, r2, r, q, t1, t2, t;
	r1 = r2 = r = q = t1 = t2 = t = 0;

	t1 = 0;
	t2 = 1;
	r1 = b;
	r2 = a;

	q = r1 / r2;
	r = r1 % r2;
	t = t1 - q * t2;
	while(r2 != 1)
	{
		r1 = r2;
		r2 = r;
		r = r1 % r2;
		q = r1 / r2;
		t1 = t2;
		t2 = t;
		t = t1 - q * t2;
	}
	while(t2 < 0)
	{
		t2 += b;
	}
	return (UINT64)t2;
}

UINT64 RNSLogic::RNS2dec( UINT64 * nRNS, UINT64 mCount ) const
{
	if ( mCount < 1 )
		return 0;

	UINT64 resDec = 0, mulAll = 1, mulPart;
	UINT64 contPart;

	for ( unsigned int i = 0; i < mCount; i++ )
		mulAll *= moduli[ i ];

	for ( unsigned int i = 0; i < mCount; i++ )
	{
		mulPart = 1;
		for ( unsigned int j = 0; j < mCount; j++ )
			if ( j != i ) 
				mulPart *= moduli[ j ];

		contPart = getContinuantEuclid( mulPart % moduli[ i ], moduli[ i ] );
		//contPart = getContinuantEuclid2( mulPart % moduli[ i ], moduli[ i ] ); // альтернативный способ

		resDec += ( ( nRNS[ i ] % mulAll ) * ( mulPart % mulAll ) * ( contPart % mulAll) ) % mulAll;
	}

	resDec = resDec % mulAll;

	return resDec;
}

void RNSLogic::dec2RNS( const UINT64 nDec, UINT64 * nRNS, UINT64 mCount ) const
{
	if ( mCount < 1 )
		return;

	for ( unsigned int i = 0; i < mCount; i++ )
		nRNS[ i ] = nDec % moduli[ i ];
}

void RNSLogic::SumRNS( const uint operationType )
{
	if ( moduliCount < 1 )
		return;

	for ( unsigned int index = 0; index < moduliCount; index++ )
		switch ( operationType )
	{
		case 1:
			resultmods[ index ] = ( operand1mods[ index ] + operand2mods[ index ] ) % moduli[ index ];
			break;
		case 2:
			resultmods[ index ] = ( moduli[ index ] + operand1mods[ index ] - operand2mods[ index ] ) % moduli[ index ];
			break;
		case 3:
			resultmods[ index ] = ( operand1mods[ index ] * operand2mods[ index ] ) % moduli[ index ];
			break;
		case 4:
			UINT64 invVal = getContinuantEuclid( operand2mods[ index ], moduli[ index ] );
			if ( moduli[ index ] )
				resultmods[ index ] = ( operand1mods[ index ] * invVal ) % moduli[ index ];
			else resultmods[ index ] = 0;

            break;
	}

	emit consoleOut( "Результат получен при использовании 1 CPU" );
}

void RNSLogic::SumRNSCUDA( const uint operationType )
{
	unsigned int numBytes = moduliCount * sizeof( UINT64 );

	if ( operationType == 4 )
		for( int i = 0; i < moduliCount; i++ )
			operand2mods[ i ] = getContinuantEuclid( operand2mods[ i ], moduli[ i ] );

	UINT64 * aDev = NULL;
	UINT64 * bDev = NULL;
	UINT64 * mDev = NULL;
	UINT64 * cDev = NULL;

	// выделить память GPU
	cudaMalloc( ( void** )&aDev, numBytes );
	cudaMalloc( ( void** )&bDev, numBytes );
	cudaMalloc( ( void** )&mDev, numBytes );
	cudaMalloc( ( void** )&cDev, numBytes );

	// конфигурация нитей
	dim3 threads = dim3( moduliCount, 1 );
	dim3 blocks = dim3( moduliCount / threads.x, 1 );

	// скопировать массивы из памяти CPU в память GPU
	cudaMemcpy( aDev, &operand1mods, numBytes, cudaMemcpyHostToDevice );
	cudaMemcpy( bDev, &operand2mods, numBytes, cudaMemcpyHostToDevice );
	cudaMemcpy( mDev, &moduli, numBytes, cudaMemcpyHostToDevice );

	// вызывть ядро с заданной конфигурацией для обработки данных
	doOperation( aDev, bDev, mDev, cDev, operationType, threads, blocks );

	// скопировать результаты в обратно в память CPU
	cudaMemcpy( &resultmods, cDev, numBytes, cudaMemcpyDeviceToHost );

	// освободить память GPU
	cudaFree( aDev );
	cudaFree( bDev );
	cudaFree( cDev );

    emit consoleOut( "Результат получен при использовании GPU" );
}
