#ifndef RNSLOGIC_H
#define RNSLOGIC_H

#define UINT64 long long

#define MAX_MODULI_COUNT 10
#define MAX_MODULI_VALUE 50000
#define MAX_CONT_COUNT 30

#include <QObject>

class RNSLogic : public QObject
{
	Q_OBJECT

public:
	RNSLogic();
    ~RNSLogic() {
        for ( int i = 0; i < MAX_CONT_COUNT; i++ )
            delete [] contArray[i];
    }

private:
	UINT64 moduliCount;

	UINT64 moduli[ MAX_MODULI_COUNT ];
	UINT64 operand1mods[ MAX_MODULI_COUNT ];
	UINT64 operand2mods[ MAX_MODULI_COUNT ];
	UINT64 resultmods[ MAX_MODULI_COUNT ];

	int** contArray;	

    unsigned int FoperationType;

	// перевод числа из СОК в десятичную систему по Китайской Теореме Остатков (КТО) через континуанты нахождения НОД методом Евклида 
    UINT64 RNS2dec( UINT64 * nRNS, UINT64 mCount ) const;

	// перевод числа из десятичной системы в СОК
    void dec2RNS( const UINT64 nDec, UINT64 * nRNS, UINT64 mCount ) const;

public:
	// функция для получения континуанты нахождения наибольшего общего делителя (GCD) в Расширенном алгоритме Евклида
    UINT64 getContinuantEuclid( const UINT64 a, const UINT64 b ) const;
//	UINT64 getContinuantEuclid2( const UINT64 a, const UINT64 b ); // альтернативный способ

	bool checkModuliCoprime();

//	const QString & getConsoleOut() { return consoleOut; }
	
	// находим максимальное число в полученной СОК
    UINT64 getMaxRNSValue() const;

	// выполнение операции в CPU в одном потоке
    void SumRNS( const unsigned int operationType );

	// функция вызова выполнения операции в GPU CUDA
    void SumRNSCUDA( const unsigned int operationType );

	//-----------------------------------------------------------------------------------------
	// операции - обертки над приватными переменными - properties
	//-----------------------------------------------------------------------------------------

    unsigned int getOperationType() const { return FoperationType; }

    void setModuliCount( const UINT64 newC ) { moduliCount = newC; }
	
    void setModuliValue( const int index, const UINT64 nValue )
	{ 
		if ( ( index < 0 ) || ( index > ( MAX_MODULI_COUNT - 1 ) ) )
			return;
		moduli[ index ] = nValue;
    }

    UINT64 getModuliBitValue( const int index ) const
	{ 
		if ( ( index < 0 ) || ( index > ( moduliCount - 1 ) ) )
			return 0;
		return moduli[ index ]; 
	} 

	//---
    void setFirstOpBitValue( const int index, const UINT64 nValue )
	{ 
		if ( ( index < 0 ) || ( index > ( MAX_MODULI_COUNT - 1 ) ) )
			return;
		operand1mods[ index ] = nValue;
    }
	
    UINT64 getFirstOpBitValue( const int index ) const
	{ 
		if ( ( index < 0 ) || ( index > ( moduliCount - 1 ) ) )
			return 0;
		return operand1mods[ index ]; 
	} 

	//---
	void setFirstResult( const UINT64 nVal ) 
	{ 
		if ( ( nVal < 0 ) || ( nVal > getMaxRNSValue() ) )
			return;
		dec2RNS( nVal, operand1mods, moduliCount );
	} 
	
    UINT64 getFirstResult() const { return RNS2dec( (UINT64 *)operand1mods, moduliCount ); }

	//---
    void setSecondOpBitValue( const int index, const UINT64 nValue )
	{ 
		if ( ( index < 0 ) || ( index > ( MAX_MODULI_COUNT - 1 ) ) )
			return;
		operand2mods[ index ] = nValue;
    }

    UINT64 getSecondOpBitValue( const int index ) const
	{ 
		if ( ( index < 0 ) || ( index > ( moduliCount - 1 ) ) )
			return 0;
		return operand2mods[ index ]; 
	} 

	//---
	void setSecondResult( const UINT64 nVal ) 
	{ 
		if ( ( nVal < 0 ) || ( nVal > getMaxRNSValue() ) )
			return;
		dec2RNS( nVal, operand2mods, moduliCount );
	} 

    UINT64 getSecondResult() const { return RNS2dec( (UINT64 *)operand2mods, moduliCount ); }

	//---
    UINT64 getResultBitValue( const int index ) const
	{ 
		if ( ( index < 0 ) || ( index > ( moduliCount - 1 ) ) )
			return 0;
		return resultmods[ index ]; 
	} 

    UINT64 getTotalResult() const { return RNS2dec( (UINT64 *)resultmods, moduliCount ); }

	//---
    bool getSecondDividedResult( const UINT64 val ) const
	{
		for ( uint i = 0; i < moduliCount; i++ )
			if ( !( val % moduli[ i ] ) )
					return true;
		
		return false;
	}

signals:
	void consoleOut( const QString & strOut );
};


#endif // RNSQT_H
