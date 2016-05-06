#ifndef USEFULFUNCTIONS_H
#define USEFULFUNCTIONS_H

#define UINT64 long long

// функция для получения наибольшего общего делителя (GCD) - Расширенный алгоритм Евклида
UINT64 getGCDEuclid( const UINT64 a, const UINT64 b );

//нахождение определителя матрицы методом Гаусса
UINT64 getDeterminantGauss( int** Arr, int size );

#endif // USEFULFUNCTIONS_H
