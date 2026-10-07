#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "omp.h"

using matrix_int8_t = std::vector<int8_t>;
using matrix_int32_t = std::vector<int32_t>;

void setup_matrixes( matrix_int8_t& A, matrix_int8_t& B, matrix_int32_t& C, size_t MATRIX_SIZE,
                     uint8_t values_range ) {
  std::random_device rd;
  std::mt19937 gen( rd() );
  std::bernoulli_distribution dist( 0.5 );

#pragma omp parallel for
  for ( size_t i = 0; i < MATRIX_SIZE; i++ ) {
    for ( size_t j = 0; j < MATRIX_SIZE; j++ ) {
      A[ i * MATRIX_SIZE + j ] = ( dist( gen ) ? 1 : -1 ) * ( std::rand() % values_range );
      B[ i * MATRIX_SIZE + j ] = ( dist( gen ) ? 1 : -1 ) * ( std::rand() % values_range );
      C[ i * MATRIX_SIZE + j ] = 0;
    }
  }
}

void multiply( const matrix_int8_t& A, const matrix_int8_t& B, matrix_int32_t& C,
               size_t MATRIX_SIZE ) {
  for ( size_t i = 0; i < MATRIX_SIZE; i++ ) {
    for ( size_t k = 0; k < MATRIX_SIZE; k++ ) {
      int32_t a_ik = A[ i * MATRIX_SIZE + k ];
      for ( size_t j = 0; j < MATRIX_SIZE; j++ ) {
        C[ i * MATRIX_SIZE + j ] += a_ik * B[ k * MATRIX_SIZE + j ];
      }
    }
  }
}

void multiply_omp( const matrix_int8_t& A, const matrix_int8_t& B, matrix_int32_t& C,
                   size_t MATRIX_SIZE ) {
#pragma omp parallel for
  for ( size_t i = 0; i < MATRIX_SIZE; i++ ) {
    for ( size_t k = 0; k < MATRIX_SIZE; k++ ) {
      int32_t a_ik = A[ i * MATRIX_SIZE + k ];
      for ( size_t j = 0; j < MATRIX_SIZE; j++ ) {
        C[ i * MATRIX_SIZE + j ] += a_ik * B[ k * MATRIX_SIZE + j ];
      }
    }
  }
}

int main( int argc, char** argv ) {
  const uint8_t VALUES_RANGE{ 10 };
  const size_t MATRIX_SIZE{ 500 };
  const uint8_t N_SAMPLES{ 5 };

  std::printf( "* Initializing matrixes of size %d*%d with values in range [-%d,%d]*\n\n",
               MATRIX_SIZE, MATRIX_SIZE, VALUES_RANGE, VALUES_RANGE );

  matrix_int8_t A( MATRIX_SIZE * MATRIX_SIZE ), B( MATRIX_SIZE * MATRIX_SIZE );
  matrix_int32_t C( MATRIX_SIZE * MATRIX_SIZE );

  std::printf( "* Number of samples: %d\n\n", N_SAMPLES );

  /*
   * Calculate C = A * B using a single thread
   */
  std::printf( "* Calculating matrix product on a single thread *\n" );
  double single_thread_time{ 0 };

  for ( size_t i = 0; i < N_SAMPLES; i++ ) {
    std::printf( "* Running sample #%d\n", i + 1 );
    setup_matrixes( A, B, C, MATRIX_SIZE, VALUES_RANGE );

    auto t1 = std::chrono::high_resolution_clock::now();
    multiply( A, B, C, MATRIX_SIZE );
    auto t2 = std::chrono::high_resolution_clock::now();

    single_thread_time += std::chrono::duration<double, std::milli>( t2 - t1 ).count();
  }

  std::printf( "Avarage time needed (ms): %.2f\n\n", ( single_thread_time / N_SAMPLES ) );

  /*
   * Calculate C = A * B using OpenMP
   */

  std::printf( "* Calculating matrix product using OpenMP *\n" );
  double omp_time{ 0 };

  for ( size_t i = 0; i < N_SAMPLES; i++ ) {
    std::printf( "* Running sample #%d\n", i + 1 );
    setup_matrixes( A, B, C, MATRIX_SIZE, VALUES_RANGE );

    auto t1 = std::chrono::high_resolution_clock::now();
    multiply_omp( A, B, C, MATRIX_SIZE );
    auto t2 = std::chrono::high_resolution_clock::now();

    omp_time += std::chrono::duration<double, std::milli>( t2 - t1 ).count();
  }
  std::printf( "Time needed (ms): %.2f\n\n", ( omp_time / N_SAMPLES ) );

  std::printf( "Total speedup (%): %.2f\n\n", ( single_thread_time / omp_time * 100 ) );

  return 0;
}