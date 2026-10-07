#include <sys/resource.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include "omp.h"

using matrix_int8_t = std::vector<int8_t>;
using matrix_int32_t = std::vector<int32_t>;

struct Config {
    static constexpr uint8_t VALUES_RANGE{ 10 };
    static constexpr size_t MATRIX_SIZE{ 2000 };
    static constexpr uint8_t N_SAMPLES{ 5 };
};

void setup_matrixes( matrix_int8_t& A, matrix_int8_t& B, matrix_int32_t& C ) {
  std::random_device rd;
  std::mt19937 gen( rd() );
  std::bernoulli_distribution dist( 0.5 );

#pragma omp parallel for
  for ( size_t i = 0; i < Config::MATRIX_SIZE; i++ ) {
    for ( size_t j = 0; j < Config::MATRIX_SIZE; j++ ) {
      A[ i * Config::MATRIX_SIZE + j ] =
          ( dist( gen ) ? 1 : -1 ) * ( std::rand() % Config::VALUES_RANGE );
      B[ i * Config::MATRIX_SIZE + j ] =
          ( dist( gen ) ? 1 : -1 ) * ( std::rand() % Config::VALUES_RANGE );
      C[ i * Config::MATRIX_SIZE + j ] = 0;
    }
  }
}

void multiply( const matrix_int8_t& A, const matrix_int8_t& B, matrix_int32_t& C ) {
  for ( size_t i = 0; i < Config::MATRIX_SIZE; i++ ) {
    for ( size_t k = 0; k < Config::MATRIX_SIZE; k++ ) {
      int32_t a_ik = A[ i * Config::MATRIX_SIZE + k ];
      for ( size_t j = 0; j < Config::MATRIX_SIZE; j++ ) {
        C[ i * Config::MATRIX_SIZE + j ] += a_ik * B[ k * Config::MATRIX_SIZE + j ];
      }
    }
  }
}

void multiply_omp( const matrix_int8_t& A, const matrix_int8_t& B, matrix_int32_t& C ) {
#pragma omp parallel for
  for ( size_t i = 0; i < Config::MATRIX_SIZE; i++ ) {
    for ( size_t k = 0; k < Config::MATRIX_SIZE; k++ ) {
      int32_t a_ik = A[ i * Config::MATRIX_SIZE + k ];
      for ( size_t j = 0; j < Config::MATRIX_SIZE; j++ ) {
        C[ i * Config::MATRIX_SIZE + j ] += a_ik * B[ k * Config::MATRIX_SIZE + j ];
      }
    }
  }
}

int main( int argc, char** argv ) {
  std::printf( "* Initializing matrixes of size %d*%d with values in range [-%d,%d]\n\n",
               Config::MATRIX_SIZE, Config::MATRIX_SIZE, Config::VALUES_RANGE,
               Config::VALUES_RANGE );

  matrix_int8_t A( Config::MATRIX_SIZE * Config::MATRIX_SIZE ),
      B( Config::MATRIX_SIZE * Config::MATRIX_SIZE );
  matrix_int32_t C( Config::MATRIX_SIZE * Config::MATRIX_SIZE );

  std::printf( "* Number of samples: %d\n\n", Config::N_SAMPLES );

  /*
   * Calculate C = A * B using a single thread
   */
  std::printf( "* Calculating matrix product on a single thread\n" );
  double single_thread_time{ 0 };

  for ( size_t i = 0; i < Config::N_SAMPLES; i++ ) {
    std::printf( "* Running sample #%d\n", i + 1 );
    setup_matrixes( A, B, C );

    auto t1 = std::chrono::high_resolution_clock::now();
    multiply( A, B, C );
    auto t2 = std::chrono::high_resolution_clock::now();

    single_thread_time += std::chrono::duration<double, std::milli>( t2 - t1 ).count();
  }

  std::printf( "Avarage time needed (ms): %.2f\n\n", ( single_thread_time / Config::N_SAMPLES ) );

  /*
   * Calculate C = A * B using OpenMP
   */

  std::printf( "* Calculating matrix product using OpenMP\n" );
  double omp_time{ 0 };

  for ( size_t i = 0; i < Config::N_SAMPLES; i++ ) {
    std::printf( "* Running sample #%d\n", i + 1 );
    setup_matrixes( A, B, C );

    auto t1 = std::chrono::high_resolution_clock::now();
    multiply_omp( A, B, C );
    auto t2 = std::chrono::high_resolution_clock::now();

    omp_time += std::chrono::duration<double, std::milli>( t2 - t1 ).count();
  }

  std::printf( "Avarage time needed (ms): %.2f\n\n", ( omp_time / Config::N_SAMPLES ) );

  std::printf( "Total speedup (%): %.2f\n\n", ( single_thread_time / omp_time * 100 ) );

  return 0;
}
