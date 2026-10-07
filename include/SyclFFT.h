#ifndef __SYCLFFT_H__
#define __SYCLFFT_H__



//=========================================================================================================
// SyclFFT : SYCL-native, block-level, batched, in-place complex-to-complex FFT
//
// Replacement for cuFFTDx in the ELBDM Gram-Fourier extension (GramFE_FFT) GPU solver
// (see doc/SyclFFT/DESIGN.md)
//
// Algorithm   :  Mixed-radix (4, 2, 3, 5, 7) decimation-in-time Cooley-Tukey FFT
//                1. Digit-reversal permutation (gather into registers --> barrier --> scatter)
//                2. One in-place butterfly stage per radix factor, each followed by a work-group barrier
//
// Execution   :  All work-items of ONE work-group cooperatively transform "FFTsPerBlock" rows of
//                "N" complex values stored contiguously (row-major, row stride = N) in local memory
//                --> data[ FFTsPerBlock ][ N ]
//                The work-group must be launched with the local range given by "block_dim"
//
// Convention  :  Forward : X[k] = sum_n x[n] exp(-2*pi*i*n*k/N)
//                Inverse : x[n] = sum_k X[k] exp(+2*pi*i*n*k/N)
//                --> NEITHER direction is normalized (same as cuFFTDx and FFTW_FORWARD/FFTW_BACKWARD)
//                --> GramFE folds the 1/N normalization into ExpCoeff
//
// Note        :  1. Only the SYCL-specific glue (work-item query, group barrier, sycl::range conversion)
//                   is guarded by SYCL_LANGUAGE_VERSION. The algorithm itself is plain C++ and can be
//                   validated on the host through execute_impl() with any barrier functor
//                2. No long double/double arithmetic is performed at run time when T = float
//                   (constant tables are converted to T at compile time)
//                3. The GAMER-specific aliases (FFT, IFFT, complex_type, ...) at the end of this file
//                   are only defined for SYCL builds of the GramFE_FFT scheme (GRAMFE_FLU_NXT defined)
//=========================================================================================================



#include <complex>
#include <cmath>

#ifdef SYCL_LANGUAGE_VERSION
#  include <sycl/sycl.hpp>
#endif



namespace syclfft
{

enum class fft_direction { forward, inverse };



// workspace is empty since the transform is performed entirely in the caller-provided local memory
struct workspace {};



//-------------------------------------------------------------------------------------------------------
// Structure   :  dim3
// Description :  (x,y,z) work-group shape usable in constant expressions
//
// Note        :  1. x is the fastest-varying dimension, i.e., x <-> local_id(2) and y <-> local_id(1)
//                2. Converts implicitly to sycl::range<3>(z,y,x)
//-------------------------------------------------------------------------------------------------------
struct dim3
{
   unsigned int x, y, z;

#  ifdef SYCL_LANGUAGE_VERSION
   operator sycl::range<3>() const { return sycl::range<3>( z, y, x ); }
#  endif
}; // struct dim3



namespace detail
{

//-------------------------------------------------------------------------------------------------------
// Function    :  pick_radix
// Description :  Return the radix to be used for the next stage of a length-n transform
//                --> prefer radix 4, then 2, 3, 5, 7
//                --> return 0 if n contains an unsupported prime factor
//-------------------------------------------------------------------------------------------------------
constexpr unsigned int pick_radix( const unsigned int n )
{
   if ( n % 4 == 0 )   return 4;
   if ( n % 2 == 0 )   return 2;
   if ( n % 3 == 0 )   return 3;
   if ( n % 5 == 0 )   return 5;
   if ( n % 7 == 0 )   return 7;
   return 0;
}

constexpr bool is_supported_size( unsigned int n )
{
   if ( n < 2 )   return false;
   while ( n > 1 )
   {
      const unsigned int p = pick_radix( n );
      if ( p == 0 )   return false;
      n /= p;
   }
   return true;
}



// factorization N = radix[0] * radix[1] * ... * radix[NStage-1]
// span[s] = radix[0] * ... * radix[s-1] = length of the sub-transforms entering stage s
constexpr unsigned int MaxStage = 32;

struct plan_data
{
   unsigned int NStage;
   unsigned int radix[MaxStage];
   unsigned int span [MaxStage];
};

constexpr plan_data make_plan( const unsigned int N )
{
   plan_data plan {};
   unsigned int n = N, L = 1;

   while ( n > 1  &&  plan.NStage < MaxStage )
   {
      const unsigned int p = pick_radix( n );
      if ( p == 0 )   break;

      plan.radix[ plan.NStage ] = p;
      plan.span [ plan.NStage ] = L;
      plan.NStage ++;

      L *= p;
      n /= p;
   }

   return plan;
}




//-------------------------------------------------------------------------------------------------------
// Function    :  sincospi
// Description :  s = sin(pi*x), c = cos(pi*x)
//-------------------------------------------------------------------------------------------------------
template <typename T>
inline void sincospi( const T x, T &s, T &c )
{
#  ifdef SYCL_LANGUAGE_VERSION
   s = sycl::sinpi( x );
   c = sycl::cospi( x );
#  else
   const T a = T(3.14159265358979323846) * x;
   s = std::sin( a );
   c = std::cos( a );
#  endif
}



//-------------------------------------------------------------------------------------------------------
// Structure   :  roots
// Description :  cos(2*pi*j/P) and sin(2*pi*j/P) for j = [0 ... P-1] used by the odd-radix butterflies
//
// Note        :  Stored directly in precision T to avoid double/long double arithmetic in device code
//-------------------------------------------------------------------------------------------------------
template <unsigned int P, typename T> struct roots;

template <typename T> struct roots<3,T>
{
   static constexpr T c[3] = { T(1.0), T(-0.5), T(-0.5) };
   static constexpr T s[3] = { T(0.0), T(+0.866025403784438646763723170752936), T(-0.866025403784438646763723170752936) };
};

template <typename T> struct roots<5,T>
{
   static constexpr T c[5] = { T(1.0),
                               T(+0.309016994374947424102293417182819),
                               T(-0.809016994374947424102293417182819),
                               T(-0.809016994374947424102293417182819),
                               T(+0.309016994374947424102293417182819) };
   static constexpr T s[5] = { T(0.0),
                               T(+0.951056516295153572116439333379382),
                               T(+0.587785252292473129168705954639073),
                               T(-0.587785252292473129168705954639073),
                               T(-0.951056516295153572116439333379382) };
};

template <typename T> struct roots<7,T>
{
   static constexpr T c[7] = { T(1.0),
                               T(+0.623489801858733530525004884004240),
                               T(-0.222520933956314404288902564496795),
                               T(-0.900968867902419126236102319507445),
                               T(-0.900968867902419126236102319507445),
                               T(-0.222520933956314404288902564496795),
                               T(+0.623489801858733530525004884004240) };
   static constexpr T s[7] = { T(0.0),
                               T(+0.781831482468029808708444526674058),
                               T(+0.974927912181823607018131682993931),
                               T(+0.433883739117558120475768332848359),
                               T(-0.433883739117558120475768332848359),
                               T(-0.974927912181823607018131682993931),
                               T(-0.781831482468029808708444526674058) };
};



//-------------------------------------------------------------------------------------------------------
// Function    :  dft_small
// Description :  In-place length-P DFT on register arrays: X[t] = sum_q x[q] exp(Sign*2*pi*i*q*t/P)
//-------------------------------------------------------------------------------------------------------
template <unsigned int P, int Sign, typename T>
inline void dft_small( T (&re)[P], T (&im)[P] )
{
   if constexpr ( P == 2 )
   {
      const T r0 = re[0], i0 = im[0];
      re[0] = r0 + re[1];   im[0] = i0 + im[1];
      re[1] = r0 - re[1];   im[1] = i0 - im[1];
   }

   else if constexpr ( P == 4 )
   {
      const T t0r = re[0] + re[2],   t0i = im[0] + im[2];
      const T t1r = re[0] - re[2],   t1i = im[0] - im[2];
      const T t2r = re[1] + re[3],   t2i = im[1] + im[3];
      const T t3r = re[1] - re[3],   t3i = im[1] - im[3];

//    X1 = t1 + Sign*i*t3, X3 = t1 - Sign*i*t3
      re[0] = t0r + t2r;   im[0] = t0i + t2i;
      re[2] = t0r - t2r;   im[2] = t0i - t2i;
      if constexpr ( Sign < 0 )
      {
         re[1] = t1r + t3i;   im[1] = t1i - t3r;
         re[3] = t1r - t3i;   im[3] = t1i + t3r;
      }
      else
      {
         re[1] = t1r - t3i;   im[1] = t1i + t3r;
         re[3] = t1r + t3i;   im[3] = t1i - t3r;
      }
   }

   else
   {
      T out_re[P], out_im[P];

#     pragma unroll
      for (unsigned int t=0; t<P; t++)
      {
         T sr = re[0], si = im[0];

#        pragma unroll
         for (unsigned int q=1; q<P; q++)
         {
            const unsigned int j = (q*t) % P;
            const T c = roots<P,T>::c[j];
            const T s = ( Sign < 0 ) ? -roots<P,T>::s[j] : roots<P,T>::s[j];

            sr += re[q]*c - im[q]*s;
            si += re[q]*s + im[q]*c;
         }

         out_re[t] = sr;
         out_im[t] = si;
      }

#     pragma unroll
      for (unsigned int t=0; t<P; t++)
      {
         re[t] = out_re[t];
         im[t] = out_im[t];
      }
   }
} // FUNCTION : dft_small

} // namespace detail




#ifdef SYCL_LANGUAGE_VERSION
// allow "sycl::range<3>(1,1,NPatch) * FFT::block_dim" at the kernel launch site (found through ADL)
inline sycl::range<3> operator*( const sycl::range<3> &r, const dim3 &d ) { return r * sycl::range<3>(d); }
inline sycl::range<3> operator*( const dim3 &d, const sycl::range<3> &r ) { return sycl::range<3>(d) * r; }
#endif



//-------------------------------------------------------------------------------------------------------
// Structure   :  BlockFFT
// Description :  Batched in-place block FFT executed by one work-group (cuFFTDx-like interface)
//
// Note        :  1. block_dim = { ceil(N/ElementsPerThread), FFTsPerBlock, 1 }, as in cuFFTDx
//                   --> the GramFE kernel relies on block_dim.y == FFTsPerBlock
//                2. shared_memory_size is in BYTES (cuFFTDx convention) and only covers the FFT data
//                   itself (FFTsPerBlock*N complex values); callers storing extra data in the same
//                   local buffer must add it themselves
//
// Template    :  T                 : Floating-point type (float or double)
//                N                 : Transform size (must factor into 2, 3, 5 and 7)
//                Direction         : fft_direction::forward / fft_direction::inverse
//                FFTsPerBlock      : Number of rows (transforms) per work-group
//                ElementsPerThread : Number of elements per work-item per row
//-------------------------------------------------------------------------------------------------------
template <typename T, unsigned int N, fft_direction Direction, unsigned int FFTsPerBlock, unsigned int ElementsPerThread>
struct BlockFFT
{
   static_assert( detail::is_supported_size(N), "SyclFFT : N must be >= 2 and factor into 2, 3, 5 and 7 only !!" );
   static_assert( FFTsPerBlock > 0  &&  ElementsPerThread > 0, "SyclFFT : FFTsPerBlock and ElementsPerThread must be > 0 !!" );

// public interface (mirrors cuFFTDx)
   using value_type     = std::complex<T>;
   using workspace_type = workspace;

   static constexpr unsigned int  size                  = N;              // replaces cufftdx::size_of<FFT>::value
   static constexpr fft_direction direction             = Direction;
   static constexpr unsigned int  ffts_per_block        = FFTsPerBlock;
   static constexpr unsigned int  elements_per_thread   = ElementsPerThread;
   static constexpr dim3          block_dim             = { (N + ElementsPerThread - 1)/ElementsPerThread, FFTsPerBlock, 1 };
   static constexpr unsigned int  max_threads_per_block = block_dim.x*block_dim.y*block_dim.z;
   static constexpr unsigned int  storage_size          = FFTsPerBlock*N;                    // number of complex elements
   static constexpr unsigned int  shared_memory_size    = storage_size*sizeof(value_type);   // in bytes

#  ifdef SYCL_LANGUAGE_VERSION
   static sycl::range<3> block_range() { return sycl::range<3>( block_dim ); }
#  endif


// internal parameters
   static constexpr int               Sign    = ( Direction == fft_direction::forward ) ? -1 : +1;
   static constexpr detail::plan_data Plan    = detail::make_plan( N );
   static constexpr unsigned int      NThread = max_threads_per_block;
   static constexpr unsigned int      NReg    = (storage_size + NThread - 1)/NThread;  // elements per work-item in the permutation



//-------------------------------------------------------------------------------------------------------
// Function    :  digit_reverse
// Description :  Return the input index whose value must be stored at position "pos" before the
//                decimation-in-time stages
//
// Note        :  pos = d_0 + r_0*( d_1 + r_1*( d_2 + ... ) )  -->  n = d_{m-1} + r_{m-1}*( d_{m-2} + ... + r_1*d_0 )
//-------------------------------------------------------------------------------------------------------
   static inline unsigned int digit_reverse( unsigned int pos )
   {
      unsigned int n = 0;

      for (unsigned int s=0; s<Plan.NStage; s++)
      {
         const unsigned int p = Plan.radix[s];
         n   = n*p + pos%p;
         pos = pos/p;
      }

      return n;
   } // FUNCTION : digit_reverse



//-------------------------------------------------------------------------------------------------------
// Function    :  stage
// Description :  In-place butterfly stage S
//                --> combine radix[S] consecutive sub-transforms of length span[S] into one sub-transform
//                    of length span[S]*radix[S]
//                --> every butterfly reads and writes its own disjoint index set, so no intra-stage
//                    synchronization is needed
//-------------------------------------------------------------------------------------------------------
   template <unsigned int S>
   static inline void stage( T *data, const unsigned int tid )
   {
      constexpr unsigned int P     = Plan.radix[S];
      constexpr unsigned int Ls    = Plan.span [S];
      constexpr unsigned int L     = Ls*P;
      constexpr unsigned int NBfly = N/P;                  // butterflies per row
      constexpr unsigned int NTask = FFTsPerBlock*NBfly;   // butterflies per work-group

      for (unsigned int t=tid; t<NTask; t+=NThread)
      {
         const unsigned int row = t / NBfly;
         const unsigned int b   = t % NBfly;
         const unsigned int g   = b / Ls;                  // sub-transform group within the row
         const unsigned int k   = b % Ls;                  // frequency index within the sub-transforms

         T *base = data + 2*( row*N + g*L + k );
         T re[P], im[P];

#        pragma unroll
         for (unsigned int q=0; q<P; q++)
         {
            re[q] = base[ 2*q*Ls     ];
            im[q] = base[ 2*q*Ls + 1 ];
         }

//       twiddle factors exp(Sign*2*pi*i*q*k/L) (none in the first stage since Ls = 1 --> k = 0)
         if constexpr ( Ls > 1 )
         {
            if ( k != 0 )
            {
#              pragma unroll
               for (unsigned int q=1; q<P; q++)
               {
                  T s, c;
                  detail::sincospi( T(2*q*k)/T(L), s, c );
                  if constexpr ( Sign < 0 )   s = -s;

                  const T r = re[q];
                  re[q] = r*c - im[q]*s;
                  im[q] = r*s + im[q]*c;
               }
            }
         }

         detail::dft_small<P, Sign>( re, im );

#        pragma unroll
         for (unsigned int q=0; q<P; q++)
         {
            base[ 2*q*Ls     ] = re[q];
            base[ 2*q*Ls + 1 ] = im[q];
         }
      }
   } // FUNCTION : stage



//-------------------------------------------------------------------------------------------------------
// Function    :  run_stages
// Description :  Apply stages [S ... NStage-1], each followed by a barrier
//-------------------------------------------------------------------------------------------------------
   template <unsigned int S, typename Barrier>
   static inline void run_stages( T *data, const unsigned int tid, Barrier &barrier )
   {
      if constexpr ( S < Plan.NStage )
      {
         stage<S>( data, tid );
         barrier();
         run_stages<S+1>( data, tid, barrier );
      }
   } // FUNCTION : run_stages



//-------------------------------------------------------------------------------------------------------
// Function    :  execute_impl
// Description :  Backend-agnostic body of execute()
//
// Note        :  1. Must be called by all NThread work-items of the group with distinct tid in [0, NThread)
//                2. "barrier()" must synchronize all these work-items and make local-memory writes visible
//                3. Ends with a barrier, so results can be read by any work-item on return
//
// Parameter   :  data_void : Pointer to value_type[FFTsPerBlock][N]
//                tid       : Linear work-item index within the work-group
//                barrier   : Work-group barrier functor
//-------------------------------------------------------------------------------------------------------
   template <typename Barrier>
   static inline void execute_impl( void *data_void, const unsigned int tid, Barrier barrier )
   {
//    std::complex<T> is guaranteed to be layout-compatible with T[2]
      T *data = reinterpret_cast<T*>( data_void );

//    1. digit-reversal permutation (gather --> barrier --> scatter)
      T re[NReg], im[NReg];

#     pragma unroll
      for (unsigned int j=0; j<NReg; j++)
      {
         const unsigned int e = tid + j*NThread;
         if ( e < storage_size )
         {
            const unsigned int row = e / N;
            const unsigned int src = row*N + digit_reverse( e % N );
            re[j] = data[ 2*src     ];
            im[j] = data[ 2*src + 1 ];
         }
      }

      barrier();

#     pragma unroll
      for (unsigned int j=0; j<NReg; j++)
      {
         const unsigned int e = tid + j*NThread;
         if ( e < storage_size )
         {
            data[ 2*e     ] = re[j];
            data[ 2*e + 1 ] = im[j];
         }
      }

      barrier();

//    2. butterfly stages
      run_stages<0>( data, tid, barrier );
   } // FUNCTION : execute_impl



#  ifdef SYCL_LANGUAGE_VERSION
//-------------------------------------------------------------------------------------------------------
// Function    :  execute
// Description :  Transform FFTsPerBlock rows of N complex values in local memory (in place)
//
// Note        :  1. Must be called by every work-item of a work-group launched with local range block_dim
//                2. Contains work-group barriers --> must not be called in divergent control flow
//
// Parameter   :  data  : Pointer to value_type[FFTsPerBlock][N] in local memory
//                (2nd) : Workspace (unused, kept for call-site compatibility)
//-------------------------------------------------------------------------------------------------------
   inline void execute( void *data, const workspace_type & = workspace_type() ) const
   {
      const auto item  = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
      const auto group = item.get_group();

      execute_impl( data, (unsigned int)item.get_local_linear_id(), [&group]() { sycl::group_barrier( group ); } );
   } // FUNCTION : execute
#  endif // #ifdef SYCL_LANGUAGE_VERSION

}; // struct BlockFFT

} // namespace syclfft



//=========================================================================================================
// GAMER GramFE_FFT aliases (replace the former cuFFTDx typedefs in FLU.h)
//=========================================================================================================
#if ( defined(SYCL_LANGUAGE_VERSION)  &&  defined(GRAMFE_FLU_NXT) )

#  ifndef GRAMFE_CUSTOM_ELEMENTS_PER_THREAD
#  define GRAMFE_CUSTOM_ELEMENTS_PER_THREAD  4
#  endif
#  ifndef GRAMFE_CUSTOM_FFTS_PER_BLOCK
#  define GRAMFE_CUSTOM_FFTS_PER_BLOCK       12
#  endif

static constexpr unsigned int elements_per_thread = GRAMFE_CUSTOM_ELEMENTS_PER_THREAD;
static constexpr unsigned int ffts_per_block      = GRAMFE_CUSTOM_FFTS_PER_BLOCK;

using FFT          = syclfft::BlockFFT< gramfe_fft_float, GRAMFE_FLU_NXT, syclfft::fft_direction::forward,
                                        ffts_per_block, elements_per_thread >;
using IFFT         = syclfft::BlockFFT< gramfe_fft_float, GRAMFE_FLU_NXT, syclfft::fft_direction::inverse,
                                        ffts_per_block, elements_per_thread >;

using complex_type = typename FFT::value_type;

#endif // #if ( defined(SYCL_LANGUAGE_VERSION)  &&  defined(GRAMFE_FLU_NXT) )



#endif // #ifndef __SYCLFFT_H__
