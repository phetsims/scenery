// Copyright 2023, University of Colorado Boulder

import Snippet from './Snippet.js';

const padLeft = ( input, padding, length ) => {
  let result = input;

  const padLength = length - input.length;
  for ( let i = 0; i < padLength; i++ ) {
    result = padding + result;
  }

  return result;
};

const toU32Hex = n => {
  return padLeft( n.toString( 16 ), '0', 8 );
};
const toU32Binary = n => {
  return padLeft( n.toString( 2 ), '0', 32 );
};

const logInputOutput = ( inputArrayBuffer, outputArrayBuffer ) => {
  const inputInt32Array = new Int32Array( inputArrayBuffer );
  const inputUint32Array = new Uint32Array( inputArrayBuffer );
  const outputUInt32Array = new Uint32Array( outputArrayBuffer );
  const outputInt32Array = new Int32Array( outputArrayBuffer );

  console.log( 'in (s)', [ ...inputInt32Array ].join( ', ' ) );
  console.log( 'ou (s)', [ ...outputInt32Array ].join( ', ' ) );
  console.log( 'in (u)', [ ...inputUint32Array ].join( ', ' ) );
  console.log( 'ou (u)', [ ...outputUInt32Array ].join( ', ' ) );
  console.log( 'in', [ ...inputUint32Array ].map( toU32Hex ).join( ', ' ) );
  console.log( 'ou', [ ...outputUInt32Array ].map( toU32Hex ).join( ', ' ) );
  console.log( 'in', [ ...inputUint32Array ].map( toU32Binary ).join( ', ' ) );
  console.log( 'ou', [ ...outputUInt32Array ].map( toU32Binary ).join( ', ' ) );
};

// TODO: look into overflow detection (since we want certain operations to not overflow)

const n16 = ( 2n ** 16n );
const n32 = ( 2n ** 32n );
const n64 = ( 2n ** 64n );
// const n16Mask = n16 - 1n;
const n32Mask = n32 - 1n;
// const n64Mask = n64 - 1n;

const nToU32s = n => {
  return [ Number( n & n32Mask ), Number( ( n >> 32n ) & n32Mask ) ];
};
window.nToU32s = nToU32s;

// u64, backed by two u32 (little-endian)
const u64Snippet = new Snippet( 'alias u64 = vec2<u32>;' );

// i64, backed by two u32 (little-endian)
const i64Snippet = new Snippet( 'alias i64 = vec2<u32>;' );

// rational signed number, backed by four u32 (little-endian, signed numerator then unsigned divisor)
const q128Snippet = new Snippet( 'alias q128 = vec4<u32>;' );

// TODO use this
const ZERO_u64Snippet = new Snippet( `
const ZERO_u64 = vec2( 0u, 0u );
` );

// TODO use this
const ONE_u64Snippet = new Snippet( `
const ONE_u64 = vec2( 1u, 0u );
` );

const ZERO_q128Snippet = new Snippet( `
const ZERO_q128 = vec4( 0u, 0u, 1u, 0u );
` );

const ONE_q128Snippet = new Snippet( `
const ONE_q128 = vec4( 1u, 0u, 1u, 0u );
` );

const u32_to_u64Snippet = new Snippet( `
fn u32_to_u64( x: u32 ) -> u64 {
  return vec2<u32>( x, 0u );
}
`, [ u64Snippet ] );

const i32_to_i64Snippet = new Snippet( `
fn i32_to_i64( x: i32 ) -> i64 {
  return vec2<u32>( u32( x ), select( 0u, 0xffffffffu, x < 0i ) );
}
`, [ i64Snippet ] );

const add_u32_u32_to_u64Snippet = new Snippet( `
fn add_u32_u32_to_u64( a: u32, b: u32 ) -> u64 {
  let sum = a + b;
  return vec2( sum, select( 0u, 1u, sum < a ) );
}
`, [ u64Snippet ] );

// TODO: check all adds used internally, to see if we are overflowing things

// ( a_low + a_high * 2^16 ) * ( b_low + b_high * 2^16 )
// ( a_low * b_low ) + ( a_low * b_high + a_high * b_low ) * 2^16 + ( a_high * b_high ) * 2^32
const mul_u32_u32_to_u64Snippet = new Snippet( `
fn mul_u32_u32_to_u64( a: u32, b: u32 ) -> u64 {
  let a_low = a & 0xffffu;
  let a_high = a >> 16u;
  let b_low = b & 0xffffu;
  let b_high = b >> 16u;
  let c_low = a_low * b_low;
  let c_mid = a_low * b_high + a_high * b_low;
  let c_high = a_high * b_high;
  let low = add_u32_u32_to_u64( c_low, c_mid << 16u );
  let high = vec2( 0u, ( c_mid >> 16u ) + c_high );
  return low + high;
}
`, [ u64Snippet, add_u32_u32_to_u64Snippet ] );

// ( a_low + a_high * 2^32 ) + ( b_low + b_high * 2^32 ) mod 2^64
// a_low + b_low + ( a_high + b_high ) * 2^32 mod 2^64
const add_u64_u64Snippet = new Snippet( `
fn add_u64_u64( a: u64, b: u64 ) -> u64 {
  return add_u32_u32_to_u64( a.x, b.x ) + vec2( 0u, add_u32_u32_to_u64( a.y, b.y ).x );
}
`, [ u64Snippet, add_u32_u32_to_u64Snippet ] );

const add_i64_i64Snippet = new Snippet( `
fn add_i64_i64( a: i64, b: i64 ) -> i64 {
  return add_u64_u64( a, b );
}
`, [ i64Snippet, add_u64_u64Snippet ] );

const negate_i64Snippet = new Snippet( `
fn negate_i64( a: i64 ) -> i64 {
  return add_u64_u64( ~a, vec2( 1u, 0u ) );
}
`, [ i64Snippet, add_u64_u64Snippet ] );

const is_zero_u64Snippet = new Snippet( `
fn is_zero_u64( a: u64 ) -> bool {
  return a.x == 0u && a.y == 0u;
}
`, [ u64Snippet ] );

const is_negative_i64Snippet = new Snippet( `
fn is_negative_i64( a: i64 ) -> bool {
  return ( a.y >> 31u ) == 1u;
}
`, [ i64Snippet ] );

const abs_i64Snippet = new Snippet( `
fn abs_i64( a: i64 ) -> i64 {
  return select( a, negate_i64( a ), is_negative_i64( a ) );
}
`, [ i64Snippet, negate_i64Snippet, is_negative_i64Snippet ] );

const left_shift_u64Snippet = new Snippet( `
fn left_shift_u64( a: u64, b: u32 ) -> u64 {
  if ( b == 0u ) {
    return a;
  }
  else if ( b < 32u ) {
    return vec2( a.x << b, ( a.y << b ) | ( a.x >> ( 32u - b ) ) );
  }
  else {
    return vec2( 0u, a.x << ( b - 32u ) );
  }
}
`, [ u64Snippet ] );

const right_shift_u64Snippet = new Snippet( `
fn right_shift_u64( a: u64, b: u32 ) -> u64 {
  if ( b == 0u ) {
    return a;
  }
  else if ( b < 32u ) {
    return vec2( ( a.x >> b ) | ( a.y << ( 32u - b ) ), a.y >> b );
  }
  else {
    return vec2( a.y >> ( b - 32u ), 0u );
  }
}
`, [ u64Snippet ] );

// TODO: signed right shift?

// NOTE: ASSUMES NONZERO
const first_leading_bit_u64Snippet = new Snippet( `
fn first_leading_bit_u64( a: u64 ) -> u32 {
  if ( a.y != 0u ) {
    return firstLeadingBit( a.y ) + 32u;
  }
  else {
    return firstLeadingBit( a.x );
  }
}
`, [ u64Snippet ] );

// NOTE: ASSUMES NONZERO
const first_trailing_bit_u64Snippet = new Snippet( `
fn first_trailing_bit_u64( a: u64 ) -> u32 {
  if ( a.x != 0u ) {
    return firstTrailingBit( a.x );
  }
  else {
    return firstTrailingBit( a.y ) + 32u;
  }
}
`, [ u64Snippet ] );

const subtract_i64_i64Snippet = new Snippet( `
fn subtract_i64_i64( a: i64, b: i64 ) -> i64 {
  return add_i64_i64( a, negate_i64( b ) );
}
`, [ i64Snippet, add_i64_i64Snippet, negate_i64Snippet ] );

const cmp_u64_u64Snippet = new Snippet( `
fn cmp_u64_u64( a: u64, b: u64 ) -> i32 {
  if ( a.y < b.y ) {
    return -1i;
  }
  else if ( a.y > b.y ) {
    return 1i;
  }
  else {
    if ( a.x < b.x ) {
      return -1i;
    }
    else if ( a.x > b.x ) {
      return 1i;
    }
    else {
      return 0i;
    }  
  }
}
`, [ u64Snippet ] );

const cmp_i64_i64Snippet = new Snippet( `
fn cmp_i64_i64( a: i64, b: i64 ) -> i32 {
  let diff = subtract_i64_i64( a, b );
  if ( is_zero_u64( diff ) ) {
    return 0i;
  }
  else {
    return select( 1i, -1i, is_negative_i64( diff ) );
  }
}
`, [ i64Snippet, subtract_i64_i64Snippet, is_negative_i64Snippet, is_zero_u64Snippet ] );

// ( a_low + a_high * 2^32 ) * ( b_low + b_high * 2^32 ) mod 2^64
// = a_low * b_low + a_low * b_high * 2^32 + a_high * b_low * 2^32 + a_high * b_high * 2^64 mod 2^64
// = a_low * b_low + ( a_low * b_high + a_high * b_low ) * 2^32 mod 2^64
const mul_u64_u64Snippet = new Snippet( `
fn mul_u64_u64( a: u64, b: u64 ) -> u64 {
  let low = mul_u32_u32_to_u64( a.x, b.x );
  let mid0 = vec2( 0u, mul_u32_u32_to_u64( a.x, b.y ).x );
  let mid1 = vec2( 0u, mul_u32_u32_to_u64( a.y, b.x ).x );
  return add_u64_u64( add_u64_u64( low, mid0 ), mid1 );
}
`, [ u64Snippet, mul_u32_u32_to_u64Snippet, add_u64_u64Snippet ] );

const mul_i64_i64Snippet = new Snippet( `
fn mul_i64_i64( a: i64, b: i64 ) -> i64 {
  var result = mul_u64_u64( abs_i64( a ), abs_i64( b ) );
  result.y &= 0x7fffffffu; // remove the sign bit
  if ( is_negative_i64( a ) != is_negative_i64( b ) ) {
    return negate_i64( result );
  }
  else {
    return result;
  }
}
`, [ i64Snippet, mul_u64_u64Snippet, abs_i64Snippet, is_negative_i64Snippet, negate_i64Snippet ] );

// TODO: we can ignore division with https://en.wikipedia.org/wiki/Binary_GCD_algorithm perhaps?

// Packed quotient, remainder
// See https://stackoverflow.com/questions/18448343/divdi3-division-used-for-long-long-by-gcc-on-x86
// and https://stackoverflow.com/questions/11548070/x86-64-big-integer-representation/18202791#18202791
// TODO: eeek, will this work, we're using our signed subtraction on unsigned where we guarantee the top bit
// TODO: could optimize the left shift
// TODO: omg, are we going to overflow?
const div_u64_u64Snippet = new Snippet( `
fn div_u64_u64( a: u64, b: u64 ) -> vec4<u32> {
  if ( is_zero_u64( a ) ) {
    return vec4( 0u, 0u, 0u, 0u );
  }
  else if ( is_zero_u64( b ) ) {
    // TODO: HOW to better complain loudly? OR do we just not check, because we should have checked before?
    return vec4( 0u, 0u, 0u, 0u );
  }
  var result = vec2( 0u, 0u );
  var remainder = a;

  let high_bit = min( first_leading_bit_u64( a ), first_leading_bit_u64( b ) );
  var count = 63u - high_bit;
  var divisor = left_shift_u64( b, count );
  
  while( !is_zero_u64( remainder ) ) {
    if ( cmp_u64_u64( remainder, divisor ) >= 0i ) {
      remainder = subtract_i64_i64( remainder, divisor );
      result = result | left_shift_u64( vec2( 1u, 0u ), count );
    }
    if ( count == 0u ) {
      break;
    }
    divisor = right_shift_u64( divisor, 1u );
    count -= 1u;
  }
  
  return vec4( result, remainder );
}
`, [ u64Snippet, first_leading_bit_u64Snippet, left_shift_u64Snippet, is_zero_u64Snippet, cmp_u64_u64Snippet, subtract_i64_i64Snippet, right_shift_u64Snippet ] );

// binary GCD
const gcd_u64_u64Snippet = new Snippet( `
fn gcd_u64_u64( a: u64, b: u64 ) -> u64 {
  if ( is_zero_u64( a ) ) {
    return b;
  }
  else if ( is_zero_u64( b ) ) {
    return a;
  }
  
  let gcd_two = first_trailing_bit_u64( a | b );
  
  var u = right_shift_u64( a, gcd_two );
  var v = right_shift_u64( b, gcd_two );
  
  while ( u.x != v.x || u.y != v.y ) {
    if ( cmp_u64_u64( u, v ) == -1i ) {
      let t = u;
      u = v;
      v = t;
    }
    
    u = subtract_i64_i64( u, v );
    u = right_shift_u64( u, first_trailing_bit_u64( u ) );
  }
  
  return left_shift_u64( u, gcd_two );
}
`, [ u64Snippet, is_zero_u64Snippet, first_trailing_bit_u64Snippet, left_shift_u64Snippet, cmp_u64_u64Snippet, subtract_i64_i64Snippet, right_shift_u64Snippet ] );

const i64_to_q128Snippet = new Snippet( `
fn i64_to_q128( numerator: i64, denominator: i64 ) -> q128 {
  if ( is_negative_i64( denominator ) ) {
    return vec4( negate_i64( numerator ), negate_i64( denominator ) );
  }
  else {
    return vec4( numerator, denominator );
  }
}
`, [ q128Snippet, i64Snippet, is_negative_i64Snippet, negate_i64Snippet ] );

const whole_i64_to_q128Snippet = new Snippet( `
fn whole_i64_to_q128( numerator: i64 ) -> q128 {
  return vec4( numerator, 1u, 0u );
}
`, [ q128Snippet, i64Snippet ] );

// TODO: test
// Check fraction equality with cross-multiply (if we have the bits to spare to avoid reduction... reduction would also
// work).
const equals_cross_mul_q128Snippet = new Snippet( `
fn equals_cross_mul_q128( a: q128, b: q128 ) -> bool {
  return is_zero_u64( subtract_i64_i64( mul_i64_i64( a.xy, b.zw ), mul_i64_i64( a.zw, b.xy ) ) );
}
`, [ q128Snippet, is_zero_u64Snippet, subtract_i64_i64Snippet, mul_i64_i64Snippet ] );

const is_zero_q128Snippet = new Snippet( `
fn is_zero_q128( a: q128 ) -> bool {
  return a.x == 0u && a.y == 0u;
}
`, [ q128Snippet ] );

// 2i means totally internal (0<q<1), 1i means on an endpoint (q=0 or q=1), 0i means totally external (q<0 or q>1)
const ratio_test_q128Snippet = new Snippet( `
fn ratio_test_q128( q: q128 ) -> i32 {
  return cmp_i64_i64( q.xy, vec2( 0u, 0u ) ) + cmp_i64_i64( q.zw, q.xy );
}
`, [ q128Snippet, cmp_i64_i64Snippet ] );

const reduce_q128Snippet = new Snippet( `
fn reduce_q128( a: q128 ) -> q128 {
  let numerator = a.xy;
  let denominator = a.zw;
  if ( numerator.x == 0u && numerator.y == 0u ) {
    return vec4( 0u, 0u, 1u, 0u ); // 0/1
  }
  else if ( denominator.x == 1 && denominator.y == 0u ) {
    return a; // we're already reduced, x/1
  }
  let abs_numerator = abs_i64( numerator );
  let gcd = gcd_u64_u64( abs_numerator, denominator );
  if ( gcd.x == 1u && gcd.y == 0u ) {
    return a;
  }
  else {
    let reduced_numerator = div_u64_u64( abs_numerator, gcd ).xy;
    let reduced_denominator = div_u64_u64( denominator, gcd ).xy;
    if ( is_negative_i64( numerator ) ) {
      return vec4( negate_i64( reduced_numerator ), reduced_denominator );
    }
    else {
      return vec4( reduced_numerator, reduced_denominator );
    }
  }
}
`, [ q128Snippet, gcd_u64_u64Snippet, abs_i64Snippet, div_u64_u64Snippet, is_negative_i64Snippet, negate_i64Snippet ] );

const intersectionPointSnippet = new Snippet( `
struct IntersectionPoint {
  t0: q128,
  t1: q128,
  px: q128,
  py: q128
}
`, [ q128Snippet ] );

const line_segment_intersectionSnippet = new Snippet( `
struct LineSegmentIntersection {
  num_points: u32, // can include overlap points
  p0: IntersectionPoint,
  p1: IntersectionPoint
}
`, [ intersectionPointSnippet ] );

// TODO: isolate constants
// NOTE: Should handle zero-length segments fine (will report no intersections, denominator will be 0)
const intersect_line_segmentsSnippet = new Snippet( `
const not_rational = vec4( 0u, 0u, 0u, 0u );
const not_point = IntersectionPoint( not_rational, not_rational, not_rational, not_rational );
const not_intersection = LineSegmentIntersection( 0u, not_point, not_point );

fn intersect_line_segments( p0: vec2i, p1: vec2i, p2: vec2i, p3: vec2i ) -> LineSegmentIntersection {
  let p0x = i32_to_i64( p0.x );
  let p0y = i32_to_i64( p0.y );
  let p1x = i32_to_i64( p1.x );
  let p1y = i32_to_i64( p1.y );
  let p2x = i32_to_i64( p2.x );
  let p2y = i32_to_i64( p2.y );
  let p3x = i32_to_i64( p3.x );
  let p3y = i32_to_i64( p3.y );
  
  let d0x = subtract_i64_i64( p1x, p0x );
  let d0y = subtract_i64_i64( p1y, p0y );
  let d1x = subtract_i64_i64( p3x, p2x );
  let d1y = subtract_i64_i64( p3y, p2y );
  
  let cdx = subtract_i64_i64( p2x, p0x );
  let cdy = subtract_i64_i64( p2y, p0y );

  let denominator = subtract_i64_i64( mul_i64_i64( d0x, d1y ), mul_i64_i64( d0y, d1x ) );
  
  if ( is_zero_u64( denominator ) ) {
    // such that p0 + t * ( p1 - p0 ) = p2 + ( a * t + b ) * ( p3 - p2 )
    // an equivalency between lines
    var a: q128;
    var b: q128;

    let d1x_zero = is_zero_u64( d1x );
    let d1y_zero = is_zero_u64( d1y );
    
    // if ( d0s === 0 || d1s === 0 ) {
    //   return NO_OVERLAP;
    // }
    //
    // a = d0s / d1s;
    // b = ( p0s - p2s ) / d1s;
    
    // TODO: can we reduce the branching here?
    // Find a dimension where our line is not degenerate (e.g. covers multiple values in that dimension)
    // Compute line equivalency there
    if ( d1x_zero && d1y_zero ) {
      // DEGENERATE case for second line, it's a point, bail out
      return not_intersection;
    }
    else if ( d1x_zero ) {
      // if d1x is zero AND our denominator is zero, that means d0x or d1y must be zero. We checked d1y above, so d0x must be zero
      if ( p0.x != p2.x ) {
        // vertical lines, BUT not same x, so no intersection
        return not_intersection;
      }
      a = i64_to_q128( d0y, d1y );
      b = i64_to_q128( negate_i64( cdy ), d1y );
    }
    else if ( d1y_zero ) {
      // if d1y is zero AND our denominator is zero, that means d0y or d1x must be zero. We checked d1x above, so d0y must be zero
      if ( p0.y != p2.y ) {
        // horizontal lines, BUT not same y, so no intersection
        return not_intersection;
      }
      a = i64_to_q128( d0x, d1x );
      b = i64_to_q128( negate_i64( cdx ), d1x );
    }
    else {
      // we have non-axis-aligned second line, use that to compute a,b for each dimension, and we're the same "line"
      // iff those are consistent  
      if ( is_zero_u64( d0x ) && is_zero_u64( d0y ) ) {
        // DEGENERATE first line, it's a point, bail out
        return not_intersection;
      }
      let ax = i64_to_q128( d0x, d1x );
      let ay = i64_to_q128( d0y, d1y );
      if ( !equals_cross_mul_q128( ax, ay ) ) {
        return not_intersection;
      }
      let bx = i64_to_q128( negate_i64( cdx ), d1x );
      let by = i64_to_q128( negate_i64( cdy ), d1y );
      if ( !equals_cross_mul_q128( bx, by ) ) {
        return not_intersection;
      }
      
      // Pick the one with a non-zero a, so it is invertible
      if ( is_zero_q128( ax ) ) {
        a = ay;
        b = by;
      }
      else {
        a = ax;
        b = bx;
      }
    }
    
    var points: u32 = 0u;
    var results = array<IntersectionPoint, 2u>( not_point, not_point );
    
    // p0 + t * ( p1 - p0 ) = p2 + ( a * t + b ) * ( p3 - p2 )
    // i.e. line0( t ) = line1( a * t + b )
    // replacements for endpoints:
    // t=0       =>  t0=0,        t1=b
    // t=1       =>  t0=1,        t1=a+b
    // t=-b/a    =>  t0=-b/a,     t1=0
    // t=(1-b)/a =>  t0=(1-b)/a,  t1=1
    
    // NOTE: cases become identical if b=0, b=1, b=-a, b=1-a, HOWEVER these would not be internal, so they would be
    // excluded, and we can ignore them
    
    // t0=0, t1=b, p0
    let case1t1 = b;
    if ( ratio_test_q128( case1t1 ) == 2i ) {
      let p = IntersectionPoint( ZERO_q128, reduce_q128( case1t1 ), whole_i64_to_q128( p0x ), whole_i64_to_q128( p0y ) );
      results[ points ] = p;
      points += 1u;
    }
    
    // t0=1, t1=a+b, p1
    let case2t1 = vec4( add_i64_i64( a.xy, b.xy ), a.zw ); // abuse a,b having same denominator
    if ( ratio_test_q128( case2t1 ) == 2i ) {
      let p = IntersectionPoint( ONE_q128, reduce_q128( case2t1 ), whole_i64_to_q128( p1x ), whole_i64_to_q128( p1y ) );
      results[ points ] = p;
      points += 1u;
    }
    
    // t0=-b/a, t1=0, p2
    let case3t0 = i64_to_q128( negate_i64( b.xy ), a.xy ); // abuse a,b having same denominator
    if ( ratio_test_q128( case3t0 ) == 2i ) {
      let p = IntersectionPoint( reduce_q128( case3t0 ), ZERO_q128, whole_i64_to_q128( p2x ), whole_i64_to_q128( p2y ) );
      results[ points ] = p;
      points += 1u;
    }
    
    // t0=(1-b)/a, t1=1, p3
    // ( 1 - b ) / a = ( denom - b_numer ) / denom / ( a_numer / denom ) = ( denom - b_numer ) / a_numer
    let case4t0 = i64_to_q128( subtract_i64_i64( a.zw, b.xy ), a.xy );
    if ( ratio_test_q128( case4t0 ) == 2i ) {
      let p = IntersectionPoint( reduce_q128( case4t0 ), ONE_q128, whole_i64_to_q128( p3x ), whole_i64_to_q128( p3y ) );
      results[ points ] = p;
      points += 1u;
    }
    
    return LineSegmentIntersection( points, results[ 0 ], results[ 1 ] );
  }
  else {
    let t_numerator = subtract_i64_i64( mul_i64_i64( cdx, d1y ), mul_i64_i64( cdy, d1x ) );
    let u_numerator = subtract_i64_i64( mul_i64_i64( cdx, d0y ), mul_i64_i64( cdy, d0x ) );
    
    // This will move the sign to the numerator, BUT won't do the reduction (let us first see if there is an intersection)
    let t_raw = i64_to_q128( t_numerator, denominator );
    let u_raw = i64_to_q128( u_numerator, denominator );
    
    // 2i means totally internal, 1i means on an endpoint, 0i means totally external
    let t_cmp = ratio_test_q128( t_raw );
    let u_cmp = ratio_test_q128( u_raw );
    
    if ( t_cmp <= 0i || u_cmp <= 0i ) {
      return not_intersection; // outside one or both segments
    }
    else if ( t_cmp == 1i && u_cmp == 1i ) {
      return not_intersection; // on endpoints of both segments (we ignore that, we only want something internal to one)
    }
    else {
      // use parametric segment definition to get the intersection point
      // x0 + t * (x1 - x0)
      // p0x + t_numerator / denominator * d0x
      // ( denominator * p0x + t_numerator * d0x ) / denominator
      let x_numerator = add_i64_i64( mul_i64_i64( denominator, p0x ), mul_i64_i64( t_numerator, d0x ) );
      let y_numerator = add_i64_i64( mul_i64_i64( denominator, p0y ), mul_i64_i64( t_numerator, d0y ) );
      
      let x_raw = i64_to_q128( x_numerator, denominator );
      let y_raw = i64_to_q128( y_numerator, denominator );
      
      let x = reduce_q128( x_raw );
      let y = reduce_q128( y_raw );
      
      let t = reduce_q128( t_raw );
      let u = reduce_q128( u_raw );
      
      // NOTE: will t/u be exactly 0,1 for endpoints if they are endpoints, no?
      return LineSegmentIntersection( 1u, IntersectionPoint( t, u, x, y ), not_point );
    }
  }
}
`, [ intersectionPointSnippet, line_segment_intersectionSnippet, q128Snippet, i32_to_i64Snippet, mul_i64_i64Snippet, subtract_i64_i64Snippet, is_zero_u64Snippet, i64_to_q128Snippet, reduce_q128Snippet, cmp_i64_i64Snippet, equals_cross_mul_q128Snippet, negate_i64Snippet, is_zero_q128Snippet, ratio_test_q128Snippet, add_i64_i64Snippet, whole_i64_to_q128Snippet, ZERO_q128Snippet, ONE_q128Snippet ] );

const gcdBigInt = ( a, b ) => {
  while ( b !== 0n ) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
};

class Rational {
  // BigInts
  constructor( numerator, denominator ) {
    this.numerator = BigInt( numerator );
    this.denominator = BigInt( denominator );

    if ( denominator === 0n ) {
      throw new Error( 'Division by zero' );
    }
    if ( denominator < 0n ) {
      this.numerator = -this.numerator;
      this.denominator = -this.denominator;
    }
  }

  // @public
  copy() {
    return new Rational( this.numerator, this.denominator );
  }

  // @public - lazy implementation NOT meant to be in JS due to excess reduction
  plus( rational ) {
    return new Rational(
      this.numerator * rational.denominator + this.denominator * rational.numerator,
      this.denominator * rational.denominator
    ).reduced();
  }

  // @public - lazy implementation NOT meant to be in JS due to excess reduction
  minus( rational ) {
    return new Rational(
      this.numerator * rational.denominator - this.denominator * rational.numerator,
      this.denominator * rational.denominator
    ).reduced();
  }

  // @public - lazy implementation NOT meant to be in JS due to excess reduction
  times( rational ) {
    return new Rational(
      this.numerator * rational.numerator,
      this.denominator * rational.denominator
    ).reduced();
  }

  // @public
  reduce() {
    if ( this.numerator === 0n ) {
      this.denominator = 1n;
      return;
    }
    else if ( this.denominator === 1n ) {
      return;
    }

    const absNumerator = this.numerator < 0n ? -this.numerator : this.numerator;
    const gcd = gcdBigInt( absNumerator, this.denominator );

    if ( gcd !== 1n ) {
      this.numerator /= gcd;
      this.denominator /= gcd;
    }
  }

  // @public
  reduced() {
    const result = this.copy();
    result.reduce();
    return result;
  }

  // @public
  isZero() {
    return this.numerator === 0n;
  }

  // @public
  ratioTest() {
    if ( this.numerator === 0n || this.numerator === this.denominator ) {
      return 1n;
    }
    else if ( this.numerator > 0n && this.numerator < this.denominator ) {
      return 2n;
    }
    else {
      return 0n;
    }
  }

  // @public
  equalsCrossMul( other ) {
    return this.numerator * other.denominator === this.denominator * other.numerator;
  }

  // @public --- NOT for WGSL, slow
  equals( other ) {
    return this.reduced().numerator === other.reduced().numerator && this.reduced().denominator === other.reduced().denominator;
  }

  // @public
  toString() {
    return this.denominator === 1n ? `${this.numerator}` : `${this.numerator}/${this.denominator}`;
  }

  // @public
  static ZERO = new Rational( 0n, 1n );
  static ONE = new Rational( 1n, 1n );

  // @public
  static whole( numerator ) {
    return new Rational( numerator, 1n );
  }
}
window.Rational = Rational;

class BigIntPoint {
  // BigInt/numbers
  constructor( x, y ) {
    this.x = BigInt( x );
    this.y = BigInt( y );
  }
}

// TODO: could use this?
class RationalPoint {
  // Rationals
  constructor( x, y ) {
    this.x = x;
    this.y = y;
  }
}

class IntersectionPoint {
  constructor( t0, t1, px, py ) {
    this.t0 = t0;
    this.t1 = t1;
    this.px = px;
    this.py = py;
  }

  // @public
  toString() {
    return `t0=${this.t0}, t1=${this.t1}, px=${this.px}, py=${this.py}`;
  }

  // @public
  verify( p0, p1, p2, p3 ) {
    const px0 = Rational.whole( p0.x ).plus( this.t0.times( Rational.whole( p1.x - p0.x ) ) );
    const py0 = Rational.whole( p0.y ).plus( this.t0.times( Rational.whole( p1.y - p0.y ) ) );
    const px1 = Rational.whole( p2.x ).plus( this.t1.times( Rational.whole( p3.x - p2.x ) ) );
    const py1 = Rational.whole( p2.y ).plus( this.t1.times( Rational.whole( p3.y - p2.y ) ) );
    if ( !px0.equals( px1 ) || !py0.equals( py1 ) ) {
      throw new Error( 'Intersection point does not match' );
    }
  }
}

const intersectLineSegments = ( p0, p1, p2, p3 ) => {
  const p0x = BigInt( p0.x );
  const p0y = BigInt( p0.y );
  const p1x = BigInt( p1.x );
  const p1y = BigInt( p1.y );
  const p2x = BigInt( p2.x );
  const p2y = BigInt( p2.y );
  const p3x = BigInt( p3.x );
  const p3y = BigInt( p3.y );

  const d0x = p1x - p0x;
  const d0y = p1y - p0y;
  const d1x = p3x - p2x;
  const d1y = p3y - p2y;

  const cdx = p2x - p0x;
  const cdy = p2y - p0y;

  const denominator = d0x * d1y - d0y * d1x;

  if ( denominator === 0n ) {
    // such that p0 + t * ( p1 - p0 ) = p2 + ( a * t + b ) * ( p3 - p2 )
    // an equivalency between lines
    let a;
    let b;

    const d1x_zero = d1x === 0n;
    const d1y_zero = d1y === 0n;

    // if ( d0s === 0 || d1s === 0 ) {
    //   return NO_OVERLAP;
    // }
    //
    // a = d0s / d1s;
    // b = ( p0s - p2s ) / d1s;

    // TODO: can we reduce the branching here?
    // Find a dimension where our line is not degenerate (e.g. covers multiple values in that dimension)
    // Compute line equivalency there
    if ( d1x_zero && d1y_zero ) {
      // DEGENERATE case for second line, it's a point, bail out
      return [];
    }
    else if ( d1x_zero ) {
      // if d1x is zero AND our denominator is zero, that means d0x or d1y must be zero. We checked d1y above, so d0x must be zero
      if ( p0.x !== p2.x ) {
        // vertical lines, BUT not same x, so no intersection
        return [];
      }
      a = new Rational( d0y, d1y );
      b = new Rational( -cdy, d1y );
    }
    else if ( d1y_zero ) {
      // if d1y is zero AND our denominator is zero, that means d0y or d1x must be zero. We checked d1x above, so d0y must be zero
      if ( p0.y !== p2.y ) {
        // horizontal lines, BUT not same y, so no intersection
        return [];
      }
      a = new Rational( d0x, d1x );
      b = new Rational( -cdx, d1x );
    }
    else {
      // we have non-axis-aligned second line, use that to compute a,b for each dimension, and we're the same "line"
      // iff those are consistent
      if ( d0x === 0n && d0y === 0n ) {
        // DEGENERATE first line, it's a point, bail out
        return [];
      }
      const ax = new Rational( d0x, d1x );
      const ay = new Rational( d0y, d1y );
      if ( !ax.equalsCrossMul( ay ) ) {
        return [];
      }
      const bx = new Rational( -cdx, d1x );
      const by = new Rational( -cdy, d1y );
      if ( !bx.equalsCrossMul( by ) ) {
        return [];
      }

      // Pick the one with a non-zero a, so it is invertible
      if ( ax.isZero() ) {
        a = ay;
        b = by;
      }
      else {
        a = ax;
        b = bx;
      }
    }

    const points = [];

    // p0 + t * ( p1 - p0 ) = p2 + ( a * t + b ) * ( p3 - p2 )
    // i.e. line0( t ) = line1( a * t + b )
    // replacements for endpoints:
    // t=0       =>  t0=0,        t1=b
    // t=1       =>  t0=1,        t1=a+b
    // t=-b/a    =>  t0=-b/a,     t1=0
    // t=(1-b)/a =>  t0=(1-b)/a,  t1=1

    // NOTE: cases become identical if b=0, b=1, b=-a, b=1-a, HOWEVER these would not be internal, so they would be
    // excluded, and we can ignore them

    // t0=0, t1=b, p0
    const case1t1 = b;
    if ( case1t1.ratioTest() === 2n ) {
      const p = new IntersectionPoint( Rational.ZERO, case1t1.reduced(), Rational.whole( p0x ), Rational.whole( p0y ) );
      p.verify( p0, p1, p2, p3 );
      points.push( p );
    }

    // t0=1, t1=a+b, p1
    const case2t1 = new Rational( a.numerator + b.numerator, a.denominator ); // abuse a,b having same denominator
    if ( case2t1.ratioTest() === 2n ) {
      const p = new IntersectionPoint( Rational.ONE, case2t1.reduced(), Rational.whole( p1x ), Rational.whole( p1y ) );
      p.verify( p0, p1, p2, p3 );
      points.push( p );
    }

    // t0=-b/a, t1=0, p2
    const case3t0 = new Rational( -b.numerator, a.numerator ); // abuse a,b having same denominator
    if ( case3t0.ratioTest() === 2n ) {
      const p = new IntersectionPoint( case3t0.reduced(), Rational.ZERO, Rational.whole( p2x ), Rational.whole( p2y ) );
      p.verify( p0, p1, p2, p3 );
      points.push( p );
    }

    // t0=(1-b)/a, t1=1, p3
    // ( 1 - b ) / a = ( denom - b_numer ) / denom / ( a_numer / denom ) = ( denom - b_numer ) / a_numer
    const case4t0 = new Rational( a.denominator - b.numerator, a.numerator );
    if ( case4t0.ratioTest() === 2n ) {
      const p = new IntersectionPoint( case4t0.reduced(), Rational.ONE, Rational.whole( p3x ), Rational.whole( p3y ) );
      p.verify( p0, p1, p2, p3 );
      points.push( p );
    }

    return points;
  }
  else {
    const t_numerator = cdx * d1y - cdy * d1x;
    const u_numerator = cdx * d0y - cdy * d0x;

    // This will move the sign to the numerator, BUT won't do the reduction (let us first see if there is an intersection)
    const t_raw = new Rational( t_numerator, denominator );
    const u_raw = new Rational( u_numerator, denominator );

    // 2i means totally internal, 1i means on an endpoint, 0i means totally external
    const t_cmp = t_raw.ratioTest();
    const u_cmp = u_raw.ratioTest();

    if ( t_cmp <= 0n || u_cmp <= 0n ) {
      return []; // outside one or both segments
    }
    else if ( t_cmp === 1n && u_cmp === 1n ) {
      return []; // on endpoints of both segments (we ignore that, we only want something internal to one)
    }
    else {
      // use parametric segment definition to get the intersection point
      // x0 + t * (x1 - x0)
      // p0x + t_numerator / denominator * d0x
      // ( denominator * p0x + t_numerator * d0x ) / denominator
      const x_numerator = denominator * p0x + t_numerator * d0x;
      const y_numerator = denominator * p0y + t_numerator * d0y;

      const x_raw = new Rational( x_numerator, denominator );
      const y_raw = new Rational( y_numerator, denominator );

      const x = x_raw.reduced();
      const y = y_raw.reduced();

      const t = t_raw.reduced();
      const u = u_raw.reduced();

      // NOTE: will t/u be exactly 0,1 for endpoints if they are endpoints, no?
      const point = new IntersectionPoint( t, u, x, y );
      point.verify( p0, p1, p2, p3 );
      return [ point ];
    }
  }
};
window.intersectLineSegments = intersectLineSegments;

// window.div_u64_u64 = ( a, b ) => {
//   if ( a === 0n ) {
//     return [ 0n, 0n ];
//   }
//   if ( b === 0n ) {
//     throw new Error( 'Division by zero' );
//   }
//
//   let result = 0n;
//   let remainder = a;
//
//   const highBit = Math.min( a.toString( 2 ).length - 1, b.toString( 2 ).length - 1 );
//   let count = 63 - highBit;
//   let divisor = b << BigInt( count );
//
//   while ( remainder !== 0n ) {
//     if ( remainder >= divisor ) {
//       remainder -= divisor;
//       result |= 1n << BigInt( count );
//     }
//     if ( count === 0 ) {
//       break;
//     }
//     divisor >>= 1n;
//     count -= 1;
//   }
//
//   return [ result, remainder ];
// };
// // TODO: div_u64_u64 JS seems to be working, what is our problem here? Check dependencies

const runInOut = async ( device, mainCode, dependencies, dispatchSize, inputArrayBuffer, outputArrayBuffer ) => {
  const code = new Snippet( `
@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1) fn main(
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let i = id.x;
  ${mainCode}
}
`, dependencies ).toString();
  // console.log( code );
  const module = device.createShaderModule( {
    label: 'shader module',
    code: code
  } );

  const bindGroupLayout = device.createBindGroupLayout( {
    entries: [ {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'storage'
      }
    }, {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'storage'
      }
    } ]
  } );

  const pipeline = device.createComputePipeline( {
    label: 'compute pipeline',
    layout: device.createPipelineLayout( {
      bindGroupLayouts: [ bindGroupLayout ]
    } ),
    compute: {
      module: module,
      entryPoint: 'main'
    }
  } );

  const inputSize = inputArrayBuffer.byteLength;
  const outputSize = outputArrayBuffer.byteLength;

  const inputBuffer = device.createBuffer( {
    label: 'work buffer',
    size: inputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  } );
  device.queue.writeBuffer( inputBuffer, 0, inputArrayBuffer );

  const outputBuffer = device.createBuffer( {
    label: 'output buffer',
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  } );

  const resultBuffer = device.createBuffer( {
    label: 'result buffer',
    size: outputSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  } );

  const bindGroup = device.createBindGroup( {
    label: 'bindGroup',
    layout: pipeline.getBindGroupLayout( 0 ),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } }
    ]
  } );

  const encoder = device.createCommandEncoder( {
    label: 'encoder'
  } );
  const pass = encoder.beginComputePass( {
    label: 'compute pass'
  } );
  pass.setPipeline( pipeline );
  pass.setBindGroup( 0, bindGroup );
  pass.dispatchWorkgroups( dispatchSize );
  pass.end();

  encoder.copyBufferToBuffer( outputBuffer, 0, resultBuffer, 0, resultBuffer.size );

  const commandBuffer = encoder.finish();
  device.queue.submit( [ commandBuffer ] );

  await resultBuffer.mapAsync( GPUMapMode.READ );
  const resultArrayBuffer = resultBuffer.getMappedRange();
  new Uint8Array( outputArrayBuffer ).set( new Uint8Array( resultArrayBuffer ) );

  resultBuffer.unmap();

  inputBuffer.destroy();
  outputBuffer.destroy();
  resultBuffer.destroy();
};

// const printInOut = async ( device, mainCode, dependencies, dispatchSize, inputArrayBuffer, outputArrayBuffer ) => {
//   await runInOut( device, mainCode, dependencies, dispatchSize, inputArrayBuffer, outputArrayBuffer );
//
//   logInputOutput( inputArrayBuffer, outputArrayBuffer );
// };

const expectInOut = async ( device, mainCode, dependencies, dispatchSize, inputArrayBuffer, expectedOutputArrayBuffer, message ) => {
  const actualOutputArrayBuffer = new ArrayBuffer( expectedOutputArrayBuffer.byteLength );

  await runInOut( device, mainCode, dependencies, dispatchSize, inputArrayBuffer, actualOutputArrayBuffer );

  const inputUint32Array = new Uint32Array( inputArrayBuffer );
  const expectedOutputUInt32Array = new Uint32Array( expectedOutputArrayBuffer );
  const actualOutputUInt32Array = new Uint32Array( actualOutputArrayBuffer );

  if ( [ ...expectedOutputUInt32Array ].every( ( v, i ) => v === actualOutputUInt32Array[ i ] ) ) {
    console.log( `[PASS] ${message}` );
    return true;
  }
  else {
    console.log( `[FAIL] ${message}` );
    console.log( 'in', [ ...inputUint32Array ].map( toU32Hex ).join( ', ' ) );
    console.log( 'in', [ ...inputUint32Array ].map( toU32Binary ).join( ', ' ) );
    console.log( 'ex', [ ...expectedOutputUInt32Array ].map( toU32Hex ).join( ', ' ) );
    console.log( 'ex', [ ...expectedOutputUInt32Array ].map( toU32Binary ).join( ', ' ) );
    console.log( 'ac', [ ...actualOutputUInt32Array ].map( toU32Hex ).join( ', ' ) );
    console.log( 'ac', [ ...actualOutputUInt32Array ].map( toU32Binary ).join( ', ' ) );
    return false;
  }
};

const main = async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();

  {
    // i32_to_i64
    await expectInOut( device, `
      let in = i * 1u;
      let out = i * 2u;
      let a = bitcast<i32>( input[ in ] );
      let c = i32_to_i64( a );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      i32_to_i64Snippet
    ], 4, new Int32Array( [
      0, 25, -7, -1024
    ] ).buffer, new Uint32Array( [
      ...nToU32s( 0n ),
      ...nToU32s( 25n ),
      ...nToU32s( -7n ),
      ...nToU32s( -1024n )
    ] ).buffer, 'i32_to_i64' );
  }

  {
    // negate_i64
    await expectInOut( device, `
      let in = i * 2u;
      let out = i * 2u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let c = negate_i64( a );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      negate_i64Snippet
    ], 4, new Uint32Array( [
      ...nToU32s( 0n ),
      ...nToU32s( 25n ),
      ...nToU32s( -7n ),
      ...nToU32s( -1024n )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( -0n ),
      ...nToU32s( -25n ),
      ...nToU32s( 7n ),
      ...nToU32s( 1024n )
    ] ).buffer, 'negate_i64' );
  }

  {
    // left_shift_u64
    await expectInOut( device, `
      let out = i * 2u;
      let a = vec2( input[ 0u ], input[ 1u ] );
      let c = left_shift_u64( a, i );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      left_shift_u64Snippet
    ], 64, new Uint32Array( [
      ...nToU32s( 0xf9fe432c7aca8bfan )
    ] ).buffer, new Uint32Array( [
      // TODO: simplify, I'm lazy
      ...nToU32s( 0xf9fe432c7aca8bfan << 0n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 1n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 2n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 3n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 4n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 5n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 6n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 7n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 8n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 9n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 10n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 11n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 12n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 13n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 14n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 15n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 16n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 17n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 18n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 19n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 20n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 21n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 22n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 23n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 24n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 25n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 26n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 27n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 28n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 29n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 30n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 31n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 32n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 33n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 34n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 35n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 36n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 37n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 38n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 39n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 40n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 41n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 42n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 43n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 44n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 45n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 46n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 47n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 48n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 49n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 50n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 51n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 52n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 53n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 54n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 55n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 56n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 57n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 58n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 59n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 60n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 61n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 62n ),
      ...nToU32s( 0xf9fe432c7aca8bfan << 63n )
    ] ).buffer, 'left_shift_u64' );
  }

  {
    // right_shift_u64
    await expectInOut( device, `
      let out = i * 2u;
      let a = vec2( input[ 0u ], input[ 1u ] );
      let c = right_shift_u64( a, i );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      right_shift_u64Snippet
    ], 64, new Uint32Array( [
      ...nToU32s( 0xf9fe432c7aca8bfan )
    ] ).buffer, new Uint32Array( [
      // TODO: simplify, I'm lazy
      ...nToU32s( 0xf9fe432c7aca8bfan >> 0n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 1n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 2n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 3n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 4n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 5n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 6n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 7n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 8n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 9n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 10n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 11n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 12n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 13n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 14n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 15n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 16n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 17n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 18n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 19n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 20n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 21n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 22n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 23n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 24n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 25n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 26n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 27n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 28n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 29n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 30n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 31n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 32n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 33n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 34n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 35n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 36n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 37n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 38n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 39n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 40n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 41n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 42n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 43n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 44n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 45n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 46n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 47n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 48n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 49n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 50n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 51n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 52n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 53n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 54n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 55n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 56n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 57n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 58n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 59n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 60n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 61n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 62n ),
      ...nToU32s( 0xf9fe432c7aca8bfan >> 63n )
    ] ).buffer, 'right_shift_u64' );
  }

  {
    // is_negative_i64
    await expectInOut( device, `
      let in = i * 2u;
      let out = i * 1u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let c = is_negative_i64( a );
      output[ out ] = select( 0u, 1u, c );
    `, [
      is_negative_i64Snippet
    ], 4, new Uint32Array( [
      ...nToU32s( 0n ),
      ...nToU32s( 25n ),
      ...nToU32s( -7n ),
      ...nToU32s( -1024n )
    ] ).buffer, new Uint32Array( [
      0,
      0,
      1,
      1
    ] ).buffer, 'is_negative_i64' );
  }

  {
    // cmp_u64_u64
    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 1u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = cmp_u64_u64( a, b );
      output[ out ] = bitcast<u32>( c );
    `, [
      cmp_u64_u64Snippet
    ], 3, new Uint32Array( [
      ...nToU32s( 5n ),
      ...nToU32s( 7n ),
      ...nToU32s( 7n ),
      ...nToU32s( 5n ),
      ...nToU32s( 12n ),
      ...nToU32s( 12n )
    ] ).buffer, new Int32Array( [
      -1, // 5 < 7
      1, // 7 > 5
      0 // -12 = -12
    ] ).buffer, 'cmp_u64_u64' );
  }

  {
    // cmp_i64_i64
    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 1u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = cmp_i64_i64( a, b );
      output[ out ] = bitcast<u32>( c );
    `, [
      cmp_i64_i64Snippet
    ], 3, new Uint32Array( [
      ...nToU32s( 5n ),
      ...nToU32s( 7n ),
      ...nToU32s( 5n ),
      ...nToU32s( -7n ),
      ...nToU32s( -12n ),
      ...nToU32s( -12n )
    ] ).buffer, new Int32Array( [
      -1, // 5 < 7
      1, // 5 > -7
      0 // -12 = -12
    ] ).buffer, 'cmp_i64_i64' );
  }

  {
    // add_u64_u64
    const an = 0xf9fe432c7aca8bfan;
    const bn = 0x583b15971ad94165n;
    const cn = an + bn;

    const dn = 0xddddddddddddddddn;
    const en = 0xababababababababn;
    const fn = dn + en;

    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 2u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = add_u64_u64( a, b );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      add_u64_u64Snippet
    ], 2, new Uint32Array( [
      ...nToU32s( an ),
      ...nToU32s( bn ),
      ...nToU32s( dn ),
      ...nToU32s( en )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( cn ),
      ...nToU32s( fn )
    ] ).buffer, `add_u64_u64 ${an.toString( 16 )} ${bn.toString( 16 )} = ${cn.toString( 16 )}` );
  }

  {
    // add_i64_i64
    const an = 0xf9fe432c7aca8bfan;
    const bn = 0x583b15971ad94165n;
    const cn = an + bn;

    const dn = 0xddddddddddddddddn;
    const en = 0xababababababababn;
    const fn = dn + en;

    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 2u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = add_i64_i64( a, b );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      add_i64_i64Snippet
    ], 2, new Uint32Array( [
      ...nToU32s( an ),
      ...nToU32s( bn ),
      ...nToU32s( dn ),
      ...nToU32s( en )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( cn ),
      ...nToU32s( fn )
    ] ).buffer, 'add_i64_i64' );
  }

  {
    // subtract_i64_i64
    const an = 0xf9fe432c7aca8bfan;
    const bn = 0x583b15971ad94165n;
    const cn = an - bn;

    const dn = 0xddddddddddddddddn;
    const en = 0xababababababababn;
    const fn = dn - en;

    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 2u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = subtract_i64_i64( a, b );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      subtract_i64_i64Snippet
    ], 2, new Uint32Array( [
      ...nToU32s( an ),
      ...nToU32s( bn ),
      ...nToU32s( dn ),
      ...nToU32s( en )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( cn ),
      ...nToU32s( fn )
    ] ).buffer, 'subtract_i64_i64' );
  }

  {
    // mul_u64_u64
    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 2u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = mul_u64_u64( a, b );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      mul_u64_u64Snippet
    ], 2, new Uint32Array( [
      ...nToU32s( 0xf9fe432c7aca8bfan ),
      ...nToU32s( 0x583b15971ad94165n ),
      ...nToU32s( 0x1a951ef9n ),
      ...nToU32s( 0xa629b1b2n )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( 0xf9fe432c7aca8bfan * 0x583b15971ad94165n ),
      ...nToU32s( 0x1a951ef9n * 0xa629b1b2n )
    ] ).buffer, 'mul_u64_u64' );
  }

  {
    // mul_i64_i64
    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 2u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = mul_i64_i64( a, b );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      mul_i64_i64Snippet
    ], 5, new Uint32Array( [
      ...nToU32s( 0x1a951ef9n ),
      ...nToU32s( 0xa629b1b2n ),
      ...nToU32s( 5n ),
      ...nToU32s( 7n ),
      ...nToU32s( -5n ),
      ...nToU32s( 7n ),
      ...nToU32s( 5n ),
      ...nToU32s( -7n ),
      ...nToU32s( -5n ),
      ...nToU32s( -7n )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( 0x1a951ef9n * 0xa629b1b2n ),
      ...nToU32s( 35n ),
      ...nToU32s( -35n ),
      ...nToU32s( -35n ),
      ...nToU32s( 35n )
    ] ).buffer, 'mul_i64_i64' );
  }

  {
    // div_u64_u64
    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 4u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = div_u64_u64( a, b );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
      output[ out + 2u ] = c.z;
      output[ out + 3u ] = c.w;
    `, [
      div_u64_u64Snippet
    ], 3, new Uint32Array( [
      ...nToU32s( 32n ),
      ...nToU32s( 5n ),
      ...nToU32s( 0xf9fe432c7aca8bfan ),
      ...nToU32s( 0x583b15971ad94165n ),
      ...nToU32s( 0x19fe432c7aca8bfan ),
      ...nToU32s( 0x1b5dcn )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( 6n ),
      ...nToU32s( 2n ),
      ...nToU32s( 0xf9fe432c7aca8bfan / 0x583b15971ad94165n ),
      ...nToU32s( 0xf9fe432c7aca8bfan % 0x583b15971ad94165n ),
      ...nToU32s( 0x19fe432c7aca8bfan / 0x1b5dcn ),
      ...nToU32s( 0x19fe432c7aca8bfan % 0x1b5dcn )
    ] ).buffer, 'div_u64_u64' );
  }

  {
    // gcd_u64_u64
    const gcd0 = 0xa519bc952f7n;
    const a0 = gcd0 * 0x1542n;
    const b0 = gcd0 * 0xa93n; // chosen as relatively prime

    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 2u;
      let a = vec2( input[ in + 0u ], input[ in + 1u ] );
      let b = vec2( input[ in + 2u ], input[ in + 3u ] );
      let c = gcd_u64_u64( a, b );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
    `, [
      gcd_u64_u64Snippet
    ], 2, new Uint32Array( [
      ...nToU32s( 35n ),
      ...nToU32s( 10n ),
      ...nToU32s( a0 ),
      ...nToU32s( b0 )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( 5n ),
      ...nToU32s( gcd0 )
    ] ).buffer, 'gcd_u64_u64' );
  }

  {
    // reduce_q128
    await expectInOut( device, `
      let in = i * 4u;
      let out = i * 4u;
      let a = vec4( input[ in + 0u ], input[ in + 1u ], input[ in + 2u ], input[ in + 3u ] );
      let c = reduce_q128( a );
      output[ out + 0u ] = c.x;
      output[ out + 1u ] = c.y;
      output[ out + 2u ] = c.z;
      output[ out + 3u ] = c.w;
    `, [
      reduce_q128Snippet
    ], 3, new Uint32Array( [
      ...nToU32s( 4n ),
      ...nToU32s( 12n ),
      ...nToU32s( -32n ),
      ...nToU32s( 100n ),
      ...nToU32s( 0n ),
      ...nToU32s( 100n )
    ] ).buffer, new Uint32Array( [
      ...nToU32s( 1n ), // 4/12 => 1/3
      ...nToU32s( 3n ),
      ...nToU32s( -8n ), // -32/100 => -8/25
      ...nToU32s( 25n ),
      ...nToU32s( 0n ), // 0/100 => 0/1
      ...nToU32s( 1n )
    ] ).buffer, 'reduce_q128' );
  }

  {
    // intersect_line_segments
    await expectInOut( device, `
      let in = i * 8u;
      let out = i * 32u;
      let p0 = bitcast<vec2i>( vec2( input[ in + 0u ], input[ in + 1u ] ) );
      let p1 = bitcast<vec2i>( vec2( input[ in + 2u ], input[ in + 3u ] ) );
      let p2 = bitcast<vec2i>( vec2( input[ in + 4u ], input[ in + 5u ] ) );
      let p3 = bitcast<vec2i>( vec2( input[ in + 6u ], input[ in + 7u ] ) );
      let c = intersect_line_segments( p0, p1, p2, p3 );
      output[ out + 0u ] = c.p0.t0.x;
      output[ out + 1u ] = c.p0.t0.y;
      output[ out + 2u ] = c.p0.t0.z;
      output[ out + 3u ] = c.p0.t0.w;
      output[ out + 4u ] = c.p0.t1.x;
      output[ out + 5u ] = c.p0.t1.y;
      output[ out + 6u ] = c.p0.t1.z;
      output[ out + 7u ] = c.p0.t1.w;
      output[ out + 8u ] = c.p0.px.x;
      output[ out + 9u ] = c.p0.px.y;
      output[ out + 10u ] = c.p0.px.z;
      output[ out + 11u ] = c.p0.px.w;
      output[ out + 12u ] = c.p0.py.x;
      output[ out + 13u ] = c.p0.py.y;
      output[ out + 14u ] = c.p0.py.z;
      output[ out + 15u ] = c.p0.py.w;
      output[ out + 16u ] = c.p1.t0.x;
      output[ out + 17u ] = c.p1.t0.y;
      output[ out + 18u ] = c.p1.t0.z;
      output[ out + 19u ] = c.p1.t0.w;
      output[ out + 20u ] = c.p1.t1.x;
      output[ out + 21u ] = c.p1.t1.y;
      output[ out + 22u ] = c.p1.t1.z;
      output[ out + 23u ] = c.p1.t1.w;
      output[ out + 24u ] = c.p1.px.x;
      output[ out + 25u ] = c.p1.px.y;
      output[ out + 26u ] = c.p1.px.z;
      output[ out + 27u ] = c.p1.px.w;
      output[ out + 28u ] = c.p1.py.x;
      output[ out + 29u ] = c.p1.py.y;
      output[ out + 30u ] = c.p1.py.z;
      output[ out + 31u ] = c.p1.py.w;
    `, [
      intersect_line_segmentsSnippet
    ], 7, new Int32Array( [
      // an X (diagonal)
      0, 0, 100, 100, 0, 100, 100, 0,

      // overlap (skewed)
      // --------
      //     --------
      0, 0, 100, 200, 50, 100, 150, 300,

      // overlap (skewed)
      //     --------
      // --------
      50, 100, 150, 300, 0, 0, 100, 200,

      // overlap (and horizontal)
      //   ----
      // --------
      25, 0, 75, 0, 0, 0, 100, 0,

      // overlap (and vertical)
      // |
      // | |
      // | |
      // |
      0, 0, 0, 15, 0, 5, 0, 10,

      // no intersection
      0, 0, 40, 50, 0, 100, 100, 0,

      // T intersection
      0, 0, 12, 8, 3, 2, 5, -2
    ] ).buffer, new Uint32Array( [
      // p0 t0
      ...nToU32s( 1n ), ...nToU32s( 2n ), // 1/2
      // p0 t1
      ...nToU32s( 1n ), ...nToU32s( 2n ), // 1/2
      // p0 px
      ...nToU32s( 50n ), ...nToU32s( 1n ), // 50
      // p0 py
      ...nToU32s( 50n ), ...nToU32s( 1n ), // 50
      // p1 t0
      0, 0, 0, 0,
      // p1 t1
      0, 0, 0, 0,
      // p1 px
      0, 0, 0, 0,
      // p1 py
      0, 0, 0, 0,

      // p0 t0
      ...nToU32s( 1n ), ...nToU32s( 1n ), // 1
      // p0 t1
      ...nToU32s( 1n ), ...nToU32s( 2n ), // 1/2
      // p0 px
      ...nToU32s( 100n ), ...nToU32s( 1n ), // 100
      // p0 py
      ...nToU32s( 200n ), ...nToU32s( 1n ), // 200
      // p1 t0
      ...nToU32s( 1n ), ...nToU32s( 2n ), // 1/2
      // p1 t1
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p1 px
      ...nToU32s( 50n ), ...nToU32s( 1n ), // 50
      // p1 py
      ...nToU32s( 100n ), ...nToU32s( 1n ), // 100

      // p0 t0
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p0 t1
      ...nToU32s( 1n ), ...nToU32s( 2n ), // 1/2
      // p0 px
      ...nToU32s( 50n ), ...nToU32s( 1n ), // 50
      // p0 py
      ...nToU32s( 100n ), ...nToU32s( 1n ), // 100
      // p1 t0
      ...nToU32s( 1n ), ...nToU32s( 2n ), // 1/2
      // p1 t1
      ...nToU32s( 1n ), ...nToU32s( 1n ), // 1
      // p1 px
      ...nToU32s( 100n ), ...nToU32s( 1n ), // 100
      // p1 py
      ...nToU32s( 200n ), ...nToU32s( 1n ), // 200

      // p0 t0
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p0 t1
      ...nToU32s( 1n ), ...nToU32s( 4n ), // 1/4
      // p0 px
      ...nToU32s( 25n ), ...nToU32s( 1n ), // 25
      // p0 py
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p1 t0
      ...nToU32s( 1n ), ...nToU32s( 1n ), // 1
      // p1 t1
      ...nToU32s( 3n ), ...nToU32s( 4n ), // 3/4
      // p1 px
      ...nToU32s( 75n ), ...nToU32s( 1n ), // 75
      // p1 py
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0

      // p0 t0
      ...nToU32s( 1n ), ...nToU32s( 3n ), // 1/3
      // p0 t1
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p0 px
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p0 py
      ...nToU32s( 5n ), ...nToU32s( 1n ), // 5
      // p1 t0
      ...nToU32s( 2n ), ...nToU32s( 3n ), // 2/3
      // p1 t1
      ...nToU32s( 1n ), ...nToU32s( 1n ), // 1
      // p1 px
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p1 py
      ...nToU32s( 10n ), ...nToU32s( 1n ), // 10

      // p0 t0
      0, 0, 0, 0,
      // p0 t1
      0, 0, 0, 0,
      // p0 px
      0, 0, 0, 0,
      // p0 py
      0, 0, 0, 0,
      // p1 t0
      0, 0, 0, 0,
      // p1 t1
      0, 0, 0, 0,
      // p1 px
      0, 0, 0, 0,
      // p1 py
      0, 0, 0, 0,

      // p0 t0
      ...nToU32s( 1n ), ...nToU32s( 4n ), // 1/4
      // p0 t1
      ...nToU32s( 0n ), ...nToU32s( 1n ), // 0
      // p0 px
      ...nToU32s( 3n ), ...nToU32s( 1n ), // 3
      // p0 py
      ...nToU32s( 2n ), ...nToU32s( 1n ), // 2
      // p1 t0
      0, 0, 0, 0,
      // p1 t1
      0, 0, 0, 0,
      // p1 px
      0, 0, 0, 0,
      // p1 py
      0, 0, 0, 0
    ] ).buffer, 'intersect_line_segments' );
  }
};
main();
