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
    // TODO: zero denominator! parallel lines, check for overlap or not
    return not_intersection;
  }
  else {
    let t_numerator = subtract_i64_i64( mul_i64_i64( cdx, d1y ), mul_i64_i64( cdy, d1x ) );
    let u_numerator = subtract_i64_i64( mul_i64_i64( cdx, d0y ), mul_i64_i64( cdy, d0x ) );
    
    // This will move the sign to the numerator, BUT won't do the reduction (let us first see if there is an intersection)
    let t_raw = i64_to_q128( t_numerator, denominator );
    let u_raw = i64_to_q128( u_numerator, denominator );
    
    // 2i means totally internal, 1i means on an endpoint, 0i means totally external
    let t_cmp = cmp_i64_i64( t_raw.xy, vec2( 0u, 0u ) ) + cmp_i64_i64( t_raw.zw, t_raw.xy );
    let u_cmp = cmp_i64_i64( u_raw.xy, vec2( 0u, 0u ) ) + cmp_i64_i64( u_raw.zw, u_raw.xy );
    
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
`, [ intersectionPointSnippet, line_segment_intersectionSnippet, q128Snippet, i32_to_i64Snippet, mul_i64_i64Snippet, subtract_i64_i64Snippet, is_zero_u64Snippet, i64_to_q128Snippet, reduce_q128Snippet, cmp_i64_i64Snippet ] );

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


// window.gcd_u64_u64 = ( a, b ) => {
//   while ( b !== 0n ) {
//     const t = b;
//     b = a % b;
//     a = t;
//   }
//   return a;
// };

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
    ], 1, new Int32Array( [
      0, 0, 100, 100, 0, 100, 100, 0
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
      0, 0, 0, 0
    ] ).buffer, 'intersect_line_segments' );
  }
};
main();
