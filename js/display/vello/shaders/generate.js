/* eslint-disable */

// Compiles and minifies the WGSL Vello shaders, turning them into ES6 modules
// TODO: better minification with 3rd party library (if/when it exists?)

/*
 * List of things I've had to patch for the shaders:
 *
 * Shaders from 31f8d9ffa046d5c8cb2da8d99f24f65f00073215
 *
 * Don't premultiply output:
 * - fine.wgsl
 *
 * Color Matrix filter support, changed DrawTag values, moving the scene_offset, and swapping it for alpha
 * - coarse.ts
 * - fine.ts
 * - draw_leaf.ts
 * - shared/drawtag.ts
 * - shared/ptcl.ts
 *
 * Use constants in case expressions (assorted files)
 */

// TODO: could look at places where abstract int/float could be swapped in for the explicit types
// TODO: could wrap long builtin function calls with a shorter named function (but that might reduce performance?)
// TODO: looking at you, bitcast!!!
// TODO: vec2(0.0, 0.0) => vec2(0.0) (and similar) -- doesn't happen often enough to bother

// go to this directory, then run `node generate.js` to generate the shaders (output is also in this directory)

const fs = require( 'fs' );
const _ = require( 'lodash' );

// Enable opting out of minification for debugging
const MINIFY = true;

const KEYWORDS = [
  'alias',
  'break',
  'case',
  'const',
  'const_assert',
  'continue',
  'continuing',
  'default',
  'diagnostic',
  'discard',
  'else',
  'enable',
  'false',
  'fn',
  'for',
  'if',
  'let',
  'loop',
  'override',
  'requires',
  'return',
  'struct',
  'switch',
  'true',
  'var',
  'while'
];

const RESERVED = [
  'NULL',
  'Self',
  'abstract',
  'active',
  'alignas',
  'alignof',
  'as',
  'asm',
  'asm_fragment',
  'async',
  'attribute',
  'auto',
  'await',
  'become',
  'binding_array',
  'cast',
  'catch',
  'class',
  'co_await',
  'co_return',
  'co_yield',
  'coherent',
  'column_major',
  'common',
  'compile',
  'compile_fragment',
  'concept',
  'const_cast',
  'consteval',
  'constexpr',
  'constinit',
  'crate',
  'debugger',
  'decltype',
  'delete',
  'demote',
  'demote_to_helper',
  'do',
  'dynamic_cast',
  'enum',
  'explicit',
  'export',
  'extends',
  'extern',
  'external',
  'fallthrough',
  'filter',
  'final',
  'finally',
  'friend',
  'from',
  'fxgroup',
  'get',
  'goto',
  'groupshared',
  'highp',
  'impl',
  'implements',
  'import',
  'inline',
  'instanceof',
  'interface',
  'layout',
  'lowp',
  'macro',
  'macro_rules',
  'match',
  'mediump',
  'meta',
  'mod',
  'module',
  'move',
  'mut',
  'mutable',
  'namespace',
  'new',
  'nil',
  'noexcept',
  'noinline',
  'nointerpolation',
  'noperspective',
  'null',
  'nullptr',
  'of',
  'operator',
  'package',
  'packoffset',
  'partition',
  'pass',
  'patch',
  'pixelfragment',
  'precise',
  'precision',
  'premerge',
  'priv',
  'protected',
  'pub',
  'public',
  'readonly',
  'ref',
  'regardless',
  'register',
  'reinterpret_cast',
  'require',
  'resource',
  'restrict',
  'self',
  'set',
  'shared',
  'sizeof',
  'smooth',
  'snorm',
  'static',
  'static_assert',
  'static_cast',
  'std',
  'subroutine',
  'super',
  'target',
  'template',
  'this',
  'thread_local',
  'throw',
  'trait',
  'try',
  'type',
  'typedef',
  'typeid',
  'typename',
  'typeof',
  'union',
  'unless',
  'unorm',
  'unsafe',
  'unsized',
  'use',
  'using',
  'varying',
  'virtual',
  'volatile',
  'wgsl',
  'where',
  'with',
  'writeonly',
  'yield'
];

const ATTRIBUTES = [
  'align',
  'binding',
  'builtin',
  'compute',
  'const',
  'fragment',
  'group',
  'id',
  'interpolate',
  'invariant',
  'location',
  'size',
  'vertex',
  'workgroup_size'
];

const SWIZZLES = _.flatten( [ 'rgba', 'xyzw' ].map( rgba => {
  const result = [];
  const recur = ( prefix, remaining ) => {
    prefix && result.push( prefix );
    if ( remaining > 0 ) {
      for ( let i = 0; i < rgba.length; i++ ) {
        recur( prefix + rgba[ i ], remaining - 1 );
      }
    }
  };
  recur( '', 4 );
  return result;
} ) );

const OTHER = [
  'array',
  'bool',
  'f16',
  'f32',
  'i32',
  'mat2x2',
  'mat2x3',
  'mat2x4',
  'mat3x2',
  'mat3x3',
  'mat3x4',
  'mat4x2',
  'mat4x3',
  'mat4x4',
  'u32',
  'vec2',
  'vec3',
  'vec4',
  'bitcast',
  'all',
  'any',
  'select',
  'arrayLength',
  'abs',
  'acos',
  'acosh',
  'asin',
  'asinh',
  'atan',
  'atanh',
  'atan2',
  'ceil',
  'clamp',
  'cos',
  'cosh',
  'countLeadingZeros',
  'countOneBits',
  'countTrailingZeros',
  'cross',
  'degrees',
  'determinant',
  'distance',
  'dot',
  'exp',
  'exp2',
  'extractBits',
  'extractBits',
  'faceForward',
  'firstLeadingBit',
  'firstLeadingBit',
  'firstTrailingBit',
  'floor',
  'fma',
  'fract',
  'frexp',
  'insertBits',
  'inverseSqrt',
  'ldexp',
  'length',
  'log',
  'log2',
  'max',
  'min',
  'mix',
  'modf',
  'normalize',
  'pow',
  'quantizeToF16',
  'radians',
  'reflect',
  'refract',
  'reverseBits',
  'round',
  'saturate',
  'sign',
  'sin',
  'sinh',
  'smoothstep',
  'sqrt',
  'step',
  'tan',
  'tanh',
  'transpose',
  'trunc',
  'dpdx',
  'dpdxCoarse',
  'dpdxFine',
  'dpdy',
  'dpdyCoarse',
  'dpdyFine',
  'fwidth',
  'fwidthCoarse',
  'fwidthFine',
  'textureDimensions',
  'textureGather',
  'textureGatherCompare',
  'textureLoad',
  'textureNumLayers',
  'textureNumLevels',
  'textureNumSamples',
  'textureSample',
  'textureSampleBias',
  'textureSampleCompare',
  'textureSampleCompareLevel',
  'textureSampleGrad',
  'textureSampleLevel',
  'textureSampleBaseClampToEdge',
  'textureStore',
  'atomicLoad',
  'atomicStore',
  'atomicAdd',
  'atomicSub',
  'atomicMax',
  'atomicMin',
  'atomicAnd',
  'atomicOr',
  'atomicXor',
  'atomicExchange',
  'atomicCompareExchangeWeak',
  'pack4x8snorm',
  'pack4x8unorm',
  'pack2x16snorm',
  'pack2x16unorm',
  'pack2x16float',
  'unpack4x8snorm',
  'unpack4x8unorm',
  'unpack2x16snorm',
  'unpack2x16unorm',
  'unpack2x16float',
  'storageBarrier',
  'workgroupBarrier',
  'workgroupUniformLoad',
  'vec2i',
  'vec3i',
  'vec4i',
  'vec2u',
  'vec3u',
  'vec4u',
  'vec2f',
  'vec3f',
  'vec4f',
  'vec2h',
  'vec3h',
  'vec4h',
  'mat2x2f',
  'mat2x3f',
  'mat2x4f',
  'mat3x2f',
  'mat3x3f',
  'mat3x4f',
  'mat4x2f',
  'mat4x3f',
  'mat4x4f',
  'mat2x2h',
  'mat2x3h',
  'mat2x4h',
  'mat3x2h',
  'mat3x3h',
  'mat3x4h',
  'mat4x2h',
  'mat4x3h',
  'mat4x4h',
  'atomic',
  'read',
  'write',
  'read_write',
  'function',
  'private',
  'workgroup',
  'uniform',
  'storage',
  'perspective',
  'linear',
  'flat',
  'center',
  'centroid',
  'sample',
  'vertex_index',
  'instance_index',
  'position',
  'front_facing',
  'frag_depth',
  'local_invocation_id',
  'local_invocation_index',
  'global_invocation_id',
  'workgroup_id',
  'num_workgroups',
  'sample_index',
  'sample_mask',
  'rgba8unorm',
  'rgba8snorm',
  'rgba8uint',
  'rgba8sint',
  'rgba16uint',
  'rgba16sint',
  'rgba16float',
  'r32uint',
  'r32sint',
  'r32float',
  'rg32uint',
  'rg32sint',
  'rg32float',
  'rgba32uint',
  'rgba32sint',
  'rgba32float',
  'bgra8unorm',
  'texture_1d',
  'texture_2d',
  'texture_2d_array',
  'texture_3d',
  'texture_cube',
  'texture_cube_array',
  'texture_multisampled_2d',
  'texture_depth_multisampled_2d',
  'texture_external',
  'texture_storage_1d',
  'texture_storage_2d',
  'texture_storage_2d_array',
  'texture_storage_3d',
  'texture_depth_2d',
  'texture_depth_2d_array',
  'texture_depth_cube',
  'texture_depth_cube_array',
  'sampler',
  'sampler_comparison',
  'alias',
  'ptr',
  'vertex_index',
  'instance_index',
  'position',
  'fragment',
  'front_facing',
  'frag_depth',
  'sample_index',
  'sample_mask',
  'fragment',
  'local_invocation_id',
  'local_invocation_index',
  'global_invocation_id',
  'workgroup_id',
  'num_workgroups',
  'align',
  'binding',
  'builtin',
  'compute',
  'const',
  'diagnostic',
  'fragment',
  'group',
  'id',
  'interpolate',
  'invariant',
  'location',
  'must_use',
  'size',
  'vertex',
  'workgroup_size',
  'true',
  'false',
  'diagnostic',
  'error',
  'info',
  'off',
  'warning',
];

const AVOID_SYMBOLS = _.uniq( [
  ...KEYWORDS,
  ...RESERVED,
  ...ATTRIBUTES,
  ...SWIZZLES,
  ...OTHER
] ).sort();

const REPLACEMENT_MAP = {
  'vec2<i32>': 'vec2i',
  'vec3<i32>': 'vec3i',
  'vec4<i32>': 'vec4i',
  'vec2<u32>': 'vec2u',
  'vec3<u32>': 'vec3u',
  'vec4<u32>': 'vec4u',
  'vec2<f32>': 'vec2f',
  'vec3<f32>': 'vec3f',
  'vec4<f32>': 'vec4f',
  'vec2<f16>': 'vec2h',
  'vec3<f16>': 'vec3h',
  'vec4<f16>': 'vec4h',
  'mat2x2<f32>': 'mat2x2f',
  'mat2x3<f32>': 'mat2x3f',
  'mat2x4<f32>': 'mat2x4f',
  'mat3x2<f32>': 'mat3x2f',
  'mat3x3<f32>': 'mat3x3f',
  'mat3x4<f32>': 'mat3x4f',
  'mat4x2<f32>': 'mat4x2f',
  'mat4x3<f32>': 'mat4x3f',
  'mat4x4<f32>': 'mat4x4f',
  'mat2x2<f16>': 'mat2x2h',
  'mat2x3<f16>': 'mat2x3h',
  'mat2x4<f16>': 'mat2x4h',
  'mat3x2<f16>': 'mat3x2h',
  'mat3x3<f16>': 'mat3x3h',
  'mat3x4<f16>': 'mat3x4h',
  'mat4x2<f16>': 'mat4x2h',
  'mat4x3<f16>': 'mat4x3h',
  'mat4x4<f16>': 'mat4x4h'
};

const GLOBALLY_ALIASABLE_TYPES = [
  'u32',
  'i32',
  'f32',
  'bool',
  'f16',
  'vec2i',
  'vec3i',
  'vec4i',
  'vec2u',
  'vec3u',
  'vec4u',
  'vec2f',
  'vec3f',
  'vec4f',
  'vec2h',
  'vec3h',
  'vec4h',
  'mat2x2f',
  'mat2x3f',
  'mat2x4f',
  'mat3x2f',
  'mat3x3f',
  'mat3x4f',
  'mat4x2f',
  'mat4x3f',
  'mat4x4f',
  'mat2x2h',
  'mat2x3h',
  'mat2x4h',
  'mat3x2h',
  'mat3x3h',
  'mat3x4h',
  'mat4x2h',
  'mat4x3h',
  'mat4x4h',
  'atomic<u32>',
  'atomic<i32>',
  'array<u32>',
  'array<i32>',
  'array<f32>'
  // TODO: potentially other arrays?
  // TODO: potentially we can insert aliases AFTER struct defs that are for arrays of them?
];

const firstCharAlphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_';
const otherCharAlphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789';

// Generator to iterate through possible replacement symbols
const generateSymbol = function*() {
  let length = 0;
  let indexStack = [ firstCharAlphabet.length ];
  while ( true ) {
    let pushedNext = false;
    while ( indexStack.length > 0 ) {
      const index = indexStack.pop();
      const nextIndex = index + 1;
      if ( nextIndex < ( indexStack.length ? otherCharAlphabet : firstCharAlphabet ).length ) {
        indexStack.push( nextIndex );
        pushedNext = true;
        break;
      }
    }
    if ( !pushedNext ) {
      length++;
    }
    while ( indexStack.length < length ) {
      indexStack.push( 0 );
    }
    const symbol = indexStack.map( ( index, i ) => i === 0 ? firstCharAlphabet[ index ] : otherCharAlphabet[ index ] ).join( '' );
    if ( !AVOID_SYMBOLS.includes( symbol ) && symbol !== '_' && !symbol.startsWith( '__' ) ) {
      yield symbol;
    }
  }
};

const stripComments = str => {
  return str.replace( /\r\n/g, '\n' ).split( '\n' ).map( line => {
    const index = line.indexOf( '//' );
    return index >= 0 ? line.substring( 0, index ) : line;
  } ).join( '\n' );
};

let symbols = [];
let totalStr = '';
const scanSymbols = str => {
  // TODO: don't require this specific formatting! Search for symbols otherwise?
  totalStr += str;
  str = stripComments( str );
  [ ...str.matchAll( /struct ([\w]+) {/g ) ].forEach( match => {
    symbols.push( match[ 1 ] );
  } );
  [ ...str.matchAll( /fn ([\w]+)\(/g ) ].forEach( match => {
    symbols.push( match[ 1 ] );
  } );
  [ ...str.matchAll( /let ([\w]+) = /g ) ].forEach( match => {
    symbols.push( match[ 1 ] );
  } );
  [ ...str.matchAll( /var ([\w]+) = /g ) ].forEach( match => {
    symbols.push( match[ 1 ] );
  } );
  [ ...str.matchAll( /alias ([\w]+) = /g ) ].forEach( match => {
    symbols.push( match[ 1 ] );
  } );
  [ ...str.matchAll( /\s([\w]+):/g ) ].forEach( match => {
    symbols.push( match[ 1 ] );
  } );
};

fs.readdirSync( '../wgsl/shared' ).forEach( filename => {
  if ( filename.endsWith( '.wgsl' ) ) {
    scanSymbols( fs.readFileSync( `../wgsl/shared/${filename}`, 'utf8' ) );
  }
} );
fs.readdirSync( '../wgsl' ).forEach( filename => {
  if ( filename.endsWith( '.wgsl' ) ) {
    scanSymbols( fs.readFileSync( `../wgsl/${filename}`, 'utf8' ) );
  }
} );

symbols = _.uniq( symbols ).filter( symbol => {
  if ( _.some( _.range( 0, 10 ), i => symbol.startsWith( `${i}` ) ) ) {
    return false;
  }

  if ( AVOID_SYMBOLS.includes( symbol ) ) {
    return false;
  }

  // OUR main entry point (NOT general)
  return symbol !== 'main';


} );
const symbolCounts = {};
// Count symbols, and sort by the count. We'll use the count later to remove unused constants!
symbols = _.sortBy( symbols, symbol => {
  const count = [ ...totalStr.matchAll( new RegExp( `[^\\w]${symbol}[^\\w]`, 'g' ) ) ].length;
  symbolCounts[ symbol ] = count;
  return count;
} ).reverse();

const globalAliasesCounts = {};
const globalAliases = _.sortBy( GLOBALLY_ALIASABLE_TYPES.filter( alias => {
  let count = [ ...totalStr.matchAll( new RegExp( `[^\\w]${alias}[^\\w]`, 'g' ) ) ].length;

  // If vec2f, also check vec2<f32>
  const expandedAlias = Object.keys( REPLACEMENT_MAP ).find( before => REPLACEMENT_MAP[ before ] === alias );
  if ( expandedAlias ) {
    count += [ ...totalStr.matchAll( new RegExp( `[^\\w]${expandedAlias}[^\\w]`, 'g' ) ) ].length
  }

  globalAliasesCounts[ alias ] = count;
  // Just anticipate 2 characters per alias (though some might get 1 char?) - we don't want to blow up our preamble
  // with useless things.
  return count * ( alias.length - 2 ) > `alias ${alias}=XX;`.length;
} ), alias => {
  return globalAliasesCounts[ alias ];
} ).reverse();

const combinedSymbolEntries = _.sortBy( [
  ...symbols.map( symbol => ( {
    type: 'symbol',
    name: symbol
  } ) ),
  ...globalAliases.map( alias => ( {
    type: 'alias',
    name: alias
  } ) )
], symbolEntry => {
  return ( symbolEntry.type === 'symbol' ? symbolCounts : globalAliasesCounts )[ symbolEntry.name ];
} ).reverse();

const newSymbols = [];
const newGlobalAliases = [];
const symbolGenerator = generateSymbol();

// TODO: this is a hack, order things correctly
const floatZeroSymbol = symbolGenerator.next().value;
const floatOneSymbol = symbolGenerator.next().value;
const intZeroSymbol = symbolGenerator.next().value;
const intOneSymbol = symbolGenerator.next().value;

for ( let i = 0; i < combinedSymbolEntries.length; i++ ) {
  const entry = combinedSymbolEntries[ i ];
  if ( entry.type === 'symbol' ) {
    newSymbols.push( symbolGenerator.next().value );
  }
  else {
    newGlobalAliases.push( symbolGenerator.next().value );
  }
}

const preamble = MINIFY ? globalAliases.map( ( alias, index ) => {
  return `alias ${newGlobalAliases[ index ]}=${alias};`;
} ).join( '' ) + `const ${floatZeroSymbol}=0.;const ${floatOneSymbol}=1.;const ${intZeroSymbol}=0u;const ${intOneSymbol}=1u;` : '';

symbols.push( ...globalAliases );
newSymbols.push( ...newGlobalAliases );

const minify = str => {
  str = str.replace( /\r\n/g, '\n' );

  // // Naga does not yet recognize `const` but web does not allow global `let`.
  str = str.replace( /\nlet /g, '\nconst ' );

  if ( MINIFY ) {
    // According to WGSL spec:
    // line breaks: \u000A\u000B\u000C\u000D\u0085\u2028\u2029
    // white space: \u0020\u0009\u000A\u000B\u000C\u000D\u0085\u200E\u200F\u2028\u2029

    // TODO: actually use a parser for all of this (make a rust crate for it)
    // TODO: That's actually harder, because we want to parse fragments. THEN we have to actually infer all the types
    // TODO: to determine structure members. Much easier to do this hacky method

    const linebreak = '[\u000A\u000B\u000C\u000D\u0085\u2028\u2029]';
    const whitespace = '[\u0020\u0009\u0085\u200E\u200F\u2028\u2029]'; // don't include most the linebreak ones
    const linebreakOrWhitespace = '[\u000A\u000B\u000C\u000D\u0085\u2028\u2029\u0020\u0009\u0085\u200E\u200F]';

    // Collapse newlines
    str = str.replace( new RegExp( `${whitespace}*${linebreak}+${whitespace}*`, 'g' ), '\n' );
    str = str.trim();

    // Collapse other whitespace
    str = str.replace( new RegExp( `${whitespace}+`, 'g' ), ' ' );

    // Semicolon + newline => semicolon
    str = str.replace( new RegExp( `;${linebreak}`, 'g' ), ';' );

    // Comma + newline => comma
    str = str.replace( new RegExp( `,${linebreak}`, 'g' ), ',' );

    // whitespace around {}
    str = str.replace( new RegExp( `${linebreakOrWhitespace}*([\\{\\}])${linebreakOrWhitespace}*`, 'g' ), ( _, m ) => m );

    // Remove whitespace after :;,
    str = str.replace( new RegExp( `([:;,])${linebreakOrWhitespace}+`, 'g' ), ( _, m ) => m );

    // Remove trailing commas before }])
    str = str.replace( new RegExp( `,([\\}\\]\\)])`, 'g' ), ( _, m ) => m );

    // It's safe to remove whitespace before '-', however Firefox's tokenizer doesn't like 'x-1u' (presumably identifier + literal number, no operator)
    // So we'll only replace whitespace after '-' if it's not followed by a digit
    str = str.replace( new RegExp( `${linebreakOrWhitespace}*-`, 'g' ), '-' );
    str = str.replace( new RegExp( `-${linebreakOrWhitespace}+([^0-9])`, 'g' ), ( _, m ) => `-${m}` );

    // Operators don't need whitespace around them in general
    str = str.replace( new RegExp( `${linebreakOrWhitespace}*([\\+\\*/<>&\\|=\\(\\)!])${linebreakOrWhitespace}*`, 'g' ), ( _, m ) => m );

    // e.g. 0.5 => .5, 10.0 => 10.
    str = str.replace( /\d+\.\d+/g, m => {
      if ( m.endsWith( '.0' ) ) {
        m = m.substring( 0, m.length - 1 );
      }
      if ( m.startsWith( '0.' ) && m.length > 2 ) {
        m = m.substring( 1 );
      }
      return m;
    } );

    // Replace hex literals with decimal literals if they are shorter
    str = str.replace( /0x([0-9abcdefABCDEF]+)u/g, ( m, digits ) => {
      const str = '' + parseInt( digits, 16 ) + 'u';
      if ( str.length < m.length ) {
        return str;
      }
      else {
        return m;
      }
    } );

    // Detect cases where abstract int can be used safely, instead of the explicit ones
    // str = str.replace( /(==|!=)([0-9.])+[uif]/g, ( m, op, digits ) => {
    //   return `${op}${digits}`;
    // } );

    // Replace some predeclared aliases (vec2<f32> => vec2f)
    Object.keys( REPLACEMENT_MAP ).forEach( key => {
      while ( true ) {
        const match = new RegExp( `[^\\w](${key})[^\\w]`, 'g' ).exec( str );

        if ( match ) {
          const index0 = match.index + 1;
          const index1 = index0 + key.length;
          const before = str.substring( 0, index0 );
          const after = str.substring( index1 );
          str = before + REPLACEMENT_MAP[ key ] + after;
        }
        else {
          break;
        }
      }
    } );

    // Replace symbols[ i ] with newSymbols[ i ], but with:
    // 1. No going back, since some newSymbols might be in symbols.
    // 2. Ignore imports
    // 3. Strip out unused constants

    // Regexp that should match any symbol that is to be replaced (with 1char on each side)
    let firstIndex = 0;
    while ( true ) {
      const match = new RegExp( `[^\\w](${symbols.join( '|' )})[^\\w]` ).exec( str.slice( firstIndex ) );
      if ( !match ) {
        break;
      }

      const name = match[ 1 ];
      const index = symbols.indexOf( name );
      if ( index < 0 ) {
        throw new Error( 'bad regexp' );
      }
      const index0 = firstIndex + match.index + 1;
      const index1 = index0 + name.length;
      const before = str.substring( 0, index0 );
      const after = str.substring( index1 );

      // We still have to do import stuff
      if ( !before.endsWith( '#import ' ) ) {
        // Try to strip out unused variables
        const afterMatch = /=\d+u;/.exec( after );

        if ( symbolCounts[ name ] === 1 && before.endsWith( 'const ' ) && afterMatch ) {
          const newBefore = before.slice( 0, before.length - 'const '.length );
          str = newBefore + after.slice( afterMatch[ 0 ].length );
          firstIndex = newBefore.length - 1;
        }
        else {
          const newBefore = before + newSymbols[ index ];
          str = newBefore + after;
          firstIndex = newBefore.length - 1;
        }
      }
      else {
        firstIndex = index1 - 1;
      }
    }

    // Replace some numeric literals with replacement symbols that are shorter(!)
    str = str.replace( /([^0-9a-fA-FxX])0\.([^0-9a-fA-FxXeEfh:pP])/g, ( m, before, after ) => {
      return before + floatZeroSymbol + after;
    } );
    str = str.replace( /([^0-9a-fA-FxX])1\.([^0-9a-fA-FxXeEfh:pP])/g, ( m, before, after ) => {
      return before + floatOneSymbol + after;
    } );
    str = str.replace( /([^0-9a-fA-FxX])0u([^0-9a-fA-FxXeEfh:pP])/g, ( m, before, after ) => {
      return before + intZeroSymbol + after;
    } );
    str = str.replace( /([^0-9a-fA-FxX])1u([^0-9a-fA-FxXeEfh:pP])/g, ( m, before, after ) => {
      return before + intOneSymbol + after;
    } );

    // Remove whitespace around the replacement symbols, since it won't be interpreted as a literal
    [ floatZeroSymbol, floatOneSymbol, intZeroSymbol, intOneSymbol ].forEach( symbol => {
      str = str.replace( new RegExp( `- ${symbol}` ), `-${symbol}` );
    } );
  }

  return str;
};

// does NOT handle imports (we'll need to handle those in a bit)
const preprocess = ( str, defines ) => {

  // sanity check
  str = str.replace( /\r\n/g, '\n' );

  const lines = str.split( '\n' );
  const outputLines = [];
  const stack = [];

  const ifdefString = '#ifdef ';
  const ifndefString = '#ifndef ';
  const elseString = '#else';
  const endifString = '#endif';

  lines.forEach( line => {
    let include = true;

    if ( line.startsWith( ifdefString ) ) {
      stack.push( {
        include: true,
        define: line.substring( ifdefString.length )
      } );
      include = false;
    }
    else if ( line.startsWith( ifndefString ) ) {
      stack.push( {
        include: false,
        define: line.substring( ifndefString.length )
      } );
      include = false;
    }
    else if ( line.startsWith( elseString ) ) {
      const entry = stack.pop();
      stack.push( {
        include: !entry.include,
        define: entry.define
      } );
      include = false;
    }
    else if ( line.startsWith( endifString ) ) {
      stack.pop();
      include = false;
    }

    if ( include && stack.every( entry => entry.include === defines.includes( entry.define ) ) ) {
      outputLines.push( line );
    }
  } );

  return outputLines.join( '\n' );
};

let byteSize = 0;
const outputFile = ( path, importsString, outputString ) => {
  byteSize += outputString.length;
  fs.writeFileSync(
    path,
    `/* eslint-disable */\n${importsString}\nexport default ${outputString}\n`
  );
};
const convert = ( dir, filename, defines = [], outputName ) => {
  outputName = outputName || filename;
  if ( filename.endsWith( '.wgsl' ) ) {
    let shaderString = minify( preprocess( stripComments( fs.readFileSync( `../wgsl/${dir}${filename}`, 'utf8' ) ), defines ) );

    // Replace imports
    let importNames = [];
    // Reversed, so we can do replacement "in place"
    [ ...shaderString.matchAll( /#import [a-z_]+/g ) ].reverse().forEach( match => {
      const index = match.index;
      const importName = match[ 0 ].substring( '#import '.length );
      importNames.push( importName );

      shaderString = shaderString.substring( 0, index ) + `\$\{${importName}\}` + shaderString.substring( index + match[ 0 ].length );
    } );

    // If we are top-level, add the preamble
    if ( !dir ) {
      importNames.push( 'pre' );
      shaderString = '${pre}' + shaderString;
    }

    const outputString = ( '`' + shaderString + '`' ).replace( /`` \\+ /g, '' );

    const importsString = importNames.map( name => `import ${name} from './shared/${name}.js';\n` ).join( '' );

    outputFile( `./${dir}${outputName.replace( '.wgsl', '.ts' )}`, importsString, outputString );
  }
};

// A preamble included in all shaders
outputFile( './shared/pre.ts', '', `\`${preamble}\`` );
convert( 'shared/', 'bbox.wgsl', [ 'full' ] );
convert( 'shared/', 'blend.wgsl', [ 'full' ] );
convert( 'shared/', 'bump.wgsl', [ 'full' ] );
convert( 'shared/', 'clip.wgsl', [ 'full' ] );
convert( 'shared/', 'config.wgsl', [ 'full' ] );
convert( 'shared/', 'cubic.wgsl', [ 'full' ] );
convert( 'shared/', 'drawtag.wgsl', [ 'full' ] );
convert( 'shared/', 'pathtag.wgsl', [ 'full' ] );
convert( 'shared/', 'pathdata_util.wgsl', [ 'full' ] );
convert( 'shared/', 'ptcl.wgsl', [ 'full' ] );
convert( 'shared/', 'segment.wgsl', [ 'full' ] );
convert( 'shared/', 'tile.wgsl', [ 'full' ] );
convert( 'shared/', 'transform.wgsl', [ 'full' ] );
convert( 'shared/', 'util.wgsl', [ 'full' ] );
// convert( '', 'backdrop.wgsl' ); // NOT USED
// convert( '', 'path_coarse.wgsl' ); // NOT USED?
convert( '', 'pathtag_reduce.wgsl', [ 'full' ] );
convert( '', 'pathtag_reduce2.wgsl', [ 'full' ] );
convert( '', 'pathtag_scan1.wgsl', [ 'full' ] );
convert( '', 'pathtag_scan.wgsl', [ 'full', 'small' ], 'pathtag_scan_small.wgsl' );
convert( '', 'pathtag_scan.wgsl', [ 'full' ], 'pathtag_scan_large.wgsl' );
convert( '', 'bbox_clear.wgsl', [] );
convert( '', 'pathseg.wgsl', [ 'full' ] );
convert( '', 'draw_reduce.wgsl', [] );
convert( '', 'draw_leaf.wgsl', [] );
convert( '', 'clip_reduce.wgsl', [] );
convert( '', 'clip_leaf.wgsl', [] );
convert( '', 'binning.wgsl', [] );
convert( '', 'tile_alloc.wgsl', [ 'have_uniform' ] ); // TODO: Firefox doesn't have workgroupUniformLoad
convert( '', 'path_coarse_full.wgsl', [ 'full' ] );
convert( '', 'backdrop_dyn.wgsl', [] );
convert( '', 'coarse.wgsl', [ 'have_uniform' ] ); // TODO: Firefox doesn't have workgroupUniformLoad
convert( '', 'fine.wgsl', [ 'full' ] );

console.log( `bytes: ${byteSize}` );
