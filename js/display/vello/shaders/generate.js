/* eslint-disable */

// Compiles and minifies the WGSL Vello shaders, turning them into ES6 modules
// TODO: better minification with 3rd party library (if/when it exists?) - we rely on assumptions like
// TODO: 'no variables starting with _ appended with one character are used', and don't do a lot of ideal things

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
 */

const fs = require( 'fs' );
const _ = require( 'lodash' );

// Enable opting out of minification for debugging
const MINIFY = true;

// go to this directory, then run `node generate.js` to generate the shaders (output is also in this directory)

const stripComments = str => {
  return str.replace( /\r\n/g, '\n' ).split( '\n' ).map( line => {
    const index = line.indexOf( '//' );
    return index >= 0 ? line.substring( 0, index ) : line;
  } ).join( '\n' );
};

let symbols = [];
let totalStr = '';
const scanSymbols = str => {
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

  // Don't kill swizzles
  if ( _.every( symbol.split( '' ), c => c === 'x' || c === 'y' || c === 'z' || c === 'w' ) ) {
    return false;
  }
  if ( _.every( symbol.split( '' ), c => c === 'r' || c === 'g' || c === 'b' || c === 'a' ) ) {
    return false;
  }

  // TODO: update to < 2 once we get rid of underscore (not worth changing symbols, and we assume there aren't existing
  // TODO: ones starting with underscores)
  if ( symbol.length < 3 ) {
    return false;
  }

  return ![
    'main',
    'default',
    'group',
    'binding',
    'compute',
    'workgroup_size',
    'builtin',
    'local_invocation_id',
    'global_invocation_id',
    'workgroup_id',
    'storage',
    'uniform',
    'read_write',
    'array',
    'atomic',
    'f32',
    'u32',
    'u8',
    'workgroup'
  ].includes( symbol );
} );
const symbolCounts = {};
// Count symbols, and sort by the count. We'll use the count later to remove unused constants!
symbols = _.sortBy( symbols, symbol => {
  const count = [ ...totalStr.matchAll( new RegExp( symbol, 'g' ) ) ].length;
  symbolCounts[ symbol ] = count;
  return count;
} ).reverse();
// console.log( JSON.stringify( symbols, null, 2 ) );

// TODO: we have... constants declared that aren't used! Can we just strip them out? Presumably yes

const minify = str => {
  // TODO: could improve minification significantly

  str = str.replace( /\r\n/g, '\n' );

  // // Naga does not yet recognize `const` but web does not allow global `let`.
  str = str.replace( /\nlet /g, '\nconst ' );

  if ( MINIFY ) {
    // According to WGSL spec:
    // line breaks: \u000A\u000B\u000C\u000D\u0085\u2028\u2029
    // white space: \u0020\u0009\u000A\u000B\u000C\u000D\u0085\u200E\u200F\u2028\u2029

    // TODO: actually use a parser for all of this (make a rust crate for it)

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

    // Remove trailing commas before }]
    str = str.replace( new RegExp( `,([\\}\\]])`, 'g' ), ( _, m ) => m );

    // It's safe to remove whitespace before '-', however Firefox's tokenizer doesn't like 'x-1u' (presumably identifier + literal number, no operator)
    // So we'll only replace whitespace after '-' if it's not followed by a digit
    str = str.replace( new RegExp( `${linebreakOrWhitespace}*-`, 'g' ), '-' );
    str = str.replace( new RegExp( `-${linebreakOrWhitespace}+([^0-9])`, 'g' ), ( _, m ) => `-${m}` );

    // Operators don't need whitespace around them in general
    str = str.replace( new RegExp( `${linebreakOrWhitespace}*([\\+\\*/<>&\\|=\\(\\)!])${linebreakOrWhitespace}*`, 'g' ), ( _, m ) => m );

    str = str.replace( /\d+\.\d+/g, m => {
      if ( m.endsWith( '.0' ) ) {
        m = m.substring( 0, m.length - 1 );
      }
      if ( m.startsWith( '0.' ) && m.length > 2 ) {
        m = m.substring( 1 );
      }
      return m;
    } );

    symbols.forEach( ( name, index ) => {
      // TODO: OMG why do we need to iterate this? Something is WRONG with this code. See below
      for ( let i = 0; i < 2; i++ ) {
        [ ...str.matchAll( new RegExp( `[^\\w](${name})[^\\w]`, 'g' ) ) ].reverse().forEach( match => {
          const index0 = match.index + 1;
          const index1 = index0 + match[ 1 ].length;
          const before = str.substring( 0, index0 );
          const after = str.substring( index1 );

          // We still have to do import stuff
          if ( !before.endsWith( '#import ' ) ) {
            // Try to strip out unused variables
            const afterMatch = /=\d+u;/.exec( after );
            if ( symbolCounts[ name ] === 1 && before.endsWith( 'const ' ) && afterMatch ) {
              str = before.slice( 0, before.length - 'const '.length ) + after.slice( afterMatch[ 0 ].length );
            }
            else {
              const alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';

              let variable;
              if ( index < alphabet.length ) {
                variable = alphabet[ index ];
              }
              else {
                variable = alphabet[ Math.floor( index / alphabet.length ) - 1 ] + alphabet[ index % alphabet.length ];
              }
              // TODO: figure out how we could get rid of this underscore. What are we hitting?
              variable = `_${variable}`;
              str = before + variable + after;
            }
          }
        } );
      }
      // THIS is why the loop above is needed
      // if ( name === 'linewidth' && str.includes( 'linewidth' ) ) {
      //   console.log( 'EEEEEK' );
      //   console.log( [ ...str.matchAll( new RegExp( `.linewidth.`, 'g' ) ) ].map( match => match[ 0 ] ) );
      //   console.log( [ ...str.matchAll( new RegExp( `[^\\w](${name})[^\\w]`, 'g' ) ) ].map( match => match[ 0 ] ) );
      // }
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
const convert = ( dir, filename, defines = [], outputName ) => {
  outputName = outputName || filename;
  if ( filename.endsWith( '.wgsl' ) ) {
    let shaderString = minify( preprocess( stripComments( fs.readFileSync( `../wgsl/${dir}${filename}`, 'utf8' ) ), defines ) );

    // Replace imports
    let importNames = [];
    // Reversed, so we can do replacement "in place"
    [ ...shaderString.matchAll( /#import [a-z]+/g ) ].reverse().forEach( match => {
      const index = match.index;
      const importName = match[ 0 ].substring( '#import '.length );
      importNames.push( importName );

      shaderString = shaderString.substring( 0, index ) + `\$\{${importName}\}` + shaderString.substring( index + match[ 0 ].length );
    } );

    const outputString = ( '`' + shaderString + '`' ).replace( /`` \\+ /g, '' );
    byteSize += outputString.length;

    const importsString = importNames.map( name => `import ${name} from './shared/${name}.js';\n` ).join( '' );

    fs.writeFileSync(
      `./${dir}${outputName.replace( '.wgsl', '.ts' )}`,
      `/* eslint-disable */\n${importsString}\nexport default ${outputString}\n`
    );
  }
};

convert( 'shared/', 'bbox.wgsl', [ 'full' ] );
convert( 'shared/', 'blend.wgsl', [ 'full' ] );
convert( 'shared/', 'bump.wgsl', [ 'full' ] );
convert( 'shared/', 'clip.wgsl', [ 'full' ] );
convert( 'shared/', 'config.wgsl', [ 'full' ] );
convert( 'shared/', 'cubic.wgsl', [ 'full' ] );
convert( 'shared/', 'drawtag.wgsl', [ 'full' ] );
convert( 'shared/', 'pathtag.wgsl', [ 'full' ] );
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
