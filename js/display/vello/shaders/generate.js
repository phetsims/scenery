/* eslint-disable */

// Compiles and minifies the WGSL Vello shaders, turning them into ES6 modules
// TODO: better minification with 3rd party library (if/when it exists?) - we rely on assumptions like
// TODO: 'no variables starting with _ appended with one character are used', and don't do a lot of ideal things

/*
 * List of things I've had to patch for the shaders:
 * - fine.wgsl: don't premultiply output. For some reason (even though we're requesting premultiplied canvas texture)
 */

const fs = require( 'fs' );
const _ = require( 'lodash' );

// Enable opting out of minification for debugging
const MINIFY = false;

// go to this directory, then run `node generate.js` to generate the shaders (output is also in this directory)

const stripComments = str => {
  return str.replaceAll( '\r\n', '\n' ).split( '\n' ).map( line => {
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

  // TODO: update to < 2 once we get rid of underscore
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
symbols = _.sortBy( symbols, symbol => {
  return [ ...totalStr.matchAll( new RegExp( symbol, 'g' ) ) ].length;
} ).reverse();
// console.log( JSON.stringify( symbols, null, 2 ) );

// TODO: we have... constants declared that aren't used! Can we just strip them out?

const minify = str => {
  // TODO: could improve minification

  str = str.replaceAll( '\r\n', '\n' );

  // // Naga does not yet recognize `const` but web does not allow global `let`.
  str = str.replaceAll( '\nlet ', '\nconst ' );

  if ( MINIFY ) {
    // According to WGSL spec:
    // line breaks: \u000A\u000B\u000C\u000D\u0085\u2028\u2029
    // white space: \u0020\u0009\u000A\u000B\u000C\u000D\u0085\u200E\u200F\u2028\u2029

    const linebreak = '[\u000A\u000B\u000C\u000D\u0085\u2028\u2029]';
    const whitespace = '[\u0020\u0009\u0085\u200E\u200F\u2028\u2029]';

    // Collapse newlines
    str = str.replaceAll( new RegExp( `${whitespace}*${linebreak}+${whitespace}*`, 'g' ), '\n' );
    str = str.trim();

    // Collapse other whitespace
    str = str.replaceAll( new RegExp( `${whitespace}+`, 'g' ), ' ' );

    // Semicolon + newline => semicolon
    str = str.replaceAll( new RegExp( `;${linebreak}`, 'g' ), ';' );

    // Comma + newline => comma
    str = str.replaceAll( new RegExp( `,${linebreak}`, 'g' ), ',' );

    // newlines around {}
    str = str.replaceAll( new RegExp( `${linebreak}*\u007B${linebreak}*`, 'g' ), '{' );
    str = str.replaceAll( new RegExp( `${linebreak}*\u007D${linebreak}*`, 'g' ), '}' );
    str = str.replaceAll( new RegExp( `${whitespace}*\u007B${whitespace}*`, 'g' ), '{' );
    str = str.replaceAll( new RegExp( `${whitespace}*\u007D${whitespace}*`, 'g' ), '}' );

    str = str.replaceAll( new RegExp( `: `, 'g' ), ':' );
    str = str.replaceAll( new RegExp( `; `, 'g' ), ';' );
    str = str.replaceAll( new RegExp( `, `, 'g' ), ',' );
    str = str.replaceAll( new RegExp( `,}`, 'g' ), '}' );
    str = str.replaceAll( new RegExp( `,]`, 'g' ), ']' );

    str = str.replaceAll( new RegExp( `${whitespace}*\\+${whitespace}*`, 'g' ), '+' );
    str = str.replaceAll( new RegExp( `${whitespace}*-${whitespace}*`, 'g' ), '-' );
    str = str.replaceAll( new RegExp( `${whitespace}*\\*${whitespace}*`, 'g' ), '*' );
    str = str.replaceAll( new RegExp( `${whitespace}*/${whitespace}*`, 'g' ), '/' );
    str = str.replaceAll( new RegExp( `${whitespace}*<${whitespace}*`, 'g' ), '<' );
    str = str.replaceAll( new RegExp( `${whitespace}*>${whitespace}*`, 'g' ), '>' );
    str = str.replaceAll( new RegExp( `${whitespace}*&${whitespace}*`, 'g' ), '&' );
    str = str.replaceAll( new RegExp( `${whitespace}*\\|${whitespace}*`, 'g' ), '|' );
    str = str.replaceAll( new RegExp( `${whitespace}*=${whitespace}*`, 'g' ), '=' );
    str = str.replaceAll( new RegExp( `${whitespace}*\\(${whitespace}*`, 'g' ), '(' );
    str = str.replaceAll( new RegExp( `${whitespace}*\\)${whitespace}*`, 'g' ), ')' );

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
  str = str.replaceAll( '\r\n', '\n' );

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

    const outputString = ( '`' + shaderString + '`' ).replaceAll( '`` + ', '' );
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
