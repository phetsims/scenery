/* eslint-disable */

const fs = require( 'fs' );

// go to this directory, then run `node generate.js` to generate the shaders (output is also in this directory)

const stripComments = str => {
  return str.replaceAll( '\r\n', '\n' ).split( '\n' ).map( line => {
    const index = line.indexOf( '//' );
    return index >= 0 ? line.substring( 0, index ) : line;
  } ).join( '\n' );
};

const minify = str => {
  // TODO: could improve minification

  str = str.replaceAll( '\r\n', '\n' );

  // // Naga does not yet recognize `const` but web does not allow global `let`.
  str = str.replaceAll( '\nlet ', '\nconst ' );

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
  str = str.replaceAll( new RegExp( `, `, 'g' ), ',' );

  str = str.replaceAll( new RegExp( `${whitespace}*\\+${whitespace}*`, 'g' ), '+' );
  str = str.replaceAll( new RegExp( `${whitespace}*-${whitespace}*`, 'g' ), '-' );
  str = str.replaceAll( new RegExp( `${whitespace}*\\*${whitespace}*`, 'g' ), '*' );
  str = str.replaceAll( new RegExp( `${whitespace}*/${whitespace}*`, 'g' ), '/' );
  str = str.replaceAll( new RegExp( `${whitespace}*<${whitespace}*`, 'g' ), '<' );
  str = str.replaceAll( new RegExp( `${whitespace}*>${whitespace}*`, 'g' ), '>' );
  str = str.replaceAll( new RegExp( `${whitespace}*&${whitespace}*`, 'g' ), '&' );
  str = str.replaceAll( new RegExp( `${whitespace}*\\|${whitespace}*`, 'g' ), '|' );
  str = str.replaceAll( new RegExp( `${whitespace}*=${whitespace}*`, 'g' ), '=' );

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

const convert = ( dir, filename, defines = [ 'full' ], outputName ) => {
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

    const importsString = importNames.map( name => `import ${name} from './shared/${name}.js';\n` ).join( '' );

    fs.writeFileSync(
      `./${dir}${outputName.replace( '.wgsl', '.ts' )}`,
      `/* eslint-disable */\n${importsString}\nexport default ${outputString}\n`
    );
  }
};

convert( 'shared/', 'bbox.wgsl' );
convert( 'shared/', 'blend.wgsl' );
convert( 'shared/', 'bump.wgsl' );
convert( 'shared/', 'clip.wgsl' );
convert( 'shared/', 'config.wgsl' );
convert( 'shared/', 'cubic.wgsl' );
convert( 'shared/', 'drawtag.wgsl' );
convert( 'shared/', 'pathtag.wgsl' );
convert( 'shared/', 'ptcl.wgsl' );
convert( 'shared/', 'segment.wgsl' );
convert( 'shared/', 'tile.wgsl' );
convert( 'shared/', 'transform.wgsl' );
convert( '', 'backdrop.wgsl' );
convert( '', 'backdrop_dyn.wgsl' );
convert( '', 'bbox_clear.wgsl' );
convert( '', 'binning.wgsl' );
convert( '', 'clip_leaf.wgsl' );
convert( '', 'clip_reduce.wgsl' );
convert( '', 'coarse.wgsl' );
convert( '', 'draw_leaf.wgsl' );
convert( '', 'draw_reduce.wgsl' );
convert( '', 'fine.wgsl' );
convert( '', 'path_coarse.wgsl' );
convert( '', 'path_coarse_full.wgsl' );
convert( '', 'pathseg.wgsl' );
convert( '', 'pathtag_reduce.wgsl' );
convert( '', 'pathtag_reduce2.wgsl' );
convert( '', 'pathtag_scan.wgsl', [ 'full' ], 'pathtag_scan_large.wgsl' );
convert( '', 'pathtag_scan.wgsl', [ 'full', 'small' ], 'pathtag_scan_small.wgsl' );
convert( '', 'pathtag_scan1.wgsl' );
convert( '', 'tile_alloc.wgsl' );
