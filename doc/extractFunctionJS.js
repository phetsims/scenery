// Copyright 2022, University of Colorado Boulder

window.extractFunctionJS = func => {
  const functionString = func.toString();
  let js = functionString.match( /\/\*START\*\/((.|\n)*)\/\*END\*\// )[ 1 ];
  let jsBefore = '';
  let jsAfter = '';
  if ( js.length !== functionString.length ) {
    jsBefore = functionString.slice( 0, functionString.indexOf( js ) );
    jsAfter = functionString.slice( functionString.indexOf( js ) + js.length );
  }
  let lines = js.split( '\n' );
  let minPadding = Number.POSITIVE_INFINITY;
  lines = lines.map( line => {
    const index = line.search( /[^ ]/ );
    if ( index === -1 ) {
      return '';
    }
    else {
      minPadding = Math.min( minPadding, index );
      return line;
    }
  } );
  while ( lines[ 0 ].length === 0 ) {
    lines.shift();
  }
  while ( lines[ lines.length - 1 ].length === 0 ) {
    lines.pop();
  }
  lines = lines.map( line => line.slice( minPadding ) );
  js = lines.join( '\n' );
  return {
    js: js,
    jsBefore: jsBefore,
    jsAfter: jsAfter
  };
};
