(function() {
  if ( !window.hasOwnProperty( '_' ) ) {
    throw new Error( 'Underscore/Lodash not found: _' );
  }
  if ( !window.hasOwnProperty( '$' ) ) {
    throw new Error( 'jQuery not found: $' );
  }
