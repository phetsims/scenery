

(function(){
  'use strict';
  
  module( 'Scenery: JSHint' );
  
  var filenames = _.filter( $( 'head script' ).map( function( i, script ) { return script.src; } ), function( src ) {
    return src.indexOf( 'scenery/js' ) !== -1 ||
           src.indexOf( 'scenery/common/kite/js' ) !== -1 ||
           src.indexOf( 'scenery/common/dot/js' ) !== -1 ||
           src.indexOf( 'scenery/common/phet-core/js' ) !== -1 ||
           src.indexOf( 'scenery/common/assert/js' ) !== -1;
  } );
  
  var options = window.jshintOptions;
  var globals = options.globals;
  delete options.globals; // it's technically an invalid option when passed to qHint
  
  _.each( filenames, function( filename ) {
    var name = filename.slice( filename.lastIndexOf( '/' ) + 1, filename.indexOf( '?' ) );
    var lib = filename.slice( 0, filename.lastIndexOf( '/js/' ) );
    lib = lib.slice( lib.lastIndexOf( '/' ) + 1 );
    qHint( lib + ': ' + name, filename, options, globals );
  } );
  
})();



