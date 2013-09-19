
(function(){
  'use strict';
  
  module( 'Scenery: JSHint' );
  
  unitTestLintFilesMatching( function( src ) {
    return src.indexOf( 'scenery/js' ) !== -1;
  } );
})();
