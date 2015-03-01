// Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: JSHint' );

  unitTestLintFilesMatching( function( src ) {
    return src.indexOf( 'scenery/js' ) !== -1;
  } );
})();
