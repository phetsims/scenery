// Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';
  
  // add elements to the QUnit fixture for our Scenery-specific tests
  var $fixture = $( '#qunit-fixture' );
  $fixture.append( $( '<div>' ).attr( 'id', 'main' ).attr( 'style', 'position: absolute; left: 0; top: 0; background-color: white; z-index: 1; width: 640px; height: 480px;' ) );
  $fixture.append( $( '<div>' ).attr( 'id', 'secondary' ).attr( 'style', 'position: absolute; left: 0; top: 0; background-color: white; z-index: 0; width: 640px; height: 480px;' ) );
})();
