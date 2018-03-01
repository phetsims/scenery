// Copyright 2017, University of Colorado Boulder

/**
 * DragListener tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var DragListener = require( 'SCENERY/listeners/DragListener' );
  var ListenerTestUtils = require( 'SCENERY/listeners/ListenerTestUtils' );

  QUnit.module( 'DragListener' );

  QUnit.test( 'translateNode', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var listener = new DragListener( {
        translateNode: true
      } );
      rect.addInputListener( listener );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      ListenerTestUtils.mouseMove( display, 20, 15 );
      ListenerTestUtils.mouseUp( display, 20, 15 );
      assert.equal( rect.x, 10, 'Drag with translateNode should have changed the x translation' );
      assert.equal( rect.y, 5, 'Drag with translateNode should have changed the y translation' );
    } );
  } );
} );
