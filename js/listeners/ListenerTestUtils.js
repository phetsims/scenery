// Copyright 2017, University of Colorado Boulder

/**
 * Utilities for listener tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var Display = require( 'SCENERY/display/Display' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Vector2 = require( 'DOT/Vector2' );

  return {
    /**
     * Sends a synthesized mouseDown event at the given coordinates.
     * @public
     *
     * @param {Display} display
     * @param {number} x
     * @param {number} y
     */
    mouseDown: function( display, x, y ) {
      var domEvent = document.createEvent( 'MouseEvent' );

      // technically deprecated, but DOM4 event constructors not out yet. people on #whatwg said to use it
      domEvent.initMouseEvent( 'mousedown', true, true, window, 1, // click count
        x, y, x, y,
        false, false, false, false,
        0, // button
        null );

      display._input.validatePointers();
      display._input.mouseDown( new Vector2( x, y ), domEvent );
    },

    /**
     * Sends a synthesized mouseUp event at the given coordinates.
     * @public
     *
     * @param {Display} display
     * @param {number} x
     * @param {number} y
     */
    mouseUp: function( display, x, y ) {
      var domEvent = document.createEvent( 'MouseEvent' );

      // technically deprecated, but DOM4 event constructors not out yet. people on #whatwg said to use it
      domEvent.initMouseEvent( 'mouseup', true, true, window, 1, // click count
        x, y, x, y,
        false, false, false, false,
        0, // button
        null );

      display._input.validatePointers();
      display._input.mouseUp( new Vector2( x, y ), domEvent );
    },

    /**
     * Sends a synthesized mouseMove event at the given coordinates.
     * @public
     *
     * @param {Display} display
     * @param {number} x
     * @param {number} y
     */
    mouseMove: function( display, x, y ) {
      var domEvent = document.createEvent( 'MouseEvent' );

      // technically deprecated, but DOM4 event constructors not out yet. people on #whatwg said to use it
      domEvent.initMouseEvent( 'mousemove', true, true, window, 0, // click count
        x, y, x, y,
        false, false, false, false,
        0, // button
        null );


      display._input.validatePointers();
      display._input.mouseMove( new Vector2( x, y ), domEvent );
    },

    /**
     * Runs a simple test with a 20x20 rectangle in a 640x480 display.
     * @public
     *
     * @param {Function} callback - Called with callback( {Display}, {Node}, {Node} ) - First node is the draggable rect
     */
    simpleRectangleTest: function( callback ) {
      var node = new Node();
      var display = new Display( node, { width: 640, height: 480 } );
      display.initializeEvents();
      display.updateDisplay();

      var rect = new Rectangle( 0, 0, 20, 20, { fill: 'red' } );
      node.addChild( rect );

      callback( display, rect, node );

      // Cleanup, so we don't leak listeners/memory
      display.dispose();
    }
  };
} );
