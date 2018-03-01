// Copyright 2017, University of Colorado Boulder

/**
 * PressListener tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var ListenerTestUtils = require( 'SCENERY/listeners/ListenerTestUtils' );
  var PressListener = require( 'SCENERY/listeners/PressListener' );

  QUnit.module( 'PressListener' );

  QUnit.test( 'Basics', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var pressCount = 0;
      var releaseCount = 0;
      var dragCount = 0;
      var listener = new PressListener( {
        press: function( event, listener ) {
          pressCount++;
        },
        release: function( event, listener ) {
          releaseCount++;
        },
        drag: function( event, listener ) {
          dragCount++;
        }
      } );
      rect.addInputListener( listener );

      assert.equal( pressCount, 0, '[1] Has not been pressed yet' );
      assert.equal( releaseCount, 0, '[1] Has not been released yet' );
      assert.equal( dragCount, 0, '[1] Has not been dragged yet' );
      assert.equal( listener.isPressedProperty.value, false, '[1] Is not pressed' );
      assert.equal( listener.isOverProperty.value, false, '[1] Is not over' );
      assert.equal( listener.isHoveringProperty.value, false, '[1] Is not hovering' );
      assert.equal( listener.isHighlightedProperty.value, false, '[1] Is not highlighted' );
      assert.equal( listener.interrupted, false, '[1] Is not interrupted' );

      ListenerTestUtils.mouseMove( display, 10, 10 );

      assert.equal( pressCount, 0, '[2] Has not been pressed yet' );
      assert.equal( releaseCount, 0, '[2] Has not been released yet' );
      assert.equal( dragCount, 0, '[2] Has not been dragged yet' );
      assert.equal( listener.isPressedProperty.value, false, '[2] Is not pressed' );
      assert.equal( listener.isOverProperty.value, true, '[2] Is over' );
      assert.equal( listener.isHoveringProperty.value, true, '[2] Is hovering' );
      assert.equal( listener.isHighlightedProperty.value, true, '[2] Is highlighted' );
      assert.equal( listener.interrupted, false, '[2] Is not interrupted' );

      ListenerTestUtils.mouseDown( display, 10, 10 );

      assert.equal( pressCount, 1, '[3] Pressed once' );
      assert.equal( releaseCount, 0, '[3] Has not been released yet' );
      assert.equal( dragCount, 0, '[3] Has not been dragged yet' );
      assert.equal( listener.isPressedProperty.value, true, '[3] Is pressed' );
      assert.equal( listener.isOverProperty.value, true, '[3] Is over' );
      assert.equal( listener.isHoveringProperty.value, true, '[3] Is hovering' );
      assert.equal( listener.isHighlightedProperty.value, true, '[3] Is highlighted' );
      assert.equal( listener.interrupted, false, '[3] Is not interrupted' );

      assert.ok( listener.pressedTrail.lastNode() === rect, '[3] Dragging the proper rectangle' );

      // A move that goes "outside" the node
      ListenerTestUtils.mouseMove( display, 50, 10 );

      assert.equal( pressCount, 1, '[4] Pressed once' );
      assert.equal( releaseCount, 0, '[4] Has not been released yet' );
      assert.equal( dragCount, 1, '[4] Dragged once' );
      assert.equal( listener.isPressedProperty.value, true, '[4] Is pressed' );
      assert.equal( listener.isOverProperty.value, false, '[4] Is NOT over anymore' );
      assert.equal( listener.isHoveringProperty.value, false, '[4] Is NOT hovering' );
      assert.equal( listener.isHighlightedProperty.value, true, '[4] Is highlighted' );
      assert.equal( listener.interrupted, false, '[4] Is not interrupted' );

      ListenerTestUtils.mouseUp( display, 50, 10 );

      assert.equal( pressCount, 1, '[5] Pressed once' );
      assert.equal( releaseCount, 1, '[5] Released once' );
      assert.equal( dragCount, 1, '[5] Dragged once' );
      assert.equal( listener.isPressedProperty.value, false, '[5] Is NOT pressed' );
      assert.equal( listener.isOverProperty.value, false, '[5] Is NOT over anymore' );
      assert.equal( listener.isHoveringProperty.value, false, '[5] Is NOT hovering' );
      assert.equal( listener.isHighlightedProperty.value, false, '[5] Is NOT highlighted' );
      assert.equal( listener.interrupted, false, '[5] Is not interrupted' );
    } );
  } );

  QUnit.test( 'Interruption', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var listener = new PressListener();
      rect.addInputListener( listener );

      ListenerTestUtils.mouseDown( display, 10, 10 );

      assert.equal( listener.isPressedProperty.value, true, 'Is pressed before the interruption' );
      listener.interrupt();
      assert.equal( listener.isPressedProperty.value, false, 'Is NOT pressed after the interruption' );
    } );
  } );
} );
