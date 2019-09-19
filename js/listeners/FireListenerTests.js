// Copyright 2018-2019, University of Colorado Boulder

/**
 * FireListener tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( require => {
  'use strict';

  // modules
  const FireListener = require( 'SCENERY/listeners/FireListener' );
  const ListenerTestUtils = require( 'SCENERY/listeners/ListenerTestUtils' );

  QUnit.module( 'FireListener' );

  QUnit.test( 'Basics', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      let fireCount = 0;
      const listener = new FireListener( {
        fire: function() {
          fireCount++;
        }
      } );
      rect.addInputListener( listener );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      assert.equal( fireCount, 0, 'Not yet fired on move' );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      assert.equal( fireCount, 0, 'Not yet fired on initial press' );
      ListenerTestUtils.mouseUp( display, 10, 10 );
      assert.equal( fireCount, 1, 'It fired on release' );

      ListenerTestUtils.mouseMove( display, 50, 10 );
      ListenerTestUtils.mouseDown( display, 50, 10 );
      ListenerTestUtils.mouseUp( display, 50, 10 );
      assert.equal( fireCount, 1, 'Should not fire when the mouse totally misses' );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      ListenerTestUtils.mouseMove( display, 50, 10 );
      ListenerTestUtils.mouseUp( display, 50, 10 );
      assert.equal( fireCount, 1, 'Should NOT fire when pressed and then moved away' );

      ListenerTestUtils.mouseMove( display, 50, 10 );
      ListenerTestUtils.mouseDown( display, 50, 10 );
      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseUp( display, 10, 10 );
      assert.equal( fireCount, 1, 'Should NOT fire when the press misses (even if the release is over)' );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      listener.interrupt();
      ListenerTestUtils.mouseUp( display, 10, 10 );
      assert.equal( fireCount, 1, 'Should NOT fire on an interruption' );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      ListenerTestUtils.mouseMove( display, 50, 10 );
      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseUp( display, 10, 10 );
      assert.equal( fireCount, 2, 'Should fire if the mouse is moved away after press (but moved back before release)' );
    } );
  } );
} );
