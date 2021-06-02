// Copyright 2018-2021, University of Colorado Boulder

/**
 * FireListener tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Tandem from '../../../tandem/js/Tandem.js';
import FireListener from './FireListener.js';
import ListenerTestUtils from './ListenerTestUtils.js';

QUnit.module( 'FireListener' );

QUnit.test( 'Basics', assert => {
  ListenerTestUtils.simpleRectangleTest( ( display, rect, node ) => {
    let fireCount = 0;
    const listener = new FireListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'myListener' ),
      fire: () => {
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

    listener.dispose();
  } );
} );