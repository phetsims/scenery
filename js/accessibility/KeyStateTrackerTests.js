// Copyright 2017, University of Colorado Boulder

/**
 * QUnit tests for the KeyStateTracker.
 *
 * @author Jesse Greenberg
 * @author Michael Barlow
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  // const KeyboardFuzzer = require( 'SCENERY/accessibility/KeyboardFuzzer' ); // TODO: can we use this in testing? https://github.com/phetsims/scenery/issues/850
  const KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  const KeyStateTracker = require( 'SCENERY/accessibility/KeyStateTracker' );
  const timer = require( 'PHET_CORE/timer' );

  // mock DOM KeyboardEvents
  const tabKeyEvent = { keyCode: KeyboardUtil.KEY_TAB };
  const spaceKeyEvent = { keyCode: KeyboardUtil.KEY_SPACE };
  const shiftTabKeyEvent = { keyCode: KeyboardUtil.KEY_TAB, shiftKey: true };
  const shiftKeyEvent = { keyCode: KeyboardUtil.KEY_SHIFT };

  const testTracker = new KeyStateTracker();

  let intervalID = null;

  QUnit.module( 'KeyStateTracker', {

    before() {

      // step the timer, because utteranceQueue runs on timer
      intervalID = setInterval( () => {
        timer.emit1( 1 / 100 ); // step timer in seconds, every millisecond
      }, 10 );

    },
    after() {
      clearInterval( intervalID );
    },

    beforeEach() {
      testTracker.clearState();
    }
  } );

  QUnit.test( 'basic state tracking of keys', assert => {

    // mock sending "keydown" events to the tracker
    testTracker.keydownUpdate( tabKeyEvent );
    assert.ok( testTracker.isKeyDown( tabKeyEvent.keyCode ), 'tab key should be down in tracker' );

    testTracker.keyupUpdate( tabKeyEvent );
    assert.ok( !testTracker.isKeyDown( tabKeyEvent.keyCode ), 'tab key should be up in tracker' );

    testTracker.keydownUpdate( spaceKeyEvent );
    assert.ok( testTracker.isAnyKeyInListDown( [ tabKeyEvent.keyCode, spaceKeyEvent.keyCode ] ), 'tab or space are down' );
    assert.ok( !testTracker.areKeysDown( [ tabKeyEvent.keyCode, spaceKeyEvent.keyCode ] ), 'tab and space are not down' );

    testTracker.keydownUpdate( tabKeyEvent );
    assert.ok( testTracker.isAnyKeyInListDown( [ tabKeyEvent.keyCode, spaceKeyEvent.keyCode ] ), 'tab and/or space are down' );
    assert.ok( testTracker.areKeysDown( [ tabKeyEvent.keyCode, spaceKeyEvent.keyCode ] ), 'tab and space are down' );

    testTracker.keydownUpdate( spaceKeyEvent );
  } );


  QUnit.test( 'tracking of shift key', assert => {

    // mock sending "keydown" events to the tracker
    window.assert && assert.throws( () => {
      testTracker.keydownUpdate( shiftTabKeyEvent );

    }, 'event has shift key down when key state tracker does did not get a shift down key event.' );


    testTracker.keydownUpdate( shiftKeyEvent );
    testTracker.keydownUpdate( shiftTabKeyEvent );
    assert.ok( testTracker.isKeyDown( tabKeyEvent.keyCode ), 'tab key should be down in tracker' );
    assert.ok( testTracker.isKeyDown( shiftKeyEvent.keyCode ), 'shift key should be down in tracker' );
    assert.ok( testTracker.shiftKeyDown, 'shift key should be down in tracker getter' );

    testTracker.keyupUpdate( shiftKeyEvent );
    testTracker.keyupUpdate( tabKeyEvent );


    assert.ok( !testTracker.isKeyDown( shiftKeyEvent.keyCode ), 'shift key should not be down in tracker' );
    assert.ok( !testTracker.shiftKeyDown, 'shift key should not be down in tracker getter' );
    assert.ok( !testTracker.isKeyDown( tabKeyEvent.keyCode ), 'tab key should not be down in tracker' );

    assert.ok( !testTracker.keysAreDown(), 'no keys should be down' );

    testTracker.keydownUpdate( tabKeyEvent );
    testTracker.keyupUpdate( shiftTabKeyEvent );
    assert.ok( !testTracker.isKeyDown( tabKeyEvent.keyCode ), 'tab key should not be down in tracker' );
    assert.ok( testTracker.isKeyDown( shiftKeyEvent.keyCode ), 'shift key should have been set to be down in tracker because of tab up' );
    assert.ok( testTracker.shiftKeyDown, 'shift key should be down in tracker getter' );

    testTracker.keyupUpdate( shiftKeyEvent );
    testTracker.keyupUpdate( tabKeyEvent );
  } );


  QUnit.test( 'test tracking with time', async assert => {

    var done = assert.async();

    testTracker.keydownUpdate( spaceKeyEvent );
    let currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.keyCode );
    assert.ok( currentTimeDown === 0, 'should be zero, has not been down any time' );


    timer.setTimeout( () => {
      currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.keyCode );

      assert.ok( currentTimeDown >= 95 && currentTimeDown <= 115, 'key pressed for 100ms' );

      timer.setTimeout( () => {
        currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.keyCode );
        assert.ok( currentTimeDown >= 146 && currentTimeDown <= 170, 'key pressed for 51 more ms.' );

        testTracker.keyupUpdate( spaceKeyEvent.keyCode );

        done();
      }, 51 );
    }, 100 );
  } );
} );