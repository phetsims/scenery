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

  // mock scenery Event, type has a domEvent field for the dom event we
  const tabKeyEvent = { domEvent: { keyCode: KeyboardUtil.KEY_TAB } };
  const spaceKeyEvent = { domEvent: { keyCode: KeyboardUtil.KEY_SPACE } };
  const shiftTabKeyEvent = { domEvent: { keyCode: KeyboardUtil.KEY_TAB, shiftKey: true } };
  const shiftKeyEvent = { domEvent: { keyCode: KeyboardUtil.KEY_SHIFT } };

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
    assert.ok( testTracker.isKeyDown( tabKeyEvent.domEvent.keyCode ), 'tab key should be down in tracker' );

    testTracker.keyupUpdate( tabKeyEvent );
    assert.ok( !testTracker.isKeyDown( tabKeyEvent.domEvent.keyCode ), 'tab key should be up in tracker' );

    testTracker.keydownUpdate( spaceKeyEvent );
    assert.ok( testTracker.isAnyKeyInListDown( [ tabKeyEvent.domEvent.keyCode, spaceKeyEvent.domEvent.keyCode ] ), 'tab or space are down' );
    assert.ok( !testTracker.areKeysDown( [ tabKeyEvent.domEvent.keyCode, spaceKeyEvent.domEvent.keyCode ] ), 'tab and space are not down' );

    testTracker.keydownUpdate( tabKeyEvent );
    assert.ok( testTracker.isAnyKeyInListDown( [ tabKeyEvent.domEvent.keyCode, spaceKeyEvent.domEvent.keyCode ] ), 'tab and/or space are down' );
    assert.ok( testTracker.areKeysDown( [ tabKeyEvent.domEvent.keyCode, spaceKeyEvent.domEvent.keyCode ] ), 'tab and space are down' );

    testTracker.keydownUpdate( spaceKeyEvent );
  } );


  QUnit.test( 'tracking of shift key', assert => {

    // mock sending "keydown" events to the tracker
    testTracker.keydownUpdate( shiftTabKeyEvent );
    assert.ok( testTracker.shiftKeyDown, 'tab key with shift modifier should produce a keystate with shift key down' );

    testTracker.keydownUpdate( shiftKeyEvent );
    testTracker.keydownUpdate( shiftTabKeyEvent );
    assert.ok( testTracker.isKeyDown( tabKeyEvent.domEvent.keyCode ), 'tab key should be down in tracker' );
    assert.ok( testTracker.isKeyDown( shiftKeyEvent.domEvent.keyCode ), 'shift key should be down in tracker' );
    assert.ok( testTracker.shiftKeyDown, 'shift key should be down in tracker getter' );

    testTracker.keyupUpdate( shiftKeyEvent );
    testTracker.keyupUpdate( tabKeyEvent );


    assert.ok( !testTracker.isKeyDown( shiftKeyEvent.domEvent.keyCode ), 'shift key should not be down in tracker' );
    assert.ok( !testTracker.shiftKeyDown, 'shift key should not be down in tracker getter' );
    assert.ok( !testTracker.isKeyDown( tabKeyEvent.domEvent.keyCode ), 'tab key should not be down in tracker' );

    assert.ok( !testTracker.keysAreDown(), 'no keys should be down' );

    testTracker.keydownUpdate( tabKeyEvent );
    testTracker.keyupUpdate( shiftTabKeyEvent );
    assert.ok( !testTracker.isKeyDown( tabKeyEvent.domEvent.keyCode ), 'tab key should not be down in tracker' );

    // This test originally came before a global key state tracker, when changing focus might begin to update a new
    // key state tracker and we need to make sure the new tracker was initialized with the state of the previous
    // tracker. With a global tracker, this isn't necessary.
    assert.ok( !testTracker.isKeyDown( shiftKeyEvent.domEvent.keyCode ), 'shift key should not be down in tracker just because of tab up with shift key code' );
    assert.ok( !testTracker.shiftKeyDown, 'shift key should not be down in tracker getter' );
  } );


  QUnit.test( 'test tracking with time', async assert => {

    const done = assert.async();

    let previousTime = Date.now();
    const intervalID = window.setInterval( () => {
      const currentTime = Date.now();
      testTracker.step( ( currentTime - previousTime ) / 1000 ); // convert to seconds
      previousTime = currentTime;
    }, 10 );

    testTracker.keydownUpdate( spaceKeyEvent );
    let currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.domEvent.keyCode );
    assert.ok( currentTimeDown === 0, 'should be zero, has not been down any time' );

    timer.setTimeout( () => {
      currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.domEvent.keyCode );


      assert.ok( currentTimeDown >= 480 && currentTimeDown <= 505, 'key pressed for 100ms' );

      timer.setTimeout( () => {
        currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.domEvent.keyCode );

        assert.ok( currentTimeDown >= 560 && currentTimeDown <= 590, 'key pressed for 51 more ms.' );

        testTracker.keyupUpdate( spaceKeyEvent );

        window.clearInterval( intervalID );
        done();
      }, 71 );
    }, 500 );
  } );
} );