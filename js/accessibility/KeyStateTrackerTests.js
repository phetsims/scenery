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

  const tabKeyEvent = { keyCode: KeyboardUtil.KEY_TAB };
  const spaceKeyEvent = { keyCode: KeyboardUtil.KEY_SPACE };

  const shiftTabKeyEvent = { keyCode: KeyboardUtil.KEY_TAB, shiftKey: true };
  const shiftKeyEvent = { keyCode: KeyboardUtil.KEY_SHIFT };

  const testTracker = new KeyStateTracker();

  // helper es6 functions from  https://stackoverflow.com/questions/33289726/combination-of-async-function-await-settimeout/33292942
  function timeout( ms ) {
    return new Promise( resolve => setTimeout( resolve, ms ) );
  }

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

    testTracker.keydownUpdate( spaceKeyEvent );
    let currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.keyCode );
    assert.ok( currentTimeDown === 0, 'should be zero, has not been down any time' );

    await timeout( 100 );
    currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.keyCode );
    assert.ok( currentTimeDown >= 99 && currentTimeDown <= 101, 'key pressed for 100ms' );

    await timeout( 51 );
    currentTimeDown = testTracker.timeDownForKey( spaceKeyEvent.keyCode );
    assert.ok( currentTimeDown >= 150 && currentTimeDown <= 152, 'key pressed for 51 more ms.' );

    testTracker.keyupUpdate( spaceKeyEvent.keyCode );
  } );
} );