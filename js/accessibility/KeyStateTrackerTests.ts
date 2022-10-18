// Copyright 2018-2022, University of Colorado Boulder

/**
 * QUnit tests for the KeyStateTracker.
 *
 * @author Jesse Greenberg
 * @author Michael Barlow
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */


// modules
import stepTimer from '../../../axon/js/stepTimer.js';
import KeyboardUtils from './KeyboardUtils.js';
import KeyStateTracker from './KeyStateTracker.js';

// Reusable KeyboardEvents to update the KeyStateTracker with
// const tabKeyEvent = { key: KeyboardUtils.KEY_TAB };
// const spaceKeyEvent = { key: KeyboardUtils.KEY_SPACE };
// const shiftTabKeyEvent = { key: KeyboardUtils.KEY_TAB, shiftKey: true };
// const shiftKeyEvent = { key: KeyboardUtils.KEY_SHIFT };

const tabKeyDownEvent = new KeyboardEvent( 'keydown', { code: KeyboardUtils.KEY_TAB } );
const tabKeyUpEvent = new KeyboardEvent( 'keyup', { code: KeyboardUtils.KEY_TAB } );
const spaceKeyDownEvent = new KeyboardEvent( 'keydown', { code: KeyboardUtils.KEY_SPACE } );
const spaceKeyUpEvent = new KeyboardEvent( 'keyup', { code: KeyboardUtils.KEY_SPACE } );
const shiftTabKeyDownEvent = new KeyboardEvent( 'keydown', { code: KeyboardUtils.KEY_TAB, shiftKey: true } );
const shiftTabKeyUpEvent = new KeyboardEvent( 'keyup', { code: KeyboardUtils.KEY_TAB, shiftKey: true } );
const shiftKeyLeftDownEvent = new KeyboardEvent( 'keydown', { code: KeyboardUtils.KEY_SHIFT_LEFT } );
const shiftKeyLeftUpEvent = new KeyboardEvent( 'keyup', { code: KeyboardUtils.KEY_SHIFT_LEFT } );

const testTracker = new KeyStateTracker();

let intervalID: number;

QUnit.module( 'KeyStateTracker', {

  before() {

    // step the stepTimer, because utteranceQueue runs on stepTimer
    let previousTime = Date.now();
    intervalID = window.setInterval( () => { // eslint-disable-line bad-sim-text
      const currentTime = Date.now();
      const timeStep = ( currentTime - previousTime ) / 1000; // convert to seconds
      previousTime = currentTime;

      // step timer
      stepTimer.emit( timeStep );
    }, 10 );
  },
  after() {
    testTracker.dispose();
    clearInterval( intervalID );
  },

  beforeEach() {
    testTracker.clearState();
  }
} );

QUnit.test( 'basic state tracking of keys', assert => {

  // mock sending "keydown" events to the tracker
  testTracker[ 'keydownUpdate' ]( tabKeyDownEvent );
  assert.ok( testTracker.isKeyDown( tabKeyDownEvent.code ), 'tab key should be down in tracker' );

  testTracker[ 'keyupUpdate' ]( tabKeyUpEvent );
  assert.ok( !testTracker.isKeyDown( tabKeyUpEvent.code ), 'tab key should be up in tracker' );

  testTracker[ 'keydownUpdate' ]( spaceKeyDownEvent );
  assert.ok( testTracker.isAnyKeyInListDown( [ spaceKeyDownEvent.code, tabKeyDownEvent.code ] ), 'tab or space are down' );
  assert.ok( !testTracker.areKeysDown( [ tabKeyDownEvent.code, spaceKeyDownEvent.code ] ), 'tab and space are not down' );

  testTracker[ 'keydownUpdate' ]( tabKeyDownEvent );
  assert.ok( testTracker.isAnyKeyInListDown( [ tabKeyDownEvent.code, spaceKeyDownEvent.code ] ), 'tab and/or space are down' );
  assert.ok( testTracker.areKeysDown( [ tabKeyDownEvent.code, spaceKeyDownEvent.code ] ), 'tab and space are down' );

  testTracker[ 'keydownUpdate' ]( spaceKeyUpEvent );
} );


QUnit.test( 'tracking of shift key', assert => {

  // mock sending "keydown" events to the tracker
  testTracker[ 'keydownUpdate' ]( shiftTabKeyDownEvent );
  assert.ok( testTracker.shiftKeyDown, 'tab key with shift modifier should produce a keystate with shift key down' );

  testTracker[ 'keydownUpdate' ]( shiftKeyLeftDownEvent );
  testTracker[ 'keydownUpdate' ]( shiftTabKeyDownEvent );
  assert.ok( testTracker.isKeyDown( tabKeyDownEvent.code ), 'tab key should be down in tracker' );
  assert.ok( testTracker.isKeyDown( shiftKeyLeftDownEvent.code ), 'shift key should be down in tracker' );
  assert.ok( testTracker.shiftKeyDown, 'shift key should be down in tracker getter' );

  testTracker[ 'keyupUpdate' ]( shiftKeyLeftUpEvent );
  testTracker[ 'keyupUpdate' ]( tabKeyUpEvent );


  assert.ok( !testTracker.isKeyDown( shiftKeyLeftUpEvent.code ), 'shift key should not be down in tracker' );
  assert.ok( !testTracker.shiftKeyDown, 'shift key should not be down in tracker getter' );
  assert.ok( !testTracker.isKeyDown( tabKeyUpEvent.code ), 'tab key should not be down in tracker' );

  assert.ok( !testTracker.keysAreDown(), 'no keys should be down' );

  testTracker[ 'keydownUpdate' ]( tabKeyDownEvent );
  testTracker[ 'keyupUpdate' ]( shiftTabKeyUpEvent );
  assert.ok( !testTracker.isKeyDown( tabKeyDownEvent.code ), 'tab key should not be down in tracker' );

  // KeyStateTracker should correctly update when modifier keys like "shift" are attached to the event - if shift
  // is down on keyUpUpdate, shift should be considered down
  assert.ok( testTracker.isKeyDown( shiftKeyLeftDownEvent.code ), 'shift key should update from modifier' );
  assert.ok( testTracker.shiftKeyDown, 'shift key should update from modifier getter' );
} );


QUnit.test( 'test tracking with time', async assert => {

  const done = assert.async();

  // we will test holding down a key for these lengths of time in ms
  const firstPressTime = 500;
  const secondPressTime = 71;
  const totalPressTime = firstPressTime + secondPressTime;

  testTracker[ 'keydownUpdate' ]( spaceKeyDownEvent );
  let currentTimeDown = testTracker.timeDownForKey( spaceKeyDownEvent.code );
  assert.ok( currentTimeDown === 0, 'should be zero, has not been down any time' );

  stepTimer.setTimeout( () => {
    currentTimeDown = testTracker.timeDownForKey( spaceKeyDownEvent.code );

    assert.ok( currentTimeDown >= firstPressTime && currentTimeDown <= totalPressTime, `key pressed for ${firstPressTime} ms` );

    stepTimer.setTimeout( () => {
      currentTimeDown = testTracker.timeDownForKey( spaceKeyDownEvent.code );

      assert.ok( currentTimeDown >= totalPressTime, `key pressed for ${secondPressTime} more ms.` );

      testTracker[ 'keyupUpdate' ]( spaceKeyUpEvent );
      done();
    }, secondPressTime );
  }, firstPressTime );
} );


QUnit.test( 'KeyStateTracker.enabled', async assert => {

  const keyStateTracker = new KeyStateTracker();

  keyStateTracker[ 'keydownUpdate' ]( tabKeyDownEvent );

  assert.ok( keyStateTracker.enabled, 'default enabled' );
  assert.ok( keyStateTracker.isKeyDown( KeyboardUtils.KEY_TAB ), 'tab key down' );

  keyStateTracker.enabled = false;


  assert.ok( !keyStateTracker.enabled, 'disabled' );
  assert.ok( !keyStateTracker.isKeyDown( KeyboardUtils.KEY_TAB ), 'tab key down cleared upon disabled' );
  assert.ok( !keyStateTracker.keysAreDown(), 'no keys down' );

  keyStateTracker[ 'keydownUpdate' ]( tabKeyDownEvent );
  assert.ok( !keyStateTracker.isKeyDown( KeyboardUtils.KEY_TAB ), 'tab key not registered when disabled' );

  keyStateTracker[ 'keydownUpdate' ]( shiftTabKeyDownEvent );

  assert.ok( !keyStateTracker.isKeyDown( KeyboardUtils.KEY_SHIFT_LEFT ), 'shift key should not be down' );


  keyStateTracker.enabled = true;

  keyStateTracker[ 'keydownUpdate' ]( shiftTabKeyDownEvent );

  assert.ok( keyStateTracker.isKeyDown( KeyboardUtils.KEY_SHIFT_LEFT ), 'shift key should be down' );
  assert.ok( keyStateTracker.isKeyDown( KeyboardUtils.KEY_TAB ), 'tab key should  be down' );

} );