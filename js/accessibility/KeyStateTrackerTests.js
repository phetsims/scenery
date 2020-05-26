// Copyright 2018-2020, University of Colorado Boulder

/**
 * QUnit tests for the KeyStateTracker.
 *
 * @author Jesse Greenberg
 * @author Michael Barlow
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */


// modules
import timer from '../../../axon/js/timer.js';
import KeyboardUtils from './KeyboardUtils.js';
import KeyStateTracker from './KeyStateTracker.js';

// Reusable KeyboardEvents to update the KeyStateTracker with
// const tabKeyEvent = { keyCode: KeyboardUtils.KEY_TAB };
// const spaceKeyEvent = { keyCode: KeyboardUtils.KEY_SPACE };
// const shiftTabKeyEvent = { keyCode: KeyboardUtils.KEY_TAB, shiftKey: true };
// const shiftKeyEvent = { keyCode: KeyboardUtils.KEY_SHIFT };

const tabKeyDownEvent = new KeyboardEvent( 'keydown', { keyCode: KeyboardUtils.KEY_TAB } );
const tabKeyUpEvent = new KeyboardEvent( 'keyup', { keyCode: KeyboardUtils.KEY_TAB } );
const spaceKeyDownEvent = new KeyboardEvent( 'keydown', { keyCode: KeyboardUtils.KEY_SPACE } );
const spaceKeyUpEvent = new KeyboardEvent( 'keyup', { keyCode: KeyboardUtils.KEY_SPACE } );
const shiftTabKeyDownEvent = new KeyboardEvent( 'keydown', { keyCode: KeyboardUtils.KEY_TAB, shiftKey: true } );
const shiftTabKeyUpEvent = new KeyboardEvent( 'keyup', { keyCode: KeyboardUtils.KEY_TAB, shiftKey: true } );
const shiftKeyDownEvent = new KeyboardEvent( 'keydown', { keyCode: KeyboardUtils.KEY_SHIFT } );
const shiftKeyUpEvent = new KeyboardEvent( 'keyup', { keyCode: KeyboardUtils.KEY_SHIFT } );

const testTracker = new KeyStateTracker();

let intervalID = null;

QUnit.module( 'KeyStateTracker', {

  before() {

    // step the timer, because utteranceQueue runs on timer
    let previousTime = Date.now();
    intervalID = setInterval( () => { // eslint-disable-line bad-sim-text
      const currentTime = Date.now();
      const timeStep = ( currentTime - previousTime ) / 1000; // convert to seconds
      previousTime = currentTime;

      // step timer
      timer.emit( timeStep );
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
  testTracker.keydownUpdate( tabKeyDownEvent );
  assert.ok( testTracker.isKeyDown( tabKeyDownEvent.keyCode ), 'tab key should be down in tracker' );

  testTracker.keyupUpdate( tabKeyUpEvent );
  assert.ok( !testTracker.isKeyDown( tabKeyUpEvent.keyCode ), 'tab key should be up in tracker' );

  testTracker.keydownUpdate( spaceKeyDownEvent );
  assert.ok( testTracker.isAnyKeyInListDown( [ spaceKeyDownEvent.keyCode, tabKeyDownEvent.keyCode ] ), 'tab or space are down' );
  assert.ok( !testTracker.areKeysDown( [ tabKeyDownEvent.keyCode, spaceKeyDownEvent.keyCode ] ), 'tab and space are not down' );

  testTracker.keydownUpdate( tabKeyDownEvent );
  assert.ok( testTracker.isAnyKeyInListDown( [ tabKeyDownEvent.keyCode, spaceKeyDownEvent.keyCode ] ), 'tab and/or space are down' );
  assert.ok( testTracker.areKeysDown( [ tabKeyDownEvent.keyCode, spaceKeyDownEvent.keyCode ] ), 'tab and space are down' );

  testTracker.keydownUpdate( spaceKeyUpEvent );
} );


QUnit.test( 'tracking of shift key', assert => {

  // mock sending "keydown" events to the tracker
  testTracker.keydownUpdate( shiftTabKeyDownEvent );
  assert.ok( testTracker.shiftKeyDown, 'tab key with shift modifier should produce a keystate with shift key down' );

  testTracker.keydownUpdate( shiftKeyDownEvent );
  testTracker.keydownUpdate( shiftTabKeyDownEvent );
  assert.ok( testTracker.isKeyDown( tabKeyDownEvent.keyCode ), 'tab key should be down in tracker' );
  assert.ok( testTracker.isKeyDown( shiftKeyDownEvent.keyCode ), 'shift key should be down in tracker' );
  assert.ok( testTracker.shiftKeyDown, 'shift key should be down in tracker getter' );

  testTracker.keyupUpdate( shiftKeyUpEvent );
  testTracker.keyupUpdate( tabKeyUpEvent );


  assert.ok( !testTracker.isKeyDown( shiftKeyUpEvent.keyCode ), 'shift key should not be down in tracker' );
  assert.ok( !testTracker.shiftKeyDown, 'shift key should not be down in tracker getter' );
  assert.ok( !testTracker.isKeyDown( tabKeyUpEvent.keyCode ), 'tab key should not be down in tracker' );

  assert.ok( !testTracker.keysAreDown(), 'no keys should be down' );

  testTracker.keydownUpdate( tabKeyDownEvent );
  testTracker.keyupUpdate( shiftTabKeyUpEvent );
  assert.ok( !testTracker.isKeyDown( tabKeyDownEvent.keyCode ), 'tab key should not be down in tracker' );

  // KeyStateTracker should correctly update when modifier keys like "shift" are attached to the event - if shift
  // is down on keyUpUpdate, shift should be considered down
  assert.ok( testTracker.isKeyDown( shiftKeyDownEvent.keyCode ), 'shift key should update from modifier' );
  assert.ok( testTracker.shiftKeyDown, 'shift key should update from modifier getter' );
} );


QUnit.test( 'test tracking with time', async assert => {

  const done = assert.async();

  // we will test holding down a key for these lenghts of time in ms
  const firstPressTime = 500;
  const secondPressTime = 71;
  const totalPressTime = firstPressTime + secondPressTime;

  testTracker.keydownUpdate( spaceKeyDownEvent );
  let currentTimeDown = testTracker.timeDownForKey( spaceKeyDownEvent.keyCode );
  assert.ok( currentTimeDown === 0, 'should be zero, has not been down any time' );

  timer.setTimeout( () => { // eslint-disable-line bad-sim-text
    currentTimeDown = testTracker.timeDownForKey( spaceKeyDownEvent.keyCode );

    assert.ok( currentTimeDown >= firstPressTime && currentTimeDown <= totalPressTime, 'key pressed for ' + firstPressTime + ' ms' );

    timer.setTimeout( () => { // eslint-disable-line bad-sim-text
      currentTimeDown = testTracker.timeDownForKey( spaceKeyDownEvent.keyCode );

      assert.ok( currentTimeDown >= totalPressTime, 'key pressed for ' + secondPressTime + ' more ms.' );

      testTracker.keyupUpdate( spaceKeyUpEvent );
      done();
    }, secondPressTime );
  }, firstPressTime );
} );