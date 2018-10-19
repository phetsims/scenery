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
  const KeyStateTracker = require( 'SCENERY/accessibility/KeyStateTracker' );
  // const KeyboardFuzzer = require( 'SCENERY/accessibility/KeyboardFuzzer' ); // can we use this in testing?

  QUnit.module( 'KeyStateTracker' );

  const tabKeyEvent = { keyCode: 9 };
  const spaceKeyEvent = { keyCode: 32 };
  // const aKeyEvent = { keyCode: 65 };
  // const nKeyEvent = { keyCode: 78 };

  const testTracker = new KeyStateTracker();

  QUnit.test( 'basic state tracking of keys', assert => {

    // mock sending "keydown" events to the tracker
    testTracker.keydownUpdate( tabKeyEvent );
    assert.ok( testTracker.isKeyDown( tabKeyEvent.keyCode ), 'tab key should be down in tracker' );

    testTracker.keydownUpdate( spaceKeyEvent );
    assert.ok( testTracker.isKeyInListDown( [ tabKeyEvent.keyCode, spaceKeyEvent.keyCode ] ), 'tab or space are down' );
  } );
} );