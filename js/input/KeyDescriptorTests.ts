// Copyright 2024-2025, University of Colorado Boulder

/**
 * Tests for KeyDescriptor.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import type { EnglishKeyString } from '../accessibility/EnglishStringToCodeMap.js';
import KeyDescriptor from '../input/KeyDescriptor.js';

QUnit.module( 'KeyDescriptor' );

// Helper function to compare two arrays with QUnit assertions, ignoring order.
const arraysEqualIgnoringOrder = function( array1: EnglishKeyString[], array2: EnglishKeyString[], assert: Assert, message: string ) {
  array1 = array1.slice().sort();
  array2 = array2.slice().sort();

  // Use deepEqual to compare the sorted arrays
  assert.deepEqual( array1, array2, message );
};

// Tests for the keyStrokeToKeyDescriptor method, which converts a keyStroke string to a key descriptor object.
QUnit.test( 'keyStrokeToKeyDescriptor', assert => {

  const descriptor1 = KeyDescriptor.keyStrokeToKeyDescriptor( 'r' );
  assert.equal( descriptor1.key, 'r', 'r key (keyStroke r)' );
  assert.deepEqual( descriptor1.modifierKeys, [], 'no modifier keys (keyStroke r)' );
  assert.deepEqual( descriptor1.ignoredModifierKeys, [], 'no ignored modifier keys (keyStroke r)' );

  const descriptor2 = KeyDescriptor.keyStrokeToKeyDescriptor( 'alt+r' );
  assert.equal( descriptor2.key, 'r', 'r key (keyStroke alt+r)' );
  assert.deepEqual( descriptor2.modifierKeys, [ 'alt' ], 'alt modifier key (keyStroke alt+r)' );
  assert.deepEqual( descriptor2.ignoredModifierKeys, [], 'no ignored modifier keys (keyStroke alt+r)' );

  const descriptor3 = KeyDescriptor.keyStrokeToKeyDescriptor( 'alt+j+r' );
  assert.equal( descriptor3.key, 'r', 'r key (keyStroke alt+j+r)' );
  assert.deepEqual( descriptor3.modifierKeys, [ 'alt', 'j' ], 'alt and j modifier keys (keyStroke alt+j+r)' );
  assert.deepEqual( descriptor3.ignoredModifierKeys, [], 'no ignored modifier keys (keyStroke alt+j+r)' );

  const descriptor4 = KeyDescriptor.keyStrokeToKeyDescriptor( 'alt?+j+r' );
  assert.equal( descriptor4.key, 'r', 'r key (keyStroke alt?+j+r)' );
  assert.deepEqual( descriptor4.modifierKeys, [ 'j' ], 'j modifier key (keyStroke alt?+j+r)' );
  assert.deepEqual( descriptor4.ignoredModifierKeys, [ 'alt' ], 'alt ignored modifier key (keyStroke alt?+j+r)' );

  const descriptor5 = KeyDescriptor.keyStrokeToKeyDescriptor( 'shift?+t' );
  assert.equal( descriptor5.key, 't', 't key (keyStroke shift?+t)' );
  assert.deepEqual( descriptor5.modifierKeys, [], 'no modifier keys (keyStroke shift?+t)' );
  assert.deepEqual( descriptor5.ignoredModifierKeys, [ 'shift' ], 'shift ignored modifier key (keyStroke shift?+t)' );

  const descriptor6 = KeyDescriptor.keyStrokeToKeyDescriptor( '?shift+t' );
  assert.equal( descriptor6.key, 't', 't key (keyStroke ?shift+t)' );
  assert.deepEqual( descriptor6.modifierKeys, [ 'shift' ], 'shift modifier key (keyStroke ?shift+t)' );
  arraysEqualIgnoringOrder( descriptor6.ignoredModifierKeys, [ 'alt', 'ctrl', 'meta' ], assert, 'all other ignored modifier keys (keyStroke ?shift+t)' );

  const descriptor7 = KeyDescriptor.keyStrokeToKeyDescriptor( '?shift+t+j' );
  assert.equal( descriptor7.key, 'j', 'j key (keyStroke ?shift+t+j)' );
  assert.deepEqual( descriptor7.modifierKeys, [ 'shift', 't' ], 'shift and j modifier keys (keyStroke ?shift+t+j)' );
  arraysEqualIgnoringOrder( descriptor7.ignoredModifierKeys, [ 'alt', 'ctrl', 'meta' ], assert, 'all other ignored modifier keys (keyStroke ?shift+t+j)' );
} );