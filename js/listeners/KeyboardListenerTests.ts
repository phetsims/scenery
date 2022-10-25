// Copyright 2018-2021, University of Colorado Boulder

/**
 * KeyboardListener tests.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Agustín Vallejo (PhET Interactive Simulations)
 */

import { KeyboardListener, Node, Display, KeyboardUtils } from '../imports.js';

QUnit.module( 'KeyboardListener' );

QUnit.test( 'Basics', assert => {

  let callbackFired = false;
  const listener = new KeyboardListener( {
    keys: [ 'enter' ],
    callback: () => {
      assert.ok( !callbackFired, 'callback cannot be fired' );
      callbackFired = true;
    }
  } );

  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  display.initializeEvents();
  document.body.appendChild( display.domElement );
  const a = new Node( { tagName: 'div' } );
  rootNode.addChild( a );
  a.addInputListener( listener );

  const domElement = a.pdomInstances[ 0 ].peer.primarySibling;
  assert.ok( domElement, 'pdom element needed' );

  domElement.dispatchEvent( new KeyboardEvent( 'keydown', {
    code: KeyboardUtils.KEY_TAB,
    bubbles: true
  } ) );

  assert.ok( !callbackFired, 'should not fire on tab' );

  domElement.dispatchEvent( new KeyboardEvent( 'keydown', {
    code: KeyboardUtils.KEY_ENTER,
    bubbles: true
  } ) );
  assert.ok( callbackFired, 'should fire on enter' );

  document.body.removeChild( display.domElement );
  display.dispose();
} );