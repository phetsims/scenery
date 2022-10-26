// Copyright 2022, University of Colorado Boulder

/**
 * KeyboardListener tests.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Agustín Vallejo (PhET Interactive Simulations)
 */

import { Display, KeyboardListener, KeyboardUtils, Node } from '../imports.js';

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

  //////////////////////////////////////////////////////

  a.removeInputListener( listener );

  let pFired = false;
  let ctrlPFired = false;
  a.addInputListener( new KeyboardListener( {
    keys: [ 'p', 'ctrl+p' ],

    callback: ( event, keysPressed ) => {
      if ( keysPressed === 'p' ) {
        pFired = true;
      }
      else if ( keysPressed === 'ctrl+p' ) {
        ctrlPFired = true;
      }
      else {
        assert.ok( false, 'never again' );
      }
    }
  } ) );
  domElement.dispatchEvent( new KeyboardEvent( 'keydown', {
    code: KeyboardUtils.KEY_P,
    ctrlKey: true,
    bubbles: true
  } ) );
  assert.ok( pFired, 'p should have fired' );
  assert.ok( ctrlPFired, 'ctrl P should have fired' );

  //////////////////////////////////////////////////////

  document.body.removeChild( display.domElement );
  display.dispose();
} );