// Copyright 2022-2025, University of Colorado Boulder

/**
 * KeyboardListener tests.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Agustín Vallejo (PhET Interactive Simulations)
 */

import Display from '../display/Display.js';
import globalKeyStateTracker from '../accessibility/globalKeyStateTracker.js';
import KeyboardListener from '../listeners/KeyboardListener.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import Node from '../nodes/Node.js';

QUnit.module( 'KeyboardListener', {
  before() {

    // clear in case other tests didn't finish with a keyup event
    globalKeyStateTracker.clearState();
  }
} );

const triggerKeydownEvent = ( target: HTMLElement, code: string, ctrlKey = false ) => {
  target.dispatchEvent( new KeyboardEvent( 'keydown', {
    code: code,
    bubbles: true,
    ctrlKey: ctrlKey
  } ) );
};

const triggerKeyupEvent = ( target: HTMLElement, code: string, ctrlKey = false ) => {
  target.dispatchEvent( new KeyboardEvent( 'keyup', {
    code: code,
    bubbles: true,
    ctrlKey: ctrlKey
  } ) );
};

QUnit.test( 'KeyboardListener Tests', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  //////////////////////////////////////////////////

  let callbackFired = false;
  const listener = new KeyboardListener( {
    keys: [ 'enter' ],
    fire: () => {
      assert.ok( !callbackFired, 'callback cannot be fired' );
      callbackFired = true;
    }
  } );

  // Test putting a key in keys that is not supported (error only thrown with assertions enabled)
  window.assert && assert.throws( () => {
    const bogusListener = new KeyboardListener( {

      // @ts-expect-error - Typescript should catch bad keys too
      keys: [ 'badKey' ],
      fire: () => {

        // just testing the typing, no work to do here
      }
    } );
    bogusListener.dispose();
  }, Error, 'Constructor should catch providing bad keys at runtime' );

  const a = new Node( { tagName: 'div', focusable: true } );
  rootNode.addChild( a );
  a.addInputListener( listener );

  const domElementA = a.pdomInstances[ 0 ].peer!.primarySibling!;
  assert.ok( domElementA, 'pdom element needed' );

  // Hotkey uses the focused Trail to determine if it should fire, so we need to focus the element
  a.focus();

  triggerKeydownEvent( domElementA, KeyboardUtils.KEY_TAB );
  assert.ok( !callbackFired, 'should not fire on tab' );
  triggerKeyupEvent( domElementA, KeyboardUtils.KEY_TAB );

  triggerKeydownEvent( domElementA, KeyboardUtils.KEY_ENTER );
  assert.ok( callbackFired, 'should fire on enter' );
  triggerKeyupEvent( domElementA, KeyboardUtils.KEY_ENTER );

  //////////////////////////////////////////////////////
  // Test an overlap of keys in two keygroups. The callback should fire for only the keygroup where every key
  // is down and only every key is down.
  a.removeInputListener( listener );

  let pFired = false;
  let ctrlPFired = false;
  const listenerWithOverlappingKeys = new KeyboardListener( {
    keys: [ 'p', 'ctrl+p' ],

    fire: ( event, keysPressed ) => {
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
  } );

  a.addInputListener( listenerWithOverlappingKeys );

  triggerKeydownEvent( domElementA, KeyboardUtils.KEY_P, true );
  assert.ok( !pFired, 'p should not fire because control key is down' );
  assert.ok( ctrlPFired, 'ctrl P should have fired' );
  //////////////////////////////////////////////////////

  // test interrupt/cancel
  // TODO: This test fails but that is working as expected. interrupt/cancel are only relevant for the https://github.com/phetsims/scenery/issues/1581
  // listener for press and hold functionality. Interrupt/cancel cannot clear the keystate because the listener
  // does not own its KeyStateTracker, it is using the global one.
  // let pbFiredFromA = false;
  // let pbjFiredFromA = false;
  // const listenerToInterrupt = new KeyboardListener( {
  //   keys: [ 'p+b', 'p+b+j' ],
  // callback: ( event, listener ) => {
  //   const keysPressed = listener.keysPressed;
  //     if ( keysPressed === 'p+b' ) {
  //       pbFiredFromA = true;
  //       listenerToInterrupt.interrupt();
  //     }
  //     else if ( keysPressed === 'p+b+j' ) {
  //       pbjFiredFromA = true;
  //     }
  //   }
  // } );
  // a.addInputListener( listenerToInterrupt );
  //
  // domElementB.dispatchEvent( new KeyboardEvent( 'keydown', {
  //   code: KeyboardUtils.KEY_P,
  //   bubbles: true
  // } ) );
  // domElementB.dispatchEvent( new KeyboardEvent( 'keydown', {
  //   code: KeyboardUtils.KEY_B,
  //   bubbles: true
  // } ) );
  // domElementB.dispatchEvent( new KeyboardEvent( 'keydown', {
  //   code: KeyboardUtils.KEY_J,
  //   bubbles: true
  // } ) );
  //
  // assert.ok( pbFiredFromA, 'p+b receives the event and interrupts the listener' );
  // assert.ok( !pbjFiredFromA, 'interruption clears the keystate so p+b+j does not fire' );

  //////////////////////////////////////////////////////

  document.body.removeChild( display.domElement );
  display.dispose();
} );

//
// QUnit.test( 'KeyboardListener Callback timing', assert => {
//   const rootNode = new Node( { tagName: 'div' } );
//   const display = new Display( rootNode );
//   display.initializeEvents();
//   document.body.appendChild( display.domElement );
//
//
//   //
//   // a -> callback timer
//   //
//   // wait
//   // b -> callback timer
//   //
//   // release before b
//   //
//   // ensure a fires
//
//
//   document.body.removeChild( display.domElement );
//   display.dispose();
// });