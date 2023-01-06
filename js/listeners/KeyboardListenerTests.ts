// Copyright 2022-2023, University of Colorado Boulder

/**
 * KeyboardListener tests.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Agustín Vallejo (PhET Interactive Simulations)
 */

import { Display, globalKeyStateTracker, KeyboardListener, KeyboardUtils, Node, SceneryEvent } from '../imports.js';

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
    callback: () => {
      assert.ok( !callbackFired, 'callback cannot be fired' );
      callbackFired = true;
    }
  } );

  const a = new Node( { tagName: 'div' } );
  rootNode.addChild( a );
  a.addInputListener( listener );

  const domElementA = a.pdomInstances[ 0 ].peer!.primarySibling!;
  assert.ok( domElementA, 'pdom element needed' );

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

    callback: ( event, listener ) => {
      const keysPressed = listener.keysPressed;
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

  // test handle/abort
  a.removeInputListener( listenerWithOverlappingKeys );
  const b = new Node( { tagName: 'div' } );
  a.addChild( b );

  const domElementB = b.pdomInstances[ 0 ].peer!.primarySibling!;

  // test handled - event should no longer bubble, b listener should handle and a listener should not fire
  let pFiredFromA = false;
  let pFiredFromB = false;
  const listenerPreventedByHandle = new KeyboardListener( {
    keys: [ 'p' ],
    callback: ( event, listener ) => {
      const keysPressed = listener.keysPressed;
      if ( keysPressed === 'p' ) {
        pFiredFromA = true;
      }
    }
  } );
  a.addInputListener( listenerPreventedByHandle );

  const handlingListener = new KeyboardListener( {
    keys: [ 'p' ],
    callback: ( event, listener ) => {
      const keysPressed = listener.keysPressed;
      if ( keysPressed === 'p' ) {
        pFiredFromB = true;

        assert.ok( !!event, 'An event should be provided to the callback in this case.' );
        event!.handle();
      }
    }
  } );
  b.addInputListener( handlingListener );

  triggerKeydownEvent( domElementB, KeyboardUtils.KEY_P );
  assert.ok( !pFiredFromA, 'A should not have received the event because of event handling' );
  assert.ok( pFiredFromB, 'B received the event and handled it (stopping bubbling)' );
  triggerKeyupEvent( domElementB, KeyboardUtils.KEY_P );

  a.removeInputListener( listenerPreventedByHandle );
  b.removeInputListener( handlingListener );
  pFiredFromA = false;
  pFiredFromB = false;

  // test abort
  const listenerPreventedByAbort = new KeyboardListener( {
    keys: [ 'p' ],
    callback: ( event, listener ) => {
      const keysPressed = listener.keysPressed;
      if ( keysPressed === 'p' ) {
        pFiredFromA = true;
      }
    }
  } );
  a.addInputListener( listenerPreventedByAbort );

  const abortingListener = new KeyboardListener( {
    keys: [ 'p' ],
    callback: ( event, listener ) => {
      const keysPressed = listener.keysPressed;
      if ( keysPressed === 'p' ) {
        pFiredFromB = true;

        assert.ok( !!event, 'An event should be provided to the callback in this case.' );
        event!.abort();
      }
    }
  } );
  b.addInputListener( abortingListener );

  let pFiredFromExtraListener = false;
  const otherListenerPreventedByAbort = {
    keydown: ( event: SceneryEvent<KeyboardEvent> ) => {
      pFiredFromExtraListener = true;
    }
  };
  b.addInputListener( otherListenerPreventedByAbort );

  triggerKeydownEvent( domElementB, KeyboardUtils.KEY_P );
  assert.ok( !pFiredFromA, 'A should not have received the event because of abort' );
  assert.ok( pFiredFromB, 'B received the event and handled it (stopping bubbling)' );
  assert.ok( !pFiredFromExtraListener, 'Other listener on B did not fire because of abort (stopping all listeners)' );
  triggerKeyupEvent( domElementB, KeyboardUtils.KEY_P );

  a.removeInputListener( listenerPreventedByAbort );
  b.removeInputListener( abortingListener );
  b.removeInputListener( otherListenerPreventedByAbort );
  pFiredFromA = false;
  pFiredFromB = false;

  //////////////////////////////////////////////////////

  // test interrupt/cancel
  // TODO: This test fails but that is working as expected. interrupt/cancel are only relevant for the
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