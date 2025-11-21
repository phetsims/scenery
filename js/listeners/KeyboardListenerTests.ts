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
import type SceneryEvent from '../input/SceneryEvent.js';

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
    keys: [ 'a' ],
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

  triggerKeydownEvent( domElementA, KeyboardUtils.KEY_A );
  assert.ok( callbackFired, 'should fire on "a"' );
  triggerKeyupEvent( domElementA, KeyboardUtils.KEY_A );

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

QUnit.test( 'KeyboardListener fireOnClick mode respects enabled state', assert => {

  let fireCount = 0;
  let pressCalled = false;
  let releaseCalled = false;

  const listener = new KeyboardListener( {
    fireOnClick: true,
    fire: () => fireCount++,
    press: () => { pressCalled = true; },
    release: () => { releaseCalled = true; }
  } );

  const createClickEvent = () => ( {
    domEvent: new MouseEvent( 'click' )
  } ) as unknown as SceneryEvent;

  listener.click( createClickEvent() );
  assert.strictEqual( fireCount, 1, 'fires when enabled' );
  assert.ok( !pressCalled, 'press is never called in click mode' );
  assert.ok( !releaseCalled, 'release is never called in click mode' );

  listener.enabledProperty.value = false;
  listener.click( createClickEvent() );
  assert.strictEqual( fireCount, 1, 'disabled prevents firing' );

  listener.enabledProperty.value = true;
  listener.click( createClickEvent() );
  assert.strictEqual( fireCount, 2, 'fires again when enabled' );

  listener.dispose();
} );

QUnit.test( 'KeyboardListener Enter/Space assertion overrides', assert => {

  window.assert && assert.throws( () => {
    const badListener = new KeyboardListener( {
      keys: [ 'enter' ],
      fire: _.noop
    } );

    const nodeWithClickSemantics = new Node( { tagName: 'button' } );
    nodeWithClickSemantics.addInputListener( badListener );

    // trigger a keydown event for enter
    const domElement = nodeWithClickSemantics.pdomInstances[ 0 ].peer!.primarySibling!;
    triggerKeydownEvent( domElement, KeyboardUtils.KEY_ENTER );

    nodeWithClickSemantics.dispose();
    badListener.dispose();
  }, 'enter key should require click activation or allowEnterSpaceWithoutApplicationRole' );

  const allowListener = new KeyboardListener( {
    keys: [ 'enter' ],
    allowEnterSpaceWithoutApplicationRole: true,
    fire: _.noop
  } );
  allowListener.dispose();

  const clickListener = new KeyboardListener( {
    fireOnClick: true,
    fire: _.noop
  } );

  assert.strictEqual( clickListener.hotkeys.length, 0, 'click mode skips hotkeys' );
  clickListener.dispose();
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
