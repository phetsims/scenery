// Copyright 2026, University of Colorado Boulder

/**
 * Unit tests for KeyboardDragListener.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Property from '../../../axon/js/Property.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Vector2Property from '../../../dot/js/Vector2Property.js';
import Tandem from '../../../tandem/js/Tandem.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import type PDOMInstance from '../accessibility/pdom/PDOMInstance.js';
import Display from '../display/Display.js';
import Node from '../nodes/Node.js';
import Rectangle from '../nodes/Rectangle.js';
import KeyboardDragListener from './KeyboardDragListener.js';

/**
 * Sends a synthesized keyup event to the target element.
 *
 * @param target - DOM element that receives the event
 * @param type - 'keydown' or 'keyup'
 * @param code - KeyboardEvent.code
 */
const triggerKeyboardEvent = ( target: HTMLElement, type: 'keydown' | 'keyup', code: string ) => {
  target.dispatchEvent( new KeyboardEvent( type, {
    code: code,
    bubbles: true
  } ) );
};

/**
 * Sends synthesized keydown and keyup event pair to the target element for the specified code.
 */
const triggerKeyboardDownUpEvents = ( target: HTMLElement, code: string ) => {
  target.focus();
  triggerKeyboardEvent( target, 'keydown', code );
  triggerKeyboardEvent( target, 'keyup', code );
};

/**
 * Returns the primary sibling for the Node's first PDOM instance.
 */
const getPrimarySibling = ( nodeOrInstance: Node | PDOMInstance ): HTMLElement => {
  const pdomInstance = nodeOrInstance instanceof Node ? nodeOrInstance.pdomInstances[ 0 ] : nodeOrInstance;
  return pdomInstance.peer!.primarySibling!;
};

/**
 * Sets up a test harness by creating a display, and focusable Node. Add a keyboard drag listener to
 * the focusable for tests.
 *
 * @param callback - Invoked after setup with display, test rectangle, PDOM instance, and its dom element
 */
const prepareTest = ( callback: ( display: Display, rect: Rectangle, pdomInstance: PDOMInstance, domElement: HTMLElement ) => void ) => {
  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode, { width: 640, height: 480 } );
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  const rect = new Rectangle( 0, 0, 20, 20, {
    tagName: 'div',
    focusable: true
  } );
  rootNode.addChild( rect );
  display.updateDisplay();

  const pdomInstance = rect.pdomInstances[ 0 ];
  const domElement = getPrimarySibling( pdomInstance );
  callback( display, rect, pdomInstance, domElement );

  document.body.removeChild( display.domElement );
  display.dispose();
};

QUnit.test( 'modelPoint initializes from positionProperty', assert => {
  prepareTest( ( _display, rect, _pdomInstance, domElement ) => {
    const positionProperty = new Vector2Property( new Vector2( 5, 7 ) );
    let startCalled = false;

    const listener = new KeyboardDragListener( {
      positionProperty: positionProperty,
      tandem: Tandem.ROOT_TEST.createTandem( 'modelPointPositionPropertyListener' ),
      start: ( event, dragListener ) => {
        startCalled = true;
        assert.ok( dragListener.modelPoint.equals( positionProperty.value ),
          'modelPoint should match positionProperty at start' );
      }
    } );
    rect.addInputListener( listener );

    triggerKeyboardDownUpEvents( domElement, KeyboardUtils.KEY_RIGHT_ARROW );

    // Sanity check to make sure input events are run.
    assert.ok( startCalled, 'start should have been called' );
    listener.dispose();
  } );
} );

QUnit.test( 'modelPoint applies mapPosition at start', assert => {
  prepareTest( ( _display, rect, _pdomInstance, domElement ) => {
    const positionProperty = new Vector2Property( new Vector2( 10, 12 ) );
    let startCalled = false;
    const mapPosition = ( point: Vector2 ) => new Vector2( 3, 4 );

    const listener = new KeyboardDragListener( {
      positionProperty: positionProperty,
      mapPosition: mapPosition,
      tandem: Tandem.ROOT_TEST.createTandem( 'modelPointMapPositionListener' ),
      start: ( event, dragListener ) => {
        startCalled = true;
        assert.ok( dragListener.modelPoint.equals( new Vector2( 3, 4 ) ),
          'modelPoint should apply mapPosition at start' );
      }
    } );
    rect.addInputListener( listener );

    triggerKeyboardDownUpEvents( domElement, KeyboardUtils.KEY_RIGHT_ARROW );

    // Sanity check to make sure input events are run.
    assert.ok( startCalled, 'start should have been called' );
    listener.dispose();
  } );
} );

QUnit.test( 'modelPoint applies dragBounds at start', assert => {
  prepareTest( ( _display, rect, _pdomInstance, domElement ) => {
    const positionProperty = new Vector2Property( new Vector2( 10, 12 ) );
    let startCalled = false;

    const listener = new KeyboardDragListener( {
      positionProperty: positionProperty,
      dragBoundsProperty: new Property( new Bounds2( 0, 0, 5, 6 ) ),
      tandem: Tandem.ROOT_TEST.createTandem( 'modelPointDragBoundsListener' ),
      start: ( event, dragListener ) => {
        startCalled = true;
        assert.ok( dragListener.modelPoint.equals( new Vector2( 5, 6 ) ),
          'modelPoint should apply dragBounds at start' );
      }
    } );
    rect.addInputListener( listener );

    triggerKeyboardDownUpEvents( domElement, KeyboardUtils.KEY_RIGHT_ARROW );

    // Sanity check to make sure input events are run.
    assert.ok( startCalled, 'start should have been called' );
    listener.dispose();
  } );
} );

QUnit.test( 'modelPoint initializes from target translation without positionProperty', assert => {
  prepareTest( ( _display, rect, _pdomInstance, domElement ) => {
    rect.translation = new Vector2( 2, 4 );
    let startCalled = false;

    const listener = new KeyboardDragListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'modelPointTargetTranslationListener' ),
      start: ( event, dragListener ) => {
        startCalled = true;
        assert.ok( dragListener.modelPoint.equals( new Vector2( 2, 4 ) ),
          'modelPoint should match target translation at start' );
      }
    } );
    rect.addInputListener( listener );

    triggerKeyboardDownUpEvents( domElement, KeyboardUtils.KEY_RIGHT_ARROW );

    // Sanity check to make sure input events are run.
    assert.ok( startCalled, 'start should have been called' );
    listener.dispose();
  } );
} );

QUnit.test( 'modelPoint uses full inverse transform for translation', assert => {
  prepareTest( ( _display, rect, _pdomInstance, domElement ) => {
    rect.translation = new Vector2( 10, 15 );
    let startCalled = false;
    const transform = new Transform3( Matrix3.translation( 5, 3 ) );

    const listener = new KeyboardDragListener( {
      transform: transform,
      tandem: Tandem.ROOT_TEST.createTandem( 'modelPointTransformListener' ),
      start: ( event, dragListener ) => {
        startCalled = true;
        assert.ok( dragListener.modelPoint.equals( new Vector2( 5, 12 ) ),
          'modelPoint should use full inverse transform (including translation)' );
      }
    } );
    rect.addInputListener( listener );

    triggerKeyboardDownUpEvents( domElement, KeyboardUtils.KEY_RIGHT_ARROW );

    // Sanity check to make sure input events are run.
    assert.ok( startCalled, 'start should have been called' );
    listener.dispose();
  } );
} );

QUnit.test( 'modelPoint updates during drag', assert => {
  prepareTest( ( _display, rect, _pdomInstance, domElement ) => {
    const positionProperty = new Vector2Property( new Vector2( 0, 0 ) );
    let dragCalled = false;

    const listener = new KeyboardDragListener( {
      positionProperty: positionProperty,
      dragDelta: 2,
      shiftDragDelta: 1,
      tandem: Tandem.ROOT_TEST.createTandem( 'modelPointDragListener' ),
      drag: ( event, dragListener ) => {
        dragCalled = true;
        assert.ok( dragListener.modelPoint.equals( new Vector2( 2, 0 ) ),
          'modelPoint should update during drag' );
      }
    } );
    rect.addInputListener( listener );

    triggerKeyboardDownUpEvents( domElement, KeyboardUtils.KEY_RIGHT_ARROW );

    // Sanity check to make sure input events are run.
    assert.ok( dragCalled, 'drag should have been called' );
    listener.dispose();
  } );
} );
