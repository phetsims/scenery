// Copyright 2018-2022, University of Colorado Boulder

/**
 * A type that will manage the state of the keyboard. This will track which keys are being held down and for how long.
 * It also offers convenience methods to determine whether or not specific keys are down like shift or enter using
 * KeyboardUtils' key schema.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Barlow
 */

import PhetioAction from '../../../tandem/js/PhetioAction.js';
import Emitter from '../../../axon/js/Emitter.js';
import stepTimer from '../../../axon/js/stepTimer.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { scenery, KeyboardUtils, EventIO } from '../imports.js';

class KeyStateTracker {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    options = merge( {

      // {Tandem}
      tandem: Tandem.OPTIONAL
    }, options );

    // @private {Object.<string,{ key: {string}, isDown: {boolean}, timeDown: [boolean] }>} - where the Object
    // keys are the event.code string. JavaScript doesn't handle multiple key presses, so we track
    // which keys are currently down and update based on state of this collection of objects. Cleared when disabled.
    this.keyState = {};

    // @private {boolean} - whether or not this KeyStateTracker is attached to the document
    this.attachedToDocument = false;

    // @private {null|function} - Listeners potentially attached to the document to update the state of this
    // KeyStateTracker, see attachToWindow()
    this.documentKeyupListener = null;
    this.documentKeydownListener = null;

    // @private - if the key state tracker is enabled. If disabled, the keyState will be cleared, and listeners will noop.
    this._enabled = true;

    // @public - Emits events when keyup/keydown updates are received. These will emit after any updates to the
    // keyState so that keyState is up to date in time for listeners.
    this.keydownEmitter = new Emitter( { parameters: [ { valueType: Event } ] } ); // valueType is a native DOM event
    this.keyupEmitter = new Emitter( { parameters: [ { valueType: Event } ] } );

    // @private {Action} - Action which updates the KeyStateTracker, when it is time to do so - the update
    // is wrapped by an Action so that the KeyStateTracker state is captured for PhET-iO
    this.keydownUpdateAction = new PhetioAction( domEvent => {

      // The dom event might have a modifier key that we weren't able to catch, if that is the case update the keyState.
      // This is likely to happen when pressing browser key commands like "ctrl + tab" to switch tabs.
      this.correctModifierKeys( domEvent );

      const key = KeyboardUtils.getEventCode( domEvent );

      if ( assert && !KeyboardUtils.isShiftKey( domEvent ) ) {
        assert( !!domEvent.shiftKey === !!this.shiftKeyDown, 'shift key inconsistency between event and keyState.' );
      }
      if ( assert && !KeyboardUtils.isAltKey( domEvent ) ) {
        assert( !!domEvent.altKey === !!this.altKeyDown, 'alt key inconsistency between event and keyState.' );
      }
      if ( assert && !KeyboardUtils.isControlKey( domEvent ) ) {
        assert( !!domEvent.ctrlKey === !!this.ctrlKeyDown, 'ctrl key inconsistency between event and keyState.' );
      }

      // if the key is already down, don't do anything else (we don't want to create a new keyState object
      // for a key that is already being tracked and down)
      if ( !this.isKeyDown( key ) ) {
        const key = KeyboardUtils.getEventCode( domEvent );
        this.keyState[ key ] = {
          keyDown: true,
          key: key,
          timeDown: 0 // in ms
        };
      }

      // keydown update received, notify listeners
      this.keydownEmitter.emit( domEvent );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'keydownUpdateAction' ),
      parameters: [ { name: 'event', phetioType: EventIO } ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Action that executes whenever a keydown occurs from the input listeners this keyStateTracker adds (most likely to the document).'
    } );

    // @private {Action} - Action which updates the state of the KeyStateTracker on key release. This
    // is wrapped in an Action so that state is captured for PhET-iO
    this.keyupUpdateAction = new PhetioAction( domEvent => {

      const key = KeyboardUtils.getEventCode( domEvent );

      // correct keyState in case browser didn't receive keydown/keyup events for a modifier key
      this.correctModifierKeys( domEvent );

      // Remove this key data from the state - There are many cases where we might receive a keyup before keydown like
      // on first tab into scenery Display or when using specific operating system keys with the browser or PrtScn so
      // an assertion for this is too strict. See https://github.com/phetsims/scenery/issues/918
      if ( this.isKeyDown( key ) ) {
        delete this.keyState[ key ];
      }

      // keyup event received, notify listeners
      this.keyupEmitter.emit( domEvent );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'keyupUpdateAction' ),
      parameters: [ { name: 'event', phetioType: EventIO } ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Action that executes whenever a keyup occurs from the input listeners this keyStateTracker adds (most likely to the document).'
    } );

    const stepListener = this.step.bind( this );
    stepTimer.addListener( stepListener );

    // @private
    this._disposeKeyStateTracker = () => {
      stepTimer.removeListener( stepListener );

      if ( this.attachedToDocument ) {
        this.detachFromDocument();
      }
    };
  }

  /**
   * Implements keyboard dragging when listener is attached to the Node, public so listener is attached
   * with addInputListener(). Only updated when enabled.
   *
   *
   * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
   * `Node.addInputListener` only supports type properties as event listeners, and not the event keys as
   * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
   * @public
   * @param {Event} domEvent
   */
  keydownUpdate( domEvent ) {
    this.enabled && this.keydownUpdateAction.execute( domEvent );
  }

  /**
   * Modifier keys might be part of the domEvent but the browser may or may not have received a keydown/keyup event
   * with specifically for the modifier key. This will add or remove modifier keys in that case.
   * @private
   *
   * @param  {Event} domEvent
   */
  correctModifierKeys( domEvent ) {

    const key = KeyboardUtils.getEventCode( domEvent );

    // add modifier keys if they aren't down
    if ( domEvent.shiftKey && !this.shiftKeyDown ) {
      this.keyState[ KeyboardUtils.KEY_SHIFT_LEFT ] = {
        keyDown: true,
        key: key,
        timeDown: 0 // in ms
      };
    }
    if ( domEvent.altKey && !this.altKeyDown ) {
      this.keyState[ KeyboardUtils.KEY_ALT_LEFT ] = {
        keyDown: true,
        key: key,
        timeDown: 0 // in ms
      };
    }
    if ( domEvent.ctrlKey && !this.ctrlKeyDown ) {
      this.keyState[ KeyboardUtils.KEY_CONTROL_LEFT ] = {
        keyDown: true,
        key: key,
        timeDown: 0 // in ms
      };
    }

    // delete modifier keys if we think they are down
    if ( !domEvent.shiftKey && this.shiftKeyDown ) {
      delete this.keyState[ KeyboardUtils.KEY_SHIFT_LEFT ];
      delete this.keyState[ KeyboardUtils.KEY_SHIFT_RIGHT ];
    }
    if ( !domEvent.altKey && this.altKeyDown ) {
      delete this.keyState[ KeyboardUtils.KEY_ALT_LEFT ];
      delete this.keyState[ KeyboardUtils.KEY_ALT_RIGHT ];
    }
    if ( !domEvent.ctrlKey && this.ctrlKeyDown ) {
      delete this.keyState[ KeyboardUtils.KEY_CONTROL_LEFT ];
      delete this.keyState[ KeyboardUtils.KEY_CONTROL_RIGHT ];
    }
  }

  /**
   * Behavior for keyboard 'up' DOM event. Public so it can be attached with addInputListener(). Only updated when
   * enabled.
   *
   * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
   * `Node.addInputListener` only supports type properties as event listeners, and not the event keys as
   * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
   *
   * @public
   * @param {Event} domEvent
   */
  keyupUpdate( domEvent ) {
    this.enabled && this.keyupUpdateAction.execute( domEvent );
  }

  /**
   * Returns true if any of the movement keys are down (arrow keys or WASD keys).
   *
   * @returns {boolean}
   * @public
   */
  get movementKeysDown() {
    return this.isAnyKeyInListDown( KeyboardUtils.MOVEMENT_KEYS );
  }

  /**
   * Returns true if a key with the Event.code is currently down.
   *
   * @public
   * @param  {string} key
   * @returns {boolean}
   */
  isKeyDown( key ) {
    if ( !this.keyState[ key ] ) {

      // key hasn't been pressed once yet
      return false;
    }

    return this.keyState[ key ].keyDown;
  }

  /**
   * Returns true if any of the keys in the list are currently down.
   *
   * @param  {Array.<string>} keyList - array of string key states
   * @returns {boolean}
   * @public
   */
  isAnyKeyInListDown( keyList ) {
    for ( let i = 0; i < keyList.length; i++ ) {
      if ( this.isKeyDown( keyList[ i ] ) ) {
        return true;
      }
    }

    return false;
  }

  /**
   * Returns true if and only if all of the keys in the list are currently down.
   *
   * @param  {Array.<string>} keyList - array of string key states
   * @returns {boolean}
   * @public
   */
  areKeysDown( keyList ) {
    const keysDown = true;
    for ( let i = 0; i < keyList.length; i++ ) {
      if ( !this.isKeyDown( keyList[ i ] ) ) {
        return false;
      }
    }

    return keysDown;
  }

  /**
   * @returns {boolean} if any keys in the key state are currently down
   * @public
   */
  keysAreDown() {
    return !!Object.keys( this.keyState ).length > 0;
  }

  /**
   * @returns {boolean} - true if the enter key is currently pressed down.
   * @public
   */
  get enterKeyDown() {
    return this.isKeyDown( KeyboardUtils.KEY_ENTER );
  }

  /**
   * @returns {boolean} - true if the keyState indicates that the shift key is currently down.
   * @public
   */
  get shiftKeyDown() {
    return this.isAnyKeyInListDown( KeyboardUtils.SHIFT_KEYS );
  }

  /**
   * @returns {boolean} - true if the keyState indicates that the alt key is currently down.
   * @public
   */
  get altKeyDown() {
    return this.isAnyKeyInListDown( KeyboardUtils.ALT_KEYS );
  }

  /**
   * @returns {boolean} - true if the keyState indicates that the ctrl key is currently down.
   * @public
   */
  get ctrlKeyDown() {
    return this.isAnyKeyInListDown( KeyboardUtils.CONTROL_KEYS );
  }

  /**
   * Will assert if the key isn't currently pressed down
   * @param {string} key
   * @returns {number} how long the key has been down
   * @public
   */
  timeDownForKey( key ) {
    assert && assert( this.isKeyDown( key ), 'cannot get timeDown on a key that is not pressed down' );
    return this.keyState[ key ].timeDown;
  }

  /**
   * Clear the entire state of the key tracker, basically reinitializing the instance.
   * @public
   */
  clearState() {
    this.keyState = {};
  }

  /**
   * Step function for the tracker. JavaScript does not natively handle multiple keydown events at once,
   * so we need to track the state of the keyboard in an Object and manage dragging in this function.
   * In order for the drag handler to work.
   *
   * @private
   * @param {number} dt - time in seconds that has passed since the last update
   */
  step( dt ) {

    // no-op unless a key is down
    if ( this.keysAreDown() ) {
      const ms = dt * 1000;

      // for each key that is still down, increment the tracked time that has been down
      for ( const i in this.keyState ) {
        if ( this.keyState.hasOwnProperty( i ) ) {
          if ( this.keyState[ i ].keyDown ) {
            this.keyState[ i ].timeDown += ms;
          }

        }
      }
    }
  }

  /**
   * Add this KeyStateTracker to the DOM document so that it updates whenever the document receives key events. This is
   * useful if you want to observe key presses while DOM focus not within the PDOM root.
   * @public
   */
  attachToWindow() {
    assert && assert( !this.attachedToDocument, 'KeyStateTracker is already attached to document.' );

    this.documentKeydownListener = event => {
      if ( this.blockTrustedEvents && event.isTrusted ) {
        return;
      }
      this.keydownUpdate( event );
    };

    this.documentKeyupListener = event => {
      if ( this.blockTrustedEvents && event.isTrusted ) {
        return;
      }
      this.keyupUpdate( event );
    };

    const addListenersToDocument = () => {

      // attach with useCapture so that the keyStateTracker is up to date before the events dispatch within Scenery
      window.addEventListener( 'keyup', this.documentKeyupListener, true );
      window.addEventListener( 'keydown', this.documentKeydownListener, true );
      this.attachedToDocument = true;
    };

    if ( !document ) {

      // attach listeners on window load to ensure that the document is defined
      const loadListener = () => {
        addListenersToDocument();
        window.removeEventListener( 'load', loadListener );
      };
      window.addEventListener( 'load', loadListener );
    }
    else {

      // document is defined and we won't get another load event so attach right away
      addListenersToDocument();
    }
  }

  /**
   * @public
   * @param {boolean} enabled
   */
  setEnabled( enabled ) {
    if ( this._enabled !== enabled ) {
      this._enabled = enabled;

      // clear state when disabled
      !enabled && this.clearState();
    }
  }

  // @public
  set enabled( enabled ) { this.setEnabled( enabled ); }

  /**
   * @public
   * @returns {boolean}
   */
  isEnabled() { return this._enabled; }

  // @public
  get enabled() { return this.isEnabled(); }


  /**
   * Detach listeners from the document that would update the state of this KeyStateTracker on key presses.
   *
   * @public
   */
  detachFromDocument() {
    assert && assert( this.attachedToDocument, 'KeyStateTracker is not attached to document.' );

    window.removeEventListener( 'keyup', this.documentKeyupListener );
    window.removeEventListener( 'keydown', this.documentKeydownListener );

    this.documentKeyupListener = null;
    this.documentKeydownListener = null;

    this.attachedToDocument = false;
  }

  /**
   * Make eligible for garbage collection.
   * @public
   */
  dispose() {
    this._disposeKeyStateTracker();
  }
}

scenery.register( 'KeyStateTracker', KeyStateTracker );
export default KeyStateTracker;