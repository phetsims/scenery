// Copyright 2018-2023, University of Colorado Boulder

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
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { EventIO, KeyboardUtils, scenery } from '../imports.js';
import optionize from '../../../phet-core/js/optionize.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import PickOptional from '../../../phet-core/js/types/PickOptional.js';
import TEmitter from '../../../axon/js/TEmitter.js';

// Type describing the state of a single key in the KeyState.
type KeyStateInfo = {

  // The event.code string for the key.
  key: string;

  // Is the key currently down?
  keyDown: boolean;

  // How long has the key been held down, in milliseconds
  timeDown: number;
};

// The type for the keyState Object, keys are the KeyboardEvent.code for the pressed key.
type KeyState = Record<string, KeyStateInfo>;

export type KeyStateTrackerOptions = PickOptional<PhetioObject, 'tandem'>;

class KeyStateTracker {

  // Contains info about which keys are currently pressed for how long. JavaScript doesn't handle multiple key presses,
  // with events so we have to update this object ourselves.
  private keyState: KeyState = {};

  // The KeyboardEvent.code of the last key that was pressed down when updating the key state.
  private _lastKeyDown: string | null = null;

  // Whether this KeyStateTracker is attached to the document and listening for events.
  private attachedToDocument = false;

  // Listeners potentially attached to the document to update the state of this KeyStateTracker, see attachToWindow()
  private documentKeyupListener: null | ( ( event: KeyboardEvent ) => void ) = null;
  private documentKeydownListener: null | ( ( event: KeyboardEvent ) => void ) = null;

  // If the KeyStateTracker is enabled. If disabled, keyState is cleared and listeners noop.
  private _enabled = true;

  // Emits events when keyup/keydown updates are received. These will emit after any updates to the
  // keyState so that keyState is correct in time for listeners. Note the valueType is a native KeyboardEvent event.
  public readonly keydownEmitter: TEmitter<[ KeyboardEvent ]> = new Emitter( { parameters: [ { valueType: KeyboardEvent } ] } );
  public readonly keyupEmitter: TEmitter<[ KeyboardEvent ]> = new Emitter( { parameters: [ { valueType: KeyboardEvent } ] } );

  // Action which updates the KeyStateTracker, when it is time to do so - the update is wrapped by an Action so that
  // the KeyStateTracker state is captured for PhET-iO.
  public readonly keydownUpdateAction: PhetioAction<[ KeyboardEvent ]>;

  // Action which updates the state of the KeyStateTracker on key release. This is wrapped in an Action so that state
  // is captured for PhET-iO.
  public readonly keyupUpdateAction: PhetioAction<[ KeyboardEvent ]>;

  private readonly disposeKeyStateTracker: () => void;

  public constructor( providedOptions?: KeyStateTrackerOptions ) {

    const options = optionize<KeyStateTrackerOptions>()( {
      tandem: Tandem.OPTIONAL
    }, providedOptions );

    this.keydownUpdateAction = new PhetioAction( domEvent => {

      // Not all keys have a code for the browser to use, we need to be graceful and do nothing if there isn't one.
      const key = KeyboardUtils.getEventCode( domEvent );
      if ( key ) {

        // The dom event might have a modifier key that we weren't able to catch, if that is the case update the keyState.
        // This is likely to happen when pressing browser key commands like "ctrl + tab" to switch tabs.
        this.correctModifierKeys( domEvent );

        if ( assert && !KeyboardUtils.isShiftKey( domEvent ) ) {
          assert( domEvent.shiftKey === this.shiftKeyDown, 'shift key inconsistency between event and keyState.' );
        }
        if ( assert && !KeyboardUtils.isAltKey( domEvent ) ) {
          assert( domEvent.altKey === this.altKeyDown, 'alt key inconsistency between event and keyState.' );
        }
        if ( assert && !KeyboardUtils.isControlKey( domEvent ) ) {
          assert( domEvent.ctrlKey === this.ctrlKeyDown, 'ctrl key inconsistency between event and keyState.' );
        }

        // if the key is already down, don't do anything else (we don't want to create a new keyState object
        // for a key that is already being tracked and down)
        if ( !this.isKeyDown( key ) ) {
          const key = KeyboardUtils.getEventCode( domEvent )!;
          assert && assert( key, 'Could not find key from domEvent' );
          this.keyState[ key ] = {
            keyDown: true,
            key: key,
            timeDown: 0 // in ms
          };
        }

        this._lastKeyDown = key;

        // keydown update received, notify listeners
        this.keydownEmitter.emit( domEvent );
      }

    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'keydownUpdateAction' ),
      parameters: [ { name: 'event', phetioType: EventIO } ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Action that executes whenever a keydown occurs from the input listeners this keyStateTracker adds (most likely to the document).'
    } );

    this.keyupUpdateAction = new PhetioAction( domEvent => {

      // Not all keys have a code for the browser to use, we need to be graceful and do nothing if there isn't one.
      const key = KeyboardUtils.getEventCode( domEvent );
      if ( key ) {

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
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'keyupUpdateAction' ),
      parameters: [ { name: 'event', phetioType: EventIO } ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Action that executes whenever a keyup occurs from the input listeners this keyStateTracker adds (most likely to the document).'
    } );

    const stepListener = this.step.bind( this );
    stepTimer.addListener( stepListener );

    this.disposeKeyStateTracker = () => {
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
   * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
   * `Node.addInputListener` only supports type properties as event listeners, and not the event keys as
   * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
   */
  public keydownUpdate( domEvent: KeyboardEvent ): void {
    this.enabled && this.keydownUpdateAction.execute( domEvent );
  }

  /**
   * Modifier keys might be part of the domEvent but the browser may or may not have received a keydown/keyup event
   * with specifically for the modifier key. This will add or remove modifier keys in that case.
   */
  private correctModifierKeys( domEvent: KeyboardEvent ): void {
    const key = KeyboardUtils.getEventCode( domEvent )!;
    assert && assert( key, 'key not found from domEvent' );

    // add modifier keys if they aren't down
    if ( domEvent.shiftKey && !KeyboardUtils.isShiftKey( domEvent ) && !this.shiftKeyDown ) {
      this.keyState[ KeyboardUtils.KEY_SHIFT_LEFT ] = {
        keyDown: true,
        key: key,
        timeDown: 0 // in ms
      };
    }
    if ( domEvent.altKey && !KeyboardUtils.isAltKey( domEvent ) && !this.altKeyDown ) {
      this.keyState[ KeyboardUtils.KEY_ALT_LEFT ] = {
        keyDown: true,
        key: key,
        timeDown: 0 // in ms
      };
    }
    if ( domEvent.ctrlKey && !KeyboardUtils.isControlKey( domEvent ) && !this.ctrlKeyDown ) {
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
   */
  public keyupUpdate( domEvent: KeyboardEvent ): void {
    this.enabled && this.keyupUpdateAction.execute( domEvent );
  }

  /**
   * Returns true if any of the movement keys are down (arrow keys or WASD keys).
   */
  public get movementKeysDown(): boolean {
    return this.isAnyKeyInListDown( KeyboardUtils.MOVEMENT_KEYS );
  }

  /**
   * Returns the KeyboardEvent.code from the last key down that updated the keystate.
   */
  public getLastKeyDown(): string | null {
    return this._lastKeyDown;
  }

  /**
   * Returns true if a key with the KeyboardEvent.code is currently down.
   */
  public isKeyDown( key: string ): boolean {
    if ( !this.keyState[ key ] ) {

      // key hasn't been pressed once yet
      return false;
    }

    return this.keyState[ key ].keyDown;
  }

  /**
   * Returns true if any of the keys in the list are currently down. Keys are the KeyboardEvent.code strings.
   */
  public isAnyKeyInListDown( keyList: string[] ): boolean {
    for ( let i = 0; i < keyList.length; i++ ) {
      if ( this.isKeyDown( keyList[ i ] ) ) {
        return true;
      }
    }

    return false;
  }

  /**
   * Returns true if ALL of the keys in the list are currently down. Values of the keyList array are the
   * KeyboardEvent.code for the keys you are interested in.
   */
  public areKeysDown( keyList: string[] ): boolean {
    const keysDown = true;
    for ( let i = 0; i < keyList.length; i++ ) {
      if ( !this.isKeyDown( keyList[ i ] ) ) {
        return false;
      }
    }

    return keysDown;
  }

  /**
   * Returns true if ALL keys in the list are down and ONLY the keys in the list are down. Values of keyList array
   * are the KeyboardEvent.code for keys you are interested in OR the KeyboardEvent.key in the special case of
   * modifier keys.
   *
   * (scenery-internal)
   */
  public areKeysExclusivelyDown( keyList: string [] ): boolean {
    const keyStateKeys = Object.keys( this.keyState );

    // quick sanity check for equality first
    if ( keyStateKeys.length !== keyList.length ) {
      return false;
    }

    // Now make sure that every key in the list is in the keyState
    let onlyKeyListDown = true;
    for ( let i = 0; i < keyList.length; i++ ) {
      const initialKey = keyList[ i ];
      let keysToCheck = [ initialKey ];

      // If a modifier key, need to look for the equivalent pair of left/right KeyboardEvent.codes in the list
      // because KeyStateTracker works exclusively with codes.
      if ( KeyboardUtils.isModifierKey( initialKey ) ) {
        keysToCheck = KeyboardUtils.MODIFIER_KEY_TO_CODE_MAP.get( initialKey )!;
      }

      if ( _.intersection( keyStateKeys, keysToCheck ).length === 0 ) {
        onlyKeyListDown = false;
      }
    }

    return onlyKeyListDown;
  }

  /**
   * Returns true if any keys are down according to teh keyState.
   */
  public keysAreDown(): boolean {
    return Object.keys( this.keyState ).length > 0;
  }

  /**
   * Returns true if the "Enter" key is currently down.
   */
  public get enterKeyDown(): boolean {
    return this.isKeyDown( KeyboardUtils.KEY_ENTER );
  }

  /**
   * Returns true if the shift key is currently down.
   */
  public get shiftKeyDown(): boolean {
    return this.isAnyKeyInListDown( KeyboardUtils.SHIFT_KEYS );
  }

  /**
   * Returns true if the alt key is currently down.
   */
  public get altKeyDown(): boolean {
    return this.isAnyKeyInListDown( KeyboardUtils.ALT_KEYS );
  }

  /**
   * Returns true if the control key is currently down.
   */
  public get ctrlKeyDown(): boolean {
    return this.isAnyKeyInListDown( KeyboardUtils.CONTROL_KEYS );
  }

  /**
   * Returns the amount of time that the provided key has been held down. Error if the key is not currently down.
   * @param key - KeyboardEvent.code for the key you are inspecting.
   */
  public timeDownForKey( key: string ): number {
    assert && assert( this.isKeyDown( key ), 'cannot get timeDown on a key that is not pressed down' );
    return this.keyState[ key ].timeDown;
  }

  /**
   * Clear the entire state of the key tracker, basically restarting the tracker.
   */
  public clearState(): void {
    this.keyState = {};
  }

  /**
   * Step function for the tracker. JavaScript does not natively handle multiple keydown events at once,
   * so we need to track the state of the keyboard in an Object and manage dragging in this function.
   * In order for the drag handler to work.
   *
   * @param dt - time in seconds that has passed since the last update
   */
  private step( dt: number ): void {

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
   * Add this KeyStateTracker to the window so that it updates whenever the document receives key events. This is
   * useful if you want to observe key presses while DOM focus not within the PDOM root.
   */
  public attachToWindow(): void {
    assert && assert( !this.attachedToDocument, 'KeyStateTracker is already attached to document.' );

    this.documentKeydownListener = event => {
      this.keydownUpdate( event );
    };

    this.documentKeyupListener = event => {
      this.keyupUpdate( event );
    };

    const addListenersToDocument = () => {

      // attach with useCapture so that the keyStateTracker is updated before the events dispatch within Scenery
      window.addEventListener( 'keyup', this.documentKeyupListener!, { capture: true } );
      window.addEventListener( 'keydown', this.documentKeydownListener!, { capture: true } );
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
   * The KeyState is cleared when the tracker is disabled.
   */
  public setEnabled( enabled: boolean ): void {
    if ( this._enabled !== enabled ) {
      this._enabled = enabled;

      // clear state when disabled
      !enabled && this.clearState();
    }
  }

  public set enabled( enabled: boolean ) { this.setEnabled( enabled ); }

  public get enabled(): boolean { return this.isEnabled(); }

  public isEnabled(): boolean { return this._enabled; }

  /**
   * Detach listeners from the document that would update the state of this KeyStateTracker on key presses.
   */
  public detachFromDocument(): void {
    assert && assert( this.attachedToDocument, 'KeyStateTracker is not attached to window.' );
    assert && assert( this.documentKeyupListener, 'keyup listener was not created or attached to window' );
    assert && assert( this.documentKeydownListener, 'keydown listener was not created or attached to window.' );

    window.removeEventListener( 'keyup', this.documentKeyupListener! );
    window.removeEventListener( 'keydown', this.documentKeydownListener! );

    this.documentKeyupListener = null;
    this.documentKeydownListener = null;

    this.attachedToDocument = false;
  }

  public dispose(): void {
    this.disposeKeyStateTracker();
  }
}

scenery.register( 'KeyStateTracker', KeyStateTracker );
export default KeyStateTracker;