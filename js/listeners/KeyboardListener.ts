// Copyright 2022, University of Colorado Boulder

/**
 * A listener for general keyboard input. Specify the keys with a `keys` option in a readable format that looks like
 * this: [ 'shift+t', 'alt+shift+r' ]
 *
 * - Each entry in the array represents a "group" of keys.
 * - '+' separates each key in a single group.
 * - The keys leading up to the last key in the group are considered "modifier" keys. The last key in the group needs
 *   to be pressed while the modifier keys are down.
 * - The order modifier keys are pressed does not matter for firing the callback.
 *
 * In the above example "shift+t" OR "alt+shift+r" will fire the callback when pressed.
 *
 * An example usage would like this:
 *
 * this.addInputListener( new KeyboardListener( {
 *   keys: [ 'a+b', 'a+c', 'shift+arrowLeft', 'alt+g+t', 'ctrl+3', 'alt+ctrl+t' ],
 *   callback: ( event, listener ) => {
 *     const keysPressed = listener.keysPressed;
 *
 *     if ( keysPressed === 'a+b' ) {
 *       console.log( 'you just pressed a+b!' );
 *     }
 *     else if ( keysPressed === 'a+c' ) {
 *       console.log( 'you just pressed a+c!' );
 *     }
 *     else if ( keysPressed === 'alt+g+t' ) {
 *       console.log( 'you just pressed alt+g+t' );
 *     }
 *     else if ( keysPressed === 'ctrl+3' ) {
 *       console.log( 'you just pressed ctrl+3' );
 *     }
 *     else if ( keysPressed === 'shift+arrowLeft' ) {
 *       console.log( 'you just pressed shift+arrowLeft' );
 *     }
 *   }
 * } ) );
 *
 * By default the callback will fire when the last key is pressed down. See additional options for firing on key
 * up or other press and hold behavior.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import optionize from '../../../phet-core/js/optionize.js';
import { EnglishStringToCodeMap, globalKeyStateTracker, scenery, SceneryEvent, TInputListener } from '../imports.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import assertMutuallyExclusiveOptions from '../../../phet-core/js/assertMutuallyExclusiveOptions.js';


// TODO: The typing for ModifierKey and OneKeyStroke is limited TypeScript, there is a limitation to the number of
//       entries in a union type. If that limitation is not acceptable remove this typing. OR maybe TypeScript will
//       someday support regex patterns for a type. See https://github.com/microsoft/TypeScript/issues/41160
// If we run out of union space for template strings, consider the above comment or remove some from the type.
type ModifierKey = 'q' | 'w' | 'e' | 'r' | 't' | 'y' | 'u' | 'i' | 'o' | 'p' | 'a' | 's' | 'd' |
  'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'z' | 'x' | 'c' |
  'v' | 'b' | 'n' | 'm' | 'ctrl' | 'alt' | 'shift' | 'tab';
type AllowedKeys = keyof typeof EnglishStringToCodeMap;

type OneKeyStroke = `${AllowedKeys}` |
  `${ModifierKey}+${AllowedKeys}` |
  `${ModifierKey}+${ModifierKey}+${AllowedKeys}`;
// These combinations are not supported by TypeScript: "TS2590: Expression produces a union type that is too complex to
// represent." See above note and https://github.com/microsoft/TypeScript/issues/41160#issuecomment-1287271132.
// `${AllowedKeys}+${AllowedKeys}+${AllowedKeys}+${AllowedKeys}`;
// type KeyCombinations = `${OneKeyStroke}` | `${OneKeyStroke},${OneKeyStroke}`;

type KeyboardListenerOptions<Keys extends readonly OneKeyStroke[ ]> = {

  // The keys that need to be pressed to fire the callback. In a form like `[ 'shift+t', 'alt+shift+r' ]`. See top
  // level documentation for more information and an example of providing keys.
  keys: Keys;

  // Called when the listener detects that the set of keys are pressed.
  callback?: ( event: SceneryEvent<KeyboardEvent> | null, listener: KeyboardListener<Keys> ) => void;

  // Does the listener fire when the last key in the group is pressed down or released?
  fireOnKeyUp?: boolean;

  // Does the listener fire continuously as you hold down keys?
  fireOnHold?: boolean;

  // If fireOnHold true, this is the delay in (in milliseconds) before the callback is fired continuously.
  fireOnHoldDelay?: number;

  // If fireOnHold true, this is the interval (in milliseconds) that the callback fires after the fireOnHoldDelay.
  fireOnHoldInterval?: number;
};

type KeyGroup<Keys extends readonly OneKeyStroke[]> = {

  // All must be pressed fully before the key is pressed to activate the command.
  modifierKeys: string[];

  // the final key that is pressed (after modifier keys) to trigger the listener
  key: string;

  // all keys in this KeyGroup as a KeyboardEvent.code
  allKeys: string[];

  // All keys in this KeyGroup using the readable form
  naturalKeys: Keys[number];

  // A callback timer for this KeyGroup to support press and hold timing and callbacks
  timer: CallbackTimer | null;
};

class KeyboardListener<Keys extends readonly OneKeyStroke[]> implements TInputListener {

  // The function called when a KeyGroup is pressed (or just released). Provides the SceneryEvent that fired the input
  // listeners and this the keys that were pressed from the active KeyGroup. The event may be null when using
  // fireOnHold or in cases of cancel or interrupt.
  private readonly _callback: ( event: SceneryEvent<KeyboardEvent> | null, listener: KeyboardListener<Keys> ) => void;

  // Will it the callback fire on keys up or down?
  private readonly _fireOnKeyUp: boolean;

  // Does the listener fire the callback continuously when keys are held down?
  private readonly _fireOnHold: boolean;

  // All the KeyGroups of this listener from the keys provided in natural language.
  private readonly _keyGroups: KeyGroup<Keys>[];

  // All the KeyGroups that are currently firing
  private readonly _activeKeyGroups: KeyGroup<Keys>[];

  // Current keys pressed that are having their listeners fired now.
  public keysPressed: Keys[number] | null = null;

  // Timing variables for the CallbackTimers.
  private readonly _fireOnHoldDelay: number;
  private readonly _fireOnHoldInterval: number;

  public constructor( providedOptions: KeyboardListenerOptions<Keys> ) {
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'fireOnKeyUp' ], [ 'fireOnHold', 'fireOnHoldInterval', 'fireOnHoldDelay' ] );

    const options = optionize<KeyboardListenerOptions<Keys>>()( {
      callback: _.noop,
      fireOnKeyUp: false,
      fireOnHold: false,
      fireOnHoldDelay: 400,
      fireOnHoldInterval: 100
    }, providedOptions );

    this._callback = options.callback;
    this._fireOnKeyUp = options.fireOnKeyUp;

    // convert the provided keys to data that we can respond to with scenery's Input system
    this._keyGroups = this.convertKeysToKeyGroups( options.keys );

    this._fireOnHold = options.fireOnHold;
    this._fireOnHoldDelay = options.fireOnHoldDelay;
    this._fireOnHoldInterval = options.fireOnHoldInterval;

    this._activeKeyGroups = [];
  }

  /**
   * Mostly required to fire with CallbackTimer since the callback cannot take arguments.
   */
  public fireCallback( event: SceneryEvent<KeyboardEvent> | null, naturalKeys: Keys[number] ): void {
    this.keysPressed = naturalKeys;
    this._callback( event, this );
    this.keysPressed = null;
  }

  /**
   * Part of the scenery listener API. Responding to a keydown event, update active KeyGroups and potentially
   * fire callbacks and start CallbackTimers.
   */
  public keydown( event: SceneryEvent<KeyboardEvent> ): void {

    if ( !this._fireOnKeyUp ) {

      // modifier keys can be pressed in any order but the last key in the group must be pressed last
      this._keyGroups.forEach( keyGroup => {

        if ( !this._activeKeyGroups.includes( keyGroup ) ) {
          if ( globalKeyStateTracker.areKeysDown( keyGroup.allKeys ) &&
               globalKeyStateTracker.getLastKeyDown() === keyGroup.key ) {

            this._activeKeyGroups.push( keyGroup );

            if ( keyGroup.timer ) {
              keyGroup.timer.start();
            }
            this.fireCallback( event, keyGroup.naturalKeys );
          }
        }
      } );
    }
  }

  /**
   * Part of the scenery listener API. If there are any active KeyGroup firing stop and remove if KeyGroup keys
   * are no longer down. Also, potentially fires a KeyGroup if the key that was released has all other modifier keys
   * down.
   */
  public keyup( event: SceneryEvent<KeyboardEvent> ): void {

    if ( this._activeKeyGroups.length > 0 ) {

      this._activeKeyGroups.forEach( ( activeKeyGroup, index ) => {
        if ( !globalKeyStateTracker.areKeysDown( activeKeyGroup.allKeys ) ) {
          if ( activeKeyGroup.timer ) {
            activeKeyGroup.timer.stop( false );
          }
          this._activeKeyGroups.splice( index, 1 );
        }
      } );
    }

    if ( this._fireOnKeyUp ) {
      this._keyGroups.forEach( keyGroup => {
        if ( globalKeyStateTracker.areKeysDown( keyGroup.modifierKeys ) &&
             KeyboardUtils.getEventCode( event.domEvent ) === keyGroup.key ) {
          this.fireCallback( event, keyGroup.naturalKeys );
        }
      } );
    }
  }

  /**
   * Part of the scenery listener API. On cancel, clear active KeyGroups and stop their behavior.
   */
  public cancel(): void {
    this.clearActiveKeyGroups();
  }

  /**
   * Part of the scenery listener API. Clear active KeyGroups and stop their callbacks.
   */
  public interrupt(): void {
    this.clearActiveKeyGroups();
  }

  /**
   * Dispose of this listener by disposing of any Callback timers. Then clear all KeyGroups.
   */
  public dispose(): void {
    this._keyGroups.forEach( activeKeyGroup => {
      activeKeyGroup.timer && activeKeyGroup.timer.dispose();
    } );
    this._keyGroups.length = 0;
  }

  /**
   * Clear the active KeyGroups on this listener. Stopping any active groups if they use a CallbackTimer.
   */
  private clearActiveKeyGroups(): void {
    this._activeKeyGroups.forEach( activeKeyGroup => {
      activeKeyGroup.timer && activeKeyGroup.timer.stop( false );
    } );

    this._activeKeyGroups.length = 0;
  }

  /**
   * Converts the provided keys into a collection of KeyGroups to easily track what keys are down. For example,
   * will take a string that defines the keys for this listener like 'a+c|1+2+3+4|shift+leftArrow' and return an array
   * with three KeyGroups, one describing 'a+c', one describing '1+2+3+4' and one describing 'shift+leftArrow'.
   */
  private convertKeysToKeyGroups( keys: Keys ): KeyGroup<Keys>[] {

    const keyGroups = keys.map( naturalKeys => {

      // all of the keys in this group in an array
      const groupKeys = naturalKeys.split( '+' );
      assert && assert( groupKeys.length > 0, 'no keys provided?' );

      const naturalKey = groupKeys.slice( -1 )[ 0 ];

      // @ts-ignore - because a string shouldn't be used for lookup like this in the object type
      const key = EnglishStringToCodeMap[ naturalKey ];
      assert && assert( key, `Key not found, do you need to add it to EnglishStringToCodeMap? ${naturalKey}` );

      let modifierKeys: string[] = [];
      if ( groupKeys.length > 1 ) {
        modifierKeys = groupKeys.slice( 0, groupKeys.length - 1 ).map( naturalModifierKey => {

          // @ts-ignore - because a string shouldn't be used for lookup like this in the object type
          const modifierKey = EnglishStringToCodeMap[ naturalModifierKey ];
          assert && assert( modifierKey, `Key not found, do you need to add it to EnglishStringToCodeMap? ${naturalModifierKey}` );
          return modifierKey;
        } );
      }

      // Set up the timer for triggering callbacks if this listener supports press and hold behavior
      const timer = this._fireOnHold ? new CallbackTimer( {
        callback: () => this.fireCallback( null, naturalKeys ),
        delay: this._fireOnHoldDelay,
        interval: this._fireOnHoldInterval
      } ) : null;

      return {
        key: key,
        modifierKeys: modifierKeys,
        naturalKeys: naturalKeys,
        allKeys: modifierKeys.concat( key ),
        timer: timer
      };
    } );

    return keyGroups;
  }
}

scenery.register( 'KeyboardListener', KeyboardListener );
export default KeyboardListener;
