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
 * In the above example "shift+t" OR "alt+shift + r" will fire the callback when pressed.
 *
 * An example usage would like this:
 *
 *     this.addInputListener( new KeyboardListener( {
 *       keys: [ 'a+b', 'a+c', 'shift+arrowLeft', 'alt+g+t', 'ctrl+3', 'alt+ctrl+t' ] as const,
 *       callback: ( event, keys ) => {
 *
 *         if ( keys === 'a+b' ) {
 *           console.log( 'you just pressed a+b!' );
 *         }
 *         else if ( keys === 'a+c' ) {
 *           console.log( 'you just pressed a+c!' );
 *         }
 *         else if ( keys === 'alt+g+t' ) {
 *           console.log( 'you just pressed alt+g+t' );
 *         }
 *         else if ( keys === 'ctrl+3' ) {
 *           console.log( 'you just pressed ctrl+3' );
 *         }
 *         else if ( keys === 'shift+arrowLeft' ) {
 *           console.log( 'you just pressed shift+arrowLeft' );
 *         }
 *       }
 *     } ) );
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


// TODO: If we run out of union space for template strings, then remove some of these. https://github.com/phetsims/scenery/issues/1445
type ModifierKey = 'q' | 'w' | 'e' | 'r' | 't' | 'y' | 'u' | 'i' | 'o' | 'p' | 'a' | 's' | 'd' |
  'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'z' | 'x' | 'c' |
  'v' | 'b' | 'n' | 'm' | 'ctrl' | 'alt' | 'shift' | 'tab';
type AllowedKeys = keyof typeof EnglishStringToCodeMap;

// TODO: The typing for this limits the number of keys that can exist in a group. If that limitation is not acceptable
//       remove this typing. OR maybe TypeScript will someday support regex patterns for a type.
type OneKeyStroke = `${AllowedKeys}` |
  `${ModifierKey}+${AllowedKeys}` |
  `${ModifierKey}+${ModifierKey}+${AllowedKeys}`;

// TODO: adding this extra one has Typescript all hot and bothered: "TS2590: Expression produces a union type that is too complex to represent."
// `${AllowedKeys}+${AllowedKeys}+${AllowedKeys}+${AllowedKeys}`;
// TODO: combining these has Typescript all hot and bothered: "TS2590: Expression produces a union type that is too complex to represent."
// type KeyCombinations = `${OneKeyStroke}` | `${OneKeyStroke},${OneKeyStroke}`;
//

// TODO: Automated testing, https://github.com/phetsims/scenery/issues/1445
type KeyboardListenerOptions<Keys extends readonly OneKeyStroke[ ]> = {

  // Keys that trigger functionality for this listener.
  // 'j+o+1' // all three keys need to be held down in order
  // 'j+o' // these two keys need to be pressed down
  // '1|2|3' // any of these keys are pressed
  // 'j+1|j+2' // any of these are pressed
  keys: Keys;
  callback?: ( event: SceneryEvent<KeyboardEvent> | null, keysPressed: Keys[number] ) => void;
  fireOnKeyUp?: boolean;

  fireOnHold?: boolean;
  fireOnHoldDelay?: number;
  fireOnHoldInterval?: number;
};

type KeyGroup<Keys extends readonly OneKeyStroke[]> = {

  // All must be pressed fully before the key is pressed to activate the command.
  modifierKeys: string[];
  key: string;
  allKeys: string[];
  naturalKeys: Keys[number];
  timer: CallbackTimer | null;
};

class KeyboardListener<Keys extends readonly OneKeyStroke[]> implements TInputListener {

  // The function called when a KeyGroup is pressed (or just released). Provides the SceneryEvent that fired the input
  // listeners and this the keys that were pressed from the active key group. The event may be null when using
  // fireOnHold or in cases of cancel or interrupt.
  private readonly _callback: ( event: SceneryEvent<KeyboardEvent> | null, keysPressed: Keys[number] ) => void;

  // Will it the callback fire on keys up or down?
  private readonly _fireOnKeyUp: boolean;

  // Does the listener fire the callback continuously when keys are held down?
  private readonly _fireOnHold: boolean;

  // All of the KeyGroups of this listener from the keys provided in natural language.
  private readonly _keyGroups: KeyGroup<Keys>[];

  // All of the key groups that are currently firing
  private readonly _activeKeyGroups: KeyGroup<Keys>[];

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
    this._callback( event, naturalKeys );
  }

  /**
   * Part of the scenery listener API. Responding to a keydown event, update active keygroups and potentially
   * start firing callbacks or timers.
   */
  public keydown( event: SceneryEvent<KeyboardEvent> ): void {

    if ( !this._fireOnKeyUp ) {

      // modifier keys can be pressed in any order but the last key in the group must be pressed last
      this._keyGroups.forEach( keyGroup => {

        if ( !this._activeKeyGroups.includes( keyGroup ) ) {
          // TODO: Guarantee that globalKeyStateTracker is updated first, likely by embedding that data in the event itself instead of a global, https://github.com/phetsims/scenery/issues/1445
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
   * Part of the scenery listener API. If there are any active key groups firing stop and remove if keygroup keys
   * are no longer down. Also potentially fires a keygroup if the key that was released has all other modifier keys
   * down.
   */
  public keyup( event: SceneryEvent<KeyboardEvent> ): void {

    if ( this._activeKeyGroups.length > 0 ) {
      // TODO: Guarantee that globalKeyStateTracker is updated first, likely by embedding that data in the event itself instead of a global, https://github.com/phetsims/scenery/issues/1445

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
        // TODO: Guarantee that globalKeyStateTracker is updated first, likely by embedding that data in the event itself instead of a global, https://github.com/phetsims/scenery/issues/1445
        if ( globalKeyStateTracker.areKeysDown( keyGroup.modifierKeys ) &&
             KeyboardUtils.getEventCode( event.domEvent ) === keyGroup.key ) {
          this.fireCallback( event, keyGroup.naturalKeys );
        }
      } );
    }
  }

  /**
   * Part of the scenery listener API. On cancel, clear active key groups and stop their behavior.
   */
  public cancel(): void {
    this.clearActiveKeyGroups();
  }

  /**
   * Part of the scenery listener API. Clear active key groups and stop their callbacks.
   */
  public interrupt(): void {
    this.clearActiveKeyGroups();
  }

  /**
   * Dispose of this listener by disposing of any Callback timers. Then clear all key groups.
   */
  public dispose(): void {
    this._keyGroups.forEach( activeKeyGroup => {
      activeKeyGroup.timer && activeKeyGroup.timer.dispose();
    } );
    this._keyGroups.length = 0;
  }

  /**
   * Clear the active key groups on this listener. Stopping any active groups if they use a CallbackTimer.
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
        callback: () => { this.fireCallback( null, naturalKeys ); },
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
