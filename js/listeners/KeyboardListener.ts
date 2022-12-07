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

export type OneKeyStroke = `${AllowedKeys}` |
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

  // If true, the listener will fire for keys regardless of where focus is in the document. Use this when you want
  // to add some key press behavior that will always fire no matter what the event target is. If this listener
  // is added to a Node, it will only fire if the Node (and all of its ancestors) are visible with inputEnabled: true.
  // More specifically, this uses `globalKeyUp` and `globalKeyDown`. See definitions in Input.ts for more information.
  global?: boolean;

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

  // TODO: Potential option to allow overlap between the keys of this KeyboardListener and another.
  allowKeyOverlap?: boolean;
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

  // (scenery-internal) All the KeyGroups of this listener from the keys provided in natural language.
  public readonly _keyGroups: KeyGroup<Keys>[];

  // All the KeyGroups that are currently firing
  private readonly _activeKeyGroups: KeyGroup<Keys>[];

  // Current keys pressed that are having their listeners fired now.
  public keysPressed: Keys[number] | null = null;

  // Timing variables for the CallbackTimers.
  private readonly _fireOnHoldDelay: number;
  private readonly _fireOnHoldInterval: number;

  // Will the listener respond to 'global' events or just to events targeted to where this listener was added?
  private readonly _global: boolean;

  // TODO: Potentially a flag that could allow overlaps between keys of other KeyboardListeners.
  private readonly _allowKeyOverlap: boolean;

  public constructor( providedOptions: KeyboardListenerOptions<Keys> ) {
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'fireOnKeyUp' ], [ 'fireOnHold', 'fireOnHoldInterval', 'fireOnHoldDelay' ] );

    const options = optionize<KeyboardListenerOptions<Keys>>()( {
      callback: _.noop,
      global: false,
      fireOnKeyUp: false,
      fireOnHold: false,
      fireOnHoldDelay: 400,
      fireOnHoldInterval: 100,
      allowKeyOverlap: false
    }, providedOptions );

    this._callback = options.callback;
    this._fireOnKeyUp = options.fireOnKeyUp;

    this._fireOnHold = options.fireOnHold;
    this._fireOnHoldDelay = options.fireOnHoldDelay;
    this._fireOnHoldInterval = options.fireOnHoldInterval;

    this._activeKeyGroups = [];

    this._allowKeyOverlap = options.allowKeyOverlap;

    this._global = options.global;

    // convert the provided keys to data that we can respond to with scenery's Input system
    this._keyGroups = this.convertKeysToKeyGroups( options.keys );

    ( this as unknown as TInputListener ).listener = this;
  }

  /**
   * Mostly required to fire with CallbackTimer since the callback cannot take arguments.
   */
  public fireCallback( event: SceneryEvent<KeyboardEvent> | null, keyGroup: KeyGroup<Keys> ): void {

    // TODO: Some initial work to check for overlap between other listeners that are responding to the same keys.
    // if ( assert && event && !this._allowKeyOverlap ) {
    //   this.checkForTrailKeyCollisions( event, keyGroup );
    // }

    this.keysPressed = keyGroup.naturalKeys;
    this._callback( event, this );
    this.keysPressed = null;
  }

  /**
   * Part of the scenery listener API. Responding to a keydown event, update active KeyGroups and potentially
   * fire callbacks and start CallbackTimers.
   */
  private handleKeyDown( event: SceneryEvent<KeyboardEvent> ): void {
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
            this.fireCallback( event, keyGroup );
          }
        }
      } );
    }
  }

  /**
   * If there are any active KeyGroup firing stop and remove if KeyGroup keys are no longer down. Also, potentially
   * fires a KeyGroup if the key that was released has all other modifier keys down.
   */
  private handleKeyUp( event: SceneryEvent<KeyboardEvent> ): void {

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
          this.fireCallback( event, keyGroup );
        }
      } );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the keydown event when not
   * added to the global key events. Target will be the Node, Display, or Pointer this listener was added to.
   */
  public keydown( event: SceneryEvent<KeyboardEvent> ): void {
    if ( !this._global ) {
      this.handleKeyDown( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the keyup event when not
   * added to the global key events. Target will be the Node, Display, or Pointer this listener was added to.
   */
  public keyup( event: SceneryEvent<KeyboardEvent> ): void {
    if ( !this._global ) {
      this.handleKeyUp( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the global keydown event.
   * Event has no target.
   */
  public globalkeydown( event: SceneryEvent<KeyboardEvent> ): void {
    if ( this._global ) {
      this.handleKeyDown( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the global keyup event.
   * Event has no target.
   */
  public globalkeyup( event: SceneryEvent<KeyboardEvent> ): void {
    if ( this._global ) {
      this.handleKeyUp( event );
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
   * Throws an assertion if any Node along the trail has a KeyboardListener that is listening for key presses that
   * overlap with this KeyboardListener, so both would fire. A lot of loops required to find this information,
   * only run when assertions are enabled. Or consider optimizing.
   *
   * TODO: Still working on this function. This method works decently but has not been tested well and may not
   *       cover all cases. Still need to look for collisions against active global listeners as well. For that
   *       we will probably need a registry object.
   */
  private checkForTrailKeyCollisions( event: SceneryEvent<KeyboardEvent>, myKeyGroup: KeyGroup<Keys> ): void {
    const trails = event.target.getTrails();

    for ( let i = 0; i < trails.length; i++ ) {
      const trail = trails[ i ];
      for ( let j = 0; j < trail.nodes.length; j++ ) {
        const node = trail.nodes[ j ];
        for ( let k = 0; k < node.inputListeners.length; k++ ) {
          const inputListener = node.inputListeners[ k ];
          if ( inputListener.listener instanceof KeyboardListener &&
               !inputListener.listener._allowKeyOverlap &&
               inputListener.listener !== this ) {
            const ancestorKeyGroups = inputListener.listener._keyGroups;

            for ( let l = 0; l < ancestorKeyGroups.length; l++ ) {
              const ancestorNaturalKeys = ancestorKeyGroups[ l ].naturalKeys;

              // There is an ovelrap if the last keys are the same, or if all keys of the ancestor are pressed while
              // pressing the modifier keys of the descendant keys
              const modifierOverlap = ancestorNaturalKeys.startsWith( myKeyGroup.naturalKeys );
              const finalKeysEqual = ancestorKeyGroups[ l ].key === myKeyGroup.key;

              assert && assert( !modifierOverlap && !finalKeysEqual,
                `Keys collision with another KeyboardListener along this trail. My keys: ${myKeyGroup.naturalKeys}, other keys: '${ancestorNaturalKeys}, '` );
            }
          }
        }
      }
    }
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

      // @ts-expect-error - because a string shouldn't be used for lookup like this in the object type
      const key = EnglishStringToCodeMap[ naturalKey ];
      assert && assert( key, `Key not found, do you need to add it to EnglishStringToCodeMap? ${naturalKey}` );

      let modifierKeys: string[] = [];
      if ( groupKeys.length > 1 ) {
        modifierKeys = groupKeys.slice( 0, groupKeys.length - 1 ).map( naturalModifierKey => {

          // @ts-expect-error - because a string shouldn't be used for lookup like this in the object type
          const modifierKey = EnglishStringToCodeMap[ naturalModifierKey ];
          assert && assert( modifierKey, `Key not found, do you need to add it to EnglishStringToCodeMap? ${naturalModifierKey}` );
          return modifierKey;
        } );
      }

      // Set up the timer for triggering callbacks if this listener supports press and hold behavior
      const timer = this._fireOnHold ? new CallbackTimer( {
        callback: () => this.fireCallback( null, keyGroup ),
        delay: this._fireOnHoldDelay,
        interval: this._fireOnHoldInterval
      } ) : null;

      const keyGroup: KeyGroup<Keys> = {
        key: key,
        modifierKeys: modifierKeys,
        naturalKeys: naturalKeys,
        allKeys: modifierKeys.concat( key ),
        timer: timer
      };
      return keyGroup;
    } );

    return keyGroups;
  }
}

scenery.register( 'KeyboardListener', KeyboardListener );
export default KeyboardListener;
