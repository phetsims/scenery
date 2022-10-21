// Copyright 2022, University of Colorado Boulder

/**
 * A listener keyboard input. Specify the keys that you want to listen to with the `keys` option in a format that looks
 * like this:
 *      'shift+t|alt+shift+r'
 *
 * A group of keys are assembled with
 *
 * A typical usage would like this:
 *
 *     this.addInputListener( new KeyboardListener( {
 *       keys: [ 'a+b', 'a+c', 'shift+arrowLeft', 'alt+g+t', 'ctrl+3', 'alt+ctrl+t' ] as const,
 *       callback: keys => {
 *
 *         if ( keys === 'a+b' ) {
 *           console.log( 'you just pressed a+b!' );
 *         }
 *
 *         else if ( keys === 'a+c' ) {
 *           console.log( 'you just pressed a+c!' );
 *         }
 *
 *         else if ( keys === 'alt+g+t' ) {
 *           console.log( 'you just pressed alt+g+t' );
 *         }
 *
 *         else if ( keys === 'ctrl+3' ) {
 *           console.log( 'you just pressed ctrl+3' );
 *         }
 *
 *         else if ( keys === 'shift+arrowLeft' ) {
 *           console.log( 'you just pressed shift+arrowLeft' );
 *         }
 *       }
 *     } ) );
 *
 * By default the callback will fire when the last key is pressed down.
 *
 * Modifier keys are all keys prior to the last provided key. NOTE that modifier keys are not ordered. for example:
 * ctrl+shift+f, will fire the same as shift+ctrl+f.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import optionize from '../../../phet-core/js/optionize.js';
import { EnglishStringToCodeMap, KeyStateTracker, scenery, SceneryEvent, TInputListener } from '../imports.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import assertMutuallyExclusiveOptions from '../../../phet-core/js/assertMutuallyExclusiveOptions.js';


// TODO: If we run out of union space for template strings, then remove some of these. https://github.com/phetsims/scenery/issues/1445
type ModifierKey = 'q' | 'w' | 'e' | 'r' | 't' | 'y' | 'u' | 'i' | 'o' | 'p' | 'a' | 's' | 'd' |
  'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'z' | 'x' | 'c' |
  'v' | 'b' | 'n' | 'm' | 'ctrl' | 'alt' | 'shift' | 'tab';
type AllowedKeys = keyof typeof EnglishStringToCodeMap;

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
  callback?: ( keysPressed: Keys[number] ) => void;
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
};

class KeyboardListener<Keys extends readonly OneKeyStroke[]> implements TInputListener {

  private readonly _keyStateTracker: KeyStateTracker;

  // The CallbackTimer that will manage firing when this listener supports fireOnHold.
  private readonly _timer?: CallbackTimer;

  // The KeyGroup that is currently "active", firing the callback because keys are pressed or just released.
  private _activeKeyGroup: KeyGroup<Keys> | null;

  // The function called when a KeyGroup is pressed (or just released).
  // TODO: callback should take a SCeneryEvent. https://github.com/phetsims/scenery/issues/1445
  private readonly _callback: ( keysPressed: Keys[number] ) => void;

  // Will it the callback fire on keys up or down?
  private readonly _fireOnKeyUp: boolean;

  // All of the KeyGroups of this listener from the keys provided in natural language.
  private readonly _keyGroups: KeyGroup<Keys>[];

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
    this._activeKeyGroup = null;

    // TODO: Creating a KeyStateTracker every listener seems unnecessary and inefficient. Can one globalKeyStateTracker
    //       drive everything but the listener queries it on keyboard events? FOr example, for a+b, hold a -> tab -> b, should work.
    this._keyStateTracker = new KeyStateTracker();

    // convert the provided keys to data that we can respond to with scenery's Input system
    this._keyGroups = this.convertKeysToKeyGroups( options.keys );

    if ( options.fireOnHold ) {
      this._timer = new CallbackTimer( {
        callback: this.fireCallback.bind( this ),
        delay: options.fireOnHoldDelay,
        interval: options.fireOnHoldInterval
      } );
    }
  }

  /**
   * Mostly required to fire with CallbackTimer since the callback cannot take arguments.
   */
  public fireCallback(): void {
    assert && assert( this._activeKeyGroup, 'Need an active keyGroup down to fire' );
    this._callback( this._activeKeyGroup!.naturalKeys );
  }

  public keydown( event: SceneryEvent ): void {

    // TODO: Ideally the tracker will soon go through scenery input system
    // TODO: Ideally we will query a globalKeyStateTracker instead?
    this._keyStateTracker.keydownUpdate( event.domEvent as KeyboardEvent );

    if ( !this._fireOnKeyUp && !this._activeKeyGroup ) {

      // modifier keys can be pressed in any order but the last key in the group must be pressed last
      this._keyGroups.forEach( keyGroup => {
        if ( this._keyStateTracker.areKeysDown( keyGroup.allKeys ) &&
             this._keyStateTracker.mostRecentKeyFromList( keyGroup.allKeys ) === keyGroup.key ) {

          this._activeKeyGroup = keyGroup;

          if ( this._timer ) {
            this._timer.start();
          }
          this.fireCallback();
        }
      } );
    }
  }

  public keyup( event: SceneryEvent ): void {

    // TODO: Ideally the tracker will soon go through scenery input system
    // TODO: Ideally we will query a globalKeyStateTracker instead?
    this._keyStateTracker.keyupUpdate( event.domEvent as KeyboardEvent );

    if ( this._activeKeyGroup ) {
      if ( !this._keyStateTracker.areKeysDown( this._activeKeyGroup.allKeys ) ) {
        if ( this._timer ) {
          this._timer.stop( false );
        }
        this._activeKeyGroup = null;
      }
    }

    if ( this._fireOnKeyUp ) {
      this._keyGroups.forEach( keyGroup => {
        if ( this._keyStateTracker.areKeysDown( keyGroup.modifierKeys ) &&
             KeyboardUtils.getEventCode( event.domEvent ) === keyGroup.key ) {
          this._activeKeyGroup = keyGroup;
          this.fireCallback();
          this._activeKeyGroup = null;
        }
      } );
    }
  }

  public cancel(): void {
    // TODO
  }

  public interrupt(): void {
    // TODO
  }

  public dispose(): void {
    if ( this._timer ) {
      this._timer.dispose();
    }
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
      return {
        key: key,
        modifierKeys: modifierKeys,
        naturalKeys: naturalKeys,
        allKeys: modifierKeys.concat( key )
      };
    } );

    return keyGroups;
  }
}

scenery.register( 'KeyboardListener', KeyboardListener );
export default KeyboardListener;
