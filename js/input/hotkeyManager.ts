// Copyright 2024, University of Colorado Boulder

/**
 * Manages hotkeys based on two sources:
 *
 * 1. Global hotkeys (from globalHotkeyRegistry)
 * 2. Hotkeys from the current focus trail (FocusManager.pdomFocusProperty, all hotkeys on all input listeners of
 *    nodes in the trail)
 *
 * Manages key press state using EnglishKey from globalKeyStateTracker.
 *
 * The "available" hotkeys are the union of the above two sources.
 *
 * The "enabled" hotkeys are the subset of available hotkeys whose enabledProperties are true.
 *
 * The "active" hotkeys are the subset of enabled hotkeys that are considered pressed. They will have fire-on-hold
 * behavior active.
 *
 * The set of enabled hotkeys determines the set of modifier keys that are considered "active" (in addition to
 * ctrl/alt/meta/shift, which are always included).
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EnglishKey, eventCodeToEnglishString, FocusManager, globalHotkeyRegistry, globalKeyStateTracker, Hotkey, KeyboardUtils, metaEnglishKeys, scenery } from '../imports.js';
import DerivedProperty, { UnknownDerivedProperty } from '../../../axon/js/DerivedProperty.js';
import TProperty from '../../../axon/js/TProperty.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';

const arrayComparator = <Key>( a: Key[], b: Key[] ): boolean => {
  return a.length === b.length && a.every( ( element, index ) => element === b[ index ] );
};

const setComparator = <Key>( a: Set<Key>, b: Set<Key> ) => {
  return a.size === b.size && [ ...a ].every( element => b.has( element ) );
};

class HotkeyManager {

  // All hotkeys that are either globally or under the current focus trail. They are ordered, so that the first
  // "identical key-shortcut" hotkey with override will be the one that is active.
  private readonly availableHotkeysProperty: UnknownDerivedProperty<Hotkey[]>;

  // Enabled hotkeys that are either global, or under the current focus trail
  private readonly enabledHotkeysProperty: TProperty<Hotkey[]> = new TinyProperty( [] );

  // The set of EnglishKeys that are currently pressed.
  private englishKeysDown: Set<EnglishKey> = new Set<EnglishKey>();

  // The current set of modifier keys (pressed or not) based on current enabled hotkeys
  // NOTE: Pressed modifier keys will prevent any other Hotkeys from becoming active. For example if you have a hotkey
  // with 'b+x', pressing 'b' will prevent any other hotkeys from becoming active.
  private modifierKeys: EnglishKey[] = [];

  // Hotkeys that are actively pressed
  private readonly activeHotkeys: Set<Hotkey> = new Set<Hotkey>();

  public constructor() {
    this.availableHotkeysProperty = new DerivedProperty( [
      globalHotkeyRegistry.hotkeysProperty,
      FocusManager.pdomFocusProperty
    ], ( globalHotkeys, focus ) => {
      const hotkeys: Hotkey[] = [];

      // If we have focus, include the hotkeys from the focus trail
      if ( focus ) {
        for ( const node of focus.trail.nodes.slice().reverse() ) {
          if ( !node.isInputEnabled() ) {
            break;
          }

          node.inputListeners.forEach( listener => {
            listener.hotkeys?.forEach( hotkey => {
              hotkeys.push( hotkey );
            } );
          } );
        }
      }

      // Always include global hotkeys. Use a set since we might have duplicates.
      hotkeys.push( ...globalHotkeys );

      return _.uniq( hotkeys );
    }, {
      // We want to not over-notify, so we compare the sets directly
      valueComparisonStrategy: arrayComparator
    } ) as UnknownDerivedProperty<Hotkey[]>;

    // If any of the nodes in the focus trail change inputEnabled, we need to recompute availableHotkeysProperty
    const onInputEnabledChanged = () => {
      this.availableHotkeysProperty.recomputeDerivation();
    };
    FocusManager.pdomFocusProperty.link( ( focus, oldFocus ) => {
      if ( oldFocus ) {
        oldFocus.trail.nodes.forEach( node => {
          node.inputEnabledProperty.unlink( onInputEnabledChanged );
        } );
      }

      if ( focus ) {
        focus.trail.nodes.forEach( node => {
          node.inputEnabledProperty.lazyLink( onInputEnabledChanged );
        } );
      }
    } );

    // Update enabledHotkeysProperty when availableHotkeysProperty (or any enabledProperty) changes
    const rebuildHotkeys = () => {
      const overriddenHotkeyStrings = new Set<string>();
      const enabledHotkeys: Hotkey[] = [];

      for ( const hotkey of this.availableHotkeysProperty.value ) {
        if ( hotkey.enabledProperty.value ) {
          // Each hotkey will have a canonical way to represent it, so we can check for duplicates when overridden.
          // Catch shift+ctrl+c and ctrl+shift+c as the same hotkey.
          const hotkeyCanonicalString = [
            ...hotkey.modifierKeys.slice().sort(),
            hotkey.key
          ].join( '+' );

          if ( !overriddenHotkeyStrings.has( hotkeyCanonicalString ) ) {
            enabledHotkeys.push( hotkey );

            if ( hotkey.override ) {
              overriddenHotkeyStrings.add( hotkeyCanonicalString );
            }
          }
        }
      }

      this.enabledHotkeysProperty.value = enabledHotkeys;
    };
    // Because we can't add duplicate listeners, we create extra closures to have a unique handle for each hotkey
    const hotkeyRebuildListenerMap = new Map<Hotkey, () => void>(); // eslint-disable-line no-spaced-func
    this.availableHotkeysProperty.link( ( newHotkeys, oldHotkeys ) => {
      // Track whether any hotkeys changed. If none did, we don't need to rebuild.
      let hotkeysChanged = false;

      // Any old hotkeys and aren't in new hotkeys should be unlinked
      if ( oldHotkeys ) {
        for ( const hotkey of oldHotkeys ) {
          if ( !newHotkeys.includes( hotkey ) ) {
            const listener = hotkeyRebuildListenerMap.get( hotkey )!;
            hotkeyRebuildListenerMap.delete( hotkey );
            assert && assert( listener );

            hotkey.enabledProperty.unlink( listener );
            hotkeysChanged = true;
          }
        }
      }

      // Any new hotkeys that aren't in old hotkeys should be linked
      for ( const hotkey of newHotkeys ) {
        if ( !oldHotkeys || !oldHotkeys.includes( hotkey ) ) {
          // Unfortunate. Perhaps in the future we could have an abstraction that makes a "count" of how many times we
          // are "listening" to a Property.
          const listener = () => rebuildHotkeys();
          hotkeyRebuildListenerMap.set( hotkey, listener );

          hotkey.enabledProperty.lazyLink( listener );
          hotkeysChanged = true;
        }
      }

      if ( hotkeysChanged ) {
        rebuildHotkeys();
      }
    } );

    // Update modifierKeys and whether each hotkey is currently pressed. This is how hotkeys can have their state change
    // from either themselves (or other hotkeys with modifier keys) being added/removed from enabledHotkeys.
    this.enabledHotkeysProperty.link( ( newHotkeys, oldHotkeys ) => {
      this.modifierKeys = _.uniq( [
        ...metaEnglishKeys,
        ...[ ...newHotkeys ].flatMap( hotkey => hotkey.modifierKeys )
      ] );

      // Remove any hotkeys that are no longer available or enabled
      if ( oldHotkeys ) {
        for ( const hotkey of oldHotkeys ) {
          if ( !newHotkeys.includes( hotkey ) && this.activeHotkeys.has( hotkey ) ) {
            this.removeActiveHotkey( hotkey, null, false );
          }
        }
      }

      // Re-check all hotkeys (since modifier keys might have changed, OR we need to validate that there are no conflicts).
      this.updateHotkeyStatus( null );
    } );

    // Track key state changes
    globalKeyStateTracker.keyDownStateChangedEmitter.addListener( ( keyboardEvent: KeyboardEvent | null ) => {
      const englishKeysDown = globalKeyStateTracker.getEnglishKeysDown();
      const englishKeysChanged = !setComparator( this.englishKeysDown, englishKeysDown );

      if ( englishKeysChanged ) {
        this.englishKeysDown = englishKeysDown;

        this.updateHotkeyStatus( keyboardEvent );
      }
      else {
        // No keys changed, got the browser/OS "fire on hold". See what hotkeys have the browser fire-on-hold behavior.

        // Handle re-entrancy (if something changes the state of activeHotkeys)
        for ( const hotkey of [ ...this.activeHotkeys ] ) {
          if ( hotkey.fireOnHold && hotkey.fireOnHoldTiming === 'browser' ) {
            hotkey.fire( keyboardEvent );
          }
        }
      }
    } );
  }

  /**
   * Given a main `key`, see if there is a hotkey that should be considered "active/pressed" for it.
   *
   * For a hotkey to be compatible, it needs to have:
   *
   * 1. Main key pressed
   * 2. All modifier keys in the hotkey's modifierKeys pressed
   * 3. All modifier keys not in the hotkey's modifierKeys (but in the other hotkeys above) not pressed
   */
  private getHotkeysForMainKey( mainKey: EnglishKey ): Hotkey[] {

    // If the main key isn't down, there's no way it could be active
    if ( !this.englishKeysDown.has( mainKey ) ) {
      return [];
    }

    const compatibleKeys = [ ...this.enabledHotkeysProperty.value ].filter( hotkey => {

      // Filter out hotkeys that don't have the main key
      if ( hotkey.key !== mainKey ) {
        return false;
      }

      // See whether the modifier keys match
      return this.modifierKeys.every( modifierKey => {
        return this.englishKeysDown.has( modifierKey ) === hotkey.keys.includes( modifierKey ) ||
               hotkey.ignoredModifierKeys.includes( modifierKey );
      } );
    } );

    if ( assert ) {
      const conflictingKeys = compatibleKeys.filter( hotkey => !hotkey.allowOverlap );

      assert && assert( conflictingKeys.length < 2, `Key conflict detected: ${conflictingKeys.map( hotkey => hotkey.getHotkeyString() )}` );
    }

    return compatibleKeys;
  }

  /**
   * Re-check all hotkey active/pressed states (since modifier keys might have changed, OR we need to validate that
   * there are no conflicts).
   */
  private updateHotkeyStatus( keyboardEvent: KeyboardEvent | null ): void {

    // For fireOnDown on/off cases, we only want to fire the hotkeys when we have a keyboard event specifying hotkey's
    // main `key`.
    const pressedOrReleasedKeyCode = KeyboardUtils.getEventCode( keyboardEvent );
    const pressedOrReleasedEnglishKey = pressedOrReleasedKeyCode ? eventCodeToEnglishString( pressedOrReleasedKeyCode ) : null;

    for ( const hotkey of this.enabledHotkeysProperty.value ) {

      // A hotkey should be  active if its main key is pressed. If it was interrupted, it can only become
      // active again if there was an actual key press event from the user. If a Hotkey is interrupted during
      // a press, it should remain inactive and interrupted until the NEXT press.
      const keyPressed = this.getHotkeysForMainKey( hotkey.key ).includes( hotkey );
      const notInterrupted = !hotkey.interrupted || ( keyboardEvent && keyboardEvent.type === 'keydown' );
      const shouldBeActive = keyPressed && notInterrupted;

      const isActive = this.activeHotkeys.has( hotkey );

      if ( shouldBeActive && !isActive ) {
        this.addActiveHotkey( hotkey, keyboardEvent, hotkey.key === pressedOrReleasedEnglishKey );
      }
      else if ( !shouldBeActive && isActive ) {
        this.removeActiveHotkey( hotkey, keyboardEvent, hotkey.key === pressedOrReleasedEnglishKey );
      }
    }
  }

  /**
   * Hotkey made active/pressed
   */
  private addActiveHotkey( hotkey: Hotkey, keyboardEvent: KeyboardEvent | null, triggeredFromPress: boolean ): void {
    this.activeHotkeys.add( hotkey );

    const shouldFire = triggeredFromPress && hotkey.fireOnDown;
    hotkey.onPress( keyboardEvent, shouldFire );
  }

  /**
   * Hotkey made inactive/released.
   */
  private removeActiveHotkey( hotkey: Hotkey, keyboardEvent: KeyboardEvent | null, triggeredFromRelease: boolean ): void {

    // Remove from activeHotkeys before Hotkey.onRelease so that we do not try to remove it again if there is
    // re-entrancy. This is possible if the release listener moves focus or interrupts a Hotkey.
    this.activeHotkeys.delete( hotkey );

    const shouldFire = triggeredFromRelease && !hotkey.fireOnDown;
    const interrupted = !triggeredFromRelease;
    hotkey.onRelease( keyboardEvent, interrupted, shouldFire );
  }

  /**
   * Called by Hotkey, removes the Hotkey from the active set when it is interrupted. The Hotkey cannot be active
   * again in this manager until there is an actual key press event from the user.
   */
  public interruptHotkey( hotkey: Hotkey ): void {
    assert && assert( hotkey.isPressedProperty.value, 'hotkey must be pressed to be interrupted' );
    this.removeActiveHotkey( hotkey, null, false );
  }
}

scenery.register( 'HotkeyManager', HotkeyManager );

const hotkeyManager = new HotkeyManager();

export default hotkeyManager;