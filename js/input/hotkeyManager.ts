// Copyright 2024, University of Colorado Boulder

/**
 * IMPORTANT: EXPERIMENTAL!
 * TODO DO NOT USE IN SIMULATIONS, SEE https://github.com/phetsims/scenery/issues/1621 FIRST
 *
 * Manages hotkeys based on two sources:
 *
 * 1. Global hotkeys (from globalHotkeyRegistry)
 * 2. Hotkeys from the current focus trail (FocusManager.pdomFocusProperty, all hotkeys on all input listeners of
 *    nodes in the trail)
 *
 * Manages key press state using EnglishKey from globalKeyStateTracker.
 * TODO: We need to gracefully handle when the user presses BOTH keys that correspond to an EnglishKey, e.g. https://github.com/phetsims/scenery/issues/1621
 * TODO: left-ctrl and right-ctrl, then releases one (it should still count ctrl as pressed).
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
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EnglishKey, eventCodeToEnglishString, FocusManager, globalHotkeyRegistry, globalKeyStateTracker, Hotkey, KeyboardUtils, metaEnglishKeys, scenery } from '../imports.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import TProperty from '../../../axon/js/TProperty.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';

const hotkeySetComparator = ( a: Set<Hotkey>, b: Set<Hotkey> ) => {
  return a.size === b.size && [ ...a ].every( element => b.has( element ) );
};

class HotkeyManager {

  // All hotkeys that are either globally or under the current focus trail
  private readonly availableHotkeysProperty: TReadOnlyProperty<Set<Hotkey>>;

  // Enabled hotkeys that are either global, or under the current focus trail
  private readonly enabledHotkeysProperty: TProperty<Set<Hotkey>> = new TinyProperty( new Set<Hotkey>() );

  private readonly englishKeysDownProperty: TProperty<Set<EnglishKey>> = new TinyProperty( new Set<EnglishKey>() );

  // The current set of modifier keys (pressed or not) based on current enabled hotkeys
  // TODO: Should we actually only have a set of modifier keys PER main key? https://github.com/phetsims/scenery/issues/1621
  // TODO: e.g. should "b+x" (b being pressed) prevent "y"? https://github.com/phetsims/scenery/issues/1621
  private modifierKeys: EnglishKey[] = [];

  // Hotkeys that are actively pressed
  private readonly activeHotkeys: Set<Hotkey> = new Set<Hotkey>();

  public constructor() {
    this.availableHotkeysProperty = new DerivedProperty( [
      globalHotkeyRegistry.hotkeysProperty,
      FocusManager.pdomFocusProperty
    ], ( globalHotkeys, focus ) => {
      // Always include global hotkeys. Use a set since we might have duplicates.
      const hotkeys = new Set<Hotkey>( globalHotkeys );

      // If we have focus, include the hotkeys from the focus trail
      if ( focus ) {
        for ( const node of focus.trail.nodes ) {
          // TODO: we might need to listen to things, if this is our list? https://github.com/phetsims/scenery/issues/1621
          if ( node.isDisposed || !node.isVisible() || !node.isInputEnabled() || !node.isPDOMVisible() ) {
            break;
          }

          node.inputListeners.forEach( listener => {
            listener.hotkeys?.forEach( hotkey => {
              hotkeys.add( hotkey );
            } );
          } );
        }
      }

      return hotkeys;
    }, {
      // We want to not over-notify, so we compare the sets directly
      valueComparisonStrategy: hotkeySetComparator
    } );

    // Update enabledHotkeysProperty when availableHotkeysProperty (or any enabledProperty) changes
    const rebuildHotkeys = () => {
      this.enabledHotkeysProperty.value = new Set( [ ...this.availableHotkeysProperty.value ].filter( hotkey => hotkey.enabledProperty.value ) );
    };
    // Because we can't add duplicate listeners, we create extra closures to have a unique handle for each hotkey
    const hotkeyRebuildListenerMap = new Map<Hotkey, () => void>(); // eslint-disable-line no-spaced-func
    this.availableHotkeysProperty.link( ( newHotkeys, oldHotkeys ) => {
      // Track whether any hotkeys changed. If none did, we don't need to rebuild.
      let hotkeysChanged = false;

      // Any old hotkeys and aren't in new hotkeys should be unlinked
      if ( oldHotkeys ) {
        for ( const hotkey of oldHotkeys ) {
          if ( !newHotkeys.has( hotkey ) ) {
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
        if ( !oldHotkeys || !oldHotkeys.has( hotkey ) ) {
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
          if ( !newHotkeys.has( hotkey ) && this.activeHotkeys.has( hotkey ) ) {
            this.removeActiveHotkey( hotkey, null, false );
          }
        }
      }

      // Re-check all hotkeys (since modifier keys might have changed, OR we need to validate that there are no conflicts).
      this.updateHotkeyStatus();
    } );

    // Track keydowns
    globalKeyStateTracker.keydownEmitter.addListener( keyboardEvent => {
      const keyCode = KeyboardUtils.getEventCode( keyboardEvent );

      if ( keyCode !== null ) {
        const englishKey = eventCodeToEnglishString( keyCode );
        if ( englishKey ) {
          this.onKeyDown( englishKey, keyboardEvent );
        }
      }
    } );

    // Track keyups
    globalKeyStateTracker.keyupEmitter.addListener( keyboardEvent => {
      const keyCode = KeyboardUtils.getEventCode( keyboardEvent );

      if ( keyCode !== null ) {
        const englishKey = eventCodeToEnglishString( keyCode );
        if ( englishKey ) {
          this.onKeyUp( englishKey, keyboardEvent );
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
    const englishKeysDown = this.englishKeysDownProperty.value;

    // If the main key isn't down, there's no way it could be active
    if ( !englishKeysDown.has( mainKey ) ) {
      return [];
    }

    const compatibleKeys = [ ...this.enabledHotkeysProperty.value ].filter( hotkey => {
      // Filter out hotkeys that don't have the main key
      if ( hotkey.key !== mainKey ) {
        return false;
      }

      // See whether the modifier keys match
      return this.modifierKeys.every( modifierKey => {
        return englishKeysDown.has( modifierKey ) === hotkey.modifierKeys.includes( modifierKey ) ||
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
  private updateHotkeyStatus(): void {
    for ( const hotkey of this.enabledHotkeysProperty.value ) {
      const shouldBeActive = this.getHotkeysForMainKey( hotkey.key ).includes( hotkey );
      const isActive = this.activeHotkeys.has( hotkey );

      if ( shouldBeActive && !isActive ) {
        this.addActiveHotkey( hotkey, null, false );
      }
      else if ( !shouldBeActive && isActive ) {
        this.removeActiveHotkey( hotkey, null, false );
      }
    }
  }

  /**
   * Hotkey made active/pressed
   */
  private addActiveHotkey( hotkey: Hotkey, keyboardEvent: KeyboardEvent | null, triggeredFromPress: boolean ): void {
    this.activeHotkeys.add( hotkey );
    hotkey.isPressedProperty.value = true;
    hotkey.interrupted = false;

    if ( triggeredFromPress && hotkey.fireOnDown ) {
      hotkey.fire( keyboardEvent );
    }
  }

  /**
   * Hotkey made inactive/released
   */
  private removeActiveHotkey( hotkey: Hotkey, keyboardEvent: KeyboardEvent | null, triggeredFromRelease: boolean ): void {
    hotkey.interrupted = !triggeredFromRelease;

    if ( triggeredFromRelease && !hotkey.fireOnDown ) {
      hotkey.fire( keyboardEvent );
    }

    hotkey.isPressedProperty.value = false;
    this.activeHotkeys.delete( hotkey );
  }

  private onKeyDown( englishKey: EnglishKey, keyboardEvent: KeyboardEvent ): void {
    if ( this.englishKeysDownProperty.value.has( englishKey ) ) {
      // Still pressed, got the browser/OS "fire on hold". See what hotkeys have the browser fire-on-hold behavior.

      // Handle re-entrancy (if something changes the state of activeHotkeys)
      for ( const hotkey of [ ...this.activeHotkeys ] ) {
        if ( hotkey.fireOnHold && hotkey.fireOnHoldTiming === 'browser' ) {
          hotkey.fire( keyboardEvent );
        }
      }
    }
    else {
      // Freshly pressed, was not pressed before. See if there is a hotkey to fire.

      this.englishKeysDownProperty.value = new Set( [ ...this.englishKeysDownProperty.value, englishKey ] );

      const hotkeys = this.getHotkeysForMainKey( englishKey );
      for ( const hotkey of hotkeys ) {
        this.addActiveHotkey( hotkey, keyboardEvent, true );
      }
    }

    this.updateHotkeyStatus();
  }

  private onKeyUp( englishKey: EnglishKey, keyboardEvent: KeyboardEvent ): void {

    const hotkeys = this.getHotkeysForMainKey( englishKey );
    for ( const hotkey of hotkeys ) {
      this.removeActiveHotkey( hotkey, keyboardEvent, true );
    }

    this.englishKeysDownProperty.value = new Set( [ ...this.englishKeysDownProperty.value ].filter( key => key !== englishKey ) );

    this.updateHotkeyStatus();
  }
}
scenery.register( 'HotkeyManager', HotkeyManager );

const hotkeyManager = new HotkeyManager();

export default hotkeyManager;