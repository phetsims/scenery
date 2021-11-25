// Copyright 2020, University of Colorado Boulder

/**
 * A trait to be mixed into PressListeners for identifying which SpriteInstance of a given Sprites node was interacted
 * with, AND will prevent interactions that are NOT over any SpriteInstances.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inheritance from '../../../phet-core/js/inheritance.js';
import { scenery, Sprites, PressListener } from '../imports.js';

/**
 * @param {constructor} type - Should be a PressListener-based type
 */
const SpriteListenable = type => {
  assert && assert( _.includes( inheritance( type ), PressListener ), 'Only PressListener subtypes should mix SpriteListenable' );

  return class extends type {
    /**
     * @param {*} ...args
     */
    constructor( ...args ) {
      super( ...args );

      // @public {SpriteInstance}
      this.spriteInstance = null;
    }

    /**
     * Moves the listener to the 'pressed' state if possible (attaches listeners and initializes press-related
     * properties).
     * @public
     * @override
     *
     * This can be overridden (with super-calls) when custom press behavior is needed for a type.
     *
     * This can be called by outside clients in order to try to begin a process (generally on an already-pressed
     * pointer), and is useful if a 'drag' needs to change between listeners. Use canPress( event ) to determine if
     * a press can be started (if needed beforehand).
     *
     * @param {SceneryEvent} event
     * @param {Node} [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
     *                              forwarded presses.
     * @param {function} [callback] - to be run at the end of the function, but only on success
     * @returns {boolean} success - Returns whether the press was actually started
     */
    press( event, targetNode, callback ) {
      // If pressed, then the press would be exited later AND we wouldn't want to override our spriteInstance anyway.
      if ( this.isPressed ) { return false; }

      // Zero it out, so we only respond to Sprites instances.
      this.spriteInstance = null;

      if ( event.currentTarget instanceof Sprites ) {
        const sprites = event.currentTarget;

        this.spriteInstance = sprites.getSpriteInstanceFromPoint( sprites.globalToLocalPoint( event.pointer.point ) );
      }

      // If we have no instance, don't super-call (same behavior for never starting a press)
      if ( this.spriteInstance ) {
        return super.press( event, targetNode, callback );
      }
      else {
        return false;
      }
    }
  };
};

scenery.register( 'SpriteListenable', SpriteListenable );
export default SpriteListenable;