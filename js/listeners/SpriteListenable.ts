// Copyright 2020-2025, University of Colorado Boulder

/**
 * A trait to be mixed into PressListeners for identifying which SpriteInstance of a given Sprites node was interacted
 * with, AND will prevent interactions that are NOT over any SpriteInstances.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inheritance from '../../../phet-core/js/inheritance.js';
import memoize from '../../../phet-core/js/memoize.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import Node from '../nodes/Node.js';
import PressListener from '../listeners/PressListener.js';
import type { PressListenerEvent } from '../listeners/PressListener.js';
import scenery from '../scenery.js';
import SpriteInstance from '../util/SpriteInstance.js';
import Sprites from '../nodes/Sprites.js';

type TSpriteListenable = {
  spriteInstance: SpriteInstance | null;
};

/**
 * @param type - Should be a PressListener-based type
 */
const SpriteListenable = memoize( <SuperType extends Constructor<PressListener>>( type: SuperType ): SuperType & Constructor<TSpriteListenable> => {
  assert && assert( _.includes( inheritance( type ), PressListener ), 'Only PressListener subtypes should mix SpriteListenable' );

  return class extends type implements TSpriteListenable {

    public spriteInstance: SpriteInstance | null = null;

    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );
    }

    /**
     * @override - see PressListener
     */
    public override press( event: PressListenerEvent, targetNode?: Node, callback?: () => void ): boolean {
      // If pressed, then the press would be exited later AND we wouldn't want to override our spriteInstance anyway.
      if ( ( this as unknown as PressListener ).isPressed ) { return false; }

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
} );

scenery.register( 'SpriteListenable', SpriteListenable );
export default SpriteListenable;