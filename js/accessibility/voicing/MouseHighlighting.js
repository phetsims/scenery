// Copyright 2021, University of Colorado Boulder

/**
 * A trait for Node that mixes functionality to support visual highlights that appear on hover with a mouse.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import extend from '../../../../phet-core/js/extend.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import Focus from '../Focus.js';

const MouseHighlighting = {

  /**
   * Given the constructor for Node, add MouseHighlighting functions to the prototype.
   * @public
   * @trait {Node}
   * @param {function(new:Node)} type - the constructor for Node
   */
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose MouseHighlighting' );

    const proto = type.prototype;

    /**
     * These properties and methods are put directly on the prototype of Node.
     */
    extend( proto, {

      /**
       * @public
       * This should be called in the constructor to initialize MouseHighlighting.
       */
      initializeMouseHighlighting() {

        // @private - Input listener to activate the HighlightOverlay upon pointer mouse input. Uses exit
        // and enter instead of over and out because we do not want this to fire from bubbling. The highlight
        // should be around this Node when it receives input.
        this.enterExitListener = {
          enter: this.onPointerEntered.bind( this ),
          exit: this.onPointerExited.bind( this )
        };
        this.addInputListener( this.enterExitListener );
      },

      /**
       * @public
       */
      disposeMouseHighlighting() {
        this.changedInstanceEmitter.removeListener( this.changedInstanceListener );
        this.removeInputListener( this.enterExitListener );
      },

      /**
       * When a Pointer enters this Node, signal to the Displays that the pointer is over this Node so that the
       * HighlightOverlay can be activated.
       * @private
       *
       * @param {SceneryEvent} event
       */
      onPointerEntered( event ) {
        const displays = this.getConnectedDisplays();
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];

          if ( display.pointerFocusProperty.value === null || !event.trail.equals( display.pointerFocusProperty.value.trail ) ) {
            display.pointerFocusProperty.set( new Focus( display, event.trail ) );
          }
        }
      },

      /**
       * When a pointer exits this Node, signal to the Displays that pointer focus has changed to deactivate
       * the HighlightOverlay.
       * @private
       *
       * @param {SceneryEvent} event
       */
      onPointerExited( event ) {
        const displays = this.getConnectedDisplays();
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];
          display.pointerFocusProperty.set( null );
        }
      }
    } );
  }
};

scenery.register( 'MouseHighlighting', MouseHighlighting );
export default MouseHighlighting;
