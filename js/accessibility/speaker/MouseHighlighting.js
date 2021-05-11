// Copyright 2021, University of Colorado Boulder

/**
 * A trait for Node that mixes functionality to support highlights that appear on hover with a mouse.
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
   *
   * @param {function} type - the constructor for Node
   */
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose MouseHighlighting' );

    const proto = type.prototype;

    /**
     * These properties and methods are put directly on the prototype of Node.
     */
    extend( proto, {

      /**
       * This should be called in the constructor to initialize MouseHighlighting.
       */
      initializeMouseHighlighting() {

        // @private {Display[]} - List of Displays that this Node is attached to. Mouse input will
        // activate the HighlightOverlay for each Display.
        this._displays = [];

        // @private {function} - listener that updates the list of Displays when a new Instance for this Node is
        // added/removed
        this.changedInstanceListener = this.onInstancesChanged.bind( this );
        this.changedInstanceEmitter.addListener( this.changedInstanceListener );

        // @private - input listener to update activation of HighlightOverlay upon pointer mouse input, using exit
        // and enter because the target of the event is this MouseHighlighting Node and we don't want to respond
        // to bubbled events
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
       * When the Instances for this Node are changed, update the list of Displays that this
       * Node is attached to.
       * @private
       *
       * @param {Instance} instance
       * @param {boolean} added - Was an instance added or removed?
       */
      onInstancesChanged( instance, added ) {
        const includesDisplay = _.includes( this._displays, instance.display );
        if ( added && !includesDisplay ) {
          this._displays.push( instance.display );
        }
        else if ( !added && includesDisplay ) {
          this._displays.splice( this._displays.indexOf( instance.display, 1 ) );
        }
      },

      /**
       * When a Pointer enters this Node, signal to the Displays that the pointer is over this Node so that the
       * HighlightOverlay can be activated.
       * @private
       *
       * @param {SceneryEvent} event
       */
      onPointerEntered( event ) {
        for ( let i = 0; i < this._displays.length; i++ ) {
          const display = this._displays[ i ];

          const highlightingNode = findMouseHighlightingNode( event.trail );
          const highlightTrail = event.trail.subtrailTo( highlightingNode );

          if ( display.pointerFocusProperty.value === null || !highlightTrail.equals( display.pointerFocusProperty.value.trail ) ) {
            display.pointerFocusProperty.set( new Focus( display, highlightTrail ) );
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
        for ( let i = 0; i < this._displays.length; i++ ) {
          const display = this._displays[ i ];
          display.pointerFocusProperty.set( null );
        }
      }
    } );
  }
};

/**
 * Helper function to find the Node that mixes MouseHighlighting from the provided Trail. We search backwards
 * up the trail so that the leaf-most Node of the event receives the highlighting.
 *
 * @param {Trail} trail
 * @returns {null|Trail}
 */
const findMouseHighlightingNode = trail => {
  let highlightingNode = null;
  for ( let i = trail.length - 1; i > 0; i-- ) {
    if ( trail.nodes[ i ].voicing ) {
      highlightingNode = trail.nodes[ i ];
      break;
    }
  }

  return highlightingNode;
};

scenery.register( 'MouseHighlighting', MouseHighlighting );
export default MouseHighlighting;
