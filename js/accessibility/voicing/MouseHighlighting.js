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

// REVIEW: Would we want to name this like the designed feature, i.e. "InteractiveHighlights" ? https://github.com/phetsims/scenery/issues/1223
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
       * @public
       * This should be called in the constructor to initialize MouseHighlighting.
       */
      initializeMouseHighlighting() {

        // @private {Display[]} - List of Displays that this Node is attached to. Input listeners on this Node
        // may activate the highlights of HighlightOverlay to show a highlight around this Node.
        // REVIEW: Can we get this from the Node? Like with the new getConnectedDisplays(). If not please explain why we need our own list here.
        this._displays = [];

        // @private {function} - listener that updates the list of Displays when a new Instance for this Node is
        // added/removed
        this.changedInstanceListener = this.onInstancesChanged.bind( this );
        this.changedInstanceEmitter.addListener( this.changedInstanceListener );

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
       * When the Instances for this Node are changed, update the list of Displays that this
       * Node is attached to.
       * @private
       *
       * @param {Instance} instance
       * @param {boolean} added - Was an instance added or removed?
       * // REVIEW: Likely rename this to singular "Instance" https://github.com/phetsims/scenery/issues/1223
       */
      onInstancesChanged( instance, added ) {
        const indexOfDisplay = this._displays.indexOf( instance.display );
        if ( added && indexOfDisplay < 0 ) {
          this._displays.push( instance.display );
        }
        else if ( !added && indexOfDisplay >= 0 ) {
          this._displays.splice( indexOfDisplay, 1 );
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
        for ( let i = 0; i < this._displays.length; i++ ) {
          const display = this._displays[ i ];
          display.pointerFocusProperty.set( null );
        }
      }
    } );
  }
};

scenery.register( 'MouseHighlighting', MouseHighlighting );
export default MouseHighlighting;
