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
        this.activationListener = {
          enter: this.onPointerEntered.bind( this ),
          exit: this.onPointerExited.bind( this ),
          down: this.onPointerDown.bind( this )
        };
        this.addInputListener( this.activationListener );

        const boundPointerReleaseListener = this.onPointerRelease.bind( this );
        const boundPointerCancel = this.onPointerCancel.bind( this );

        // @private - A reference to the Pointer so that we can add and remove listeners from it when necessary.
        // Since this is on the trait, only one pointer can have a listener for this Node that uses MouseHighlighting
        // at one time.
        this.pointer = null;

        // @private - Input listener that locks the HighlightOverlay so that there are no updates to the highlight
        // while the pointer is down over something that uses MouseHighlighting.
        this.pointerListener = {
          up: boundPointerReleaseListener,
          cancel: boundPointerCancel,
          interrupt: boundPointerCancel
        };
      },

      /**
       * @public
       */
      disposeMouseHighlighting() {
        this.removeInputListener( this.activationListener );
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

          if ( display.focusManager.pointerFocusProperty.value === null || !event.trail.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {
            display.focusManager.pointerFocusProperty.set( new Focus( display, event.trail ) );
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
          display.focusManager.pointerFocusProperty.set( null );
        }
      },

      /**
       * When a pointer goes down on this Node, signal to the Displays that the pointerFocus is locked
       * @private
       *
       * @param {SceneryEvent} event
       */
      onPointerDown( event ) {
        if ( this.pointer === null ) {
          const displays = this.getConnectedDisplays();
          for ( let i = 0; i < displays.length; i++ ) {
            const display = displays[ i ];
            display.focusManager.pointerFocusLockedProperty.set( true );
          }

          this.pointer = event.pointer;
          this.pointer.addInputListener( this.pointerListener );
        }
      },

      /**
       * When a Pointer goes up after going down on this Node, signal to the Displays that the pointerFocusProperty no
       * longer needs to be locked.
       * @private
       *
       * @param {SceneryEvent} [event] - may be called during interrupt or cancel, in which case there is no event
       */
      onPointerRelease( event ) {
        const displays = this.getConnectedDisplays();
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];
          display.focusManager.pointerFocusLockedProperty.set( false );
        }

        if ( this.pointer && this.pointer.listeners.includes( this.pointerListener ) ) {
          this.pointer.removeInputListener( this.pointerListener );
          this.pointer = null;
        }
      },

      /**
       * If the pointer listener is cancelled or interrupted, clear focus and remove input listeners.
       * @private
       *
       * @param event
       */
      onPointerCancel( event ) {
        const displays = this.getConnectedDisplays();
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];
          display.focusManager.pointerFocusProperty.set( null );
        }

        // unlock and remove pointer listeners
        this.onPointerRelease( event );
      }
    } );
  }
};

scenery.register( 'MouseHighlighting', MouseHighlighting );
export default MouseHighlighting;
