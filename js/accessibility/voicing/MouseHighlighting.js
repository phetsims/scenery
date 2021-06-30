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

        // @public {boolean} (read-only)
        this.isMouseHighlighting = true;

        // @private - Input listener to activate the HighlightOverlay upon pointer mouse input. Uses exit
        // and enter instead of over and out because we do not want this to fire from bubbling. The highlight
        // should be around this Node when it receives input.
        this.activationListener = {
          enter: this.onPointerEntered.bind( this ),
          move: this.onPointerMove.bind( this ),
          exit: this.onPointerExited.bind( this ),
          down: this.onPointerDown.bind( this )
        };

        // @private - A reference to the Pointer so that we can add and remove listeners from it when necessary.
        // Since this is on the trait, only one pointer can have a listener for this Node that uses MouseHighlighting
        // at one time.
        this.pointer = null;

        // @protected {Object} - A map that collects all of the Displays that this MouseHighlighting Node is
        // attached to, mapping the unique ID of the Instance Trail to the Display. We need a reference to the
        // Displays to activate the Focus Property associated with highlighting, and to add/remove listeners when
        // features that require highlighting are enabled/disabled. Note that this is updated asynchronously
        // (with updateDisplay) since Instances are added asynchronously.
        this._displays = {};

        // @private {function} - When new instances of this Node are created, adds an entry to the map of Displays.
        this.changedInstanceListener = this.onChangedInstance.bind( this );
        this.changedInstanceEmitter.addListener( this.changedInstanceListener );

        // @private {function} - Listener that adds/removes other listeners that activate highlights when
        // the feature becomes enabled/disabled so that we don't do extra work related to highlighting unless
        // it is necessary.
        this.mouseHighlightingEnabledListener = this.onMouseHighlightingEnabledChange.bind( this );

        const boundPointerReleaseListener = this.onPointerRelease.bind( this );
        const boundPointerCancel = this.onPointerCancel.bind( this );

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
        this.changedInstanceEmitter.removeListener( this.changedInstanceListener );

        // remove the activation listener if it is currently attached
        if ( this.hasInputListener( this.activationListener ) ) {
          this.removeInputListener( this.activationListener );
        }

        // remove listeners on displays and remove Displays from the map
        const trailIds = Object.keys( this._displays );
        for ( let i = 0; i < trailIds.length; i++ ) {
          const display = this._displays[ trailIds[ i ] ];

          display.focusManager.pointerHighlightsVisibleProperty.unlink( this.mouseHighlightingEnabledListener );
          delete this._displays[ trailIds[ i ] ];
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
        const displays = Object.values( this._displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];

          if ( display.focusManager.pointerFocusProperty.value === null || !event.trail.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {
            display.focusManager.pointerFocusProperty.set( new Focus( display, event.trail ) );
          }
        }
      },

      onPointerMove( event ) {

        const displays = Object.values( this._displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];

          // the SceneryEvent might have gone through a descendant of this Node
          const rootToSelf = event.trail.subtrailTo( this );

          // only do more work on move if the event indicates that pointer focus might have changed
          if ( display.focusManager.pointerFocusProperty.value === null || !rootToSelf.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {

            if ( !this.getDescendantsUseHighlighting( event.trail ) ) {
              display.focusManager.pointerFocusProperty.set( new Focus( display, rootToSelf ) );
            }
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
        const displays = Object.values( this._displays );
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
          const displays = Object.values( this._displays );
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
        const displays = Object.values( this._displays );
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
        const displays = Object.values( this._displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];
          display.focusManager.pointerFocusProperty.set( null );
        }

        // unlock and remove pointer listeners
        this.onPointerRelease( event );
      },

      /**
       * Add or remove listeners related to activating mouse highlighting when the feature becomes enabled. This way
       * we prevent doing work related to mouse highlighting unless the feature is enabled.
       * @private
       *
       * @param {boolean} enabled
       */
      onMouseHighlightingEnabledChange( enabled ) {
        const hasActivationListener = this.hasInputListener( this.activationListener );
        if ( enabled && !hasActivationListener ) {
          this.addInputListener( this.activationListener );
        }
        else if ( !enabled && hasActivationListener ) {
          this.removeInputListener( this.activationListener );
        }
      },

      /**
       * Add the Display to the collection when this Node is added to a scene graph. Also adds listeners to the
       * Display that turns on highlighting when the feature is enabled.
       *
       * @param {Instance} instance
       * @param {boolean} added - whether the instance is added or removed from the Instance tree
       */
      onChangedInstance( instance, added ) {
        if ( added ) {
          this._displays[ instance.trail.uniqueId ] = instance.display;

          // Listener may already by on the display in cases of DAG, only add if this is the first instance of this Node
          if ( !instance.display.focusManager.pointerHighlightsVisibleProperty.hasListener( this.mouseHighlightingEnabledListener ) ) {
            instance.display.focusManager.pointerHighlightsVisibleProperty.link( this.mouseHighlightingEnabledListener );
          }
        }
        else {
          const display = this._displays[ instance.trail.uniqueId ];
          assert && assert( display, 'about to remove listeners from Display Properties, but could not find Display' );

          // only unlink if there are no more instances of this Node
          if ( instance.node.instances.length === 0 ) {
            display.focusManager.pointerHighlightsVisibleProperty.unlink( this.mouseHighlightingEnabledListener );
          }

          delete this._displays[ instance.trail.uniqueId ];
        }
      },

      /**
       * Returns true if any nodes from this Node to the leaf of the Trail use Voicing features in some way. In
       * general, we do not want to activate voicing features in this case because the leaf-most Nodes in the Trail
       * should be activated instead.
       * @protected
       *
       * @param trail
       * @returns {boolean}
       */
      getDescendantsUseHighlighting( trail ) {
        const indexOfSelf = trail.nodes.indexOf( this );

        // all the way to length, end not included in slice - and if start value is greater than index range
        // an empty array is returned
        const childToLeafNodes = trail.nodes.slice( indexOfSelf + 1, trail.nodes.length );

        // if any of the nodes from leaf to self use MouseHighlighting, they should receive input, and we shouldn't
        // speak the content for this Node
        let descendantsUseVoicing = false;
        for ( let i = 0; i < childToLeafNodes.length; i++ ) {
          if ( childToLeafNodes[ i ].isMouseHighlighting ) {
            descendantsUseVoicing = true;
            break;
          }
        }

        return descendantsUseVoicing;
      }
    } );
  }
};

scenery.register( 'MouseHighlighting', MouseHighlighting );
export default MouseHighlighting;
