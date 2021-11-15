// Copyright 2021, University of Colorado Boulder

/**
 * A trait for Node that mixes functionality to support visual highlights that appear on hover with a pointer.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import Shape from '../../../../kite/js/Shape.js';
import extend from '../../../../phet-core/js/extend.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import Focus from '../Focus.js';

// constants
// option keys for InteractiveHighlighting, each of these will have a setter and getter and values are applied with mutate()
const INTERACTIVE_HIGHLIGHTING_OPTIONS = [
  'interactiveHighlight',
  'interactiveHighlightLayerable'
];

const InteractiveHighlighting = {

  /**
   * Given the constructor for Node, add InteractiveHighlighting functions to the prototype.
   * @public
   * @trait {Node}
   * @param {function(new:Node)} type - the constructor for Node
   */
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose InteractiveHighlighting' );

    const proto = type.prototype;

    /**
     * These properties and methods are put directly on the prototype of Node.
     */
    extend( proto, {

      /**
       * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in
       * the order they will be evaluated.
       * @protected
       *
       * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
       *       cases that may apply.
       */
      _mutatorKeys: INTERACTIVE_HIGHLIGHTING_OPTIONS.concat( proto._mutatorKeys ),

      /**
       * This should be called in the constructor to initialize InteractiveHighlighting.
       * @public
       *
       * @param {Object} [options]
       */
      initializeInteractiveHighlighting( options ) {

        // @private {boolean} - Indicates that the Trait was initialized, to make sure that initializeInteractiveHighlighting
        // is called before using the Trait.
        this.interactiveHighlightingInitialized = true;

        // @private - Input listener to activate the HighlightOverlay upon pointer pointer input. Uses exit
        // and enter instead of over and out because we do not want this to fire from bubbling. The highlight
        // should be around this Node when it receives input.
        this.activationListener = {
          enter: this.onPointerEntered.bind( this ),
          move: this.onPointerMove.bind( this ),
          exit: this.onPointerExited.bind( this ),
          down: this.onPointerDown.bind( this )
        };

        // @private - A reference to the Pointer so that we can add and remove listeners from it when necessary.
        // Since this is on the trait, only one pointer can have a listener for this Node that uses InteractiveHighlighting
        // at one time.
        this.pointer = null;

        // @protected {Object} - A map that collects all of the Displays that this InteractiveHighlighting Node is
        // attached to, mapping the unique ID of the Instance Trail to the Display. We need a reference to the
        // Displays to activate the Focus Property associated with highlighting, and to add/remove listeners when
        // features that require highlighting are enabled/disabled. Note that this is updated asynchronously
        // (with updateDisplay) since Instances are added asynchronously.
        this._displays = {};

        // @private {Shape|Node|null} - The highlight that will surround this Node when it is activated and a Pointer
        // is currently over it. When null, the focus highlight will be used (as defined in ParallelDOM.js).
        this._interactiveHighlight = null;

        // @private {boolean} - If true, the highlight will be layerable in the scene graph instead of drawn
        // above everything in the HighlightOverlay. If true, you are responsible for adding the interactiveHighlight
        // in the location you want in the scene graph. The interactiveHighlight will become visible when
        // this.interactiveHighlightActivated is true.
        this._interactiveHighlightLayerable = null;

        // @private {TinyEmitter} - emits an event when the interactive highlight changes for this Node
        this.interactiveHighlightChangedEmitter = new TinyEmitter();

        // @private {function} - When new instances of this Node are created, adds an entry to the map of Displays.
        this.changedInstanceListener = this.onChangedInstance.bind( this );
        this.changedInstanceEmitter.addListener( this.changedInstanceListener );

        // @private {function} - Listener that adds/removes other listeners that activate highlights when
        // the feature becomes enabled/disabled so that we don't do extra work related to highlighting unless
        // it is necessary.
        this.interactiveHighlightingEnabledListener = this.onInteractiveHighlightingEnabledChange.bind( this );

        const boundPointerReleaseListener = this.onPointerRelease.bind( this );
        const boundPointerCancel = this.onPointerCancel.bind( this );

        // @private - Input listener that locks the HighlightOverlay so that there are no updates to the highlight
        // while the pointer is down over something that uses InteractiveHighlighting.
        this.pointerListener = {
          up: boundPointerReleaseListener,
          cancel: boundPointerCancel,
          interrupt: boundPointerCancel
        };

        // support passing options through initialize
        if ( options ) {
          this.mutate( _.pick( options, INTERACTIVE_HIGHLIGHTING_OPTIONS ) );
        }
      },

      /**
       * Whether or not a Node composes InteractiveHighlighting.
       * @public
       * @returns {boolean}
       */
      get isInteractiveHighlighting() {
        return true;
      },

      /**
       * Set the interactive highlight for this node. By default, the highlight will be a pink rectangle that surrounds
       * the node's local bounds.
       * @public
       *
       * @param {Node|Shape|null} interactiveHighlight
       */
      setInteractiveHighlight( interactiveHighlight ) {
        assert && assert( interactiveHighlight === null ||
                          interactiveHighlight instanceof Node ||
                          interactiveHighlight instanceof Shape );

        if ( this._interactiveHighlight !== interactiveHighlight ) {
          this._interactiveHighlight = interactiveHighlight;

          if ( this._interactiveHighlightLayerable ) {

            // if focus highlight is layerable, it must be a node for the scene graph
            assert && assert( interactiveHighlight instanceof Node );

            // make sure the highlight is invisible, the HighlightOverlay will manage visibility
            interactiveHighlight.visible = false;
          }

          // cannot emit until initialize complete
          if ( this.interactiveHighlightingInitialized ) {
            this.interactiveHighlightChangedEmitter.emit();
          }
        }
      },
      set interactiveHighlight( interactiveHighlight ) { this.setInteractiveHighlight( interactiveHighlight ); },

      /**
       * Returns the interactive highlight for this Node.
       * @public
       *
       * @returns {null|Node|Shape}
       */
      getInteractiveHighlight() {
        return this._interactiveHighlight;
      },
      get interactiveHighlight() { return this.getInteractiveHighlight(); },

      /**
       * Sets whether the highlight is layerable in the scene graph instead of above everyting in the
       * highlight overlay. If layerable, you must provide a custom highlight and it must be a Node. The highlight
       * Node will always be invisible unless this Node is activated with a pointer.
       * @public
       *
       * @param {boolean} interactiveHighlightLayerable
       */
      setInteractiveHighlightLayerable( interactiveHighlightLayerable ) {
        if ( this._interactiveHighlightLayerable !== interactiveHighlightLayerable ) {
          this._interactiveHighlightLayerable = interactiveHighlightLayerable;

          if ( this._interactiveHighlight ) {
            assert && assert( this._interactiveHighlight instanceof Node );
            this._interactiveHighlight.visible = false;

            if ( this.interactiveHighlightingInitialized ) {
              this.interactiveHighlightChangedEmitter.emit();
            }
          }
        }
      },
      set interactiveHighlightLayerable( interactiveHighlightLayerable ) { this.setInteractiveHighlightLayerable( interactiveHighlightLayerable ); },

      /**
       * Get whether or not the interactive highlight is layerable in the scene graph.
       * @public
       *
       * @returns {null|boolean}
       */
      getInteractiveHighlightLayerable() {
        return this._interactiveHighlightLayerable;
      },
      get interactiveHighlightLayerable() { return this.getInteractiveHighlightLayerable(); },

      /**
       * Returns true if this Node is "activated" by a pointer, indicating that a Pointer is over it
       * and this Node mixes InteractiveHighlighting so an interactive highlight should surround it.
       * @public
       *
       * @returns {boolean}
       */
      isInteractiveHighlightActivated() {
        let activated = false;

        const trailIds = Object.keys( this._displays );
        for ( let i = 0; i < trailIds.length; i++ ) {
          const pointerFocus = this._displays[ trailIds[ i ] ].focusManager.pointerFocusProperty.value;
          if ( pointerFocus && pointerFocus.trail.lastNode() === this ) {
            activated = true;
            break;
          }
        }
        return activated;
      },
      get interactiveHighlightActivated() { return this.isInteractiveHighlightActivated(); },

      /**
       * @public
       */
      disposeInteractiveHighlighting() {
        this.interactiveHighlightingInitialized = false;
        this.changedInstanceEmitter.removeListener( this.changedInstanceListener );

        // remove the activation listener if it is currently attached
        if ( this.hasInputListener( this.activationListener ) ) {
          this.removeInputListener( this.activationListener );
        }

        // remove listeners on displays and remove Displays from the map
        const trailIds = Object.keys( this._displays );
        for ( let i = 0; i < trailIds.length; i++ ) {
          const display = this._displays[ trailIds[ i ] ];

          display.focusManager.pointerHighlightsVisibleProperty.unlink( this.interactiveHighlightingEnabledListener );
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
        assert && assert( this.interactiveHighlightingInitialized, 'InteractiveHighlighting should be initialized before using onPointerEntered' );

        const displays = Object.values( this._displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];

          if ( display.focusManager.pointerFocusProperty.value === null || !event.trail.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {
            display.focusManager.pointerFocusProperty.set( new Focus( display, event.trail ) );
          }
        }
      },

      onPointerMove( event ) {
        assert && assert( this.interactiveHighlightingInitialized, 'InteractiveHighlighting should be initialized before using onPointerMove' );

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
        assert && assert( this.interactiveHighlightingInitialized, 'InteractiveHighlighting should be initialized before using onPointerExited' );

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
        assert && assert( this.interactiveHighlightingInitialized, 'InteractiveHighlighting should be initialized before using onPointerDown' );

        if ( this.pointer === null ) {
          const displays = Object.values( this._displays );
          for ( let i = 0; i < displays.length; i++ ) {
            const display = displays[ i ];
            const focus = display.focusManager.pointerFocusProperty.value;

            // focus should generally be defined when pointer enters the Node, but it may be null in cases of
            // cancel or interrupt
            if ( focus ) {

              // Set the lockedPointerFocusProperty with a copy of the Focus (as deep as possible) because we want
              // to keep a reference to the old Trail while pointerFocusProperty changes.
              display.focusManager.lockedPointerFocusProperty.set( new Focus( focus.display, focus.trail.copy() ) );
            }
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
        assert && assert( this.interactiveHighlightingInitialized, 'InteractiveHighlighting should be initialized before using onPointerRelease' );

        const displays = Object.values( this._displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];
          display.focusManager.lockedPointerFocusProperty.value = null;
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
        assert && assert( this.interactiveHighlightingInitialized, 'InteractiveHighlighting should be initialized before using onPointerCancel' );

        const displays = Object.values( this._displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];
          display.focusManager.pointerFocusProperty.set( null );
        }

        // unlock and remove pointer listeners
        this.onPointerRelease( event );
      },

      /**
       * Add or remove listeners related to activating interactive highlighting when the feature becomes enabled.
       * This way we prevent doing work related to interactive highlighting unless the feature is enabled.
       * @private
       *
       * @param {boolean} enabled
       */
      onInteractiveHighlightingEnabledChange( enabled ) {
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
        assert && assert( this.interactiveHighlightingInitialized, 'InteractiveHighlighting should be initialized before the onChangedInstance lister is called' );

        if ( added ) {
          this._displays[ instance.trail.uniqueId ] = instance.display;

          // Listener may already by on the display in cases of DAG, only add if this is the first instance of this Node
          if ( !instance.display.focusManager.pointerHighlightsVisibleProperty.hasListener( this.interactiveHighlightingEnabledListener ) ) {
            instance.display.focusManager.pointerHighlightsVisibleProperty.link( this.interactiveHighlightingEnabledListener );
          }
        }
        else {
          const display = this._displays[ instance.trail.uniqueId ];

          // If the node was disposed, this display reference has already been cleaned up, but instances are updated
          // (disposed) on the next frame after the node was disposed. Only unlink if there are no more instances of
          // this node;
          if ( display && instance.node.instances.length === 0 ) {
            display.focusManager.pointerHighlightsVisibleProperty.unlink( this.interactiveHighlightingEnabledListener );
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

        // if any of the nodes from leaf to self use InteractiveHighlighting, they should receive input, and we shouldn't
        // speak the content for this Node
        let descendantsUseVoicing = false;
        for ( let i = 0; i < childToLeafNodes.length; i++ ) {
          if ( childToLeafNodes[ i ].isInteractiveHighlighting ) {
            descendantsUseVoicing = true;
            break;
          }
        }

        return descendantsUseVoicing;
      }
    } );
  }
};

scenery.register( 'InteractiveHighlighting', InteractiveHighlighting );
export default InteractiveHighlighting;
