// Copyright 2021-2022, University of Colorado Boulder

/**
 * A trait for Node that mixes functionality to support visual highlights that appear on hover with a pointer.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import Shape from '../../../../kite/js/Shape.js';
import Constructor from '../../../../phet-core/js/Constructor.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import { Display, Focus, IInputListener, Instance, Node, Pointer, scenery, SceneryEvent, Trail } from '../../imports.js';

// constants
// option keys for InteractiveHighlighting, each of these will have a setter and getter and values are applied with mutate()
const INTERACTIVE_HIGHLIGHTING_OPTIONS = [
  'interactiveHighlight',
  'interactiveHighlightLayerable'
];

type InteractiveHighlightingOptions = {
  interactiveHighlight?: Node | Shape | null,
  interactiveHighlightLayerable?: boolean
};

/**
 * @param Type
 * @param optionsArgPosition - zero-indexed number that the options argument is provided at
 */
const InteractiveHighlighting = <SuperType extends Constructor>( Type: SuperType, optionsArgPosition: number ) => {
  assert && assert( _.includes( inheritance( Type ), Node ), 'Only Node subtypes should compose InteractiveHighlighting' );

  const InteractiveHighlightingClass = class extends Type {
    activationListener: IInputListener; // TODO: use underscore so that there is a "private" convention. https://github.com/phetsims/scenery/issues/1348
    pointer: null | Pointer; // TODO: use underscore so that there is a "private" convention. https://github.com/phetsims/scenery/issues/1348
    _displays: { [ key: string ]: Display }; // TODO: this should be protected, how to conventionize this?. https://github.com/phetsims/scenery/issues/1340
    _interactiveHighlight: Shape | Node | null;
    _interactiveHighlightLayerable: boolean;
    interactiveHighlightChangedEmitter: TinyEmitter<[]>; // TODO: use underscore so that there is a "private" convention. https://github.com/phetsims/scenery/issues/1348
    changedInstanceListener: ( instance: Instance, added: boolean ) => void;
    interactiveHighlightingEnabledListener: ( enabled: boolean ) => void;
    pointerListener: IInputListener; // TODO: use underscore so that there is a "private" convention. https://github.com/phetsims/scenery/issues/1348

    constructor( ...args: any[] ) {

      const providedOptions = ( args[ optionsArgPosition ] || {} ) as InteractiveHighlightingOptions;

      const interactiveHighlightingOptions = _.pick( providedOptions, INTERACTIVE_HIGHLIGHTING_OPTIONS );
      args[ optionsArgPosition ] = _.omit( providedOptions, INTERACTIVE_HIGHLIGHTING_OPTIONS );

      super( ...args );

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

      // @protected - A map that collects all of the Displays that this InteractiveHighlighting Node is
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
      this._interactiveHighlightLayerable = false;

      // @private {TinyEmitter} - emits an event when the interactive highlight changes for this Node
      this.interactiveHighlightChangedEmitter = new TinyEmitter();

      // @private {function} - When new instances of this Node are created, adds an entry to the map of Displays.
      this.changedInstanceListener = this.onChangedInstance.bind( this );
      ( this as unknown as Node ).changedInstanceEmitter.addListener( this.changedInstanceListener );

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

      // @ts-ignore
      ( this as unknown as Node ).mutate( interactiveHighlightingOptions );
    }

    /**
     * Whether or not a Node composes InteractiveHighlighting.
     * @public
     * @returns {boolean}
     */
    get isInteractiveHighlighting() {
      return true;
    }

    /**
     * Set the interactive highlight for this node. By default, the highlight will be a pink rectangle that surrounds
     * the node's local bounds.
     * @public
     *
     * @param {Node|Shape|null} interactiveHighlight
     */
    setInteractiveHighlight( interactiveHighlight: Node | Shape | null ) {

      if ( this._interactiveHighlight !== interactiveHighlight ) {
        this._interactiveHighlight = interactiveHighlight;

        if ( this._interactiveHighlightLayerable ) {

          // if focus highlight is layerable, it must be a node for the scene graph
          assert && assert( interactiveHighlight instanceof Node );

          // make sure the highlight is invisible, the HighlightOverlay will manage visibility
          ( interactiveHighlight as Node ).visible = false;
        }

        this.interactiveHighlightChangedEmitter.emit();
      }
    }

    set interactiveHighlight( interactiveHighlight ) { this.setInteractiveHighlight( interactiveHighlight ); }

    /**
     * Returns the interactive highlight for this Node.
     * @public
     *
     * @returns {null|Node|Shape}
     */
    getInteractiveHighlight() {
      return this._interactiveHighlight;
    }

    get interactiveHighlight() { return this.getInteractiveHighlight(); }

    /**
     * Sets whether the highlight is layerable in the scene graph instead of above everyting in the
     * highlight overlay. If layerable, you must provide a custom highlight and it must be a Node. The highlight
     * Node will always be invisible unless this Node is activated with a pointer.
     */
    setInteractiveHighlightLayerable( interactiveHighlightLayerable: boolean ) {
      if ( this._interactiveHighlightLayerable !== interactiveHighlightLayerable ) {
        this._interactiveHighlightLayerable = interactiveHighlightLayerable;

        if ( this._interactiveHighlight ) {
          assert && assert( this._interactiveHighlight instanceof Node );
          ( this._interactiveHighlight as Node ).visible = false;

          this.interactiveHighlightChangedEmitter.emit();
        }
      }
    }

    set interactiveHighlightLayerable( interactiveHighlightLayerable ) { this.setInteractiveHighlightLayerable( interactiveHighlightLayerable ); }

    /**
     * Get whether or not the interactive highlight is layerable in the scene graph.
     * @public
     *
     * @returns {null|boolean}
     */
    getInteractiveHighlightLayerable() {
      return this._interactiveHighlightLayerable;
    }

    get interactiveHighlightLayerable() { return this.getInteractiveHighlightLayerable(); }

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

        // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
        if ( pointerFocus && pointerFocus.trail.lastNode() === this ) {
          activated = true;
          break;
        }
      }
      return activated;
    }

    get interactiveHighlightActivated() { return this.isInteractiveHighlightActivated(); }

    dispose() {
      const thisNode = this as unknown as Node;
      thisNode.changedInstanceEmitter.removeListener( this.changedInstanceListener );

      // remove the activation listener if it is currently attached
      if ( thisNode.hasInputListener( this.activationListener ) ) {
        thisNode.removeInputListener( this.activationListener );
      }

      // remove listeners on displays and remove Displays from the map
      const trailIds = Object.keys( this._displays );
      for ( let i = 0; i < trailIds.length; i++ ) {
        const display = this._displays[ trailIds[ i ] ];

        // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
        display.focusManager.pointerHighlightsVisibleProperty.unlink( this.interactiveHighlightingEnabledListener );
        delete this._displays[ trailIds[ i ] ];
      }

      // @ts-ignore
      super.dispose && super.dispose();
    }

    /**
     * When a Pointer enters this Node, signal to the Displays that the pointer is over this Node so that the
     * HighlightOverlay can be activated.
     * TODO: we want this to be @private, https://github.com/phetsims/scenery/issues/1348
     *
     */
    onPointerEntered( event: SceneryEvent ) {

      const displays = Object.values( this._displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];

        // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
        if ( display.focusManager.pointerFocusProperty.value === null || !event.trail.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {

          // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
          display.focusManager.pointerFocusProperty.set( new Focus( display, event.trail ) );
        }
      }
    }

    onPointerMove( event: SceneryEvent ) {

      const displays = Object.values( this._displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];

        // the SceneryEvent might have gone through a descendant of this Node
        const rootToSelf = event.trail.subtrailTo( this as unknown as Node );

        // only do more work on move if the event indicates that pointer focus might have changed
        // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
        if ( display.focusManager.pointerFocusProperty.value === null || !rootToSelf.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {

          if ( !this.getDescendantsUseHighlighting( event.trail ) ) {

            // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
            display.focusManager.pointerFocusProperty.set( new Focus( display, rootToSelf ) );
          }
        }
      }
    }

    /**
     * When a pointer exits this Node, signal to the Displays that pointer focus has changed to deactivate
     * the HighlightOverlay.
     * TODO: we want this to be @private, https://github.com/phetsims/scenery/issues/1348
     */
    onPointerExited( event: SceneryEvent ) {

      const displays = Object.values( this._displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];
        display.focusManager.pointerFocusProperty.set( null );
      }
    }

    /**
     * When a pointer goes down on this Node, signal to the Displays that the pointerFocus is locked
     * TODO: we want this to be @private, https://github.com/phetsims/scenery/issues/1348
     */
    onPointerDown( event: SceneryEvent ) {

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
            // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
            display.focusManager.lockedPointerFocusProperty.set( new Focus( focus.display, focus.trail.copy() ) );
          }
        }

        this.pointer = event.pointer;
        this.pointer.addInputListener( this.pointerListener );
      }
    }

    /**
     * When a Pointer goes up after going down on this Node, signal to the Displays that the pointerFocusProperty no
     * longer needs to be locked.
     * TODO: we want this to be @private, https://github.com/phetsims/scenery/issues/1348
     *
     * @param {SceneryEvent} [event] - may be called during interrupt or cancel, in which case there is no event
     */
    onPointerRelease( event?: SceneryEvent ) {

      const displays = Object.values( this._displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];
        display.focusManager.lockedPointerFocusProperty.value = null;
      }

      if ( this.pointer && this.pointer.listeners.includes( this.pointerListener ) ) {
        this.pointer.removeInputListener( this.pointerListener );
        this.pointer = null;
      }
    }

    /**
     * If the pointer listener is cancelled or interrupted, clear focus and remove input listeners.
     * TODO: we want this to be @private, https://github.com/phetsims/scenery/issues/1348
     *
     * @param event
     */
    onPointerCancel( event?: SceneryEvent ) {

      const displays = Object.values( this._displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];
        display.focusManager.pointerFocusProperty.set( null );
      }

      // unlock and remove pointer listeners
      this.onPointerRelease( event );
    }

    /**
     * Add or remove listeners related to activating interactive highlighting when the feature becomes enabled.
     * This way we prevent doing work related to interactive highlighting unless the feature is enabled.
     * TODO: we want this to be @private, https://github.com/phetsims/scenery/issues/1348
     */
    onInteractiveHighlightingEnabledChange( enabled: boolean ) {
      const thisNode = this as unknown as Node;

      const hasActivationListener = thisNode.hasInputListener( this.activationListener );
      if ( enabled && !hasActivationListener ) {
        thisNode.addInputListener( this.activationListener );
      }
      else if ( !enabled && hasActivationListener ) {
        thisNode.removeInputListener( this.activationListener );
      }
    }

    /**
     * Add the Display to the collection when this Node is added to a scene graph. Also adds listeners to the
     * Display that turns on highlighting when the feature is enabled.
     */
    onChangedInstance( instance: Instance, added: boolean ): void {
      assert && assert( instance.trail, 'should have a trail' );

      if ( added ) {
        this._displays[ instance.trail!.uniqueId ] = instance.display;

        // Listener may already by on the display in cases of DAG, only add if this is the first instance of this Node
        if ( !instance.display.focusManager.pointerHighlightsVisibleProperty.hasListener( this.interactiveHighlightingEnabledListener ) ) {
          instance.display.focusManager.pointerHighlightsVisibleProperty.link( this.interactiveHighlightingEnabledListener );
        }
      }
      else {
        assert && assert( instance.node, 'should have a node' );
        const display = this._displays[ instance.trail!.uniqueId ];

        // If the node was disposed, this display reference has already been cleaned up, but instances are updated
        // (disposed) on the next frame after the node was disposed. Only unlink if there are no more instances of
        // this node;
        if ( display && instance.node!.instances.length === 0 ) {

          // @ts-ignore // TODO: fixed once FocusManager is converted to typescript https://github.com/phetsims/scenery/issues/1340
          display.focusManager.pointerHighlightsVisibleProperty.unlink( this.interactiveHighlightingEnabledListener );
        }

        delete this._displays[ instance.trail!.uniqueId ];
      }
    }

    /**
     * Returns true if any nodes from this Node to the leaf of the Trail use Voicing features in some way. In
     * general, we do not want to activate voicing features in this case because the leaf-most Nodes in the Trail
     * should be activated instead.
     * @protected
     */
    getDescendantsUseHighlighting( trail: Trail ): boolean {
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
  };

  /**
   * {Array.<string>} - String keys for all of the allowed options that will be set by Node.mutate( options ), in
   * the order they will be evaluated.
   * @protected
   *
   * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
   *       cases that may apply.
   */
  InteractiveHighlightingClass.prototype._mutatorKeys = INTERACTIVE_HIGHLIGHTING_OPTIONS.concat( InteractiveHighlightingClass.prototype._mutatorKeys );
  assert && assert( InteractiveHighlightingClass.prototype._mutatorKeys.length === _.uniq( InteractiveHighlightingClass.prototype._mutatorKeys ).length, 'duplicate mutator keys in Voicing' );

  return InteractiveHighlightingClass;
};

scenery.register( 'InteractiveHighlighting', InteractiveHighlighting );
export default InteractiveHighlighting;
