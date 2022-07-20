// Copyright 2021-2022, University of Colorado Boulder

/**
 * A trait for Node that mixes functionality to support visual highlights that appear on hover with a pointer.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import { Display, Focus, IInputListener, Instance, Node, NodeOptions, Pointer, scenery, SceneryEvent, Trail } from '../../imports.js';
import { Highlight } from '../../overlays/HighlightOverlay.js';
import IEmitter from '../../../../axon/js/IEmitter.js';

// constants
// option keys for InteractiveHighlighting, each of these will have a setter and getter and values are applied with mutate()
const INTERACTIVE_HIGHLIGHTING_OPTIONS = [
  'interactiveHighlight',
  'interactiveHighlightLayerable'
];

type SelfOptions = {
  interactiveHighlight?: Highlight;
  interactiveHighlightLayerable?: boolean;
};

export type InteractiveHighlightingOptions = SelfOptions;

/**
 * @param Type
 * @param optionsArgPosition - zero-indexed number that the options argument is provided at
 */
const InteractiveHighlighting = <SuperType extends Constructor>( Type: SuperType, optionsArgPosition: number ) => {
  assert && assert( typeof optionsArgPosition === 'number', 'Must provide an index to access options arg from (zero-indexed)' );
  assert && assert( _.includes( inheritance( Type ), Node ), 'Only Node subtypes should compose InteractiveHighlighting' );

  // @ts-ignore
  assert && assert( !Type._mixesInteractiveHighlighting, 'InteractiveHighlighting is already added to this Type' );

  class InteractiveHighlightingClass extends Type {

    // Input listener to activate the HighlightOverlay upon pointer input. Uses exit and enter instead of over and out
    // because we do not want this to fire from bubbling. The highlight should be around this Node when it receives
    // input.
    private readonly _activationListener: IInputListener;

    // A reference to the Pointer so that we can add and remove listeners from it when necessary.
    // Since this is on the trait, only one pointer can have a listener for this Node that uses InteractiveHighlighting
    // at one time.
    private _pointer: null | Pointer;

    // A map that collects all of the Displays that this InteractiveHighlighting Node is
    // attached to, mapping the unique ID of the Instance Trail to the Display. We need a reference to the
    // Displays to activate the Focus Property associated with highlighting, and to add/remove listeners when
    // features that require highlighting are enabled/disabled. Note that this is updated asynchronously
    // (with updateDisplay) since Instances are added asynchronously.
    protected displays: Record<string, Display>;

    // The highlight that will surround this Node when it is activated and a Pointer is currently over it. When
    // null, the focus highlight will be used (as defined in ParallelDOM.js).
    private _interactiveHighlight: Highlight;

    // If true, the highlight will be layerable in the scene graph instead of drawn
    // above everything in the HighlightOverlay. If true, you are responsible for adding the interactiveHighlight
    // in the location you want in the scene graph. The interactiveHighlight will become visible when
    // this.interactiveHighlightActivated is true.
    private _interactiveHighlightLayerable: boolean;

    // Emits an event when the interactive highlight changes for this Node
    public interactiveHighlightChangedEmitter: IEmitter;

    // When new instances of this Node are created, adds an entry to the map of Displays.
    private readonly _changedInstanceListener: ( instance: Instance, added: boolean ) => void;

    // Listener that adds/removes other listeners that activate highlights when
    // the feature becomes enabled/disabled so that we don't do extra work related to highlighting unless
    // it is necessary.
    private readonly _interactiveHighlightingEnabledListener: ( enabled: boolean ) => void;

    // Input listener that locks the HighlightOverlay so that there are no updates to the highlight
    // while the pointer is down over something that uses InteractiveHighlighting.
    private readonly _pointerListener: IInputListener;

    public constructor( ...args: IntentionalAny[] ) {

      const providedOptions = ( args[ optionsArgPosition ] || {} ) as InteractiveHighlightingOptions;

      const interactiveHighlightingOptions = _.pick( providedOptions, INTERACTIVE_HIGHLIGHTING_OPTIONS );
      args[ optionsArgPosition ] = _.omit( providedOptions, INTERACTIVE_HIGHLIGHTING_OPTIONS );

      super( ...args );

      this._activationListener = {
        enter: this._onPointerEntered.bind( this ),
        move: this._onPointerMove.bind( this ),
        exit: this._onPointerExited.bind( this ),
        down: this._onPointerDown.bind( this )
      };

      this._pointer = null;
      this.displays = {};
      this._interactiveHighlight = null;
      this._interactiveHighlightLayerable = false;
      this.interactiveHighlightChangedEmitter = new TinyEmitter();

      this._changedInstanceListener = this.onChangedInstance.bind( this );
      ( this as unknown as Node ).changedInstanceEmitter.addListener( this._changedInstanceListener );

      this._interactiveHighlightingEnabledListener = this._onInteractiveHighlightingEnabledChange.bind( this );

      const boundPointerReleaseListener = this._onPointerRelease.bind( this );
      const boundPointerCancel = this._onPointerCancel.bind( this );

      this._pointerListener = {
        up: boundPointerReleaseListener,
        cancel: boundPointerCancel,
        interrupt: boundPointerCancel
      };

      ( this as unknown as Node ).mutate( interactiveHighlightingOptions as NodeOptions );
    }

    /**
     * Whether a Node composes InteractiveHighlighting.
     */
    public get isInteractiveHighlighting(): boolean {
      return true;
    }

    public static get _mixesInteractiveHighlighting(): boolean { return true;}

    /**
     * Set the interactive highlight for this node. By default, the highlight will be a pink rectangle that surrounds
     * the node's local bounds.
     */
    public setInteractiveHighlight( interactiveHighlight: Highlight ): void {

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

    public set interactiveHighlight( interactiveHighlight: Highlight ) { this.setInteractiveHighlight( interactiveHighlight ); }

    public get interactiveHighlight(): Highlight { return this.getInteractiveHighlight(); }

    /**
     * Returns the interactive highlight for this Node.
     */
    public getInteractiveHighlight(): Highlight {
      return this._interactiveHighlight;
    }

    /**
     * Sets whether the highlight is layerable in the scene graph instead of above everything in the
     * highlight overlay. If layerable, you must provide a custom highlight and it must be a Node. The highlight
     * Node will always be invisible unless this Node is activated with a pointer.
     */
    public setInteractiveHighlightLayerable( interactiveHighlightLayerable: boolean ): void {
      if ( this._interactiveHighlightLayerable !== interactiveHighlightLayerable ) {
        this._interactiveHighlightLayerable = interactiveHighlightLayerable;

        if ( this._interactiveHighlight ) {
          assert && assert( this._interactiveHighlight instanceof Node );
          ( this._interactiveHighlight as Node ).visible = false;

          this.interactiveHighlightChangedEmitter.emit();
        }
      }
    }

    public set interactiveHighlightLayerable( interactiveHighlightLayerable: boolean ) { this.setInteractiveHighlightLayerable( interactiveHighlightLayerable ); }

    public get interactiveHighlightLayerable() { return this.getInteractiveHighlightLayerable(); }

    /**
     * Get whether the interactive highlight is layerable in the scene graph.
     */
    public getInteractiveHighlightLayerable(): boolean {
      return this._interactiveHighlightLayerable;
    }

    /**
     * Returns true if this Node is "activated" by a pointer, indicating that a Pointer is over it
     * and this Node mixes InteractiveHighlighting so an interactive highlight should surround it.
     */
    public isInteractiveHighlightActivated(): boolean {
      let activated = false;

      const trailIds = Object.keys( this.displays );
      for ( let i = 0; i < trailIds.length; i++ ) {
        const pointerFocus = this.displays[ trailIds[ i ] ].focusManager.pointerFocusProperty.value;
        if ( pointerFocus && pointerFocus.trail.lastNode() === this as unknown as Node ) {
          activated = true;
          break;
        }
      }
      return activated;
    }

    public get interactiveHighlightActivated(): boolean { return this.isInteractiveHighlightActivated(); }

    public dispose(): void {
      const thisNode = this as unknown as Node;
      thisNode.changedInstanceEmitter.removeListener( this._changedInstanceListener );

      // remove the activation listener if it is currently attached
      if ( thisNode.hasInputListener( this._activationListener ) ) {
        thisNode.removeInputListener( this._activationListener );
      }

      // remove listeners on displays and remove Displays from the map
      const trailIds = Object.keys( this.displays );
      for ( let i = 0; i < trailIds.length; i++ ) {
        const display = this.displays[ trailIds[ i ] ];

        display.focusManager.pointerHighlightsVisibleProperty.unlink( this._interactiveHighlightingEnabledListener );
        delete this.displays[ trailIds[ i ] ];
      }

      // @ts-ignore
      super.dispose && super.dispose();
    }

    /**
     * When a Pointer enters this Node, signal to the Displays that the pointer is over this Node so that the
     * HighlightOverlay can be activated.
     */
    private _onPointerEntered( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];

        if ( display.focusManager.pointerFocusProperty.value === null || !event.trail.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {

          display.focusManager.pointerFocusProperty.set( new Focus( display, event.trail ) );
        }
      }
    }

    private _onPointerMove( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];

        // the SceneryEvent might have gone through a descendant of this Node
        const rootToSelf = event.trail.subtrailTo( this as unknown as Node );

        // only do more work on move if the event indicates that pointer focus might have changed
        if ( display.focusManager.pointerFocusProperty.value === null || !rootToSelf.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {

          if ( !this.getDescendantsUseHighlighting( event.trail ) ) {

            display.focusManager.pointerFocusProperty.set( new Focus( display, rootToSelf ) );
          }
        }
      }
    }

    /**
     * When a pointer exits this Node, signal to the Displays that pointer focus has changed to deactivate
     * the HighlightOverlay.
     */
    private _onPointerExited( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];
        display.focusManager.pointerFocusProperty.set( null );
      }
    }

    /**
     * When a pointer goes down on this Node, signal to the Displays that the pointerFocus is locked
     */
    private _onPointerDown( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      if ( this._pointer === null ) {
        const displays = Object.values( this.displays );
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

        this._pointer = event.pointer;
        this._pointer.addInputListener( this._pointerListener );
      }
    }

    /**
     * When a Pointer goes up after going down on this Node, signal to the Displays that the pointerFocusProperty no
     * longer needs to be locked.
     *
     * @param [event] - may be called during interrupt or cancel, in which case there is no event
     */
    private _onPointerRelease( event?: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];
        display.focusManager.lockedPointerFocusProperty.value = null;
      }

      if ( this._pointer && this._pointer.listeners.includes( this._pointerListener ) ) {
        this._pointer.removeInputListener( this._pointerListener );
        this._pointer = null;
      }
    }

    /**
     * If the pointer listener is cancelled or interrupted, clear focus and remove input listeners.
     */
    private _onPointerCancel( event?: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];
        display.focusManager.pointerFocusProperty.set( null );
      }

      // unlock and remove pointer listeners
      this._onPointerRelease( event );
    }

    /**
     * Add or remove listeners related to activating interactive highlighting when the feature becomes enabled.
     * This way we prevent doing work related to interactive highlighting unless the feature is enabled.
     */
    private _onInteractiveHighlightingEnabledChange( enabled: boolean ): void {
      const thisNode = this as unknown as Node;

      const hasActivationListener = thisNode.hasInputListener( this._activationListener );
      if ( enabled && !hasActivationListener ) {
        thisNode.addInputListener( this._activationListener );
      }
      else if ( !enabled && hasActivationListener ) {
        thisNode.removeInputListener( this._activationListener );
      }
    }

    /**
     * Add the Display to the collection when this Node is added to a scene graph. Also adds listeners to the
     * Display that turns on highlighting when the feature is enabled.
     */
    public onChangedInstance( instance: Instance, added: boolean ): void {
      assert && assert( instance.trail, 'should have a trail' );

      if ( added ) {
        this.displays[ instance.trail!.uniqueId ] = instance.display;

        // Listener may already by on the display in cases of DAG, only add if this is the first instance of this Node
        if ( !instance.display.focusManager.pointerHighlightsVisibleProperty.hasListener( this._interactiveHighlightingEnabledListener ) ) {
          instance.display.focusManager.pointerHighlightsVisibleProperty.link( this._interactiveHighlightingEnabledListener );
        }
      }
      else {
        assert && assert( instance.node, 'should have a node' );
        const display = this.displays[ instance.trail!.uniqueId ];

        // If the node was disposed, this display reference has already been cleaned up, but instances are updated
        // (disposed) on the next frame after the node was disposed. Only unlink if there are no more instances of
        // this node;
        if ( display && instance.node!.instances.length === 0 ) {

          display.focusManager.pointerHighlightsVisibleProperty.unlink( this._interactiveHighlightingEnabledListener );
        }

        delete this.displays[ instance.trail!.uniqueId ];
      }
    }

    /**
     * Returns true if any nodes from this Node to the leaf of the Trail use Voicing features in some way. In
     * general, we do not want to activate voicing features in this case because the leaf-most Nodes in the Trail
     * should be activated instead.
     */
    protected getDescendantsUseHighlighting( trail: Trail ): boolean {
      const indexOfSelf = trail.nodes.indexOf( this as unknown as Node );

      // all the way to length, end not included in slice - and if start value is greater than index range
      // an empty array is returned
      const childToLeafNodes = trail.nodes.slice( indexOfSelf + 1, trail.nodes.length );

      // if any of the nodes from leaf to self use InteractiveHighlighting, they should receive input, and we shouldn't
      // speak the content for this Node
      let descendantsUseVoicing = false;
      for ( let i = 0; i < childToLeafNodes.length; i++ ) {
        if ( ( childToLeafNodes[ i ] as InteractiveHighlightingNode ).isInteractiveHighlighting ) {
          descendantsUseVoicing = true;
          break;
        }
      }

      return descendantsUseVoicing;
    }
  }

  /**
   * {Array.<string>} - String keys for all the allowed options that will be set by Node.mutate( options ), in
   * the order they will be evaluated.
   *
   * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
   *       cases that may apply.
   */
  InteractiveHighlightingClass.prototype._mutatorKeys = INTERACTIVE_HIGHLIGHTING_OPTIONS.concat( InteractiveHighlightingClass.prototype._mutatorKeys );
  assert && assert( InteractiveHighlightingClass.prototype._mutatorKeys.length ===
                    _.uniq( InteractiveHighlightingClass.prototype._mutatorKeys ).length,
    'duplicate mutator keys in InteractiveHighlighting' );

  return InteractiveHighlightingClass;
};

// Provides a way to determine if a Node is composed with InteractiveHighlighting by type
const wrapper = () => InteractiveHighlighting( Node, 0 );
export type InteractiveHighlightingNode = InstanceType<ReturnType<typeof wrapper>>;

scenery.register( 'InteractiveHighlighting', InteractiveHighlighting );
export default InteractiveHighlighting;
