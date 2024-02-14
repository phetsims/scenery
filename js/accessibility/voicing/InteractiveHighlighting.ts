// Copyright 2021-2024, University of Colorado Boulder

/**
 * A trait for Node that mixes functionality to support visual highlights that appear on hover with a pointer.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import { DelayedMutate, Display, Focus, FocusManager, Instance, Node, Pointer, PressListener, scenery, SceneryEvent, TInputListener, Trail } from '../../imports.js';
import { Highlight } from '../../overlays/HighlightOverlay.js';
import TEmitter from '../../../../axon/js/TEmitter.js';
import memoize from '../../../../phet-core/js/memoize.js';
import TReadOnlyProperty from '../../../../axon/js/TReadOnlyProperty.js';
import TinyProperty from '../../../../axon/js/TinyProperty.js';

// constants
// option keys for InteractiveHighlighting, each of these will have a setter and getter and values are applied with mutate()
const INTERACTIVE_HIGHLIGHTING_OPTIONS = [
  'interactiveHighlight',
  'interactiveHighlightLayerable',
  'interactiveHighlightEnabled'
];

type SelfOptions = {
  interactiveHighlight?: Highlight;
  interactiveHighlightLayerable?: boolean;
  interactiveHighlightEnabled?: boolean;
};

export type InteractiveHighlightingOptions = SelfOptions;

const InteractiveHighlighting = memoize( <SuperType extends Constructor<Node>>( Type: SuperType ) => {

  // @ts-expect-error
  assert && assert( !Type._mixesInteractiveHighlighting, 'InteractiveHighlighting is already added to this Type' );

  const InteractiveHighlightingClass = DelayedMutate( 'InteractiveHighlightingClass', INTERACTIVE_HIGHLIGHTING_OPTIONS, class InteractiveHighlightingClass extends Type {

    // Input listener to activate the HighlightOverlay upon pointer input. Uses exit and enter instead of over and out
    // because we do not want this to fire from bubbling. The highlight should be around this Node when it receives
    // input.
    private readonly _activationListener: TInputListener;

    // A reference to the Pointer so that we can add and remove listeners from it when necessary.
    // Since this is on the trait, only one pointer can have a listener for this Node that uses InteractiveHighlighting
    // at one time.
    private _pointer: null | Pointer = null;

    // A map that collects all of the Displays that this InteractiveHighlighting Node is
    // attached to, mapping the unique ID of the Instance Trail to the Display. We need a reference to the
    // Displays to activate the Focus Property associated with highlighting, and to add/remove listeners when
    // features that require highlighting are enabled/disabled. Note that this is updated asynchronously
    // (with updateDisplay) since Instances are added asynchronously.
    protected displays: Record<string, Display> = {};

    // The highlight that will surround this Node when it is activated and a Pointer is currently over it. When
    // null, the focus highlight will be used (as defined in ParallelDOM.js).
    private _interactiveHighlight: Highlight = null;

    // If true, the highlight will be layerable in the scene graph instead of drawn
    // above everything in the HighlightOverlay. If true, you are responsible for adding the interactiveHighlight
    // in the location you want in the scene graph. The interactiveHighlight will become visible when
    // this.isInteractiveHighlightActiveProperty is true.
    private _interactiveHighlightLayerable = false;

    // If true, the highlight will be displayed on activation input. If false, it will not and we can remove listeners
    // that would do this work.
    private _interactiveHighlightEnabled = true;

    // Emits an event when the interactive highlight changes for this Node
    public interactiveHighlightChangedEmitter: TEmitter = new TinyEmitter();

    // This Property will be true when this node has highlights activated on it. See isInteractiveHighlightActivated().
    public readonly isInteractiveHighlightActiveProperty: TReadOnlyProperty<boolean>;
    private readonly _isInteractiveHighlightActiveProperty = new TinyProperty( false );

    // When new instances of this Node are created, adds an entry to the map of Displays.
    private readonly _changedInstanceListener: ( instance: Instance, added: boolean ) => void;

    // Listener that adds/removes other listeners that activate highlights when
    // the feature becomes enabled/disabled so that we don't do extra work related to highlighting unless
    // it is necessary.
    private readonly _interactiveHighlightingEnabledListener: ( enabled: boolean ) => void;

    // A listener that is added to the FocusManager.lockedPointerFocusProperty to clear this._pointer and its listeners from
    // this instance when the lockedPointerFocusProperty is set to null externally (not by InteractiveHighlighting).
    private readonly _boundPointerFocusClearedListener: ( lockedPointerFocus: Focus | null ) => void;

    // Input listener that locks the HighlightOverlay so that there are no updates to the highlight
    // while the pointer is down over something that uses InteractiveHighlighting.
    private readonly _pointerListener: TInputListener;

    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );

      this._activationListener = {
        enter: this._onPointerEntered.bind( this ),
        over: this._onPointerOver.bind( this ),
        move: this._onPointerMove.bind( this ),
        exit: this._onPointerExited.bind( this ),
        down: this._onPointerDown.bind( this )
      };

      this._changedInstanceListener = this.onChangedInstance.bind( this );
      this.changedInstanceEmitter.addListener( this._changedInstanceListener );

      this._interactiveHighlightingEnabledListener = this._onInteractiveHighlightingEnabledChange.bind( this );
      this._boundPointerFocusClearedListener = this.handleLockedPointerFocusCleared.bind( this );

      const boundPointerReleaseListener = this._onPointerRelease.bind( this );
      const boundPointerCancel = this._onPointerCancel.bind( this );

      this._pointerListener = {
        up: boundPointerReleaseListener,
        cancel: boundPointerCancel,
        interrupt: boundPointerCancel
      };

      this.isInteractiveHighlightActiveProperty = this._isInteractiveHighlightActiveProperty;
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
          assert && assert( interactiveHighlight instanceof Node ); // eslint-disable-line no-simple-type-checking-assertions

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
     * Set the enabled state of Interactive Highlights on this Node. When false, highlights will not activate
     * on this Node with mouse and touch input. You can also disable Interactive Highlights by making the node
     * pickable: false. Use this when you want to disable Interactive Highlights without modifying pickability.
     */
    public setInteractiveHighlightEnabled( enabled: boolean ): void {
      this._interactiveHighlightEnabled = enabled;

      // Each display has its own focusManager.pointerHighlightsVisibleProperty, so we need to go through all of them
      // and update after this enabled change
      const trailIds = Object.keys( this.displays );
      for ( let i = 0; i < trailIds.length; i++ ) {
        const display = this.displays[ trailIds[ i ] ];
        this._interactiveHighlightingEnabledListener( display.focusManager.pointerHighlightsVisibleProperty.value );
      }
    }

    /**
     * Are Interactive Highlights enabled for this Node? When false, no highlights activate from mouse and touch.
     */
    public getInteractiveHighlightEnabled(): boolean {
      return this._interactiveHighlightEnabled;
    }

    public set interactiveHighlightEnabled( enabled: boolean ) { this.setInteractiveHighlightEnabled( enabled ); }

    public get interactiveHighlightEnabled(): boolean { return this.getInteractiveHighlightEnabled(); }

    /**
     * Returns true if this Node is "activated" by a pointer, indicating that a Pointer is over it
     * and this Node mixes InteractiveHighlighting so an interactive highlight should surround it.
     *
     * This algorithm depends on the direct focus over the pointer, the "locked" focus (from an attached listener),
     * and if pointer highlights are visible at all.
     *
     * If you come to desire this private function, instead you should use isInteractiveHighlightActiveProperty.
     *
     */
    private isInteractiveHighlightActivated(): boolean {
      let activated = false;

      const trailIds = Object.keys( this.displays );
      for ( let i = 0; i < trailIds.length; i++ ) {
        const display = this.displays[ trailIds[ i ] ];

        // Only if the interactive highlights feature is enabled can we be active
        if ( display.focusManager.pointerHighlightsVisibleProperty.value ) {

          const pointerFocus = display.focusManager.pointerFocusProperty.value;
          const lockedPointerFocus = display.focusManager.lockedPointerFocusProperty.value;
          if ( lockedPointerFocus ) {
            if ( lockedPointerFocus?.trail.lastNode() === this ) {
              activated = true;
              break;
            }
          }
          else if ( pointerFocus?.trail.lastNode() === this ) {
            activated = true;
            break;
          }
        }
      }
      return activated;
    }

    public handleHighlightActiveChange(): void {

      // The performance of this is OK - what we
      this._isInteractiveHighlightActiveProperty.value = this.isInteractiveHighlightActivated();
    }

    public override dispose(): void {
      this.changedInstanceEmitter.removeListener( this._changedInstanceListener );

      // remove the activation listener if it is currently attached
      if ( this.hasInputListener( this._activationListener ) ) {
        this.removeInputListener( this._activationListener );
      }

      if ( this._pointer ) {
        this._pointer.removeInputListener( this._pointerListener );
        this._pointer = null;
      }

      // remove listeners on displays and remove Displays from the map
      const trailIds = Object.keys( this.displays );
      for ( let i = 0; i < trailIds.length; i++ ) {
        const display = this.displays[ trailIds[ i ] ];
        this.onDisplayRemoved( display );
        delete this.displays[ trailIds[ i ] ];
      }

      super.dispose && super.dispose();
    }

    /**
     * When the pointer goes 'over' a node (not including children), look for a group focus highlight to
     * activate. This is most useful for InteractiveHighlighting Nodes that act as a "group" container
     * for other nodes. When the pointer leaves a child, we get the 'exited' event on the child, immediately
     * followed by an 'over' event on the parent. This keeps the group highlight visible without any flickering.
     * The group parent must be composed with InteractiveHighlighting so that it has these event listeners.
     */
    private _onPointerOver( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      // If there is an ancestor that is a group focus highlight that is composed with InteractiveHighlight (
      // (should activate with pointer input)...
      const groupHighlightNode = event.trail.nodes.find( node => ( node.groupFocusHighlight && ( node as InteractiveHighlightingNode ).isInteractiveHighlighting ) );
      if ( groupHighlightNode ) {

        // trail to the group highlight Node
        const rootToGroupNode = event.trail.subtrailTo( groupHighlightNode );
        const displays = Object.values( this.displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];

          // only set focus if current Pointer focus is not defined (from a more descendant Node)
          if ( display.focusManager.pointerFocusProperty.value === null ) {
            display.focusManager.pointerFocusProperty.set( new Focus( display, rootToGroupNode ) );
          }
        }
      }
    }

    /**
     * When a Pointer enters this Node, signal to the Displays that the pointer is over this Node so that the
     * HighlightOverlay can be activated.
     *
     * This is most likely how most pointerFocusProperty is set. First we get an `enter` event then we may get
     * a down event or move event which could do further updates on the event Pointer or FocusManager.
     */
    private _onPointerEntered( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      let lockPointer = false;

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];

        if ( display.focusManager.pointerFocusProperty.value === null ||
             !event.trail.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {

          const newFocus = new Focus( display, event.trail );
          display.focusManager.pointerFocusProperty.set( newFocus );
          if ( display.focusManager.lockedPointerFocusProperty.value === null && event.pointer.attachedListener ) {
            lockPointer = this.attemptHighlightLock( newFocus, display.focusManager, event.pointer );
          }
        }
      }

      if ( lockPointer ) {
        this.savePointer( event.pointer );
      }
    }

    /**
     * Update highlights when the Pointer moves over this Node. In general, highlights will activate on 'enter'. But
     * in cases where multiple Nodes in a Trail support InteractiveHighlighting this listener can move focus
     * to the most reasonable target (the closest ancestor or descendent that is composed with InteractiveHighlighting).
     */
    private _onPointerMove( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {
      let lockPointer = false;

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];

        // the SceneryEvent might have gone through a descendant of this Node
        const rootToSelf = event.trail.subtrailTo( this );

        // only do more work on move if the event indicates that pointer focus might have changed.
        if ( display.focusManager.pointerFocusProperty.value === null || !rootToSelf.equals( display.focusManager.pointerFocusProperty.value.trail ) ) {

          if ( !this.getDescendantsUseHighlighting( event.trail ) ) {
            const newFocus = new Focus( display, rootToSelf );
            display.focusManager.pointerFocusProperty.set( newFocus );

            if ( display.focusManager.lockedPointerFocusProperty.value === null && event.pointer.attachedListener ) {
              lockPointer = this.attemptHighlightLock( newFocus, display.focusManager, event.pointer );
            }
          }
        }

        if ( lockPointer ) {
          this.savePointer( event.pointer );
        }
      }
    }

    /**
     * When a pointer exits this Node or its children, signal to the Displays that pointer focus has changed to
     * deactivate the HighlightOverlay. This can also fire when visibility/pickability of the Node changes.
     */
    private _onPointerExited( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      const displays = Object.values( this.displays );
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];
        display.focusManager.pointerFocusProperty.set( null );

        // An exit event may come from a Node along the trail becoming invisible or unpickable. In that case unlock
        // focus and remove pointer listeners so that highlights can continue to update from new input.
        const lockedPointerFocus = display.focusManager.lockedPointerFocusProperty.value;
        if ( !event.trail.isPickable() &&
             ( lockedPointerFocus === null ||

               // We do not want to remove the lockedPointerFocus if this event trail has nothing
               // to do with the node that is receiving a locked focus.
               event.trail.containsNode( lockedPointerFocus.trail.lastNode() ) ) ) {

          // unlock and remove pointer listeners
          this._onPointerRelease( event );
        }
      }
    }

    /**
     * When a pointer goes down on this Node, signal to the Displays that the pointerFocus is locked. On the down
     * event, the pointerFocusProperty will have been set first from the `enter` event.
     */
    private _onPointerDown( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {

      if ( this._pointer === null ) {
        let lockPointer = false;

        const displays = Object.values( this.displays );
        for ( let i = 0; i < displays.length; i++ ) {
          const display = displays[ i ];
          const focus = display.focusManager.pointerFocusProperty.value;
          const locked = !!display.focusManager.lockedPointerFocusProperty.value;

          // Focus should generally be defined when pointer enters the Node, but it may be null in cases of
          // cancel or interrupt. Don't attempt to lock if the FocusManager already has a locked highlight (especially
          // important for gracefully handling multitouch).
          if ( focus && !locked ) {

            // Set the lockedPointerFocusProperty with a copy of the Focus (as deep as possible) because we want
            // to keep a reference to the old Trail while pointerFocusProperty changes.
            lockPointer = this.attemptHighlightLock( focus, display.focusManager, event.pointer );
          }
        }

        if ( lockPointer ) {
          this.savePointer( event.pointer );
        }
      }
    }

    private onDisplayAdded( display: Display ): void {

      // Listener may already by on the display in cases of DAG, only add if this is the first instance of this Node
      if ( !display.focusManager.pointerHighlightsVisibleProperty.hasListener( this._interactiveHighlightingEnabledListener ) ) {
        display.focusManager.pointerHighlightsVisibleProperty.link( this._interactiveHighlightingEnabledListener );
      }
    }

    private onDisplayRemoved( display: Display ): void {

      // Pointer focus was locked due to interaction with this listener, but unlocked because of other
      // scenery-internal listeners. But the Property still has this listener so it needs to be removed now.
      if ( display.focusManager.lockedPointerFocusProperty.hasListener( this._boundPointerFocusClearedListener ) ) {
        display.focusManager.lockedPointerFocusProperty.unlink( this._boundPointerFocusClearedListener );
      }

      display.focusManager.pointerHighlightsVisibleProperty.unlink( this._interactiveHighlightingEnabledListener );
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

        // Unlink the listener that was watching for the lockedPointerFocusProperty to be cleared externally
        if ( display.focusManager.lockedPointerFocusProperty.hasListener( this._boundPointerFocusClearedListener ) ) {
          display.focusManager.lockedPointerFocusProperty.unlink( this._boundPointerFocusClearedListener );
        }
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
     * Save the Pointer and add a listener to it to remove highlights when a pointer is released/cancelled.
     */
    private savePointer( eventPointer: Pointer ): void {
      assert && assert(
        this._pointer === null,
        'It should be impossible to already have a Pointer before locking from touchSnag'
      );

      this._pointer = eventPointer;
      this._pointer.addInputListener( this._pointerListener );
    }

    /**
     * May set the lockedPointerFocusProperty for a FocusManager if the provided Pointer indicates that this should
     * be done. The "locking" makes sure that the highlight remains active on the Node that is receiving interaction
     * even when the pointer has move away from the Node (but presumably is still down somewhere else on the screen).
     * Returns true when the lockedPointerFocusProperty is set to a new Focus so that use cases can do more work
     * in this case.
     */
    private attemptHighlightLock( newFocus: Focus, focusManager: FocusManager, eventPointer: Pointer ): boolean {
      let pointerLock = false;

      // If the event Pointer is attached to a PressListener there is some activation input happening, so
      // we should "lock" the highlight to this target until the pointer is released.
      if ( eventPointer.attachedListener && eventPointer.attachedListener.listener instanceof PressListener ) {
        assert && assert( this._pointer === null,
          'It should be impossible to already have a Pointer before locking from touchSnag' );

        // A COPY of the focus is saved to the Property because we need the value of the Trail at this event.
        focusManager.lockedPointerFocusProperty.set( new Focus( newFocus.display, newFocus.trail.copy() ) );

        // Attach a listener that will clear the pointer and its listener if the lockedPointerFocusProperty is cleared
        // externally (not by InteractiveHighlighting).
        assert && assert( !focusManager.lockedPointerFocusProperty.hasListener( this._boundPointerFocusClearedListener ),
          'this listener still on the lockedPointerFocusProperty indicates a memory leak'
        );
        focusManager.lockedPointerFocusProperty.link( this._boundPointerFocusClearedListener );

        pointerLock = true;
      }

      return pointerLock;
    }

    /**
     * FocusManager.lockedPointerFocusProperty does not belong to InteractiveHighlighting and can be cleared
     * for any reason. If it is set to null while a pointer is down we need to release the Pointer and remove input
     * listeners.
     */
    private handleLockedPointerFocusCleared( lockedPointerFocus: Focus | null ): void {
      if ( lockedPointerFocus === null ) {
        this._onPointerRelease();
      }
    }

    /**
     * Add or remove listeners related to activating interactive highlighting when the feature becomes enabled.
     * Work related to interactive highlighting is avoided unless the feature is enabled.
     */
    private _onInteractiveHighlightingEnabledChange( featureEnabled: boolean ): void {
      // Only listen to the activation listener if the feature is enabled and highlighting is enabled for this Node.
      const enabled = featureEnabled && this._interactiveHighlightEnabled;

      const hasActivationListener = this.hasInputListener( this._activationListener );
      if ( enabled && !hasActivationListener ) {
        this.addInputListener( this._activationListener );
      }
      else if ( !enabled && hasActivationListener ) {
        this.removeInputListener( this._activationListener );
      }

      // If now displayed, then we should recompute if we are active or not.
      this.handleHighlightActiveChange();
    }

    /**
     * Add the Display to the collection when this Node is added to a scene graph. Also adds listeners to the
     * Display that turns on highlighting when the feature is enabled.
     */
    public onChangedInstance( instance: Instance, added: boolean ): void {
      assert && assert( instance.trail, 'should have a trail' );

      if ( added ) {
        this.displays[ instance.trail!.uniqueId ] = instance.display;

        this.onDisplayAdded( instance.display );
      }
      else {
        assert && assert( instance.node, 'should have a node' );
        const display = this.displays[ instance.trail!.uniqueId ];

        // If the node was disposed, this display reference has already been cleaned up, but instances are updated
        // (disposed) on the next frame after the node was disposed. Only unlink if there are no more instances of
        // this node;
        instance.node!.instances.length === 0 && this.onDisplayRemoved( display );
        delete this.displays[ instance.trail!.uniqueId ];
      }
    }

    /**
     * Returns true if any nodes from this Node to the leaf of the Trail use Voicing features in some way. In
     * general, we do not want to activate voicing features in this case because the leaf-most Nodes in the Trail
     * should be activated instead.
     */
    protected getDescendantsUseHighlighting( trail: Trail ): boolean {
      const indexOfSelf = trail.nodes.indexOf( this );

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

    public override mutate( options?: SelfOptions & Parameters<InstanceType<SuperType>[ 'mutate' ]>[ 0 ] ): this {
      return super.mutate( options );
    }
  } );

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
} );

// Provides a way to determine if a Node is composed with InteractiveHighlighting by type
const wrapper = () => InteractiveHighlighting( Node );
export type InteractiveHighlightingNode = InstanceType<ReturnType<typeof wrapper>>;

scenery.register( 'InteractiveHighlighting', InteractiveHighlighting );
export default InteractiveHighlighting;
