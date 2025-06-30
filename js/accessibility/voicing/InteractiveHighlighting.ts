// Copyright 2021-2025, University of Colorado Boulder

/**
 * A trait for Node that mixes functionality to support visual highlights that appear on hover with a pointer.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TEmitter from '../../../../axon/js/TEmitter.js';
import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import TinyProperty from '../../../../axon/js/TinyProperty.js';
import TReadOnlyProperty from '../../../../axon/js/TReadOnlyProperty.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import Focus from '../../accessibility/Focus.js';
import type FocusManager from '../../accessibility/FocusManager.js';
import type Display from '../../display/Display.js';
import type Instance from '../../display/Instance.js';
import type Pointer from '../../input/Pointer.js';
import type SceneryEvent from '../../input/SceneryEvent.js';
import type TInputListener from '../../input/TInputListener.js';
import type { PressListenerEvent } from '../../listeners/PressListener.js';
import type Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import DelayedMutate from '../../util/DelayedMutate.js';
import Trail from '../../util/Trail.js';
import { Highlight } from '../Highlight.js';
import { isInteractiveHighlighting } from './isInteractiveHighlighting.js';

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

// Normally our project prefers type aliases to interfaces, but interfaces are necessary for correct usage of "this", see https://github.com/phetsims/tasks/issues/1132
// eslint-disable-next-line @typescript-eslint/consistent-type-definitions
export interface TInteractiveHighlighting<SuperType extends Node = Node> {

  // @mixin-protected - made public for use in the mixin only
  displays: Record<string, Display>;

  interactiveHighlightChangedEmitter: TEmitter;
  readonly isInteractiveHighlightActiveProperty: TReadOnlyProperty<boolean>;

  // Prefer exported function isInteractiveHighlighting() for better TypeScript support
  readonly _isInteractiveHighlighting: true;

  setInteractiveHighlight( interactiveHighlight: Highlight ): void;

  interactiveHighlight: Highlight;

  getInteractiveHighlight(): Highlight;

  setInteractiveHighlightLayerable( interactiveHighlightLayerable: boolean ): void;

  interactiveHighlightLayerable: boolean;

  getInteractiveHighlightLayerable(): boolean;

  setInteractiveHighlightEnabled( enabled: boolean ): void;

  getInteractiveHighlightEnabled(): boolean;

  interactiveHighlightEnabled: boolean;

  handleHighlightActiveChange(): void;

  onChangedInstance( instance: Instance, added: boolean ): void;

  // @mixin-protected - made public for use in the mixin only
  getDescendantsUseHighlighting( trail: Trail ): boolean;

  unlockHighlight(): void;

  forwardInteractiveHighlight( event: SceneryEvent ): void;

  // Better options type for the subtype implementation that adds mutator keys
  mutate( options?: SelfOptions & Parameters<SuperType[ 'mutate' ]>[ 0 ] ): this;
}

const InteractiveHighlighting = <SuperType extends Constructor<Node>>( Type: SuperType ): SuperType & Constructor<TInteractiveHighlighting<InstanceType<SuperType>>> => {

  // @ts-expect-error
  assert && assert( !Type._mixesInteractiveHighlighting, 'InteractiveHighlighting is already added to this Type' );

  const InteractiveHighlightingClass = DelayedMutate( 'InteractiveHighlightingClass', INTERACTIVE_HIGHLIGHTING_OPTIONS,
    class InteractiveHighlightingClass extends Type implements TInteractiveHighlighting<InstanceType<SuperType>> {

      // Input listener to activate the HighlightOverlay upon pointer input. Uses exit and enter instead of over and out
      // because we do not want this to fire from bubbling. The highlight should be around this Node when it receives
      // input.
      private readonly _activationListener: TInputListener;

      // A reference to the Pointer so that we can add and remove listeners from it when necessary.
      // Since this is on the trait, only one pointer can have a listener for this Node that uses InteractiveHighlighting
      // at one time.
      private _pointer: null | Pointer = null;

      // A reference to a Pointer that we try to lock to when manually activating this Node from a forwarded event.
      // With event forwarding, this can sometimes happen before the Node is added to the scene graph, and so we try
      // to activate the highlight once this Node's list of displays updates.
      private _forwardingHighlightForPointer: null | Pointer = null;

      // A map that collects all of the Displays that this InteractiveHighlighting Node is
      // attached to, mapping the unique ID of the Instance Trail to the Display. We need a reference to the
      // Displays to activate the Focus Property associated with highlighting, and to add/remove listeners when
      // features that require highlighting are enabled/disabled. Note that this is updated asynchronously
      // (with updateDisplay) since Instances are added asynchronously.
      // @mixin-protected - made public for use in the mixin only
      public displays: Record<string, Display> = {};

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

        // This is potentially dangerous to listen to generally, but in this case it is safe because the state we change
        // will only affect a separate display's state, not this one.
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
      public get _isInteractiveHighlighting(): true {
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
            assert && assert( typeof interactiveHighlight === 'object' && !!( interactiveHighlight as Node )._isNode );

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
            assert && assert( typeof this._interactiveHighlight === 'object' && !!( this._interactiveHighlight as Node )._isNode );
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

          // The highlight is activated if
          // 1) The pointer highlights are visible (and the feature is enabled)
          // 2) Alt input highlights are not visible - if they are visible then alt input highlights are used instead
          if (
            display.focusManager.pointerHighlightsVisibleProperty.value &&
            !display.focusManager.pdomFocusHighlightsVisibleProperty.value
          ) {

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

        // The performance of this is OK at the time of this writing. It depends greatly on how often this function is
        // called, since recalculation involves looping through all instances' displays, but since recalculation only
        // occurs from FocusManager's Property updates (and not on every pointer operation), this is acceptable.
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

        this._forwardingHighlightForPointer = null;

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
        const groupHighlightNode = event.trail.nodes.find( node => ( node.groupFocusHighlight && isInteractiveHighlighting( node ) ) );
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
              this.lockHighlight( newFocus, display.focusManager );
              lockPointer = true;
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
                this.lockHighlight( newFocus, display.focusManager );
                lockPointer = true;
              }
            }
          }
        }

        if ( lockPointer ) {
          this.savePointer( event.pointer );
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
              assert && assert( !focus.trail.lastNode().isDisposed, 'Focus should not be set to a disposed Node' );

              // Set the lockedPointerFocusProperty with a copy of the Focus (as deep as possible) because we want
              // to keep a reference to the old Trail while pointerFocusProperty changes.
              this.lockHighlight( focus, display.focusManager );
              lockPointer = true;
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

        // Attempt to activate the highlight arround this Node as soon as it is added to a display
        // because it may have been activated before it was added to the scene graph.
        if ( this._forwardingHighlightForPointer ) {

          // Only try to forward a highlight if the pointer Focus isn't already set.
          if ( display.focusManager.pointerFocusProperty.value === null ) {
            this.manualActivateAndLock( this._forwardingHighlightForPointer );
          }
          this._forwardingHighlightForPointer = null;
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
       * Sets the "locked" focus for Interactive Highlighting. The "locking" makes sure that the highlight remains
       * active on the Node that is receiving interaction even when the pointer has move away from the Node
       * (but presumably is still down somewhere else on the screen).
       */
      private lockHighlight( newFocus: Focus, focusManager: FocusManager ): void {

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
      }

      /**
       * Remove the highlight from this Node and unlock the highlight so that a new highlight can be activated.
       * Useful for cases like event forwarding where clicking on one Node should begin interaction with another.
       *
       * (scenery-internal)
       */
      public unlockHighlight(): void {
        Object.values( this.displays ).forEach( display => {
          display.focusManager.pointerFocusProperty.value = null;
        } );
        this.handleLockedPointerFocusCleared( null );
      }

      public forwardInteractiveHighlight( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent> ): void {
        if ( Object.values( this.displays ).length === 0 ) {

          // If this target is not displayed, try to activate when the Node is added to the scene graph.
          this._forwardingHighlightForPointer = event.pointer;
        }
        else {
          Object.values( this.displays ).forEach( display => {

            // Only try to forward a highlight if the pointer Focus isn't already set.
            if ( display.focusManager.pointerFocusProperty.value === null ) {
              this.manualActivateAndLock( event.pointer );
            }
          } );
        }
      }

      /**
       * Manually attempt to activate the highlight for this Node. Used when forwarding a highlight
       * from one target to another with event forwarding.
       */
      private manualActivateAndLock( pointer: Pointer ): void {

        // Trails to this interactive highlighting Node which we want to activate (different from Pointer trail).
        const trails = this.getTrails();

        // This strategy does not work for DAG, but it is graceful otherwise.
        if ( trails.length === 1 ) {
          const trail = trails[ 0 ];
          const displays = Object.values( this.displays );
          for ( let i = 0; i < displays.length; i++ ) {
            const display = displays[ i ];
            const forwardFocus = new Focus( display, trail );

            // Set the pointer focus to this new target and lock.
            display.focusManager.pointerFocusProperty.set( forwardFocus );
            this.lockHighlight( forwardFocus, display.focusManager );
            this.savePointer( pointer );
          }
        }
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
        assert && assert( instance.display, 'should have a display' );

        const uniqueId = instance.trail!.uniqueId;

        if ( added ) {
          const display = instance.display!;
          this.displays[ uniqueId ] = display;
          this.onDisplayAdded( display );
        }
        else {
          assert && assert( instance.node, 'should have a node' );
          const display = this.displays[ uniqueId ];
          assert && assert( display, `interactive highlighting does not have a Display for removed instance: ${uniqueId}` );

          // If the node was disposed, this display reference has already been cleaned up, but instances are updated
          // (disposed) on the next frame after the node was disposed. Only unlink if there are no more instances of
          // this node;
          instance.node!.instances.length === 0 && this.onDisplayRemoved( display );
          delete this.displays[ uniqueId ];
        }
      }

      /**
       * Returns true if any nodes from this Node to the leaf of the Trail use Voicing features in some way. In
       * general, we do not want to activate voicing features in this case because the leaf-most Nodes in the Trail
       * should be activated instead.
       * @mixin-protected - made public for use in the mixin only
       */
      public getDescendantsUseHighlighting( trail: Trail ): boolean {
        const indexOfSelf = trail.nodes.indexOf( this );

        // all the way to length, end not included in slice - and if start value is greater than index range
        // an empty array is returned
        const childToLeafNodes = trail.nodes.slice( indexOfSelf + 1, trail.nodes.length );

        // if any of the nodes from leaf to self use InteractiveHighlighting, they should receive input, and we shouldn't
        // speak the content for this Node
        let descendantsUseVoicing = false;
        for ( let i = 0; i < childToLeafNodes.length; i++ ) {
          if ( isInteractiveHighlighting( childToLeafNodes[ i ] ) ) {
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
};

/**
 * Attempt to forward an interactive highlight to a target Node. Usually useful when forwarding events from
 * one Node to another (like DragListener.createForwardingListener).
 * @static
 */
InteractiveHighlighting.forwardInteractiveHighlightFromPress = ( targetNode: Node, event: PressListenerEvent ): void => {
  const targetTrails = targetNode.getLeafTrails();
  targetTrails.some( trail => {
    return trail.nodes.some( node => {
      if ( isInteractiveHighlighting( node ) ) {
        node.forwardInteractiveHighlight( event );
        return true;
      }
      return false;
    } );
  } );
};

// NOTE!!! This used to be called "InteractiveHighlightingNode", which conflicts with (or is confusing) with the actual
// InteractiveHighlightingNode.ts type. Renamed here so they can be in the same namespace.
export type InteractiveHighlightingNodeType = Node & TInteractiveHighlighting;

scenery.register( 'InteractiveHighlighting', InteractiveHighlighting );
export default InteractiveHighlighting;